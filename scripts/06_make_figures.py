"""Generate report figures and summary tables (report/figures, report/tables)."""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import xarray as xr  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, LogNorm  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ghana_pm25 import config  # noqa: E402
from ghana_pm25.calibration import haversine  # noqa: E402
from ghana_pm25.grid import admin_rasters, boundaries  # noqa: E402

FIG = config.REPORT / "figures"; FIG.mkdir(parents=True, exist_ok=True)
TAB = config.REPORT / "tables"; TAB.mkdir(parents=True, exist_ok=True)
OUT = config.OUTPUTS

PM_STOPS = [(0, "#fff7d6"), (5, "#fde9a6"), (15, "#fcc868"), (25, "#f8a24a"), (35, "#ef7a3a"), (55, "#d9502f"), (75, "#b32e32"), (100, "#861e3b"), (150, "#58163f"), (250, "#2c0c2e")]
PM_CMAP = LinearSegmentedColormap.from_list("pm", [c for _, c in PM_STOPS])
C1, C2, C3, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#9a988f"
plt.rcParams.update({"font.family": "sans-serif", "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "#e1e0d9", "grid.linewidth": 0.6, "axes.edgecolor": "#8a8880",
                     "axes.titlesize": 10, "axes.titleweight": "bold", "savefig.dpi": 160, "savefig.bbox": "tight", "legend.frameon": False})

MODEL_LABEL = {"cams_raw": "CAMS raw", "cams_linear": "CAMS + linear correction", "paperlike_gbdt": "GBDT: AOD+met+DOY", "lgbm": "M1 LightGBM",
               "xgb_ratio": "M2 XGBoost (CAMS ratio)", "catboost": "M3 CatBoost", "extratrees": "M4 ExtraTrees", "stack": "GH-PM25 ensemble", "stack_rk": "GH-PM25 final"}
SCHEME_LABEL = {"random": "Random 10-fold", "temporal": "Leave-month-out", "site": "Leave-site-out", "cluster": "Leave-city-cluster-out"}


def pm_norm_value(v):
    vals = np.array([s for s, _ in PM_STOPS], float)
    return np.interp(v, vals, np.linspace(0, 1, len(vals)))


def draw_boundaries(ax, level=1, color="#3d3c39", lw=0.5):
    from shapely.geometry import shape
    for ft in boundaries(level)["features"]:
        g = shape(ft["geometry"])
        geoms = [g] if g.geom_type == "Polygon" else list(g.geoms)
        for p in geoms:
            x, y = p.exterior.xy
            ax.plot(x, y, color=color, lw=lw)


def pm_imshow(ax, arr, title=None, vmax=None):
    lons, lats = config.grid_axes()
    im = ax.imshow(pm_norm_value(np.where(np.isfinite(arr), arr, np.nan)), extent=[lons[0] - .005, lons[-1] + .005, lats[-1] - .005, lats[0] + .005],
                   cmap=PM_CMAP, vmin=0, vmax=1, interpolation="nearest")
    draw_boundaries(ax, 1)
    ax.set_xlim(-3.4, 1.35); ax.set_ylim(4.6, 11.3); ax.set_aspect("equal"); ax.grid(False)
    ax.set_xticks([-3, -2, -1, 0, 1]); ax.set_yticks([5, 6, 7, 8, 9, 10, 11])
    if title:
        ax.set_title(title)
    return im


def pm_colorbar(fig, ax_or_axes, label="PM2.5 (µg/m³)"):
    sm = plt.cm.ScalarMappable(cmap=PM_CMAP, norm=plt.Normalize(0, 1))
    cb = fig.colorbar(sm, ax=ax_or_axes, fraction=0.03, pad=0.02)
    vals = [s for s, _ in PM_STOPS]
    cb.set_ticks(np.linspace(0, 1, len(vals))); cb.set_ticklabels([str(v) for v in vals]); cb.set_label(label)


# ---------------------------------------------------------------------------
def fig_network(stations, table):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.6), gridspec_kw=dict(width_ratios=[1.35, 1]))
    ax = axes[0]
    nets = table.groupby("network").size().sort_values(ascending=False).index.tolist()
    cols = dict(zip(nets, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948", "#6f6d67", "#9e7b4f", "#35a7c4"]))
    n = table.groupby("site_id").size()
    st = stations[stations.site_id.isin(n.index)].copy(); st["n"] = st.site_id.map(n)
    draw_boundaries(ax, 0, lw=0.8)
    for net, g in st.groupby("network"):
        ref = g.is_reference.any()
        ax.scatter(g.lon, g.lat, s=12 + g.n / 12, c=cols.get(net, GREY), marker="^" if ref else "o", edgecolor="white", lw=0.6, label=f"{net} ({len(g)})", zorder=3)
    ax.set_xlim(-9, 9); ax.set_ylim(4, 13.5); ax.set_aspect("equal")
    ax.set_title("a) Ground monitors used for training (marker size ∝ station-days; ▲ BAM-1020 reference)")
    ax.legend(loc="upper right", fontsize=7, ncol=2)
    ax2 = axes[1]
    gh = st[st.country == "GH"]
    draw_boundaries(ax2, 1, lw=0.4)
    for net, g in gh.groupby("network"):
        ax2.scatter(g.lon, g.lat, s=14 + g.n / 10, c=cols.get(net, GREY), marker="^" if g.is_reference.any() else "o", edgecolor="white", lw=0.6, zorder=3)
    ax2.set_xlim(-3.4, 1.35); ax2.set_ylim(4.6, 11.3); ax2.set_aspect("equal")
    ax2.set_title(f"b) Ghana: {len(gh)} sites, {int(gh.n.sum()):,} station-days")
    fig.savefig(FIG / "fig01_network.png"); plt.close(fig)


def fig_timeline(table):
    t = table.copy(); t["month"] = t.date.dt.to_period("M").dt.to_timestamp()
    piv = t.assign(grp=t.country + " · " + t.network).groupby(["month", "grp"]).size().unstack(fill_value=0)
    top = piv.sum().sort_values(ascending=False).index[:8]
    piv = pd.concat([piv[top], piv.drop(columns=top).sum(axis=1).rename("Other")], axis=1)
    fig, ax = plt.subplots(figsize=(12, 3.6))
    cols = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948", "#b9b7ae"]
    ax.stackplot(piv.index, piv.T.values, labels=piv.columns, colors=cols[: piv.shape[1]], lw=0)
    ax.set_ylabel("station-days per month"); ax.legend(ncol=3, fontsize=7, loc="upper left")
    ax.set_title("Training data availability by country and network (after QC and calibration screening)")
    fig.savefig(FIG / "fig02_timeline.png"); plt.close(fig)


def fig_calibration():
    p = pd.read_parquet(OUT / "calibration_pairs.parquet")
    models = json.loads((config.MODELS / "lcs_calibration.json").read_text())
    from ghana_pm25.calibration import _predict
    keys = [k for k in p.net_key.value_counts().index if f"net:{k}" in models][:4]
    fig, axes = plt.subplots(2, len(keys), figsize=(3.6 * len(keys), 7), squeeze=False)
    for j, k in enumerate(keys):
        g = p[p.net_key == k]; m = models[f"net:{k}"]
        cal = _predict(m, g.pm25.to_numpy(), g.rh.to_numpy())
        for i, (x, lab) in enumerate([(g.pm25, "raw sensor"), (cal, f"calibrated ({m['kind']})")]):
            ax = axes[i, j]
            ax.scatter(g.pm25_ref, x, s=5, alpha=0.35, c=C1 if i else C2, lw=0)
            lim = max(g.pm25_ref.quantile(0.995), np.quantile(x, 0.995)) * 1.05
            ax.plot([0, lim], [0, lim], color="#6f6d67", lw=0.8); ax.set_xlim(0, lim); ax.set_ylim(0, lim)
            r2 = 1 - np.sum((np.asarray(x) - g.pm25_ref) ** 2) / np.sum((g.pm25_ref - g.pm25_ref.mean()) ** 2)
            mae = np.mean(np.abs(np.asarray(x) - g.pm25_ref))
            ax.set_title(f"{k.replace('|', ' · ')} — {lab}", fontsize=8.5)
            ax.text(0.03, 0.95, f"n={len(g)}  R²={r2:.2f}  MAE={mae:.1f}", transform=ax.transAxes, va="top", fontsize=8)
            ax.set_xlabel("BAM-1020 reference (µg/m³)"); ax.set_ylabel("sensor (µg/m³)")
    fig.suptitle("Low-cost sensor harmonisation against US Embassy BAM-1020 (≤10 km, same day; in-sample fit shown)", fontweight="bold")
    fig.tight_layout(); fig.savefig(FIG / "fig03_calibration.png"); plt.close(fig)


def fig_static():
    ds = xr.open_dataset(config.PROCESSED / "static" / "static_ghana.nc")
    layers = [("elev", "Elevation (m)", "cividis"), ("lc_tree", "Tree cover fraction", "Greens"), ("lc_crop", "Cropland fraction", "YlOrBr"),
              ("lc_built", "Built-up fraction", "Greys"), ("log_pop", "log(1+population/km²)", "magma_r"), ("log_ntl", "log(1+night lights)", "inferno"),
              ("road_density_s3km", "Major-road density, 3-km (km/km²)", "Blues"), ("dist_coast", "Distance to coast (km)", "PuBu_r")]
    fig, axes = plt.subplots(2, 4, figsize=(14, 8.5))
    for ax, (v, t, cm) in zip(axes.ravel(), layers):
        a = ds[v].values
        im = ax.imshow(a, extent=[float(ds.lon.min()), float(ds.lon.max()), float(ds.lat.min()), float(ds.lat.max())], cmap=cm, vmax=np.nanpercentile(a, 99.5))
        draw_boundaries(ax, 0, color="#333", lw=0.5); ax.set_title(t, fontsize=9); ax.grid(False)
        fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02)
    fig.suptitle("Static 1-km covariates (0.01° grid)", fontweight="bold"); fig.tight_layout()
    fig.savefig(FIG / "fig04_static.png"); plt.close(fig)


def fig_cv(met):
    order = ["cams_raw", "cams_linear", "paperlike_gbdt", "lgbm", "xgb_ratio", "catboost", "extratrees", "stack", "stack_rk"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    for ax, metric, lab in ((axes[0], "r2", "R²"), (axes[1], "rmse", "RMSE (µg/m³)")):
        sub = met[met.subset == "all"]
        x = np.arange(len(order)); w = 0.2
        for i, sch in enumerate(["random", "temporal", "site", "cluster"]):
            vals = [sub[(sub.scheme == sch) & (sub.model == m)][metric].mean() for m in order]
            ax.bar(x + (i - 1.5) * w, vals, w * 0.9, color=["#b7d3f6", "#6da7ec", "#2a78d6", "#104281"][i], label=SCHEME_LABEL[sch])
        ax.set_xticks(x); ax.set_xticklabels([MODEL_LABEL[m] for m in order], rotation=35, ha="right"); ax.set_ylabel(lab)
        if metric == "r2":
            ax.axhline(0, color="#6f6d67", lw=0.8); ax.set_ylim(min(-0.2, ax.get_ylim()[0]), 1)
    axes[0].legend(fontsize=8, loc="upper left")
    fig.suptitle("Cross-validated skill: baselines, individual learners and GH-PM25 stages under four validation designs (all stations)", fontweight="bold")
    fig.tight_layout(); fig.savefig(FIG / "fig05_cv_skill.png"); plt.close(fig)


def fig_scatter(oof, smear):
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.3))
    for ax, sch in zip(axes, ["random", "temporal", "site", "cluster"]):
        o = oof.pm25.to_numpy(); p = np.exp(oof[f"{sch}__stack_rk"].to_numpy()) * smear
        h = ax.hexbin(o, p, gridsize=55, xscale="log", yscale="log", bins="log", cmap="Blues", mincnt=1, extent=(np.log10(2), np.log10(400), np.log10(2), np.log10(400)))
        ax.plot([2, 400], [2, 400], color="#6f6d67", lw=0.8)
        r2 = 1 - np.sum((p - o) ** 2) / np.sum((o - o.mean()) ** 2); rmse = np.sqrt(np.mean((p - o) ** 2)); slope = np.polyfit(o, p, 1)[0]
        ax.text(0.04, 0.96, f"R² = {r2:.2f}\nRMSE = {rmse:.1f} µg/m³\nslope = {slope:.2f}\nn = {len(o):,}", transform=ax.transAxes, va="top", fontsize=8.5)
        ax.set_title(SCHEME_LABEL[sch]); ax.set_xlabel("Observed (µg/m³)"); ax.set_ylabel("Predicted (µg/m³)"); ax.set_xlim(2, 400); ax.set_ylim(2, 400)
    fig.colorbar(h, ax=axes, fraction=0.015, label="station-days (log)")
    fig.suptitle("GH-PM25 final predictions vs observations (out-of-fold)", fontweight="bold")
    fig.savefig(FIG / "fig06_scatter.png"); plt.close(fig)


def fig_subsets(met):
    subs = ["all", "ghana", "ghana_reference", "reference_all", "ghana_south(<=8N)", "ghana_north(>8N)", "harmattan(DJF)", "non_harmattan"]
    lab = ["All", "Ghana", "Ghana BAM", "All BAM", "Ghana south", "Ghana north", "Harmattan", "Non-Harmattan"]
    sub = met[(met.scheme == "cluster")]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
    for ax, metric, yl in ((axes[0], "r2", "R²"), (axes[1], "rmse", "RMSE (µg/m³)")):
        x = np.arange(len(subs)); w = 0.26
        for i, (m, c) in enumerate([("cams_raw", "#b9b7ae"), ("paperlike_gbdt", "#86b6ef"), ("stack_rk", "#1c5cab")]):
            vals = [sub[(sub.subset == s) & (sub.model == m)][metric].mean() for s in subs]
            ax.bar(x + (i - 1) * w, vals, w * 0.9, color=c, label=MODEL_LABEL[m])
        ax.set_xticks(x); ax.set_xticklabels(lab, rotation=25, ha="right"); ax.set_ylabel(yl)
        if metric == "r2":
            ax.axhline(0, color="#6f6d67", lw=0.8)
    axes[0].legend(fontsize=8)
    fig.suptitle("Leave-city-cluster-out skill by region, monitor type and season", fontweight="bold")
    fig.tight_layout(); fig.savefig(FIG / "fig07_subsets.png"); plt.close(fig)


def fig_shap():
    imp = pd.read_csv(OUT / "shap_importance.csv").head(20)[::-1]
    s = pd.read_parquet(OUT / "shap_sample.parquet")
    top = [c[3:] for c in s.columns if c.startswith("x__")][:8]
    fig = plt.figure(figsize=(14, 6.5))
    ax = fig.add_axes([0.05, 0.08, 0.25, 0.85])
    ax.barh(imp.feature, imp.mean_abs_shap, color=C1, height=0.6); ax.set_xlabel("mean |SHAP| (ln µg/m³)"); ax.set_title("a) Global importance (top 20)")
    for i, f in enumerate(top):
        axd = fig.add_axes([0.37 + (i % 4) * 0.155, 0.56 - (i // 4) * 0.47, 0.13, 0.36])
        x, y = s[f"x__{f}"], s[f"s__{f}"]
        axd.scatter(x, y, s=3, alpha=0.3, c=C2, lw=0)
        axd.axhline(0, color="#6f6d67", lw=0.6); axd.set_title(f, fontsize=8.5); axd.tick_params(labelsize=7)
        if i % 4 == 0:
            axd.set_ylabel("SHAP")
    fig.text(0.37, 0.95, "b) Dependence of the prediction on the leading predictors", fontweight="bold")
    fig.savefig(FIG / "fig08_shap.png"); plt.close(fig)


def fig_uncertainty(oof, summary):
    cg = summary["correlogram"]
    cov = pd.read_csv(OUT / "uncertainty_coverage.csv")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    ax = axes[0]
    ax.scatter(cg["bins_km"], cg["corr"], s=np.clip(np.array(cg["npairs"], dtype=float) / 800, 10, 120), c=C1, zorder=3, label="empirical (size ∝ pairs)")
    d = np.linspace(0, max(cg["bins_km"]), 300); ax.plot(d, cg["c0"] * np.exp(-d / cg["L_km"]), color=C2, label=f"{cg['c0']:.2f}·exp(−d/{cg['L_km']:.0f} km)")
    ax.axhline(0, color="#6f6d67", lw=0.6); ax.set_xlabel("distance (km)"); ax.set_ylabel("residual correlation"); ax.legend(fontsize=8); ax.set_title("a) Same-day residual correlogram")
    ax = axes[1]
    for nom, c in ((0.9, C1), (0.68, C3)):
        g = cov[np.isclose(cov.nominal, nom)]
        ax.barh(np.arange(len(g)) + (0.2 if nom == 0.9 else -0.2), g.coverage, 0.38, color=c, label=f"nominal {nom:.0%}")
        ax.axvline(nom, color=c, lw=1)
    ax.set_yticks(np.arange(len(g))); ax.set_yticklabels(g.subset); ax.set_xlim(0.4, 1); ax.set_xlabel("empirical coverage"); ax.legend(fontsize=8, loc="lower left")
    ax.set_title("b) Conformal interval coverage (nested, leave-cluster-out)")
    ax = axes[2]
    dn = oof["cluster__dist_nearest_km"].clip(upper=600)
    err = np.abs(np.log(oof.pm25.clip(lower=1)) - oof["cluster__stack_rk"])
    bins = [0, 5, 15, 30, 60, 120, 250, 600]
    b = pd.cut(dn, bins); gg = pd.DataFrame({"b": b, "e": err}).groupby("b", observed=True).e
    ax.plot(range(len(gg.median())), np.expm1(gg.median()) * 100, marker="o", color=C1, label="median")
    ax.plot(range(len(gg.median())), np.expm1(gg.quantile(0.9)) * 100, marker="o", color=C2, label="90th percentile")
    ax.set_xticks(range(len(gg.median()))); ax.set_xticklabels([str(i) for i in gg.median().index], rotation=30, fontsize=7)
    ax.set_xlabel("distance to nearest same-day monitor (km)"); ax.set_ylabel("absolute error (%)"); ax.legend(fontsize=8)
    ax.set_title("c) Error grows with distance from monitors")
    fig.tight_layout(); fig.savefig(FIG / "fig09_uncertainty.png"); plt.close(fig)


def fig_persite(stations):
    ps = pd.read_csv(OUT / "per_site_metrics.csv")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, (col, lab, cmap, vmin, vmax) in zip(axes, [("r2", "R²", "RdYlBu", -0.2, 1.0), ("rmse", "RMSE (µg/m³)", "YlOrRd", 0, 15)]):
        draw_boundaries(ax, 0, lw=0.8)
        sc = ax.scatter(ps.lon, ps.lat, c=ps[col], cmap=cmap, vmin=vmin, vmax=vmax, s=18 + ps.n / 15, edgecolor="#333", lw=0.4, zorder=3)
        fig.colorbar(sc, ax=ax, fraction=0.03, label=lab); ax.set_xlim(-9, 9); ax.set_ylim(4, 13.5); ax.set_aspect("equal")
        ax.set_title(f"Per-site {lab} (leave-city-cluster-out)")
    fig.tight_layout(); fig.savefig(FIG / "fig10_persite.png"); plt.close(fig)


def fig_maps():
    files = sorted(glob.glob(str(OUT / "daily" / "*" / "*.nc")))
    if not files:
        return
    stats = pd.read_parquet(OUT / "region_daily_stats.parquet")
    nat = stats[stats.level == "adm0"].set_index("date").pop_weighted
    harm = nat[nat.index.month.isin([12, 1, 2])]
    wet = nat[nat.index.month.isin([6, 7, 8, 9])]
    picks = [("Harmattan day", harm.idxmax() if len(harm) else nat.idxmax()), ("Wet-season day", wet.idxmin() if len(wet) else nat.idxmin())]
    fig, axes = plt.subplots(2, 4, figsize=(15, 9.5))
    for r, (lab, d) in enumerate(picks):
        f = OUT / "daily" / f"{d:%Y}" / f"GH-PM25_1km_{d:%Y%m%d}.nc"
        ds = xr.open_dataset(f)
        pm_imshow(axes[r, 0], ds.pm25.values, f"{lab} {d:%Y-%m-%d}: GH-PM25")
        pm_imshow(axes[r, 1], ds.pm25_lower90.values, "90% lower bound")
        pm_imshow(axes[r, 2], ds.pm25_upper90.values, "90% upper bound")
        pm_imshow(axes[r, 3], ds.cams_pm25.values, "CAMS input (bilinear)")
    pm_colorbar(fig, axes)
    fig.suptitle("Daily 1-km PM2.5, uncertainty bounds and the coarse CAMS input", fontweight="bold")
    fig.savefig(FIG / "fig11_daily_maps.png"); plt.close(fig)
    # zoom on the three largest cities: record-mean field with a local colour stretch (km-scale structure)
    import rasterio
    tifs = sorted(glob.glob(str(OUT / "monthly" / "*.tif")))
    if tifs:
        stack = []
        for t in tifs:
            with rasterio.open(t) as src:
                stack.append(src.read(1))
        mean = np.nanmean(np.stack(stack), axis=0)
        lons, lats = config.grid_axes()
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.9))
        for ax, (name, bb) in zip(axes, [("Greater Accra", (-0.45, 0.2, 5.5, 5.9)), ("Kumasi", (-1.85, -1.4, 6.5, 6.9)), ("Tamale", (-1.05, -0.65, 9.25, 9.6))]):
            j0, j1 = np.searchsorted(lons, bb[0]), np.searchsorted(lons, bb[1])
            i0, i1 = np.searchsorted(-lats, -bb[3]), np.searchsorted(-lats, -bb[2])
            sub = mean[i0:i1, j0:j1]
            vmin, vmax = np.nanpercentile(sub, 2), np.nanpercentile(sub, 98)
            im = ax.imshow(sub, extent=[lons[j0] - .005, lons[j1 - 1] + .005, lats[i1 - 1] - .005, lats[i0] + .005], cmap=PM_CMAP, vmin=vmin, vmax=vmax, interpolation="nearest")
            draw_boundaries(ax, 2, lw=0.4); ax.set_xlim(bb[0], bb[1]); ax.set_ylim(bb[2], bb[3]); ax.grid(False); ax.set_aspect("equal")
            ax.set_title(f"{name}: record mean ({len(tifs)} months)")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02, label="µg/m³ (local stretch)")
        fig.suptitle("Kilometre-scale structure within cities (district boundaries shown)", fontweight="bold")
        fig.tight_layout(); fig.savefig(FIG / "fig12_urban_zoom.png"); plt.close(fig)


def fig_climatology():
    tifs = sorted(glob.glob(str(OUT / "monthly" / "*.tif")))
    if not tifs:
        return
    import rasterio
    arrs = {}
    for t in tifs:
        ym = Path(t).stem.split("_")[-1]
        with rasterio.open(t) as src:
            arrs[ym] = src.read(1)
    seasons = {"DJF (Harmattan)": [12, 1, 2], "MAM": [3, 4, 5], "JJA (monsoon)": [6, 7, 8], "SON": [9, 10, 11]}
    fig, axes = plt.subplots(1, 5, figsize=(19, 5))
    allm = np.nanmean(np.stack(list(arrs.values())), axis=0)
    pm_imshow(axes[0], allm, f"Mean of all months ({len(arrs)})")
    for ax, (s, ms) in zip(axes[1:], seasons.items()):
        sel = [a for k, a in arrs.items() if int(k[4:]) in ms]
        if sel:
            pm_imshow(ax, np.nanmean(np.stack(sel), axis=0), f"{s} mean")
    pm_colorbar(fig, axes)
    fig.suptitle("Long-term and seasonal mean PM2.5 (GH-PM25)", fontweight="bold")
    fig.savefig(FIG / "fig13_climatology.png"); plt.close(fig)


def fig_exposure():
    stats = pd.read_parquet(OUT / "region_daily_stats.parquet")
    nat = stats[stats.level == "adm0"].sort_values("date")
    r1 = stats[stats.level == "adm1"].copy()
    north = ["Northern Region", "Savannah Region", "North East Region", "Upper East Region", "Upper West Region", "Northern", "Savannah", "North East", "Upper East", "Upper West"]
    r1["grp"] = np.where(r1.name.isin(north), "North", "South & middle belt")
    g = r1.assign(w=r1.pop_weighted * r1.population).groupby(["date", "grp"]).agg(w=("w", "sum"), p=("population", "sum"))
    g = (g.w / g.p).unstack()
    fig, axes = plt.subplots(2, 1, figsize=(13, 7.5), sharex=True)
    ax = axes[0]
    s = nat.set_index("date")
    ax.fill_between(s.index, s.lower90.rolling(7, min_periods=3).mean(), s.upper90.rolling(7, min_periods=3).mean(), color=C1, alpha=0.18, lw=0, label="90% interval (7-day)")
    ax.plot(s.index, s.pop_weighted.rolling(7, min_periods=3).mean(), color=C1, lw=1.6, label="national population-weighted (7-day mean)")
    for y, l in ((15, "WHO 24-h 15"), (35, "Ghana EPA 24-h 35")):
        ax.axhline(y, color="#6f6d67", lw=0.8); ax.text(s.index[0], y + 1, l, fontsize=7.5, color="#6f6d67")
    ax.set_ylabel("µg/m³"); ax.legend(fontsize=8, loc="upper right"); ax.set_title("a) National population-weighted daily PM2.5")
    ax = axes[1]
    for c, col in (("North", C2), ("South & middle belt", C1)):
        if c in g:
            ax.plot(g.index, g[c].rolling(30, min_periods=10).mean(), color=col, lw=1.8, label=c)
    ax.set_ylabel("µg/m³ (30-day mean)"); ax.legend(fontsize=8); ax.set_title("b) North–south gradient")
    fig.tight_layout(); fig.savefig(FIG / "fig14_exposure.png"); plt.close(fig)
    # annual exposure table
    nat = nat.assign(year=nat.date.dt.year)
    tab = nat.groupby("year").agg(days=("date", "size"), pop_weighted_mean=("pop_weighted", "mean"), days_gt35=("pop_weighted", lambda v: int((v > 35).sum())),
                                  mean_pop_frac_gt15=("pop_frac_gt15", "mean"), mean_pop_frac_gt35=("pop_frac_gt35", "mean")).round(3)
    tab.to_csv(TAB / "annual_exposure_national.csv")
    r1y = r1.assign(year=r1.date.dt.year).groupby(["name", "year"]).pop_weighted.mean().unstack().round(1)
    r1y.to_csv(TAB / "annual_regional_means.csv")


CITY_COORDS = {"Accra": (5.6037, -0.1870), "Kumasi": (6.6885, -1.6244), "Tamale": (9.4075, -0.8533), "Tema": (5.6698, -0.0166),
               "Koforidua": (6.0941, -0.2591), "Cape Coast": (5.1053, -1.2466), "Sekondi-Takoradi": (4.9340, -1.7137), "Wa": (10.0601, -2.5099),
               "Ho": (6.6008, 0.4713), "Sunyani": (7.3349, -2.3123), "Bolgatanga": (10.7856, -0.8514), "Techiman": (7.5909, -1.9390), "Somanya": (6.1043, -0.0150)}


def fig_baseline_comparison(oof, smear):
    """Anand et al. (2026) city-level product vs our leave-cluster-out predictions at the same station-days."""
    files = glob.glob(str(config.RAW / "baseline_anand2026" / "CSV_PM25_Ghana_Cities_Daily_2005-2025" / "PM25_*.csv"))
    if not files:
        return None
    base = pd.concat([pd.read_csv(f) for f in files]); base["date"] = pd.to_datetime(base.Date)
    o = oof[oof.country == "GH"].copy()
    o["ours"] = np.exp(o["cluster__stack_rk"]) * smear
    rows = []
    for city, (la, lo) in CITY_COORDS.items():
        d = haversine(la, lo, o.lat.to_numpy(), o.lon.to_numpy())
        sub = o[d < 15]
        b = base[base.City == city][["date", "PM25"]]
        m = sub.merge(b, on="date")
        if len(m) < 30:
            continue
        daily = m.groupby("date").agg(obs=("pm25", "mean"), ours=("ours", "mean"), anand=("PM25", "first"), cams=("cams_pm25", "mean"), n=("site_id", "nunique")).reset_index()
        for name, col in (("GH-PM25 (this work, leave-cluster-out)", "ours"), ("Anand et al. 2026 product", "anand"), ("CAMS raw", "cams")):
            e = daily[col] - daily.obs
            rows.append(dict(city=city, product=name, days=len(daily), r=float(np.corrcoef(daily.obs, daily[col])[0, 1]),
                             r2=float(1 - np.sum(e ** 2) / np.sum((daily.obs - daily.obs.mean()) ** 2)), rmse=float(np.sqrt(np.mean(e ** 2))),
                             mae=float(np.mean(np.abs(e))), bias=float(e.mean()), mean_obs=float(daily.obs.mean()), period=f"{daily.date.min():%Y-%m}–{daily.date.max():%Y-%m}"))
        if city == "Accra":
            fig, ax = plt.subplots(figsize=(13, 3.8))
            ax.plot(daily.date, daily.obs, color="#0b0b0b", lw=1.0, label="observed (city mean of calibrated monitors)")
            ax.plot(daily.date, daily.ours, color=C1, lw=1.2, label="GH-PM25 (leave-cluster-out)")
            ax.plot(daily.date, daily.anand, color=C2, lw=1.2, label="Anand et al. (2026) product")
            ax.set_ylabel("µg/m³"); ax.legend(fontsize=8, ncol=3); ax.set_title("Accra: daily PM2.5 — observations vs products")
            fig.savefig(FIG / "fig15_accra_products.png"); plt.close(fig)
    res = pd.DataFrame(rows)
    res.to_csv(TAB / "comparison_anand2026.csv", index=False)
    return res


def cams_bias_table(oof):
    ref = oof[oof.is_reference].copy()
    ref["month"] = ref.date.dt.month
    ref["site"] = ref.site_id
    t = ref.groupby(["site", "month"]).apply(lambda g: pd.Series(dict(n=len(g), obs=g.pm25.mean(), cams=g.cams_pm25.mean(), ratio=(g.cams_pm25 / g.pm25).median()))).reset_index()
    t.to_csv(TAB / "cams_bias_reference_monthly.csv", index=False)
    fig, ax = plt.subplots(figsize=(8, 4))
    for i, (s, g) in enumerate(t.groupby("site")):
        if g.n.sum() >= 150:
            ax.plot(g.month, g.ratio, marker="o", lw=1.4, label=s)
    ax.axhline(1, color="#6f6d67", lw=0.8); ax.set_xticks(range(1, 13)); ax.set_xticklabels(list("JFMAMJJASOND"))
    ax.set_ylabel("median CAMS / BAM ratio"); ax.set_yscale("log"); ax.legend(fontsize=7); ax.set_title("CAMS PM2.5 bias at BAM-1020 reference monitors")
    fig.savefig(FIG / "fig16_cams_bias.png"); plt.close(fig)


def fig_ablation():
    a = pd.read_csv(OUT / "ablation_metrics.csv")
    fig, ax = plt.subplots(figsize=(8, 3.6))
    for i, (s, c) in enumerate((("all", C1), ("ghana", C2))):
        g = a[a.subset == s]
        ax.barh(np.arange(len(g)) + (i - 0.5) * 0.38, g.r2, 0.36, color=c, label=s)
    ax.set_yticks(np.arange(len(g))); ax.set_yticklabels(g.variant); ax.set_xlabel("R² (LightGBM, leave-cluster-out)"); ax.legend(fontsize=8)
    ax.set_title("Feature-group ablation")
    fig.savefig(FIG / "fig17_ablation.png"); plt.close(fig)


def fig_fire_qa():
    f1, f2 = OUT / "qa_fire_selection.csv", OUT / "qa_fire_efire_vs_firms.csv"
    if not (f1.exists() and f2.exists()):
        return
    a, b = pd.read_csv(f1), pd.read_csv(f2)
    b = b[(b.efire_frp > 0) & (b.firms_frp > 0)]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].bar(a.date, a.capture_frp * 100, color=C1); axes[0].set_ylim(0, 105); axes[0].set_ylabel("FRP captured (%)")
    axes[0].tick_params(axis="x", rotation=45, labelsize=7); axes[0].set_title("a) Orbit-model granule selection vs full window scan")
    axes[1].scatter(b.firms_frp / 1e3, b.efire_frp / 1e3, c=C2, s=25); lim = max(b.firms_frp.max(), b.efire_frp.max()) / 1e3 * 1.05
    axes[1].plot([0, lim], [0, lim], color="#6f6d67", lw=0.8); axes[1].set_xlabel("FIRMS VNP14IMG daily FRP (GW)"); axes[1].set_ylabel("NOAA EFIRE daily FRP (GW)")
    axes[1].set_title("b) Product consistency, S-NPP 2024 overlap")
    fig.tight_layout(); fig.savefig(FIG / "fig18_fire_qa.png"); plt.close(fig)


def main():
    stations = pd.read_parquet(config.PROCESSED / "stations.parquet")
    table = pd.read_parquet(config.PROCESSED / "training_table.parquet")
    oof = pd.read_parquet(OUT / "cv_oof.parquet")
    met = pd.read_csv(OUT / "cv_metrics.csv")
    summary = json.loads((OUT / "run_summary.json").read_text())
    smear = summary["smear"]
    steps = [("network", lambda: fig_network(stations, table)), ("timeline", lambda: fig_timeline(table)), ("calibration", fig_calibration),
             ("static", fig_static), ("cv", lambda: fig_cv(met)), ("scatter", lambda: fig_scatter(oof, smear)), ("subsets", lambda: fig_subsets(met)),
             ("shap", fig_shap), ("uncertainty", lambda: fig_uncertainty(oof, summary)), ("persite", lambda: fig_persite(stations)),
             ("maps", fig_maps), ("climatology", fig_climatology), ("exposure", fig_exposure), ("baseline", lambda: fig_baseline_comparison(oof, smear)),
             ("cams_bias", lambda: cams_bias_table(oof)), ("ablation", fig_ablation), ("fire_qa", fig_fire_qa)]
    for name, fn in steps:
        try:
            fn(); print("ok  ", name, flush=True)
        except Exception as e:
            import traceback
            print("FAIL", name, e, flush=True); traceback.print_exc()
    # tables
    met.to_csv(TAB / "cv_metrics_all.csv", index=False)
    met[(met.subset.isin(["all", "ghana", "ghana_reference"]))].pivot_table(index=["subset", "model"], columns="scheme", values=["r2", "rmse"]).round(3).to_csv(TAB / "cv_summary.csv")


if __name__ == "__main__":
    main()
