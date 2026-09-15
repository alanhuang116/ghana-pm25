"""Build the GH-PM25 technical report (report/GH-PM25_Technical_Report.html) from pipeline outputs."""
from __future__ import annotations

import datetime as dt
import html
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ghana_pm25 import __version__, config  # noqa: E402
from ghana_pm25 import features as F  # noqa: E402

OUT, REP = config.OUTPUTS, config.REPORT
TAB = REP / "tables"


# ---------------------------------------------------------------------------
def esc(s):
    return html.escape(str(s))


def fnum(v, nd=2):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "—"
    return f"{v:,.{nd}f}"


def table(df: pd.DataFrame, cols: list[tuple], caption: str | None = None, cls: str = "") -> str:
    head = "".join(f'<th class="{"num" if c[2] else ""}">{esc(c[1])}</th>' for c in cols)
    rows = []
    for _, r in df.iterrows():
        tds = []
        for key, _, nd, *rest in cols:
            v = r[key]
            if rest and callable(rest[0]):
                txt = rest[0](v, r)
            elif nd is not None and isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, bool):
                txt = fnum(float(v), nd) if nd > 0 else f"{int(v):,}"
            else:
                txt = esc(v)
            tds.append(f'<td class="{"num" if nd is not None else ""}">{txt}</td>')
        rows.append(f"<tr>{''.join(tds)}</tr>")
    cap = f"<caption>{caption}</caption>" if caption else ""
    return f'<div class="tablewrap {cls}"><table>{cap}<thead><tr>{head}</tr></thead><tbody>{"".join(rows)}</tbody></table></div>'


def figure(name: str, num: int, caption: str, wide: bool = True) -> str:
    if not (REP / "figures" / name).exists():
        return f'<p class="missing">[Figure {num} not generated: {esc(name)}]</p>'
    return (f'<figure class="{"wide" if wide else ""}" id="fig{num}"><img src="figures/{name}" alt="Figure {num}. {esc(caption[:120])}" loading="lazy">'
            f"<figcaption><span class=\"fignum\">Figure {num}.</span> {caption}</figcaption></figure>")


def m(met, scheme, model, subset, key):
    r = met[(met.scheme == scheme) & (met.model == model) & (met.subset == subset)]
    return float(r[key].iloc[0]) if len(r) else float("nan")


# ---------------------------------------------------------------------------
def main():
    met = pd.read_csv(OUT / "cv_metrics.csv")
    summ = json.loads((OUT / "run_summary.json").read_text())
    cal = pd.read_csv(OUT / "calibration_report.csv")
    cov = pd.read_csv(OUT / "uncertainty_coverage.csv")
    abl = pd.read_csv(OUT / "ablation_metrics.csv")
    shap = pd.read_csv(OUT / "shap_importance.csv")
    ps = pd.read_csv(OUT / "per_site_metrics.csv")
    stations = pd.read_parquet(config.PROCESSED / "stations.parquet")
    tt = pd.read_parquet(config.PROCESSED / "training_table.parquet", columns=["site_id", "date", "country", "network", "is_reference", "pm25", "pm25_raw", "cal_model"])
    comp = pd.read_csv(TAB / "comparison_anand2026.csv") if (TAB / "comparison_anand2026.csv").exists() else pd.DataFrame()
    ann = pd.read_csv(TAB / "annual_exposure_national.csv") if (TAB / "annual_exposure_national.csv").exists() else pd.DataFrame()
    regy = pd.read_csv(TAB / "annual_regional_means.csv") if (TAB / "annual_regional_means.csv").exists() else pd.DataFrame()
    qa_sel = pd.read_csv(OUT / "qa_fire_selection.csv") if (OUT / "qa_fire_selection.csv").exists() else pd.DataFrame()
    qa_ov = pd.read_csv(OUT / "qa_fire_efire_vs_firms.csv") if (OUT / "qa_fire_efire_vs_firms.csv").exists() else pd.DataFrame()
    fires = pd.read_parquet(config.INTERIM / "firms_hist.parquet", columns=["date"])
    stats = pd.read_parquet(OUT / "region_daily_stats.parquet") if (OUT / "region_daily_stats.parquet").exists() else pd.DataFrame()
    cams_nodes = pd.read_parquet(config.INTERIM / "cams_nodes.parquet")

    n_train, n_sites = summ["n_train"], summ["n_sites"]
    p0, p1 = summ["period"]
    gh = tt[tt.country == "GH"]
    n_gh_sites, n_gh_days = gh.site_id.nunique(), len(gh)
    cg = summ["correlogram"]
    stack = summ["stack"]["weights"]

    def S(scheme, model="stack_rk", subset="all", key="r2"):
        return m(met, scheme, model, subset, key)

    head_r2, head_rmse, head_mae = S("cluster"), S("cluster", key="rmse"), S("cluster", key="mae")
    gh_r2, gh_rmse = S("cluster", subset="ghana"), S("cluster", subset="ghana", key="rmse")
    cams_r2, cams_rmse = S("cluster", "cams_raw"), S("cluster", "cams_raw", key="rmse")
    pl_r2 = S("cluster", "paperlike_gbdt")
    rnd_r2, rnd_rmse = S("random"), S("random", key="rmse")
    cov90 = cov[(np.isclose(cov.nominal, 0.9)) & (cov.subset == "all")].coverage
    cov90 = float(cov90.iloc[0]) if len(cov90) else float("nan")
    dates_mapped = stats.date.nunique() if len(stats) else 0
    today = dt.date.today()

    # ---------------- tables ----------------
    order = ["cams_raw", "cams_linear", "paperlike_gbdt", "lgbm", "xgb_ratio", "catboost", "extratrees", "stack", "stack_rk"]
    lab = {"cams_raw": "CAMS global (raw, bilinear)", "cams_linear": "CAMS + log-linear bias correction", "paperlike_gbdt": "GBDT with AOD + meteorology + day-of-year (baseline-style)",
           "lgbm": "M1 LightGBM, ln PM", "xgb_ratio": "M2 XGBoost, ln(PM/CAMS)", "catboost": "M3 CatBoost, ln PM", "extratrees": "M4 ExtraTrees, ln PM",
           "stack": "GH-PM25 ensemble (Stage 1)", "stack_rk": "GH-PM25 final (operational)"}
    rows = []
    for mod in order:
        r = {"model": lab[mod]}
        for sch in ("random", "temporal", "site", "cluster"):
            r[f"{sch}_r2"] = S(sch, mod); r[f"{sch}_rmse"] = S(sch, mod, key="rmse")
        rows.append(r)
    cvt = pd.DataFrame(rows)
    cv_table = table(cvt, [("model", "Model", None), ("random_r2", "Random R²", 2), ("random_rmse", "RMSE", 1), ("temporal_r2", "Month-out R²", 2), ("temporal_rmse", "RMSE", 1),
                           ("site_r2", "Site-out R²", 2), ("site_rmse", "RMSE", 1), ("cluster_r2", "Cluster-out R²", 2), ("cluster_rmse", "RMSE", 1)],
                     caption=f"Table 5. Cross-validated skill for all {n_train:,} station-days (RMSE in µg/m³). Bold row: operational product.", cls="cvtable")

    subs = ["all", "ghana", "ghana_reference", "reference_all", "ghana_south(<=8N)", "ghana_north(>8N)", "harmattan(DJF)", "non_harmattan"]
    sl = {"all": "All stations", "ghana": "Ghana", "ghana_reference": "Ghana, BAM-1020 reference", "reference_all": "All BAM-1020 reference", "ghana_south(<=8N)": "Ghana south of 8°N",
          "ghana_north(>8N)": "Ghana north of 8°N", "harmattan(DJF)": "Harmattan (Dec–Feb)", "non_harmattan": "Mar–Nov"}
    srows = []
    for s in subs:
        r = met[(met.scheme == "cluster") & (met.model == "stack_rk") & (met.subset == s)]
        c = met[(met.scheme == "cluster") & (met.model == "cams_raw") & (met.subset == s)]
        if len(r):
            r = r.iloc[0]
            srows.append(dict(subset=sl[s], n=r.n, mean_obs=r.mean_obs, r2=r.r2, r=r.r, rmse=r.rmse, mae=r.mae, mb=r.mb, slope=r.slope,
                              cams_r2=float(c.r2.iloc[0]) if len(c) else np.nan, cams_rmse=float(c.rmse.iloc[0]) if len(c) else np.nan))
    sub_table = table(pd.DataFrame(srows), [("subset", "Subset", None), ("n", "n", 0), ("mean_obs", "Mean obs.", 1), ("r2", "R²", 2), ("r", "r", 2), ("rmse", "RMSE", 1),
                                            ("mae", "MAE", 1), ("mb", "Bias", 1), ("slope", "Slope", 2), ("cams_r2", "CAMS R²", 2), ("cams_rmse", "CAMS RMSE", 1)],
                      caption="Table 6. Final-product skill by subset under leave-city-cluster-out cross-validation (µg/m³).")

    calt = cal[cal.level == "net"].copy()
    cal_table = table(calt, [("key", "Network", None, lambda v, r: esc(str(v).replace("|", " · "))), ("n_pairs", "Pairs", 0), ("n_sensors", "Sensors", 0), ("mean_dist_km", "Mean dist. km", 1),
                             ("rh_range", "RH range %", None), ("max_ref", "Max ref.", 0), ("selected", "Selected model", None), ("raw_r2", "R² raw", 2), ("cv_r2", "R² cal.", 2),
                             ("raw_mae", "MAE raw", 1), ("cv_mae", "MAE cal.", 1), ("raw_bias", "Bias raw", 1), ("cv_bias", "Bias cal.", 1),
                             ("excluded", "Excluded", None, lambda v, r: "yes" if bool(v) else "no")],
                      caption="Table 3. Low-cost sensor harmonisation: raw vs month-blocked cross-validated calibrated agreement with BAM-1020 (µg/m³).")
    cov_table = table(cov, [("subset", "Subset", None, lambda v, r: esc(sl.get(v, v))), ("nominal", "Nominal", None, lambda v, r: f"{v:.0%}"),
                            ("coverage", "Empirical coverage", None, lambda v, r: f"{v:.1%}"), ("median_width", "Median width µg/m³", 1), ("n", "n", 0)],
                      caption="Table 8. Empirical coverage of the conformal prediction intervals (nested leave-city-cluster-out).")
    abl_table = table(abl, [("variant", "Variant", None), ("subset", "Subset", None), ("n_features", "Features", 0), ("r2", "R²", 3), ("rmse", "RMSE", 2), ("mae", "MAE", 2), ("mb", "Bias", 2)],
                      caption="Table 9. Feature-group ablation (single LightGBM, leave-city-cluster-out).")
    shap_table = table(shap.head(15), [("feature", "Predictor", None), ("mean_abs_shap", "mean |SHAP| (ln µg/m³)", 4)], caption="Table 10. Top-15 predictors by mean absolute SHAP value.")

    net_sum = tt.groupby(["country", "network"]).agg(sites=("site_id", "nunique"), days=("date", "size"), first=("date", "min"), last=("date", "max"),
                                                     mean_raw=("pm25_raw", "mean"), mean_cal=("pm25", "mean"), ref=("is_reference", "max")).reset_index()
    net_sum["first"] = net_sum["first"].dt.strftime("%Y-%m"); net_sum["last"] = net_sum["last"].dt.strftime("%Y-%m")
    net_table = table(net_sum.sort_values(["country", "days"], ascending=[True, False]),
                      [("country", "Country", None), ("network", "Network / operator", None), ("ref", "Grade", None, lambda v, r: "Reference BAM-1020" if v else "Low-cost optical"),
                       ("sites", "Sites", 0), ("days", "Station-days", 0), ("first", "From", None), ("last", "To", None), ("mean_raw", "Mean raw", 1), ("mean_cal", "Mean calibrated", 1)],
                      caption=f"Table 2. Ground monitoring data used for training after QC ({n_train:,} station-days, {n_sites} sites).")

    comp_html = ""
    if len(comp):
        comp_html = table(comp, [("city", "City", None), ("product", "Product", None), ("days", "Days", 0), ("period", "Period", None), ("mean_obs", "Mean obs.", 1),
                                 ("r", "r", 2), ("r2", "R²", 2), ("rmse", "RMSE", 1), ("mae", "MAE", 1), ("bias", "Bias", 1)],
                          caption="Table 7. Head-to-head evaluation against city-mean observations on identical days. GH-PM25 values are out-of-sample (leave-city-cluster-out); "
                                  "the Anand et al. (2026) product was trained with monitors in these cities, so its scores are partly in-sample.")

    ann_html = table(ann.reset_index(drop=True), [("year", "Year", None), ("days", "Days", 0), ("pop_weighted_mean", "Pop.-weighted mean µg/m³", 1),
                                                  ("days_gt35", "Days national pop.-weighted >35", 0), ("mean_pop_frac_gt15", "Mean share of pop. >15", 3),
                                                  ("mean_pop_frac_gt35", "Mean share of pop. >35", 3)], caption="Table 11. National exposure summary by year (partial years flagged by day count).") if len(ann) else ""
    reg_html = ""
    if len(regy):
        cols = [("name", "Region", None)] + [(c, c, 1) for c in regy.columns[1:]]
        reg_html = table(regy.sort_values(regy.columns[-1], ascending=False), cols, caption="Table 12. Annual mean population-weighted PM2.5 by region (µg/m³).")

    ps_top = ps[ps.country == "GH"].sort_values("n", ascending=False)
    ps_table = table(ps_top, [("name", "Site", None), ("network", "Network", None), ("lat", "Lat", 3), ("lon", "Lon", 3), ("n", "Days", 0), ("mean_obs", "Mean obs.", 1),
                              ("r2", "R²", 2), ("r", "r", 2), ("rmse", "RMSE", 1), ("mae", "MAE", 1), ("mb", "Bias", 1)],
                     caption="Table D1. Per-site leave-city-cluster-out metrics, Ghana (µg/m³).", cls="tall")

    feat_desc = {
        "cams_pm25": "CAMS PM2.5 (µg/m³), daily mean", "cams_pm10": "CAMS PM10", "cams_aod": "CAMS total AOD 550 nm", "cams_dust": "CAMS dust surface concentration",
        "cams_co": "CAMS carbon monoxide (combustion/biomass-burning tracer)", "cams_no2": "CAMS NO₂ (traffic/urban tracer)", "cams_pm25_max": "CAMS hourly PM2.5 maximum",
        "t2m": "2-m temperature (°C)", "t2m_range": "diurnal temperature range", "rh": "2-m relative humidity (%)", "qv2m": "specific humidity (g/kg)", "td2m": "dew point",
        "precip": "precipitation (mm/day, IMERG-corrected)", "ws10": "10-m wind speed", "u10": "10-m zonal wind", "v10": "10-m meridional wind", "ps": "surface pressure",
        "ssrd": "surface shortwave down (kWh/m²/day)", "cloud": "cloud fraction", "merra_aod": "MERRA-2 AOD 550 nm", "blh": "PBL depth from MERRA-2 PBLTOP (m)",
        "log_cams_pm25": "ln CAMS PM2.5", "cams_pm25_lag1": "CAMS PM2.5 day t−1", "cams_pm25_3d": "CAMS PM2.5 3-day mean", "cams_dust_lag1": "CAMS dust t−1",
        "dust_frac": "dust / (PM10 + 1)", "log_cams_dust": "ln(1+dust)", "precip_3d": "3-day precipitation", "precip_lag1": "precipitation t−1", "log_blh": "ln PBL depth",
        "ventilation": "ventilation coefficient PBL × wind (m²/s)", "rh_lag1": "RH t−1", "ws10_lag1": "wind speed t−1", "merra_aod_3d": "3-day MERRA-2 AOD",
        "wind_dir_sin": "sin wind direction", "wind_dir_cos": "cos wind direction", "frp_25km": "ln(1+FRP) within 25 km, t and t−1", "frp_100km": "ln(1+FRP) within 100 km",
        "fire_upwind": "upwind-weighted fire influence within 500 km", "doy_sin": "sin day-of-year", "doy_cos": "cos day-of-year", "dow": "day of week",
        "lat": "latitude", "lon": "longitude", "elev": "mean elevation (m)", "elev_sd": "sub-grid elevation SD", "elev_rel_10km": "elevation relative to 10-km mean",
        "lc_tree": "tree-cover fraction", "lc_shrub": "shrub fraction", "lc_grass": "grassland fraction", "lc_crop": "cropland fraction", "lc_built": "built-up fraction",
        "lc_bare": "bare/sparse fraction", "lc_water": "water fraction", "log_pop": "ln(1+population density)", "log_ntl": "ln(1+night-time radiance)",
        "road_density": "major-road density (km/km²)", "dist_road": "ln(1+distance to major road km)", "dist_coast": "ln(1+distance to coast km)",
    }
    fdf = pd.DataFrame([dict(feature=f, group=g, desc=feat_desc.get(f, f.replace("_s3km", " (3-km Gaussian)").replace("_s10km", " (10-km Gaussian)")))
                        for g, lst in (("CAMS composition", F.CAMS_COLS), ("MERRA-2 meteorology/aerosol", F.MET_COLS), ("Derived dynamic", F.DYNAMIC_DERIVED),
                                       ("Fire", F.FIRE_COLS), ("Static 1-km", F.STATIC_COLS), ("Calendar", F.TIME_COLS), ("Coordinates", F.COORD_COLS)) for f in lst])
    feat_table = table(fdf, [("group", "Group", None), ("feature", "Predictor", None), ("desc", "Definition", None)], caption=f"Table A1. Predictor dictionary ({len(fdf)} predictors).", cls="tall")

    stations_gh = stations[stations.country == "GH"].sort_values(["network", "name"])
    st_table = table(stations_gh, [("site_id", "ID", None), ("name", "Site", None), ("network", "Network", None), ("lat", "Lat", 4), ("lon", "Lon", 4), ("n_days", "Days", 0),
                                   ("is_reference", "Reference", None, lambda v, r: "BAM-1020" if v else "")], caption="Table C1. Ghanaian monitoring sites.", cls="tall")

    fire_cap = ""
    if len(qa_sel):
        fire_cap = f"Across {len(qa_sel)} independent QA days the selection read on average {qa_sel.granules_selected.mean():.0f} of {qa_sel.granules_window.mean():.0f} candidate granules and captured {qa_sel.capture_frp.min():.1%}–{qa_sel.capture_frp.max():.1%} of fire radiative power."
    ov_txt = ""
    if len(qa_ov):
        o = qa_ov[(qa_ov.efire_frp > 0) & (qa_ov.firms_frp > 0)]
        ov_txt = f"On {len(o)} overlapping days in late 2024 the median EFIRE/FIRMS ratio was {np.median(o.efire_frp / o.firms_frp):.3f} for FRP and {np.median(o.efire_n / o.firms_n):.3f} for detection counts."

    best_single = max(["lgbm", "xgb_ratio", "catboost", "extratrees"], key=lambda k: S("cluster", k))
    gain_vs_cams = (1 - head_rmse / cams_rmse) if np.isfinite(cams_rmse) and cams_rmse > 0 else np.nan
    comp_acc = comp[comp.city == "Accra"] if len(comp) else pd.DataFrame()
    acc_line = ""
    if len(comp_acc) >= 2:
        o_ = comp_acc[comp_acc["product"].str.startswith("GH-PM25")].iloc[0]; a_ = comp_acc[comp_acc["product"].str.startswith("Anand")].iloc[0]
        acc_line = (f"In Accra, over {int(o_.days)} matched days, GH-PM25 reached r = {o_.r:.2f} and RMSE = {o_.rmse:.1f} µg/m³ out-of-sample, compared with r = {a_.r:.2f} and "
                    f"RMSE = {a_.rmse:.1f} µg/m³ for the Anand et al. (2026) product.")

    hp = json.loads((config.MODELS / "hyperparams.json").read_text()) if (config.MODELS / "hyperparams.json").exists() else {}
    hp_rows = "".join(f'<tr><td>{esc(lab[k])}</td><td class="mono">{esc(", ".join(f"{a}={b}" for a, b in hp.get(k, {}).items()) or "defaults")}</td></tr>'
                      for k in ["lgbm", "xgb_ratio", "catboost", "extratrees"])

    ctx = dict(
        version=__version__, date=today.isoformat(), p0=p0, p1=p1, n_train=f"{n_train:,}", n_sites=n_sites, n_gh_sites=n_gh_sites, n_gh_days=f"{n_gh_days:,}",
        head_r2=f"{head_r2:.2f}", head_rmse=f"{head_rmse:.1f}", head_mae=f"{head_mae:.1f}", gh_r2=f"{gh_r2:.2f}", gh_rmse=f"{gh_rmse:.1f}",
        cams_r2=f"{cams_r2:.2f}", cams_rmse=f"{cams_rmse:.1f}", pl_r2=f"{pl_r2:.2f}", rnd_r2=f"{rnd_r2:.2f}", rnd_rmse=f"{rnd_rmse:.1f}", cov90=f"{cov90:.1%}",
        c0=f"{cg['c0']:.2f}", L=f"{cg['L_km']:.0f}", q90=f"{summ['q90']:.2f}", smear=f"{summ['smear']:.3f}", dates_mapped=f"{dates_mapped:,}",
        n_fires=f"{len(fires):,}", n_cams_nodes=len(cams_nodes), gain=f"{gain_vs_cams:.0%}",
        w_lgbm=f"{stack['lgbm']:.2f}", w_xgb=f"{stack['xgb_ratio']:.2f}", w_cat=f"{stack['catboost']:.2f}", w_et=f"{stack['extratrees']:.2f}",
        best_single=lab[best_single], best_single_r2=f"{S('cluster', best_single):.2f}", stage1_r2=f"{S('cluster', 'stack'):.2f}",
        site_r2=f"{S('site'):.2f}", temporal_r2=f"{S('temporal'):.2f}", n_features=len(summ["features"]),
        rk_lambda=summ.get("rk_lambda", 1.0),
    )
    def fill(template, **kw):
        for k, v in kw.items():
            template = template.replace("{" + k + "}", str(v))
        return template

    body = fill(BODY, **ctx, cv_table=cv_table, sub_table=sub_table, cal_table=cal_table, cov_table=cov_table, abl_table=abl_table, shap_table=shap_table,
                       net_table=net_table, comp_html=comp_html, ann_html=ann_html, reg_html=reg_html, ps_table=ps_table, feat_table=feat_table, st_table=st_table,
                       fire_cap=fire_cap, ov_txt=ov_txt, acc_line=acc_line, hp_rows=hp_rows,
                       f1=figure("fig01_network.png", 1, "Ground monitors used for training. (a) Regional training domain; (b) Ghana. Marker size is proportional to the number of valid station-days; triangles mark US State Department BAM-1020 reference monitors."),
                       f2=figure("fig02_timeline.png", 2, "Monthly station-days available for training by country and network. The Ghanaian network grows sharply after August 2024 (Breathe Accra) and February 2026 (national Clarity deployment by Ghana AQ)."),
                       f3=figure("fig03_calibration.png", 3, "Quasi-co-located daily pairs (sensor within 10 km of a BAM-1020, same day) before (top) and after (bottom) network-specific harmonisation."),
                       f4=figure("fig04_static.png", 4, "Examples of static predictors on the 0.01° grid."),
                       f5=figure("fig05_cv_skill.png", 5, "R² and RMSE for baselines, the four base learners and the GH-PM25 stages under four validation designs."),
                       f6=figure("fig06_scatter.png", 6, "Out-of-fold predictions of the final product against observations for each validation design (log–log density)."),
                       f7=figure("fig07_subsets.png", 7, "Leave-city-cluster-out skill by subset for raw CAMS, the baseline-style GBDT and GH-PM25."),
                       f8=figure("fig08_shap.png", 8, "SHAP analysis of the LightGBM member: (a) global importance; (b) dependence plots for leading predictors (log-space contributions)."),
                       f9=figure("fig09_uncertainty.png", 9, "(a) Pooled same-day residual correlogram and fitted exponential model used for kriging; (b) empirical coverage of 90% and 68% conformal intervals; (c) absolute error versus distance to the nearest same-day monitor."),
                       f10=figure("fig10_persite.png", 10, "Per-site R² and RMSE under leave-city-cluster-out cross-validation."),
                       f11=figure("fig11_daily_maps.png", 11, "Daily 1-km PM2.5 with 90% interval bounds, and the CAMS input, for the most polluted Harmattan day and the cleanest wet-season day in the record."),
                       f12=figure("fig12_urban_zoom.png", 12, "Kilometre-scale structure in the three largest urban areas."),
                       f13=figure("fig13_climatology.png", 13, "Long-term and seasonal mean PM2.5."),
                       f14=figure("fig14_exposure.png", 14, "(a) National population-weighted PM2.5 with 90% interval (7-day means); (b) population-weighted means for the five northern regions versus the rest of Ghana (30-day means)."),
                       f15=figure("fig15_accra_products.png", 15, "Accra: city-mean observations, GH-PM25 (out-of-sample) and the Anand et al. (2026) product on matched days."),
                       f16=figure("fig16_cams_bias.png", 16, "Median monthly ratio of CAMS to BAM-1020 PM2.5 at reference monitors."),
                       f17=figure("fig17_ablation.png", 17, "Feature-group ablation."),
                       f18=figure("fig18_fire_qa.png", 18, "Fire record quality assurance: (a) share of FRP captured by the orbit-model granule selection compared with an exhaustive scan; (b) EFIRE vs FIRMS daily regional FRP during the 2024 overlap."))
    doc = TEMPLATE.replace("{{BODY}}", body).replace("{{VERSION}}", __version__).replace("{{DATE}}", today.isoformat())
    (REP / "GH-PM25_Technical_Report.html").write_text(doc, encoding="utf-8")
    print("report written", REP / "GH-PM25_Technical_Report.html")


TEMPLATE = r"""<meta charset="utf-8">
<title>GH-PM25 Technical Report</title>
<meta name="description" content="Data sources, algorithms and evaluation of the GH-PM25 daily 1-km PM2.5 product for Ghana.">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans+Condensed:wght@500;600;700&family=Source+Serif+4:ital,opsz,wght@0,8..60,400;0,8..60,600;1,8..60,400&display=swap">
<script>window.MathJax = { tex: { inlineMath: [['\\(', '\\)']], displayMath: [['\\[', '\\]']] }, svg: { fontCache: 'global' } };</script>
<script defer src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/3.2.2/es5/tex-svg-full.min.js"></script>
<style>
:root {
  color-scheme: light;
  --ground: #f4f5f2; --paper: #ffffff; --ink: #1d2530; --ink-2: #3f4854; --muted: #5c6672; --rule: #dde1dc; --rule-2: #c7ccc6;
  --accent: #1c5cab; --accent-soft: #e6eef8; --ochre: #b8741a; --ochre-soft: #f7ecdb; --code: #eef0ec;
  --serif: "Source Serif 4", "Iowan Old Style", Georgia, serif; --cond: "IBM Plex Sans Condensed", "Arial Narrow", system-ui, sans-serif; --mono: "IBM Plex Mono", ui-monospace, Consolas, monospace;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) { color-scheme: dark; --ground: #11151a; --paper: #171c22; --ink: #e8ebee; --ink-2: #c4cad1; --muted: #98a1ab; --rule: #29313a; --rule-2: #3a434d;
    --accent: #7aaef0; --accent-soft: #1b2b40; --ochre: #e0a45a; --ochre-soft: #2e2416; --code: #1f252c; }
}
:root[data-theme="dark"] { color-scheme: dark; --ground: #11151a; --paper: #171c22; --ink: #e8ebee; --ink-2: #c4cad1; --muted: #98a1ab; --rule: #29313a; --rule-2: #3a434d;
  --accent: #7aaef0; --accent-soft: #1b2b40; --ochre: #e0a45a; --ochre-soft: #2e2416; --code: #1f252c; }
* { box-sizing: border-box; }
html { scroll-behavior: smooth; }
@media (prefers-reduced-motion: reduce) { html { scroll-behavior: auto; } }
body { margin: 0; background: var(--ground); color: var(--ink); font-family: var(--serif); font-size: 17px; line-height: 1.62; padding-inline: 16px; }
a { color: var(--accent); text-underline-offset: 2px; }
a:focus-visible, summary:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
.shell { max-width: 1240px; margin: 0 auto; display: grid; grid-template-columns: 230px minmax(0, 1fr); gap: 48px; padding-block: 40px 80px; }
nav.toc { position: sticky; top: 24px; align-self: start; max-height: calc(100vh - 48px); overflow: auto; font-family: var(--cond); font-size: 14px; }
nav.toc .toc-title { text-transform: uppercase; letter-spacing: .08em; font-size: 12px; color: var(--muted); margin-bottom: 10px; }
nav.toc ol { list-style: none; padding: 0; margin: 0; display: grid; gap: 2px; }
nav.toc ol ol { padding-left: 12px; margin: 2px 0 6px; }
nav.toc a { color: var(--ink-2); text-decoration: none; display: block; padding: 2px 0; }
nav.toc a:hover { color: var(--accent); }
main { min-width: 0; }
.doc { background: var(--paper); border: 1px solid var(--rule); padding: 56px clamp(20px, 5vw, 72px); }
.prose > * { max-width: 68ch; }
.prose > figure.wide, .prose > .tablewrap, .prose > .spec, .prose > .keyfind, .prose > .math { max-width: none; }
h1, h2, h3, h4 { font-family: var(--cond); color: var(--ink); text-wrap: balance; line-height: 1.2; }
h1 { font-size: 42px; font-weight: 700; margin: 0 0 12px; letter-spacing: -.01em; }
h2 { font-size: 27px; font-weight: 600; margin: 64px 0 14px; padding-top: 18px; border-top: 2px solid var(--ink); }
h3 { font-size: 20px; font-weight: 600; margin: 34px 0 8px; }
h4 { font-size: 16px; font-weight: 600; margin: 22px 0 6px; color: var(--ink-2); }
h2 .secnum, h3 .secnum { color: var(--accent); margin-right: .45em; font-variant-numeric: tabular-nums; }
p { margin: 0 0 14px; }
.eyebrow { font-family: var(--cond); text-transform: uppercase; letter-spacing: .1em; font-size: 13px; color: var(--accent); font-weight: 600; }
.subtitle { font-size: 21px; color: var(--ink-2); margin: 0 0 28px; max-width: 60ch; }
.spec { display: grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); border-top: 1px solid var(--ink); border-bottom: 1px solid var(--ink); margin: 28px 0 30px; }
.spec div { padding: 11px 14px 11px 0; border-bottom: 1px solid var(--rule); }
.spec dt { font-family: var(--cond); text-transform: uppercase; letter-spacing: .07em; font-size: 11.5px; color: var(--muted); }
.spec dd { margin: 2px 0 0; font-family: var(--mono); font-size: 14px; color: var(--ink); }
.keyfind { background: var(--ochre-soft); border-left: 3px solid var(--ochre); padding: 18px 22px; margin: 22px 0; }
.keyfind h4 { margin-top: 0; font-family: var(--cond); text-transform: uppercase; letter-spacing: .08em; font-size: 13px; color: var(--ochre); }
.keyfind ul { margin: 0; padding-left: 20px; }
.keyfind li { margin-bottom: 6px; }
ul, ol { padding-left: 22px; }
li { margin-bottom: 4px; }
code, .mono { font-family: var(--mono); font-size: .86em; background: var(--code); padding: 1px 5px; border-radius: 3px; }
pre { font-family: var(--mono); font-size: 13.5px; background: var(--code); padding: 14px 16px; overflow-x: auto; line-height: 1.5; max-width: 100%; }
figure { margin: 30px 0; }
figure img { width: 100%; max-width: 100%; height: auto; display: block; border: 1px solid var(--rule); background: #fff; }
figcaption { font-size: 14.5px; color: var(--ink-2); margin-top: 8px; line-height: 1.5; max-width: 90ch; }
.fignum { font-family: var(--cond); font-weight: 600; color: var(--ink); }
.tablewrap { overflow-x: auto; margin: 22px 0 28px; }
.tablewrap.tall { max-height: 560px; overflow-y: auto; }
table { border-collapse: collapse; width: 100%; font-family: var(--cond); font-size: 14px; }
caption { caption-side: top; text-align: left; font-family: var(--serif); font-size: 14.5px; color: var(--ink-2); padding-bottom: 8px; }
th { text-align: left; font-weight: 600; color: var(--muted); border-bottom: 1px solid var(--ink); padding: 6px 10px; white-space: nowrap; position: sticky; top: 0; background: var(--paper); }
td { padding: 5px 10px; border-bottom: 1px solid var(--rule); vertical-align: top; }
td.num, th.num { text-align: right; font-family: var(--mono); font-size: 13px; font-variant-numeric: tabular-nums; white-space: nowrap; }
.cvtable td:first-child, .cvtable th:first-child { min-width: 250px; }
.cvtable tbody tr:last-child td { font-weight: 700; color: var(--ink); background: var(--accent-soft); }
.math { overflow-x: auto; margin: 14px 0 18px; }
.refs li { font-size: 15px; margin-bottom: 8px; }
.missing { color: var(--muted); font-style: italic; }
.docfoot { font-family: var(--cond); color: var(--muted); font-size: 13px; margin-top: 40px; border-top: 1px solid var(--rule); padding-top: 12px; }
@media (max-width: 960px) { .shell { grid-template-columns: minmax(0, 1fr); gap: 20px; } nav.toc { position: static; max-height: none; } .doc { padding: 28px 18px; } h1 { font-size: 32px; } }
@media print { body { background: #fff; } nav.toc { display: none; } .shell { display: block; } .doc { border: 0; padding: 0; } .tablewrap.tall { max-height: none; } }
</style>
<div class="shell">
<nav class="toc" aria-label="Contents">
  <div class="toc-title">Contents</div>
  <ol>
    <li><a href="#summary">Executive summary</a></li>
    <li><a href="#s1">1 Background and objectives</a></li>
    <li><a href="#s2">2 Domain and product design</a></li>
    <li><a href="#s3">3 Data</a><ol><li><a href="#s3-1">3.1 Ground monitors</a></li><li><a href="#s3-2">3.2 Sensor harmonisation</a></li><li><a href="#s3-3">3.3 Composition and weather</a></li><li><a href="#s3-4">3.4 Fires</a></li><li><a href="#s3-5">3.5 Static covariates</a></li><li><a href="#s3-6">3.6 Provenance</a></li></ol></li>
    <li><a href="#s4">4 Methods</a><ol><li><a href="#s4-1">4.1 Predictors</a></li><li><a href="#s4-2">4.2 Stage 1 ensemble</a></li><li><a href="#s4-3">4.3 Stage 2 kriging</a></li><li><a href="#s4-4">4.4 Stage 3 uncertainty</a></li><li><a href="#s4-5">4.5 Validation design</a></li><li><a href="#s4-6">4.6 Operations</a></li></ol></li>
    <li><a href="#s5">5 Results</a><ol><li><a href="#s5-1">5.1 Calibration</a></li><li><a href="#s5-2">5.2 Cross-validation</a></li><li><a href="#s5-3">5.3 Benchmarking</a></li><li><a href="#s5-4">5.4 Drivers</a></li><li><a href="#s5-5">5.5 Uncertainty</a></li><li><a href="#s5-6">5.6 Maps</a></li><li><a href="#s5-7">5.7 Exposure</a></li></ol></li>
    <li><a href="#s6">6 Discussion and limitations</a></li>
    <li><a href="#s7">7 Reproducibility</a></li>
    <li><a href="#refs">References</a></li>
    <li><a href="#appA">Appendices</a></li>
  </ol>
</nav>
<main><article class="doc prose">
{{BODY}}
<p class="docfoot">GH-PM25 technical report · product version {{VERSION}} · compiled {{DATE}}. Figures and tables are regenerated from pipeline outputs by <code>scripts/06_make_figures.py</code> and <code>scripts/07_build_report.py</code>.</p>
</article></main>
</div>
"""

BODY = r"""
<div class="eyebrow">Technical report · Air quality · Ghana</div>
<h1>GH-PM25: daily 1-km fine particulate matter for Ghana</h1>
<p class="subtitle">Data sources, sensor harmonisation, a hybrid machine-learning–geostatistical downscaling algorithm, uncertainty quantification, validation, and the operational monitoring system.</p>

<dl class="spec">
  <div><dt>Product</dt><dd>GH-PM25 v{version}</dd></div>
  <div><dt>Variable</dt><dd>24-h mean PM2.5, µg m⁻³</dd></div>
  <div><dt>Grid</dt><dd>0.01° WGS84 (≈1.1 km), 650 × 455</dd></div>
  <div><dt>Extent</dt><dd>3.30°W–1.25°E, 4.70–11.20°N</dd></div>
  <div><dt>Training record</dt><dd>{p0} → {p1}</dd></div>
  <div><dt>Training data</dt><dd>{n_train} station-days, {n_sites} sites</dd></div>
  <div><dt>Latency</dt><dd>D+1 forecast · D0 nowcast · D−7 final</dd></div>
  <div><dt>Uncertainty</dt><dd>90% conformal interval per cell</dd></div>
  <div><dt>Spatial CV R² / RMSE</dt><dd>{head_r2} / {head_rmse} µg m⁻³</dd></div>
  <div><dt>Formats</dt><dd>NetCDF-4 CF-1.8, GeoTIFF, CSV, web</dd></div>
</dl>

<h2 id="summary">Executive summary</h2>
<p>GH-PM25 is a daily map of fine particulate matter (PM2.5) for all of Ghana at about 1 km resolution. It runs automatically each day and has a public web portal. The product combines ground measurements with atmospheric-composition modelling, reanalysis weather, satellite fire detections and detailed land-surface information. Every modelling choice (hyperparameters, ensemble composition, strength of geostatistical correction) was selected under spatial cross-validation rather than random cross-validation. Each pixel carries a distribution-free prediction interval.</p>
<div class="keyfind">
<h4>Key findings</h4>
<ul>
<li><b>Training data.</b> {n_train} daily observations from {n_sites} monitors ({n_gh_sites} in Ghana, {n_gh_days} station-days) were quality-controlled and harmonised to US State Department BAM-1020 reference monitors.</li>
<li><b>Accuracy in unseen cities.</b> Predicting whole cities the model never saw (leave-city-cluster-out cross-validation), GH-PM25 reaches R² = {head_r2}, RMSE = {head_rmse} µg m⁻³ and MAE = {head_mae} µg m⁻³. For Ghanaian stations: R² = {gh_r2}, RMSE = {gh_rmse} µg m⁻³.</li>
<li><b>Improvement over the global model.</b> The raw CAMS global model scores R² = {cams_r2} (RMSE = {cams_rmse} µg m⁻³); GH-PM25 cuts RMSE by {gain}. A gradient-boosting model with the aerosol-optical-depth, meteorology and day-of-year inputs typical of earlier work reaches R² = {pl_r2} under the same test.</li>
<li><b>Optimistic random CV.</b> Random 10-fold cross-validation, the design behind many headline numbers in the literature, gives R² = {rnd_r2}. The gap to the spatial score shows why spatial validation is needed.</li>
<li><b>Uncertainty.</b> The 90% prediction intervals cover {cov90} of held-out observations in unseen cities.</li>
<li><b>Record length.</b> {dates_mapped} daily maps have been produced so far.</li>
</ul>
</div>
<p>{acc_line}</p>

<h2 id="s1"><span class="secnum">1</span>Background and objectives</h2>
<p>Ambient PM2.5 is the leading environmental health risk in West Africa. Ghana's air-quality problem has two parts. In the south, rapidly growing cities (Greater Accra, Kumasi, Sekondi-Takoradi) add traffic, waste burning, industry and domestic biomass fuels. Nationwide, the Harmattan (roughly December–February) brings Saharan mineral dust and smoke from savanna burning. In Accra, measured PM2.5 during the Harmattan has averaged about 90 µg m⁻³, against about 22 µg m⁻³ outside it (Alli et al., 2021; 2023). The WHO 2021 guideline for 24-hour PM2.5 is 15 µg m⁻³, and Ghana EPA's national 24-hour standard (GS 1236:2019) is 35 µg m⁻³.</p>
<p>Regulatory monitoring is sparse. Until early 2025 the only continuous reference-grade monitor with open data was the US Embassy BAM-1020 in Accra. Low-cost sensor networks have since grown fast: Breathe Accra (Clarity) from 2024, AirGradient deployments by the University of Ghana, Afri-SET and Ghana EPA, and a national Clarity network in regional capitals from 2026. Without calibration and a spatial model, these networks describe only where sensors happen to be.</p>
<p>The most recent km-scale product for Ghana (Anand et al., 2026) was trained with XGBoost on MAIAC AOD, OMI/TROPOMI trace gases, ERA5-Land and MERRA-2. It reports R² = 0.72 under random cross-validation, falling to 0.49–0.57 when leaving out locations. Its hyperparameters were tuned on random folds, it gives no pixel-level uncertainty, and it uses no land-use, population, road or fire information. GH-PM25 was designed to close these gaps:</p>
<ol>
<li>Harmonise low-cost sensors to reference-grade monitors before training, with network-specific corrections chosen by cross-validation and guarded against extrapolation.</li>
<li>Use predictors that carry real kilometre-scale information (built-up area, population, night-time lights, roads, terrain) and physically meaningful dynamic information: ventilation, dust fraction, and upwind fire radiative power.</li>
<li>Train on a regional West African network, so dust and smoke regimes rare in Ghana's coastal cities are represented.</li>
<li>Combine heterogeneous learners into a robust ensemble, and test whether same-day residual kriging adds skill before using it.</li>
<li>Give every pixel a calibrated prediction interval and an area-of-applicability flag.</li>
<li>Evaluate with validation designs that stop information leaking from nearby monitors, and tune and select models under those designs only.</li>
<li>Run operationally with open, keyless data services: nowcast, one-day forecast and weekly final reprocessing, published to an interactive portal.</li>
</ol>

<h2 id="s2"><span class="secnum">2</span>Domain and product design</h2>
<p>The mapping grid is a regular 0.01° latitude–longitude lattice (≈1.11 km at the equator) covering Ghana from 3.30°W to 1.25°E and 4.70°N to 11.20°N. It has 650 rows and 455 columns; cells whose centres fall inside the geoBoundaries national outline are mapped. The day is 00–24 UTC, which is also Ghana's local time. Model training uses a wider regional domain (9°W–9°E, 4–13.5°N) that includes monitors in Côte d'Ivoire, Togo, Burkina Faso, Mali and Nigeria. The training record starts on {p0}, the first day of CAMS global composition fields in the archive used.</p>

<h2 id="s3"><span class="secnum">3</span>Data</h2>
<h3 id="s3-1"><span class="secnum">3.1</span>Ground monitors</h3>
<p>Ground data come from two keyless archives:</p>
<ul>
<li><b>OpenAQ public archive</b> (<code>s3://openaq-data-archive</code>, one gzip CSV per location per day) for low-cost networks and the State Department monitors it redistributes.</li>
<li><b>AirNow "EmbassyHistorical" files</b> (<code>files.airnowtech.org</code>) for the Lomé and Ouagadougou reference monitors, and to fill gaps at Accra and Abidjan.</li>
</ul>
<p>No OpenAQ location metadata dump exists, so Ghanaian and regional location IDs were found by listing every location prefix in the archive (55,567 locations), reading coordinates from each location's first file, and pulling operator details from the public explorer pages.</p>
<p>Hourly quality control:</p>
<ol>
<li>physical range 0 < PM2.5 ≤ 1000 µg m⁻³, and conversion of sub-hourly records to hourly means;</li>
<li>removal of stuck sensors (≥ 6 identical consecutive hours);</li>
<li>a Hampel spike filter (|x − 25-h rolling median| > max(6 × 1.4826 × MAD, 30 µg m⁻³)).</li>
</ol>
<p>A daily mean is kept only with at least 12 valid hours and at least 2 hours in each 6-hour block, so the diurnal cycle is represented. One network (Aurassure, Nigeria) was excluded after screening because its daily values were physically implausible.</p>
{net_table}
{f1}
{f2}

<h3 id="s3-2"><span class="secnum">3.2</span>Low-cost sensor harmonisation</h3>
<p>Optical sensors over-read at high humidity (hygroscopic growth) and under-read coarse dust, and the size of these errors varies by manufacturer. Each low-cost site within 10 km of a BAM-1020 was paired with the reference daily mean for the same day. For each network, three corrections were compared by month-blocked cross-validation:</p>
<ul>
<li>identity (no correction);</li>
<li>a constant multiplicative ratio;</li>
<li>a robust (Huber) log-linear model:</li>
</ul>
<div class="math">\[ \ln \mathrm{PM}_{\mathrm{ref}} = a + b \,\ln \mathrm{PM}_{\mathrm{lcs}} + c\, \mathrm{RH}^{*}, \qquad \mathrm{RH}^{*} = \operatorname{clip}\!\left(\mathrm{RH},\, \mathrm{RH}_{2\%},\, \mathrm{RH}_{98\%}\right),\quad b \in [0.4, 1.3] \]</div>
<p>RH is MERRA-2 relative humidity at the sensor, so any sensor can be corrected even without an on-board humidity channel. Humidity is clamped to the range seen in co-location. This matters because coastal Accra rarely has the very dry Harmattan air found inland, and an unconstrained humidity term produced implausible corrections in early tests.</p>
<p>Correction models are applied hierarchically: network first, then sensor family, then all low-cost sensors pooled. Each observation carries its calibration uncertainty \( \sigma_{\mathrm{cal}} \) forward into the training weights. Networks whose calibrated cross-validated R² stays below 0.30 are excluded.</p>
{cal_table}
{f3}

<h3 id="s3-3"><span class="secnum">3.3</span>Atmospheric composition and meteorology</h3>
<p><b>CAMS global composition.</b> The ECMWF CAMS global analyses and forecasts (IFS-COMPO, 0.4°, 3-hourly) assimilate MODIS and VIIRS aerosol optical depth and supply surface PM2.5, PM10, dust, AOD 550 nm, CO and NO₂. They were retrieved through the Open-Meteo air-quality API at {n_cams_nodes} lattice nodes (0.8° spacing aligned to the native grid, plus the nodes surrounding every out-of-country station) and averaged to daily means. The free service limits request volume, so downloads are paced with a token bucket.</p>
<p><b>MERRA-2 weather and aerosol.</b> Meteorology and aerosol come from NASA POWER (MERRA-2, with GEOS FP-IT for recent days; 0.5° × 0.625°):</p>
<ul>
<li>near-surface temperature, humidity and dew point;</li>
<li>precipitation (IMERG-corrected);</li>
<li>10-m winds and surface pressure;</li>
<li>downward shortwave radiation and cloud fraction;</li>
<li>MERRA-2 total AOD;</li>
<li>the pressure at the top of the planetary boundary layer, converted to a depth with the hypsometric relation \( h \approx H \ln(p_s/p_{\mathrm{PBL}}) \), H = 8.4 km.</li>
</ul>
<p><b>Forecast and nowcast days.</b> For days not yet in POWER, ECMWF IFS 0.25° forecasts are used. They are mapped to MERRA-2 by per-node linear regression over the most recent 45 overlapping days, which removes the systematic difference between the two models.</p>
<h3 id="s3-4"><span class="secnum">3.4</span>Fires</h3>
<p>Biomass burning is a major PM2.5 source in the savanna belt, yet the baseline product did not use fire data. GH-PM25 uses a single-sensor record from Suomi-NPP VIIRS at 375 m, giving {n_fires} quality-screened detections:</p>
<ul>
<li><b>2022–2024:</b> NASA FIRMS yearly country archives (VNP14IMG);</li>
<li><b>From 2025:</b> the NOAA Enterprise VIIRS I-band fire EDR (EFIRE v1r3), published keyless on the NOAA Open Data Dissemination bucket with about 1–2 h latency. The legacy AF-Iband EDR ends in February 2025.</li>
</ul>
<p>Detections with nominal or high confidence, no persistent-anomaly flag (gas flares, industry) and valid FRP are kept.</p>
<p>A full day of EFIRE holds about 1,000 granules, so GH-PM25 selects granules with a simple sun-synchronous orbit model:</p>
<ol>
<li>about a dozen probe granules each day give nadir position and pass direction from their embedded geolocation grid;</li>
<li>these yield the ascending-node time and longitude, \( \phi(t)=\arcsin(\sin i\,\sin u),\ u=2\pi(t-t_\Omega)/P \);</li>
<li>only granules whose nadir track passes within one swath of West Africa are read.</li>
</ol>
<p>A first approach based on 16-day repeat-cycle templates failed because of along-track timing drift, which is why the orbit model is used. {fire_cap} {ov_txt}</p>
{f18}

<h3 id="s3-5"><span class="secnum">3.5</span>Static 1-km covariates</h3>
<p>Kilometre-scale spatial structure comes from:</p>
<ul>
<li><b>ESA WorldCover 2021</b> (10 m): land-cover class fractions.</li>
<li><b>Copernicus DEM GLO-90</b>: mean elevation, sub-grid roughness, and elevation relative to the surrounding 10 km.</li>
<li><b>WorldPop R2025A 2024</b>: constrained, UN-adjusted population density.</li>
<li><b>VIIRS DNB monthly composites</b> (World Bank "Light Every Night"): median night-time radiance over November–February.</li>
<li><b>OpenStreetMap</b> (motorway to secondary roads): major-road density and distance to the nearest major road.</li>
<li><b>Natural Earth</b>: distance to the coast.</li>
</ul>
<p>Cloud-optimised GeoTIFFs are read in windows over HTTP and aggregated to the 0.01° lattice. Urban layers also get 3-km and 10-km Gaussian neighbourhood versions that represent upwind and area-source influence.</p>
{f4}

<h3 id="s3-6"><span class="secnum">3.6</span>Data provenance</h3>
<div class="tablewrap"><table><caption>Table 1. Input datasets, roles and access (all keyless).</caption>
<thead><tr><th>Dataset</th><th>Provider</th><th>Resolution</th><th>Access point</th><th>Latency</th><th>Role</th></tr></thead><tbody>
<tr><td>OpenAQ archive</td><td>OpenAQ; operators Clarity, AirGradient, Data354, IQAir…</td><td>point, ≤hourly</td><td><code>openaq-data-archive.s3.amazonaws.com</code></td><td>1–3 d</td><td>target (calibrated)</td></tr>
<tr><td>BAM-1020 embassy monitors</td><td>US Department of State / AirNow</td><td>point, hourly</td><td><code>files.airnowtech.org/airnow/EmbassyHistorical</code></td><td>1 d (Accra feed halted Mar 2025)</td><td>reference, calibration</td></tr>
<tr><td>CAMS global composition</td><td>ECMWF / Copernicus</td><td>0.4°, 3-hourly</td><td><code>air-quality-api.open-meteo.com</code></td><td>forecast</td><td>predictors</td></tr>
<tr><td>MERRA-2 / GEOS FP-IT</td><td>NASA GMAO via NASA POWER</td><td>0.5° × 0.625°, daily</td><td><code>power.larc.nasa.gov/api/temporal/daily/regional</code></td><td>1–3 d</td><td>predictors</td></tr>
<tr><td>ECMWF IFS 0.25°</td><td>ECMWF via Open-Meteo</td><td>0.25°, hourly</td><td><code>api.open-meteo.com/v1/forecast</code></td><td>forecast</td><td>nowcast/forecast met</td></tr>
<tr><td>VIIRS 375 m fires</td><td>NASA FIRMS; NOAA NESDIS EFIRE</td><td>375 m</td><td><code>firms.modaps.eosdis.nasa.gov/data/country</code>; <code>noaa-nesdis-snpp-pds</code></td><td>~2 h</td><td>fire predictors</td></tr>
<tr><td>ESA WorldCover v200</td><td>ESA</td><td>10 m</td><td><code>esa-worldcover.s3.eu-central-1.amazonaws.com</code></td><td>static</td><td>land cover</td></tr>
<tr><td>Copernicus DEM GLO-90</td><td>ESA / Airbus</td><td>90 m</td><td><code>copernicus-dem-90m.s3.amazonaws.com</code></td><td>static</td><td>terrain</td></tr>
<tr><td>WorldPop R2025A</td><td>University of Southampton</td><td>30″</td><td><code>data.worldpop.org</code></td><td>static</td><td>population, exposure weights</td></tr>
<tr><td>VIIRS DNB monthly</td><td>World Bank / NOAA EOG</td><td>15″</td><td><code>globalnightlight.s3.amazonaws.com</code></td><td>static</td><td>night lights</td></tr>
<tr><td>OpenStreetMap roads</td><td>OSM contributors</td><td>vector</td><td>Overpass API</td><td>static</td><td>roads</td></tr>
<tr><td>geoBoundaries GHA ADM0–2</td><td>William & Mary geoLab</td><td>vector</td><td>GitHub release</td><td>static</td><td>mask, statistics</td></tr>
</tbody></table></div>

<h2 id="s4"><span class="secnum">4</span>Methods</h2>
<h3 id="s4-1"><span class="secnum">4.1</span>Predictors</h3>
<p>{n_features} predictors are built identically for station-days and grid cells:</p>
<ul>
<li>dynamic fields bilinearly interpolated from their native lattices;</li>
<li>1–2-day lags and 3-day accumulations;</li>
<li>derived physical terms: ventilation coefficient (PBL depth × wind speed), dust fraction (dust / PM10), wind-direction components;</li>
<li>fire terms: total FRP within 25 and 100 km, and an upwind-weighted influence \( I(x)=\ln\!\big(1+\sum_j \mathrm{FRP}_j \max(\cos\theta_j,0)\, e^{-d_j/150\,\mathrm{km}}\big) \), where \( \theta_j \) is the angle between the fire-to-target vector and the mean 10-m wind;</li>
<li>static covariates and calendar terms.</li>
</ul>
<p>The full list is in Appendix A.</p>

<h3 id="s4-2"><span class="secnum">4.2</span>Stage 1: heterogeneous stacked ensemble</h3>
<p>Four learners with different inductive biases and target formulations are trained on \( y=\ln \mathrm{PM2.5} \):</p>
<ul>
<li><b>M1 LightGBM</b> (leaf-wise boosting);</li>
<li><b>M2 XGBoost</b> on the multiplicative CAMS correction \( y-\ln \mathrm{CAMS} \), so it learns where and when CAMS is biased;</li>
<li><b>M3 CatBoost</b> (ordered boosting with symmetric trees);</li>
<li><b>M4 ExtraTrees</b> (randomised, low-variance trees).</li>
</ul>
<p>Each learner's hyperparameters were tuned by random search under leave-city-cluster-out cross-validation, using a fold assignment independent of the evaluation folds (Appendix B). Tuning under random folds would reward models that memorise near-duplicate neighbours.</p>
<p>Sample weights down-weight dense city clusters and noisier observations:</p>
<div class="math">\[ w_i = \frac{1}{\sqrt{n_{\mathrm{sites}}(\mathrm{cluster}_i)}}\cdot \frac{0.2^2+0.1^2}{0.2^2+\sigma_{\mathrm{cal},i}^2} \]</div>
<p>The learners are combined in log space, with the combination rule chosen from 16 candidates:</p>
<ul>
<li>a non-negative least-squares stack \( \hat y^{(1)} = \sum_k \beta_k \hat y_k + \beta_0 \) (\( \beta_k\ge 0,\ \sum_k\beta_k = 1 \));</li>
<li>equal-weight averages of each of the 15 non-empty subsets of learners.</li>
</ul>
<p>The selection criterion was fixed before scoring: the highest mean R² across leave-city-cluster-out, leave-site-out and leave-month-out validation, computed for both all stations and Ghanaian stations. Stack weights were always fitted <em>nested</em>, without the held-out fold, because fitting them on the same out-of-fold predictions inflated R² by 0.01–0.04.</p>
<p>The NNLS stack scored highest on in-sample weights (LightGBM 0.61, ExtraTrees 0.35) but transferred poorly to Ghanaian stations. The equal-weight <b>CatBoost + ExtraTrees</b> average was the most robust and is the operational ensemble; the resulting weights are LightGBM {w_lgbm}, XGBoost-ratio {w_xgb}, CatBoost {w_cat}, ExtraTrees {w_et}. The full ranking is in <code>outputs/ensemble_selection.csv</code>. Predictions are converted back from log space with Duan's smearing factor ({smear}).</p>

<h3 id="s4-3"><span class="secnum">4.3</span>Stage 2: same-day residual kriging (evaluated, strength selected by validation)</h3>
<p>Stage-1 residuals combine a persistent per-station offset (sensor calibration error, micro-environment) with a day-specific anomaly. The offset is estimated as a shrunken mean,</p>
<div class="math">\[ r_{it} = \ln y_{it} - \hat y^{(1)}_{it} = b_i + e_{it},\qquad \hat b_i = \frac{\sum_t r_{it}}{n_i + 15} \]</div>
<p>and only the anomalies \( e_{it} \) are kriged, as a zero-mean Gaussian field with covariance \( C(d)=\sigma_s^2 e^{-d/L} + \tau^2 \mathbb{1}_{d=0} \). The parameters are fitted by weighted least squares to the pooled same-day correlogram of leave-cluster-out anomalies: \( c_0 \) = {c0}, L = {L} km.</p>
<p>The kriged anomaly enters as \( \hat y = \hat y^{(1)} + \lambda\, \hat e(x) \). The strength \( \lambda \in [0,1] \) is selected by leave-site-out validation, which mirrors the case where Stage 2 should help most: an unmonitored place inside a monitored city. The selected value was \( \lambda \) = {rk_lambda}. Section 5.2 shows why.</p>

<h3 id="s4-4"><span class="secnum">4.4</span>Stage 3: uncertainty and area of applicability</h3>
<p>Prediction intervals use locally adaptive split-conformal inference (Lei et al., 2018):</p>
<ol>
<li>A scale model \( \hat\sigma(x) \) (LightGBM) predicts the absolute log-error from the kriging variance, distance to the nearest same-day monitor, monitor density within 100 km, CAMS PM2.5, dust fraction, humidity, PBL depth, population, latitude, season and fire activity. It is cross-fitted by city cluster.</li>
<li>The conformity score is \( s_i=|r_i|/\hat\sigma(x_i) \), computed from leave-city-cluster-out residuals.</li>
<li>With \( q_{0.9} \) = {q90} the finite-sample quantile of the scores, the interval is \( [\exp(\hat y-q\hat\sigma),\ \exp(\hat y+q\hat\sigma)] \).</li>
</ol>
<p>Coverage is checked in a nested way: the quantile for each held-out cluster fold is computed only from the other folds.</p>
<p>A dissimilarity index (Meyer &amp; Pebesma, 2021) is computed on the 20 most important predictors, weighted by importance. Cells beyond the cross-validated threshold are flagged as outside the area of applicability.</p>

<h3 id="s4-5"><span class="secnum">4.5</span>Validation design</h3>
<p>Low-cost networks cluster tightly within cities, so random cross-validation leaks information between near-duplicate neighbours (Just et al., 2020; Ploton et al., 2020). Four designs are reported, from most to least optimistic:</p>
<ul>
<li><b>Random 10-fold</b> over station-days, for comparison with the literature;</li>
<li><b>Leave-month-out</b> (10 folds of calendar months);</li>
<li><b>Leave-site-out</b> (10 folds of sites);</li>
<li><b>Leave-city-cluster-out</b>: sites grouped by complete-linkage clustering at 30 km, 10 folds of clusters. This is the primary design. It asks whether the model can map a city with no monitors.</li>
</ul>
<p>All Stage-2 results use only same-day observations outside the held-out fold.</p>
<p>Baselines:</p>
<ul>
<li>raw CAMS;</li>
<li>CAMS with a log-linear bias correction on humidity, PBL depth and dust fraction;</li>
<li>a gradient-boosting model with only AOD, near-surface meteorology and day-of-year, the predictor family of earlier Ghana products.</li>
</ul>
<p>Metrics are R², Pearson r, RMSE, MAE, mean bias, normalised mean bias and the regression slope, all in µg m⁻³ on the original scale.</p>

<h3 id="s4-6"><span class="secnum">4.6</span>Operational system</h3>
<p>The daily run (<code>scripts/run_daily.py</code>, scheduled at 06:30 UTC) refreshes each input incrementally:</p>
<ol>
<li>CAMS analyses and forecast;</li>
<li>MERRA-2 from POWER, extended with bias-adjusted IFS;</li>
<li>the last 10 days of VIIRS fires;</li>
<li>the last 12 days of OpenAQ observations, calibrated with the stored network models.</li>
</ol>
<p>It then re-maps D−7 to D+1 and publishes the portal. Each day carries a product tier:</p>
<ul>
<li><b>forecast</b> (D+1): no observations;</li>
<li><b>nowcast</b> (D0);</li>
<li><b>near-real-time</b> (D−1 to D−6): re-processed as late data arrive;</li>
<li><b>final</b> (≥ D−7).</li>
</ul>
<p>A failed input is logged and the run continues with what is available; source freshness is published to the portal. The system is retrained every four weeks with the new ground data. Windows Task Scheduler and GitHub Actions definitions are provided.</p>

<h2 id="s5"><span class="secnum">5</span>Results</h2>
<h3 id="s5-1"><span class="secnum">5.1</span>Sensor harmonisation</h3>
<p>Table 3 and Figure 3 summarise calibration. Ghanaian AirGradient and Clarity sensors read higher than the Accra BAM-1020, while the Abidjan Data354 network reads well below the Abidjan reference. After correction, cross-validated bias is close to zero for the networks with co-location data, and MAE falls for every network with enough pairs. The co-location record has few days above about 60 µg m⁻³ and few dry days. That is a known limitation for the Harmattan peak, discussed in Section 6.</p>

<h3 id="s5-2"><span class="secnum">5.2</span>Cross-validated accuracy</h3>
{cv_table}
{f5}
{f6}
<p><b>Ranking of designs.</b> Skill falls in the expected order as validation gets stricter: random R² = {rnd_r2}, leave-month-out {temporal_r2}, leave-site-out {site_r2}, leave-city-cluster-out {head_r2}. The drop from random to cluster-out CV is the information leakage that random validation hides.</p>
<p><b>Ensemble vs single learners.</b> Under leave-city-cluster-out the best single learner, {best_single}, reaches R² = {best_single_r2}, while the operational ensemble reaches {stage1_r2}. The ensemble's main benefit is robustness: no single learner is best across every design and subset (Table 5, <code>ensemble_selection.csv</code>).</p>
<p><b>CAMS contributes little.</b> Raw CAMS explains almost none of the day-to-day and site-to-site variance (R² = {cams_r2}). Dropping every CAMS-derived predictor does not reduce spatial skill (Table 9). Over Ghana, the synoptic signal is carried almost entirely by MERRA-2 humidity, winds and boundary-layer depth, plus seasonality. The CAMS-ratio learner therefore receives zero weight.</p>
<h4>Stage-2 kriging: a negative result worth reporting</h4>
<p>Residual kriging is common in hybrid PM2.5 models, but here it did not add skill. Without offset removal, kriging raw residuals <em>lowered</em> leave-site-out R². With offset removal, the day-specific anomaly correlogram is short-ranged (\( c_0 \) = {c0}, L = {L} km). Leave-site-out R² still falls steadily as \( \lambda \) rises from 0 to 1 (<code>outputs/stage2_lambda_scan.csv</code>), and under leave-city-cluster-out the change is within ±0.003.</p>
<p>In dense low-cost networks the day-to-day residual at one sensor is mostly calibration noise that does not transfer to its neighbours. Stage 1 already captures the transferable spatial structure through urban-form and population predictors. The operational product therefore uses \( \lambda \) = {rk_lambda}. Observation proximity still informs the uncertainty model, and the kriging machinery is retained so \( \lambda \) is re-selected automatically at each retraining as reference monitoring improves.</p>
{sub_table}
{f7}
{f10}

<h3 id="s5-3"><span class="secnum">5.3</span>Benchmarking against the existing Ghana product</h3>
<p>The Anand et al. (2026) city-level daily series (Zenodo 19636051) was compared with city-mean calibrated observations within 15 km of each city centre, on days where both exist. GH-PM25 values in this comparison are leave-city-cluster-out predictions, meaning the whole city was withheld from training. The Anand et al. product, by contrast, included monitors in these cities during training, so on that count the comparison favours it.</p>
<p>A caveat runs the other way. The "observations" are low-cost sensors harmonised to the US Embassy BAM-1020 with this work's calibration, while Anand et al. used their own co-location corrections. Differences in bias and RMSE therefore partly reflect different calibration references. The correlation coefficient is unaffected by any linear recalibration, so it is the most robust basis for comparison.</p>
{comp_html}
{f15}
{f16}

<h3 id="s5-4"><span class="secnum">5.4</span>What drives the predictions</h3>
{f8}
{shap_table}
{abl_table}
{f17}

<h3 id="s5-5"><span class="secnum">5.5</span>Uncertainty and spatial dependence</h3>
{f9}
{cov_table}
<p>The residual correlogram shows correlation decaying over tens of kilometres ( \( c_0 \) = {c0}, L = {L} km). Kriging therefore sharpens maps inside and around monitored cities, but adds little in the sparsely monitored north. The scale model captures this: intervals widen with distance from monitors and during the Harmattan.</p>

<h3 id="s5-6"><span class="secnum">5.6</span>Maps</h3>
{f11}
{f12}
{f13}

<h3 id="s5-7"><span class="secnum">5.7</span>Population exposure</h3>
{f14}
{ann_html}
{reg_html}

<h2 id="s6"><span class="secnum">6</span>Discussion and limitations</h2>
<h4>What is new compared with existing products</h4>
<ul>
<li><b>Honest validation.</b> The ensemble is selected and weighted under leave-city-cluster-out validation, and all four designs are reported side by side.</li>
<li><b>Sensor harmonisation before training.</b> Network-specific corrections are guarded against extrapolation, and calibration uncertainty is carried into the training weights.</li>
<li><b>Kilometre-scale information.</b> Urban form, population, lights and roads, instead of downscaled 10–50 km satellite and reanalysis fields alone.</li>
<li><b>Explicit fire physics.</b> Upwind-weighted FRP from a consistent single-sensor 375 m record with near-real-time delivery.</li>
<li><b>Stage-2 residual kriging tested before use.</b> Its strength is set by validation, and the evidence that it does not help dense low-cost networks is reported.</li>
<li><b>Calibrated per-pixel uncertainty</b> and an area-of-applicability mask.</li>
<li><b>A fully keyless, reproducible operational chain</b> with product tiers and a public portal.</li>
</ul>
<h4>Limitations</h4>
<ul>
<li><b>Reference data.</b> Ghana's only open reference monitor (US Embassy Accra) stopped regular reporting in March 2025, and none has ever operated in the north. Calibration therefore rests on coastal co-location. Its humidity range is narrow and it sees few extreme dust days, so Harmattan peaks and northern concentrations carry more uncertainty than the headline figures suggest. The conformal intervals partly reflect this.</li>
<li><b>Short training record.</b> Training starts in August 2022 (CAMS availability), and most Ghanaian data are from 2024–2026. Hindcasts before 2022 are not produced, and trends over this short record should be interpreted cautiously.</li>
<li><b>No satellite AOD at 1 km.</b> The product does not use kilometre-scale satellite AOD (MAIAC or VIIRS), which needs an Earthdata account or heavy swath processing. CAMS assimilates satellite AOD at coarse resolution. Adding MAIAC as an optional input is a straightforward extension (Section 7).</li>
<li><b>Static layers.</b> Static layers are fixed (WorldCover 2021, WorldPop 2024, VIIRS 2023–24), so rapid urban change is not captured.</li>
<li><b>Operational sources.</b> Operational inputs depend on third-party free services with rate limits (Open-Meteo, NASA POWER). The system logs failures and degrades gracefully, but availability is not guaranteed.</li>
</ul>
<h4>Recommendations</h4>
<ul>
<li>Co-locate at least one sensor of each network with a BAM or FEM monitor in Tamale or Bolgatanga for a full Harmattan season.</li>
<li>Restore open reporting of a reference monitor in Accra.</li>
<li>Obtain a free OpenAQ API key to enable hourly live station layers.</li>
<li>Add MAIAC AOD when Earthdata credentials are available.</li>
<li>Recalibrate and retrain monthly as the national Clarity network grows.</li>
</ul>

<h2 id="s7"><span class="secnum">7</span>Reproducibility</h2>
<pre>ghana_pm25/            python package
  sources/ground.py      OpenAQ archive + AirNow embassy files, QC, daily means
  sources/openmeteo.py   CAMS composition (paced, cached)
  sources/power.py       MERRA-2 / GEOS via NASA POWER
  sources/firms.py       FIRMS archives, EFIRE EDR, orbit-model granule selection, fire predictors
  sources/static.py      WorldCover, DEM, WorldPop, VIIRS lights, OSM roads, coast distance
  calibration.py         sensor harmonisation
  features.py            predictor engineering
  model.py               ensemble, stacking, correlogram, kriging, conformal, AOA
  predict.py             daily 1-km mapping and regional statistics
  nrt.py                 near-real-time updates, IFS bias adjustment, product tiers
  export.py              NetCDF / GeoTIFF / portal encoders
scripts/
  01_download_ground.py  02_build_static.py  10_download_dynamic.py
  03_train_evaluate.py   04_predict_history.py  05_export_portal.py
  06_make_figures.py     07_build_report.py     run_daily.py
  qa_fire_template.py    qa_portal.py           schedule_windows_task.ps1
portal/                static web portal (MapLibre GL + d3)
outputs/daily/YYYY/GH-PM25_1km_YYYYMMDD.nc</pre>
<p>A full rebuild runs the scripts in numerical order. The operational cycle is <code>python scripts/run_daily.py</code>, and the portal is served as static files (for example <code>python -m http.server --directory portal</code>). Software: Python 3.12, LightGBM, XGBoost, CatBoost, scikit-learn, SHAP, xarray, rasterio, shapely, MapLibre GL JS and d3.</p>

<h2 id="refs">References</h2>
<ol class="refs">
<li>Adong, P., Bainomugisha, E., Okure, D., Sserunjogi, R. (2022). Applying machine learning for large scale field calibration of low-cost PM2.5 and PM10 air pollution sensors. <i>Applied AI Letters</i> 3, e76.</li>
<li>Alli, A.S., Clark, S.N., Hughes, A., et al. (2021). Spatial-temporal patterns of ambient fine particulate matter (PM2.5) and black carbon (BC) pollution in Accra. <i>Environ. Res. Lett.</i> 16, 074013.</li>
<li>Alli, A.S., et al. (2023). High-resolution patterns and inequalities in ambient fine particle mass (PM2.5) and black carbon (BC) in the Greater Accra Metropolis, Ghana. <i>Sci. Total Environ.</i> 875, 162582.</li>
<li>Anand, A., Amooli, J.A., Amoah, S., et al., Westervelt, D.M. (2026). Two decades of kilometer-scale daily PM2.5 from satellite observations and machine learning reveal geographically diverging exposure in Ghana. EarthArXiv, doi:10.31223/X5KR3D.</li>
<li>Barkjohn, K.K., Gantt, B., Clements, A.L. (2021). Development and application of a United States-wide correction for PM2.5 data collected with the PurpleAir sensor. <i>Atmos. Meas. Tech.</i> 14, 4617–4637.</li>
<li>Di, Q., Amini, H., Shi, L., et al. (2019). An ensemble-based model of PM2.5 concentration across the contiguous United States with high spatiotemporal resolution. <i>Environ. Int.</i> 130, 104909.</li>
<li>Gueymard, C.A., Yang, D. (2020). Worldwide validation of CAMS and MERRA-2 reanalysis aerosol optical depth products using 15 years of AERONET observations. <i>Atmos. Environ.</i> 225, 117216.</li>
<li>Hengl, T., Nussbaum, M., Wright, M.N., Heuvelink, G.B.M., Gräler, B. (2018). Random forest as a generic framework for predictive modeling of spatial and spatio-temporal variables. <i>PeerJ</i> 6, e5518.</li>
<li>Inness, A., et al. (2019). The CAMS reanalysis of atmospheric composition. <i>Atmos. Chem. Phys.</i> 19, 3515–3556.</li>
<li>Just, A.C., Arfer, K.B., Rush, J., et al. (2020). Advancing methodologies for applying machine learning and evaluating spatiotemporal models of PM2.5 using satellite data over large regions. <i>Atmos. Environ.</i> 239, 117649.</li>
<li>Lei, J., G'Sell, M., Rinaldo, A., Tibshirani, R.J., Wasserman, L. (2018). Distribution-free predictive inference for regression. <i>J. Am. Stat. Assoc.</i> 113, 1094–1111.</li>
<li>McFarlane, C., Raheja, G., Malings, C., et al., Westervelt, D.M. (2021). Application of Gaussian mixture regression for the correction of low cost PM2.5 monitoring data in Accra, Ghana. <i>ACS Earth Space Chem.</i> 5, 2268–2279.</li>
<li>Meyer, H., Pebesma, E. (2021). Predicting into unknown space? Estimating the area of applicability of spatial prediction models. <i>Methods Ecol. Evol.</i> 12, 1620–1633.</li>
<li>Meyer, H., Pebesma, E. (2022). Machine learning-based global maps of ecological variables and the challenge of assessing them. <i>Nat. Commun.</i> 13, 2208.</li>
<li>Ploton, P., et al. (2020). Spatial validation reveals poor predictive performance of large-scale ecological mapping models. <i>Nat. Commun.</i> 11, 4540.</li>
<li>Raheja, G., Nimo, J., Appoh, E.K., et al., Westervelt, D.M. (2023). Low-cost sensor performance intercomparison, correction factor development, and 2+ years of ambient PM2.5 monitoring in Accra, Ghana. <i>Environ. Sci. Technol.</i> 57, 10708–10720.</li>
<li>Shtein, A., Kloog, I., Schwartz, J., et al. (2020). Estimating daily PM2.5 and PM10 over Italy using an ensemble model. <i>Environ. Sci. Technol.</i> 54, 120–128.</li>
<li>van Donkelaar, A., Hammer, M.S., Bindle, L., et al. (2021). Monthly global estimates of fine particulate matter and their uncertainty. <i>Environ. Sci. Technol.</i> 55, 15287–15300.</li>
<li>Wei, J., Li, Z., Lyapustin, A., et al. (2023). First close insight into global daily gapless 1 km PM2.5 pollution, variability, and health impact. <i>Nat. Commun.</i> 14, 8349.</li>
<li>Westervelt, D.M., Amooli, J.A., Anand, A. (2025). Twenty years of high spatiotemporal resolution estimates of daily PM2.5 in West Africa using satellite data, surface monitors, and machine learning. <i>ACS ES&amp;T Air</i> 2, 1468–1477.</li>
<li>World Health Organization (2021). <i>WHO global air quality guidelines: particulate matter (PM2.5 and PM10), ozone, nitrogen dioxide, sulfur dioxide and carbon monoxide.</i> Geneva.</li>
<li>Ghana Standards Authority (2019). GS 1236:2019 Environment and health protection — requirements for ambient air quality and point source/stack emissions.</li>
</ol>

<h2 id="appA">Appendix A. Predictor dictionary</h2>
{feat_table}
<h2 id="appB">Appendix B. Model configuration</h2>
<p>Each base learner's hyperparameters were chosen by random search (12 candidates plus the default) minimising weighted log-space RMSE under 5-fold leave-city-cluster-out cross-validation. The cluster-to-fold assignment was shuffled independently of the evaluation folds; the full search history is in <code>outputs/tuning_history.csv</code>.</p>
<div class="tablewrap"><table><caption>Table B1. Selected hyperparameters (unlisted values are defaults: LightGBM n=1200, lr=0.03, leaves=48, min_child=40, colsample=0.6, λ=5; XGBoost n=1000, lr=0.03, depth=7, min_child_weight=10; CatBoost 1500 iterations, lr=0.05, depth 8, l2=5; ExtraTrees 300 trees, min_leaf=5, max_features=0.5).</caption>
<thead><tr><th>Learner</th><th>Configuration</th></tr></thead><tbody>
{hp_rows}
<tr><td>Scale model</td><td class="mono">LightGBM n=400, lr=0.03, leaves=15, min_child=100</td></tr>
<tr><td>Kriging</td><td class="mono">exponential + nugget, 300 km search radius, Cholesky solve</td></tr>
<tr><td>AOA</td><td class="mono">top-20 gain-weighted predictors, 6,000 training reference points, threshold Q3 + 1.5·IQR</td></tr>
</tbody></table></div>
<h2 id="appC">Appendix C. Ghanaian monitoring sites</h2>
{st_table}
<h2 id="appD">Appendix D. Per-site accuracy</h2>
{ps_table}
"""

if __name__ == "__main__":
    main()
