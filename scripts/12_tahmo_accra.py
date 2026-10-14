"""TAHMO weather stations in Accra: download, experiments, portal export.

    python scripts/12_tahmo_accra.py download [--start 2024-08-01 | --days 10]   # needs TAHMO_USERNAME / TAHMO_PASSWORD
    python scripts/12_tahmo_accra.py experiments [--quick]           # E0-E3 below, writes outputs/tahmo/ and report/tahmo/
    python scripts/12_tahmo_accra.py export                          # portal/data/accra/tahmo.json for the Accra view

Experiments (all restricted to the Greater Accra area, tahmo.ACCRA_BBOX)
  E0  inventory      stations, variables and daily coverage the account can read
  E1  weather bias   MERRA-2/POWER (the model's weather input, ~50 km) against TAHMO daily values
  E2  calibration    low-cost sensor correction against the US Embassy BAM-1020 using the model's
                     humidity (MERRA-2) versus humidity measured at the nearest TAHMO station
  E3  PM2.5 model    does measured local weather help the GH-PM25 learners in Accra? Accra station-days
                     with MERRA-2 weather (A) versus TAHMO weather mapped to the MERRA-2 scale (B), under
                     leave-month-out (Accra months held out) and leave-Accra-out designs
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ghana_pm25 import calibration, config, features as F, model as M  # noqa: E402
from ghana_pm25.sources import tahmo  # noqa: E402

OUT = config.OUTPUTS / "tahmo"
REP = config.REPORT / "tahmo"
MATCH_KM = 15.0                                  # PM2.5 site -> nearest TAHMO station
MET_VARS = ["t2m", "t2m_range", "rh", "precip", "ws10", "ps", "ssrd"]
SWAP_VARS = ["t2m", "t2m_range", "rh", "precip", "ws10"]   # replaced in E3 (u10/v10 too when wind direction exists)
ACCRA_CENTRE = (5.60, -0.19)


def log(msg):
    print(msg, flush=True)


def records(df: pd.DataFrame) -> list[dict]:
    """JSON-safe records (NaN -> null)."""
    return [{k: (None if isinstance(v, float) and not np.isfinite(v) else v) for k, v in r.items()} for r in df.to_dict("records")]


def load_tahmo() -> tuple[pd.DataFrame, pd.DataFrame]:
    if not tahmo.DAILY.exists() or not tahmo.STATIONS.exists():
        raise SystemExit("no TAHMO data yet: run `python scripts/12_tahmo_accra.py download` with credentials set")
    st = tahmo.in_bbox(pd.read_parquet(tahmo.STATIONS))
    d = pd.read_parquet(tahmo.DAILY)
    d["date"] = pd.to_datetime(d.date)
    d = d[d.code.isin(st.code)]
    st = st[st.code.isin(d.code.unique())].reset_index(drop=True)
    return st, d


def nearest_station(lat, lon, st: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    dist = calibration.haversine(np.asarray(lat)[:, None], np.asarray(lon)[:, None], st.lat.to_numpy()[None], st.lon.to_numpy()[None])
    k = dist.argmin(1)
    return st.code.to_numpy()[k], dist[np.arange(len(k)), k]


def merra_at(lat, lon, dates, met_lat: F.Lattice, cols) -> pd.DataFrame:
    w = met_lat.weights(np.asarray(lat, float), np.asarray(lon, float))
    t = met_lat.t_index(pd.DatetimeIndex(dates))
    out = {}
    for c in cols:
        v = np.full(len(t), np.nan, dtype="float32")
        ok = t >= 0
        v[ok] = met_lat.sample(c, t[ok], tuple(x[ok] for x in w))
        out[c] = v
    return pd.DataFrame(out)


def stats(y, yhat) -> dict:
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    ok = np.isfinite(y) & np.isfinite(yhat)
    y, yhat = y[ok], yhat[ok]
    if len(y) < 10:
        return dict(n=int(len(y)))
    return dict(n=int(len(y)), mean_obs=float(y.mean()), bias=float(np.mean(yhat - y)), rmse=float(np.sqrt(np.mean((yhat - y) ** 2))),
                mae=float(np.mean(np.abs(yhat - y))), r=float(np.corrcoef(y, yhat)[0, 1]),
                r2=float(1 - np.sum((yhat - y) ** 2) / np.sum((y - y.mean()) ** 2)))


# ---------------------------------------------------------------------------
# E0 inventory
# ---------------------------------------------------------------------------
def e0_inventory(st, d) -> pd.DataFrame:
    rows = []
    for _, s in st.iterrows():
        g = d[d.code == s.code]
        row = dict(code=s.code, name=s["name"], lat=s.lat, lon=s.lon, first=f"{g.date.min():%Y-%m-%d}", last=f"{g.date.max():%Y-%m-%d}")
        for c in MET_VARS:
            row[f"days_{c}"] = int(g[c].notna().sum()) if c in g else 0
        rows.append(row)
    inv = pd.DataFrame(rows)
    dist = calibration.haversine(inv.lat, inv.lon, *ACCRA_CENTRE)
    inv.insert(4, "km_from_centre", np.round(dist, 1))
    return inv


# ---------------------------------------------------------------------------
# E1 weather bias of the model's MERRA-2 inputs
# ---------------------------------------------------------------------------
def linear_map_cv(x, y, groups) -> np.ndarray:
    """Out-of-fold y ~ a + b x (month-blocked), used to judge whether a simple correction generalises."""
    pred = np.full(len(x), np.nan)
    k = min(6, len(np.unique(groups)))
    if k < 2:
        return pred
    for tr, te in GroupKFold(k).split(x, groups=groups):
        ok = np.isfinite(x[tr]) & np.isfinite(y[tr])
        if ok.sum() < 20:
            continue
        b, a = np.polyfit(x[tr][ok], y[tr][ok], 1)
        pred[te] = a + b * x[te]
    return pred


def e1_weather(st, d, met_lat) -> tuple[pd.DataFrame, dict]:
    dd = d.merge(st[["code", "lat", "lon"]], on="code")
    mm = merra_at(dd.lat, dd.lon, dd.date, met_lat, MET_VARS)
    rows, maps = [], {}
    months = dd.date.dt.to_period("M").astype(str).to_numpy()
    for c in MET_VARS:
        if c not in dd:
            continue
        y, x = dd[c].to_numpy(float), mm[c].to_numpy(float)
        ok = np.isfinite(x) & np.isfinite(y)
        if ok.sum() < 30:
            continue
        s = stats(y, x)
        cv = stats(y[ok], linear_map_cv(x[ok], y[ok], months[ok]))
        b, a = np.polyfit(y[ok], x[ok], 1)      # TAHMO -> MERRA scale, used by E3
        maps[c] = dict(a=float(a), b=float(b))
        rows.append(dict(variable=c, n=s["n"], tahmo_mean=float(y[ok].mean()), merra_mean=float(x[ok].mean()),
                         merra_bias=s["bias"], merra_rmse=s["rmse"], merra_r=s["r"],
                         corrected_rmse_cv=cv.get("rmse"), corrected_r_cv=cv.get("r")))
    return pd.DataFrame(rows), maps


# ---------------------------------------------------------------------------
# E2 low-cost sensor calibration with measured humidity
# ---------------------------------------------------------------------------
def e2_calibration(tt, st, d) -> pd.DataFrame:
    sites = tt.drop_duplicates("site_id")[["site_id", "lat", "lon", "is_reference", "network", "country"]]
    sites = tahmo.in_bbox(sites)
    obs = tt[tt.site_id.isin(sites.site_id)][["site_id", "date", "pm25_raw"]].rename(columns={"pm25_raw": "pm25"})
    env_merra = tt[tt.site_id.isin(sites.site_id)][["site_id", "date", "rh"]]
    code, dist = nearest_station(sites.lat, sites.lon, st)
    link = pd.DataFrame(dict(site_id=sites.site_id.to_numpy(), code=code, tahmo_km=dist))
    link = link[link.tahmo_km <= MATCH_KM]
    env_tahmo = obs[["site_id", "date"]].merge(link, on="site_id").merge(d[["code", "date", "rh"]], on=["code", "date"])[["site_id", "date", "rh"]]
    p_m = calibration.build_pairs(obs, sites, env_merra)
    p_t = calibration.build_pairs(obs, sites, env_tahmo)
    if p_m.empty or p_t.empty:
        log("E2: no co-located pairs with TAHMO humidity")
        return pd.DataFrame()
    key = ["site_id", "ref_id", "date"]
    both = p_m.merge(p_t[key + ["rh"]].rename(columns={"rh": "rh_tahmo"}), on=key).rename(columns={"rh": "rh_merra"})
    both["date"] = pd.to_datetime(both.date)
    log(f"E2: {len(both)} sensor-reference pairs with both humidity sources ({both.site_id.nunique()} sensors, "
        f"{both.date.min():%Y-%m-%d}..{both.date.max():%Y-%m-%d})")
    rows = []
    groups = [(k, g) for k, g in both.groupby("network")] + [("all low-cost", both)]
    for key_, g in groups:
        g = g.reset_index(drop=True)
        if len(g) < 60:
            continue
        for src in ("rh_merra", "rh_tahmo"):
            gg = g.assign(rh=g[src])
            for kind in ("ratio", "loglinear"):
                pred = calibration._cv(kind, gg)
                s = stats(gg.pm25_ref, pred)
                rows.append(dict(network=key_, humidity=src.replace("rh_", ""), correction=kind, n_pairs=len(g),
                                 n_sensors=g.site_id.nunique(), rh_range=f"{g[src].min():.0f}-{g[src].max():.0f}", **s))
        raw = stats(g.pm25_ref, g.pm25)
        rows.append(dict(network=key_, humidity="none", correction="raw sensor", n_pairs=len(g), n_sensors=g.site_id.nunique(), **raw))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# E3 PM2.5 model with local weather
# ---------------------------------------------------------------------------
def tahmo_features(rows: pd.DataFrame, d: pd.DataFrame, maps: dict) -> pd.DataFrame:
    """For rows (site_id, date, code) return TAHMO weather on the MERRA-2 scale incl. lags; NaN where missing."""
    dm = d.copy()
    for c, m in maps.items():
        if c in dm:
            dm[c] = m["a"] + m["b"] * dm[c]
    keep = [c for c in SWAP_VARS + ["u10", "v10"] if c in dm]
    dm = dm[["code", "date"] + keep]
    out = rows[["code", "date"]].merge(dm, on=["code", "date"], how="left")
    for lag, cols in ((1, ["rh", "ws10", "precip"]), (2, ["precip"])):
        lagged = dm[["code", "date"] + [c for c in cols if c in dm]].copy()
        lagged["date"] = lagged.date + pd.Timedelta(days=lag)
        out = out.merge(lagged.rename(columns={c: f"{c}_lag{lag}" for c in cols}), on=["code", "date"], how="left")
    out.index = rows.index
    return out


def swap_weather(df: pd.DataFrame, tf: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    """Replace MERRA-2 weather by TAHMO weather (and recompute derived features) where TAHMO is complete."""
    derived = ["u10", "v10", "wind_dir_sin", "wind_dir_cos", "rh_lag1", "ws10_lag1", "precip_lag1", "precip_3d", "ventilation"]
    df = df.astype({c: "float64" for c in SWAP_VARS + derived if c in df})
    need = [c for c in SWAP_VARS if c in tf]
    ok = tf[need].notna().all(axis=1).to_numpy()
    idx = df.index[ok]
    for c in need:
        df.loc[idx, c] = tf.loc[idx, c]
    if {"u10", "v10"} <= set(tf.columns) and tf.loc[idx, ["u10", "v10"]].notna().all(axis=1).all():
        df.loc[idx, ["u10", "v10"]] = tf.loc[idx, ["u10", "v10"]].to_numpy()
        wdir = np.arctan2(-df.loc[idx, "u10"], -df.loc[idx, "v10"])
        df.loc[idx, "wind_dir_sin"], df.loc[idx, "wind_dir_cos"] = np.sin(wdir), np.cos(wdir)
    for c, lc in (("rh", "rh_lag1"), ("ws10", "ws10_lag1"), ("precip", "precip_lag1")):
        v = tf.loc[idx, f"{c}_lag1"] if f"{c}_lag1" in tf else None
        if v is not None:
            df.loc[idx, lc] = v.fillna(df.loc[idx, lc])
    if "precip_lag2" in tf:
        p2 = tf.loc[idx, "precip_lag2"]
        df.loc[idx, "precip_3d"] = df.loc[idx, "precip"] + df.loc[idx, "precip_lag1"].fillna(0) + p2.fillna(0)
    df.loc[idx, "ventilation"] = df.loc[idx, "blh"] * df.loc[idx, "ws10"]
    return df, ok


def fit_predict(train, test, feats, which, params, w):
    bs = M.BaseSet(feats, params=params).fit(train, w, which)
    p = bs.predict(test)
    return np.exp(np.mean([p[k] for k in which], axis=0))


def e3_model(tt, st, d, maps, quick: bool) -> pd.DataFrame:
    which = ["lgbm"] if quick else ["catboost", "extratrees"]
    hp = config.MODELS / "hyperparams.json"
    params = json.loads(hp.read_text()) if hp.exists() else {}
    feats = [f for f in F.FEATURES if f in tt.columns]
    df = (tt[~tt.cal_excluded.astype(bool)] if "cal_excluded" in tt else tt).reset_index(drop=True)
    accra_sites = tahmo.in_bbox(df.drop_duplicates("site_id"))
    accra_sites = accra_sites[calibration.haversine(accra_sites.lat, accra_sites.lon, *ACCRA_CENTRE) <= 30]
    code, dist = nearest_station(accra_sites.lat, accra_sites.lon, st)
    link = pd.DataFrame(dict(site_id=accra_sites.site_id.to_numpy(), code=code, tahmo_km=dist))
    link = link[link.tahmo_km <= MATCH_KM]
    is_accra = df.site_id.isin(link.site_id).to_numpy()
    rows = df.loc[is_accra, ["site_id", "date"]].merge(link, on="site_id", how="left")
    rows.index = df.index[is_accra]
    tf = tahmo_features(rows, d, maps)
    df = df.astype({c: "float64" for c in df.columns if df[c].dtype == "float32"})
    dfB, okB = df.copy(), np.zeros(len(df), bool)
    swapped, ok_local = swap_weather(df.loc[is_accra], tf)
    dfB.loc[is_accra] = swapped
    okB[np.where(is_accra)[0][ok_local]] = True
    log(f"E3: {is_accra.sum()} Accra station-days at {len(link)} sites; TAHMO weather complete on {okB.sum()}")
    if okB.sum() < 200:
        log("E3: too few Accra station-days with TAHMO weather; skipped")
        return pd.DataFrame()
    w = M.sample_weights(df)
    res = []
    # (1) leave-month-out within Accra: the model sees Accra in other months, with the same weather source in train and test
    months = df.date.dt.to_period("M").astype(str).to_numpy()
    acc_idx = np.where(okB)[0]
    folds = GroupKFold(min(6, len(np.unique(months[acc_idx])))).split(acc_idx, groups=months[acc_idx])
    pA, pB = np.full(len(df), np.nan), np.full(len(df), np.nan)
    for k, (_, te_pos) in enumerate(folds):
        te = acc_idx[te_pos]
        tr = np.setdiff1d(np.where(~np.isin(months, np.unique(months[te])) | ~is_accra)[0], te)
        pA[te] = fit_predict(df.iloc[tr], df.iloc[te], feats, which, params, w[tr])
        pB[te] = fit_predict(dfB.iloc[tr], dfB.iloc[te], feats, which, params, w[tr])
        log(f"  leave-month-out fold {k + 1} done")
    y = df.pm25.to_numpy()
    res.append(dict(design="leave-month-out (Accra)", weather="MERRA-2 (current)", **stats(y[acc_idx], pA[acc_idx])))
    res.append(dict(design="leave-month-out (Accra)", weather="TAHMO, mapped to MERRA-2 scale", **stats(y[acc_idx], pB[acc_idx])))
    # (2) leave-Accra-out: Accra never seen in training; only the test inputs change
    tr = np.where(~is_accra)[0]
    bs = M.BaseSet(feats, params=params).fit(df.iloc[tr], w[tr], which)
    for name, src in (("MERRA-2 (current)", df), ("TAHMO, mapped to MERRA-2 scale", dfB)):
        p = bs.predict(src.iloc[acc_idx])
        res.append(dict(design="leave-Accra-out", weather=name, **stats(y[acc_idx], np.exp(np.mean([p[k] for k in which], axis=0)))))
    out = pd.DataFrame(res)
    out.insert(2, "learners", "+".join(which))
    return out


# ---------------------------------------------------------------------------
def write_report(inv, e1, e2, e3, quick):
    REP.mkdir(parents=True, exist_ok=True)
    def f(df):
        if df is None or not len(df):
            return "_not enough overlapping data_"
        df = df.round(3).astype(object).where(df.notna(), "")
        rows = [list(map(str, df.columns))] + [[str(v) for v in r] for r in df.itertuples(index=False)]
        return "\n".join(["| " + " | ".join(rows[0]) + " |", "|" + "---|" * len(rows[0])] + ["| " + " | ".join(r) + " |" for r in rows[1:]])
    lines = [
        "# TAHMO weather stations in Accra: experiments",
        "",
        f"Generated {pd.Timestamp.now("UTC"):%Y-%m-%d %H:%M} UTC by `scripts/12_tahmo_accra.py experiments{' --quick' if quick else ''}`.",
        "",
        "## E0. Stations and coverage",
        "",
        "Daily values need 75% of the 5-minute samples (00-24 UTC). Columns `days_*` count valid days per variable.",
        "",
        f(inv),
        "",
        "## E1. How far the model's weather input (MERRA-2 via NASA POWER) is from the ground in Accra",
        "",
        "`merra_*` compares MERRA-2, bilinearly interpolated to each TAHMO station, with the station's daily value. "
        "`corrected_*_cv` is a linear correction fitted on other months (month-blocked CV). "
        "TAHMO wind is measured at about 2 m, MERRA-2 wind at 10 m, so a wind bias is expected.",
        "",
        f(e1),
        "",
        "## E2. Low-cost sensor calibration: model humidity versus measured humidity",
        "",
        "Pairs are low-cost sensors within 10 km of the US Embassy BAM-1020 on the same day, restricted to days where both "
        "humidity sources exist. Errors are month-blocked cross-validated, in µg/m³ against the BAM.",
        "",
        f(e2),
        "",
        "## E3. Does local measured weather improve GH-PM25 in Accra?",
        "",
        "Accra station-days (sites within 30 km of the centre and 15 km of a TAHMO station). "
        "Weather B replaces MERRA-2 temperature, humidity, rain and wind (and their lags) with TAHMO values mapped to the "
        "MERRA-2 scale by the E1 regression. Leave-month-out lets the model learn from Accra in other months; "
        "leave-Accra-out never shows it Accra.",
        "",
        f(e3),
        "",
    ]
    (REP / "TAHMO_Accra_Experiments.md").write_text("\n".join(lines), encoding="utf-8")


def experiments(quick: bool):
    OUT.mkdir(parents=True, exist_ok=True)
    st, d = load_tahmo()
    log(f"TAHMO: {len(st)} stations in the Accra area, {len(d)} station-days")
    tt = pd.read_parquet(config.PROCESSED / "training_table.parquet")
    tt["date"] = pd.to_datetime(tt.date)
    met_lat = F.met_lattice_from_df(pd.read_parquet(config.INTERIM / "power_daily.parquet"))

    inv = e0_inventory(st, d)
    e1, maps = e1_weather(st, d, met_lat)
    log("E1:\n" + e1.round(2).to_string())
    e2 = e2_calibration(tt, st, d)
    if len(e2):
        log("E2:\n" + e2.round(3).to_string())
    e3 = e3_model(tt, st, d, maps, quick)
    if len(e3):
        log("E3:\n" + e3.round(3).to_string())

    for name, df in (("inventory", inv), ("weather_bias", e1), ("calibration_humidity", e2), ("accra_model", e3)):
        df.to_csv(OUT / f"{name}.csv", index=False)
        REP.mkdir(parents=True, exist_ok=True)
        df.to_csv(REP / f"{name}.csv", index=False)
    (OUT / "tahmo_to_merra.json").write_text(json.dumps(maps, indent=1))
    summary = dict(generated_at=pd.Timestamp.now("UTC").strftime("%Y-%m-%dT%H:%M:%S"), quick=quick,
                   n_stations=int(len(st)), n_station_days=int(len(d)),
                   weather_bias=records(e1), calibration=records(e2), model=records(e3))
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1, default=str, allow_nan=False))
    write_report(inv, e1, e2, e3, quick)
    log(f"written {OUT} and {REP / 'TAHMO_Accra_Experiments.md'}")


# ---------------------------------------------------------------------------
def export(days: int = 400):
    """Accra-view payload: station list, recent daily weather, and the experiment summary if present."""
    st, d = load_tahmo()
    d = d[d.date >= d.date.max() - pd.Timedelta(days=days)]
    series = {}
    for code, g in d.groupby("code"):
        g = g.sort_values("date")
        series[code] = {"date": g.date.dt.strftime("%Y-%m-%d").tolist(),
                        **{c: [None if not np.isfinite(v) else round(float(v), 2) for v in g[c]] for c in ["t2m", "rh", "precip", "ws10"] if c in g}}
    js = dict(generated_at=pd.Timestamp.now("UTC").strftime("%Y-%m-%dT%H:%M:%S"),
              stations=[dict(code=r.code, name=r["name"], lat=float(r.lat), lon=float(r.lon)) for _, r in st.iterrows()],
              series=series)
    s = OUT / "summary.json"
    if s.exists():
        js["experiments"] = json.loads(s.read_text())
    out = config.PORTAL_DATA / "accra"
    out.mkdir(parents=True, exist_ok=True)
    (out / "tahmo.json").write_text(json.dumps(js, separators=(",", ":"), allow_nan=False))
    log(f"portal/data/accra/tahmo.json: {len(st)} stations")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["download", "experiments", "export"])
    ap.add_argument("--start", default="2024-08-01")
    ap.add_argument("--days", type=int, help="download only the last N days (daily operation)")
    ap.add_argument("--quick", action="store_true", help="E3 with LightGBM only")
    a = ap.parse_args()
    if a.cmd == "download":
        if not tahmo.available():
            raise SystemExit("TAHMO credentials not set (TAHMO_USERNAME / TAHMO_PASSWORD)")
        v = tahmo.variables()
        if len(v):
            OUT.mkdir(parents=True, exist_ok=True)
            v.to_csv(OUT / "variables.csv", index=False)
            log("variables: " + ", ".join(f"{r.shortcode} ({r.units})" for r in v.itertuples()))
        start = (pd.Timestamp.now("UTC") - pd.Timedelta(days=a.days)).strftime("%Y-%m-%d") if a.days else a.start
        tahmo.update(start)
    elif a.cmd == "experiments":
        experiments(a.quick)
    else:
        export()


if __name__ == "__main__":
    main()
