"""Low-cost sensor (LCS) harmonisation against reference BAM-1020 monitors.

Optical LCS (Plantower PMS in AirGradient, Clarity Node-S, etc.) over-respond at high relative
humidity (hygroscopic growth) and under-respond to coarse mineral dust. Per network we build
quasi-co-located daily pairs (LCS site within R_KM of a US State Department BAM-1020, same day)
and select, by month-blocked cross-validation, the best of three candidate corrections:

  identity     PM_ref = PM_lcs
  ratio        ln PM_ref = a + ln PM_lcs                     (constant multiplicative bias)
  loglinear    ln PM_ref = a + b ln PM_lcs + c RH*           (RH* clamped to the pair range)

RH is MERRA-2 daily mean relative humidity at the sensor (NASA POWER), so every sensor can be
corrected without on-board humidity. Clamping prevents extrapolation into humidity regimes that
are absent from the co-location record (e.g. very dry Harmattan days in coastal Accra).

Networks without pairs inherit the model of their sensor family, then the pooled LCS model.
Networks whose calibrated CV R^2 is below MIN_CV_R2 are excluded from training (flagged).
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.linear_model import HuberRegressor
from sklearn.model_selection import GroupKFold

from . import config

R_KM = 10.0
MIN_PAIRS = 150
MIN_CV_R2 = 0.30


def haversine(lat1, lon1, lat2, lon2):
    p = np.pi / 180
    a = np.sin((lat2 - lat1) * p / 2) ** 2 + np.cos(lat1 * p) * np.cos(lat2 * p) * np.sin((lon2 - lon1) * p / 2) ** 2
    return 12742 * np.arcsin(np.sqrt(a))


def family(network: str) -> str:
    n = str(network).lower()
    if "clarity" in n:
        return "clarity"
    if any(s in n for s in ("airgradient", "iqair", "data354", "airqo", "aurassure", "miri", "habitat")):
        return "plantower-class"
    return "other"


def build_pairs(obs: pd.DataFrame, stations: pd.DataFrame, env: pd.DataFrame) -> pd.DataFrame:
    """LCS-reference same-day pairs within R_KM. `env` has site_id, date, rh."""
    ref = stations[stations.is_reference]
    lcs = stations[~stations.is_reference]
    links = []
    for _, s in lcs.iterrows():
        d = haversine(s.lat, s.lon, ref.lat.to_numpy(), ref.lon.to_numpy())
        for rid, dist in zip(ref.site_id, d):
            if dist <= R_KM:
                links.append((s.site_id, rid, dist))
    links = pd.DataFrame(links, columns=["site_id", "ref_id", "dist_km"])
    if links.empty:
        return pd.DataFrame()
    o = obs[["site_id", "date", "pm25"]]
    p = links.merge(o, on="site_id").merge(o.rename(columns={"site_id": "ref_id", "pm25": "pm25_ref"}), on=["ref_id", "date"])
    p = p.merge(env[["site_id", "date", "rh"]], on=["site_id", "date"], how="left")
    p = p.merge(stations[["site_id", "network", "country"]], on="site_id")
    p["net_key"] = p.country + "|" + p.network
    return p.dropna(subset=["rh"])


# ---------------------------------------------------------------------------
def _fit(kind: str, g: pd.DataFrame) -> dict:
    x = np.log(g.pm25.clip(lower=1.0)).to_numpy()
    y = np.log(g.pm25_ref.clip(lower=1.0)).to_numpy()
    if kind == "identity":
        return dict(kind=kind)
    if kind == "ratio":
        return dict(kind=kind, a=float(np.median(y - x)))
    lo, hi = float(np.percentile(g.rh, 2)), float(np.percentile(g.rh, 98))
    rh = np.clip(g.rh.to_numpy(), lo, hi) / 100.0
    m = HuberRegressor(epsilon=1.35, alpha=1e-2, max_iter=5000).fit(np.c_[x, rh], y)
    b = float(np.clip(m.coef_[0], 0.4, 1.3))
    return dict(kind=kind, a=float(m.intercept_ + (m.coef_[0] - b) * x.mean()), b=b, c=float(m.coef_[1]), rh_lo=lo, rh_hi=hi)


def _predict(model: dict, pm: np.ndarray, rh: np.ndarray) -> np.ndarray:
    x = np.log(np.clip(pm, 1.0, None))
    if model["kind"] == "identity":
        return np.exp(x)
    if model["kind"] == "ratio":
        return np.exp(x + model["a"])
    r = np.clip(rh, model["rh_lo"], model["rh_hi"]) / 100.0
    return np.exp(model["a"] + model["b"] * x + model["c"] * r)


def _metrics(y, yhat):
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    r = np.corrcoef(y, yhat)[0, 1] if len(y) > 2 else np.nan
    return dict(r2=float(1 - np.sum((yhat - y) ** 2) / np.sum((y - y.mean()) ** 2)), r=float(r),
                rmse=float(np.sqrt(np.mean((yhat - y) ** 2))), mae=float(np.mean(np.abs(yhat - y))), bias=float(np.mean(yhat - y)))


def _cv(kind: str, g: pd.DataFrame) -> np.ndarray:
    ym = g.date.dt.to_period("M").astype(str).to_numpy()
    pred = np.full(len(g), np.nan)
    k = min(6, len(np.unique(ym)))
    if k < 2:
        return pred
    for tr, te in GroupKFold(k).split(g, groups=ym):
        m = _fit(kind, g.iloc[tr])
        pred[te] = _predict(m, g.pm25.to_numpy()[te], g.rh.to_numpy()[te])
    return pred


def fit_models(pairs: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    models, report = {}, []
    pairs = pairs.copy()
    pairs["family"] = pairs.network.map(family)
    groups = [("net", k, g) for k, g in pairs.groupby("net_key")] + \
             [("family", k, g) for k, g in pairs.groupby("family")] + [("pooled", "all", pairs)]
    for level, key, g in groups:
        g = g.reset_index(drop=True)
        if len(g) < MIN_PAIRS:
            continue
        cands = {}
        for kind in ("identity", "ratio", "loglinear"):
            pred = _cv(kind, g)
            ok = np.isfinite(pred)
            cands[kind] = (_metrics(g.pm25_ref[ok], pred[ok]), pred)
        best = min(cands, key=lambda k: cands[k][0]["mae"])
        met, pred = cands[best]
        m = _fit(best, g)
        ok = np.isfinite(pred)
        m["sigma_log"] = float(np.std(np.log(g.pm25_ref[ok].clip(lower=1)) - np.log(np.clip(pred[ok], 1, None))))
        m["cv_r2"] = met["r2"]
        m["n"] = int(len(g))
        m["excluded"] = bool(level == "net" and met["r2"] < MIN_CV_R2)
        models[f"{level}:{key}"] = m
        raw = _metrics(g.pm25_ref, g.pm25)
        report.append(dict(level=level, key=key, n_pairs=len(g), n_sensors=g.site_id.nunique(), n_ref=g.ref_id.nunique(),
                           mean_dist_km=round(g.dist_km.mean(), 2), rh_range=f"{g.rh.min():.0f}-{g.rh.max():.0f}",
                           max_ref=float(g.pm25_ref.max()), selected=best, excluded=m["excluded"],
                           **{f"raw_{k}": v for k, v in raw.items()},
                           **{f"cv_{k}": v for k, v in met.items()},
                           cv_mae_identity=cands["identity"][0]["mae"], cv_mae_ratio=cands["ratio"][0]["mae"], cv_mae_loglinear=cands["loglinear"][0]["mae"],
                           **{p: m.get(p) for p in ("a", "b", "c")}))
    return models, pd.DataFrame(report)


def apply(obs: pd.DataFrame, stations: pd.DataFrame, env: pd.DataFrame, models: dict) -> pd.DataFrame:
    df = obs.merge(stations[["site_id", "network", "country", "is_reference"]], on="site_id") \
            .merge(env[["site_id", "date", "rh"]], on=["site_id", "date"], how="left")
    df["pm25_raw"] = df["pm25"]
    df["cal_model"] = "reference"
    df["cal_sigma_log"] = 0.10  # BAM-1020 daily uncertainty ~10 %
    df["cal_excluded"] = False
    lcs = ~df.is_reference
    rh_fill = float(df.rh.median()) if df.rh.notna().any() else 75.0
    for key, sub in df[lcs].groupby(df.country + "|" + df.network):
        fam = family(key.split("|", 1)[1])
        for mk in (f"net:{key}", f"family:{fam}", "pooled:all"):
            if mk in models:
                break
        else:
            df.loc[sub.index, ["cal_model", "cal_sigma_log"]] = ["uncalibrated", 0.5]
            continue
        m = models[mk]
        df.loc[sub.index, "pm25"] = _predict(m, sub.pm25.to_numpy(), sub.rh.fillna(rh_fill).to_numpy())
        df.loc[sub.index, "cal_model"] = f"{mk} ({m['kind']})"
        df.loc[sub.index, "cal_sigma_log"] = m["sigma_log"]
        df.loc[sub.index, "cal_excluded"] = bool(m.get("excluded", False))
    return df.drop(columns=["network", "country", "is_reference", "rh"])


def save(models: dict, report: pd.DataFrame):
    (config.MODELS / "lcs_calibration.json").write_text(json.dumps(models, indent=2))
    report.to_csv(config.OUTPUTS / "calibration_report.csv", index=False)
