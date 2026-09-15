"""Calibrate sensors, build the training table, cross-validate, and fit the final GH-PM25 model.

Outputs
  data/processed/training_table.parquet
  outputs/calibration_report.csv
  outputs/cv_oof.parquet                 out-of-fold predictions for every scheme/model
  outputs/cv_metrics.csv                 metrics by scheme x model x subset
  outputs/ablation_metrics.csv
  outputs/shap_importance.csv, outputs/shap_sample.parquet
  outputs/per_site_metrics.csv
  outputs/uncertainty_coverage.csv
  models/ghpm25_model.pkl (+ meta json)
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, KFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ghana_pm25 import calibration, config, features as F  # noqa: E402
from ghana_pm25 import model as M  # noqa: E402

OUT = config.OUTPUTS
N_FOLDS = 10
T0 = time.time()


def log(msg):
    print(f"[{(time.time() - T0) / 60:6.1f} min] {msg}", flush=True)


# ---------------------------------------------------------------------------
def build_table() -> pd.DataFrame:
    stations = pd.read_parquet(config.PROCESSED / "stations.parquet")
    obs = pd.read_parquet(config.PROCESSED / "obs_daily.parquet")
    obs["date"] = pd.to_datetime(obs["date"])
    cams = F.load_cams_lattice()
    met = F.met_lattice_from_df(pd.read_parquet(config.INTERIM / "power_daily.parquet"))
    fires = pd.read_parquet(config.INTERIM / "firms_hist.parquet")
    fires["date"] = pd.to_datetime(fires["date"])

    # --- LCS calibration on the full observation record (needs only MERRA-2 RH) ---
    o = obs.merge(stations[["site_id", "lat", "lon"]], on="site_id")
    wts = met.weights(o.lat.to_numpy(), o.lon.to_numpy())
    t_idx = met.t_index(o.date)
    ok = t_idx >= 0
    env = o.loc[ok, ["site_id", "date"]].copy()
    env["rh"] = met.sample("rh", t_idx[ok], tuple(x[ok] for x in wts))
    pairs = calibration.build_pairs(obs[["site_id", "date", "pm25"]], stations, env)
    pairs.to_parquet(OUT / "calibration_pairs.parquet")
    models, report = calibration.fit_models(pairs)
    calibration.save(models, report)
    log("calibration:\n" + report[["level", "key", "n_pairs", "n_sensors", "selected", "excluded", "raw_r2", "raw_mae", "raw_bias", "cv_r2", "cv_mae", "cv_bias"]].round(3).to_string())
    cal = calibration.apply(obs[["site_id", "date", "pm25"]], stations, env, models)
    n_excl = int(cal.cal_excluded.sum())
    cal = cal[~cal.cal_excluded]
    log(f"calibrated obs: {len(cal)} (excluded {n_excl} station-days from networks failing calibration QC)")

    df = cal.merge(stations[["site_id", "lat", "lon", "network", "country", "is_reference", "name"]], on="site_id")
    df = df[(df.date >= cams.date_index.min()) & (df.date <= cams.date_index.max())]
    df = df.sort_values(["date", "site_id"]).reset_index(drop=True)
    log(f"obs rows in CAMS period: {len(df)}")
    dyn = F.dynamic_features(df.lat.to_numpy(), df.lon.to_numpy(), df.date.to_numpy(), cams, met)
    fire = F.fire_block(fires, df.lat.to_numpy(), df.lon.to_numpy(), df.date.to_numpy(), dyn.u10.to_numpy(), dyn.v10.to_numpy())
    df = pd.concat([df, dyn, fire], axis=1)

    st_static = F.transform_static(pd.read_parquet(config.PROCESSED / "station_static.parquet"))
    df = df.merge(st_static, on="site_id", how="left")
    clusters = M.site_clusters(stations)
    df["cluster"] = df.site_id.map(clusters)
    df = df.dropna(subset=["cams_pm25", "t2m", "elev"]).sort_values(["date", "site_id"]).reset_index(drop=True)
    df.to_parquet(config.PROCESSED / "training_table.parquet")
    log(f"training table {df.shape}, sites={df.site_id.nunique()}, clusters={df.cluster.nunique()}")
    return df


# ---------------------------------------------------------------------------
def folds_for(df: pd.DataFrame, scheme: str) -> np.ndarray:
    f = np.zeros(len(df), dtype=int)
    if scheme == "random":
        splits = KFold(N_FOLDS, shuffle=True, random_state=42).split(df)
    elif scheme == "site":
        splits = GroupKFold(N_FOLDS).split(df, groups=df.site_id)
    elif scheme == "cluster":
        splits = GroupKFold(min(N_FOLDS, df.cluster.nunique())).split(df, groups=df.cluster)
    elif scheme == "temporal":
        splits = GroupKFold(N_FOLDS).split(df, groups=df.date.dt.to_period("M").astype(str))
    for k, (_, te) in enumerate(splits):
        f[te] = k
    return f


def nearest_obs_features(df: pd.DataFrame, holdout: np.ndarray) -> pd.DataFrame:
    """Distance to nearest available same-day observation and density of obs within 100 km."""
    dist = np.full(len(df), 2000.0); n100 = np.zeros(len(df))
    for _, g in df.groupby("date"):
        idx = g.index.to_numpy()
        tr, te = idx[~holdout[idx]], idx[holdout[idx]]
        if len(te) == 0 or len(tr) == 0:
            continue
        d = calibration.haversine(df.lat.to_numpy()[te, None], df.lon.to_numpy()[te, None], df.lat.to_numpy()[None, tr], df.lon.to_numpy()[None, tr])
        dist[te] = d.min(1); n100[te] = (d < 100).sum(1)
    return pd.DataFrame({"dist_nearest_km": np.minimum(dist, 2000), "n_obs_100km": n100})


PARAMS: dict = {}


def run_cv(df: pd.DataFrame, feats: list[str], scheme: str, which=M.BASE_MODELS) -> pd.DataFrame:
    folds = folds_for(df, scheme)
    w = M.sample_weights(df)
    oof = pd.DataFrame(index=df.index, columns=list(which), dtype=float)
    for k in np.unique(folds):
        tr, te = folds != k, folds == k
        bs = M.BaseSet(feats, params=PARAMS).fit(df[tr], w[tr], which)
        for name, p in bs.predict(df[te]).items():
            oof.loc[te, name] = p
        log(f"  {scheme} fold {k + 1}/{len(np.unique(folds))} done ({', '.join(which)})")
    oof["fold"] = folds
    return oof


def baseline_cams_linear(df: pd.DataFrame, folds: np.ndarray) -> np.ndarray:
    from sklearn.linear_model import LinearRegression
    X = np.c_[np.log(df.cams_pm25.clip(lower=1)), df.rh, df.blh.fillna(df.blh.median()) / 1000, df.dust_frac]
    y = np.log(df.pm25.clip(lower=1))
    out = np.zeros(len(df))
    for k in np.unique(folds):
        tr, te = folds != k, folds == k
        out[te] = LinearRegression().fit(X[tr], y[tr]).predict(X[te])
    return out


PAPER_LIKE = ["merra_aod", "merra_aod_3d", "t2m", "rh", "td2m", "precip", "u10", "v10", "doy_sin", "doy_cos"]


def subsets(df: pd.DataFrame) -> dict[str, np.ndarray]:
    m = df.date.dt.month
    return {
        "all": np.ones(len(df), bool),
        "ghana": (df.country == "GH").to_numpy(),
        "ghana_reference": ((df.country == "GH") & df.is_reference).to_numpy(),
        "reference_all": df.is_reference.to_numpy(),
        "ghana_north(>8N)": ((df.country == "GH") & (df.lat > 8)).to_numpy(),
        "ghana_south(<=8N)": ((df.country == "GH") & (df.lat <= 8)).to_numpy(),
        "harmattan(DJF)": m.isin([12, 1, 2]).to_numpy(),
        "non_harmattan": (~m.isin([12, 1, 2])).to_numpy(),
    }


def main():
    df = build_table() if "--reuse-table" not in sys.argv else pd.read_parquet(config.PROCESSED / "training_table.parquet")
    feats = [f for f in F.FEATURES if f in df.columns]
    y_log = np.log(df.pm25.clip(lower=1)).to_numpy()
    w = M.sample_weights(df)
    schemes = ["cluster", "site", "random", "temporal"]
    if M.QUICK:
        global N_FOLDS
        N_FOLDS = 3
        log("QUICK smoke-test mode")
    elif "--no-tune" not in sys.argv:
        log("hyper-parameter search (leave-city-cluster-out, independent fold assignment)")
        PARAMS.update(M.tune(df, feats, w, log=log))
    elif (config.MODELS / "hyperparams.json").exists():
        PARAMS.update(json.loads((config.MODELS / "hyperparams.json").read_text()))
    records, oof_all = [], {}
    reuse = "--reuse-oof" in sys.argv
    prev = pd.read_parquet(OUT / "cv_oof.parquet") if reuse else None
    if reuse:
        assert (prev.site_id.to_numpy() == df.site_id.to_numpy()).all(), "cv_oof does not match training table order"

    # ---- base learners under each CV scheme ----
    for scheme in schemes:
        if reuse:
            oof = pd.DataFrame({k: prev[f"{scheme}__{k}"].to_numpy() for k in M.BASE_MODELS}, index=df.index)
            oof["fold"] = folds_for(df, scheme)
        else:
            oof = run_cv(df, feats, scheme)
        oof_all[scheme] = oof
    # ---- ensemble selection: nested NNLS stack vs equal-weight averages of every learner subset ----
    # criterion fixed in advance: mean R2 over {cluster, site, temporal} x {all stations, Ghana}
    from itertools import combinations
    gh_mask = (df.country == "GH").to_numpy()
    cands = {"nnls_stack": None}
    for r in range(1, len(M.BASE_MODELS) + 1):
        for combo in combinations(M.BASE_MODELS, r):
            cands["avg:" + "+".join(combo)] = combo
    sel_rows = []
    for name, combo in cands.items():
        scores = []
        for sch in ("cluster", "site", "temporal"):
            o = oof_all[sch]
            lp = M.nested_stack(o[M.BASE_MODELS], y_log, w, o["fold"].to_numpy()) if combo is None else o[list(combo)].to_numpy().mean(1)
            for mask in (np.ones(len(df), bool), gh_mask):
                scores.append(M.metrics(df.pm25.to_numpy()[mask], np.exp(lp[mask]))["r2"])
        sel_rows.append(dict(candidate=name, mean_r2=float(np.mean(scores)), **{f"s{i}": s for i, s in enumerate(scores)}))
    sel = pd.DataFrame(sel_rows).sort_values("mean_r2", ascending=False)
    sel.to_csv(OUT / "ensemble_selection.csv", index=False)
    chosen = sel.candidate.iloc[0]
    if chosen == "nnls_stack":
        stack = M.stack_weights(oof_all["cluster"], y_log, w)
    else:
        members = cands[chosen]
        stack = dict(weights={m: (1.0 / len(members) if m in members else 0.0) for m in M.BASE_MODELS}, bias=0.0)
    stack["selected"] = chosen
    log("ensemble selection (top 5):\n" + sel.head(5)[["candidate", "mean_r2"]].round(4).to_string())
    log(f"selected ensemble: {chosen} -> weights {stack['weights']}")

    # ---- persistent site offsets + correlogram of day-specific residual anomalies (leave-cluster-out) ----
    resid_stage1 = y_log - M.apply_stack({k: oof_all["cluster"][k].to_numpy() for k in M.BASE_MODELS}, stack)
    offsets = M.site_offsets(df.site_id, resid_stage1)
    res = df[["date", "lat", "lon"]].copy()
    res["resid"] = resid_stage1 - df.site_id.map(offsets).to_numpy()
    cg = M.fit_correlogram(res)
    log(f"anomaly correlogram: c0={cg['c0']:.3f} L={cg['L_km']:.1f} km var={cg['var']:.3f}; site offsets sd={offsets.std():.3f}")

    oof_out = df[["site_id", "date", "lat", "lon", "country", "network", "is_reference", "cluster", "pm25", "pm25_raw", "cams_pm25"]].copy()
    for scheme in schemes:
        o = oof_all[scheme]
        folds = o["fold"].to_numpy()
        if stack["selected"] == "nnls_stack":
            stk = M.nested_stack(o[M.BASE_MODELS], y_log, w, folds)  # weights fit without the held-out fold
        else:
            stk = M.apply_stack({k: o[k].to_numpy() for k in M.BASE_MODELS}, stack)  # fixed equal weights
        oof_out[f"{scheme}__stack"] = stk
        for k in M.BASE_MODELS:
            oof_out[f"{scheme}__{k}"] = o[k].to_numpy()
        # stage 2 residual kriging: for each fold, krige from same-day rows NOT in that fold
        rk_mean = np.zeros(len(df)); rk_var = np.zeros(len(df))
        near = pd.DataFrame(index=df.index, columns=["dist_nearest_km", "n_obs_100km"], dtype=float)
        tmp = df[["site_id", "date", "lat", "lon", "pm25"]].copy(); tmp["pred"] = stk
        for k in np.unique(folds):
            hold = folds == k
            m_, v_ = M.rk_loo_same_day(tmp, "pred", cg, hold, offsets)
            rk_mean[hold], rk_var[hold] = m_[hold], v_[hold]
            nf = nearest_obs_features(df, hold)
            near.loc[hold] = nf[hold].to_numpy()
        oof_out[f"{scheme}__rk_mean"] = rk_mean
        oof_out[f"{scheme}__stack_rk"] = stk + rk_mean  # provisional (lambda = 1); rescaled below
        oof_out[f"{scheme}__krige_var"] = rk_var
        oof_out[f"{scheme}__dist_nearest_km"] = near.dist_nearest_km.to_numpy()
        oof_out[f"{scheme}__n_obs_100km"] = near.n_obs_100km.to_numpy()
        # baselines
        oof_out[f"{scheme}__cams_linear"] = baseline_cams_linear(df, folds)
        log(f"{scheme}: stage-2 kriging + baselines done")

    # paper-like feature set (XGBoost-type single GBDT with AOD+met+DOY) and single LightGBM, under cluster & random CV
    for scheme in ["cluster", "site", "random", "temporal"]:
        if reuse:
            oof_out[f"{scheme}__paperlike_gbdt"] = prev[f"{scheme}__paperlike_gbdt"].to_numpy()
        else:
            o = run_cv(df, PAPER_LIKE, scheme, which=["lgbm"])
            oof_out[f"{scheme}__paperlike_gbdt"] = o["lgbm"].to_numpy()

    # ---- Stage-2 strength: lambda chosen on leave-site-out (predicting unmonitored places inside monitored cities) ----
    lam_grid = [0.0, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0]
    site_scores = {lam: M.metrics(df.pm25.to_numpy(), np.exp(oof_out["site__stack"] + lam * oof_out["site__rk_mean"]))["r2"] for lam in lam_grid}
    rk_lambda = max(site_scores, key=site_scores.get)
    log(f"stage-2 lambda scan (leave-site-out R2): { {k: round(v, 4) for k, v in site_scores.items()} } -> lambda={rk_lambda}")
    pd.DataFrame([dict(scheme=s, lam=l, r2=M.metrics(df.pm25.to_numpy(), np.exp(oof_out[f"{s}__stack"] + l * oof_out[f"{s}__rk_mean"]))["r2"])
                  for s in schemes for l in lam_grid]).to_csv(OUT / "stage2_lambda_scan.csv", index=False)
    for scheme in schemes:
        oof_out[f"{scheme}__stack_rk"] = oof_out[f"{scheme}__stack"] + rk_lambda * oof_out[f"{scheme}__rk_mean"]

    # smearing factor (log -> linear mean) from cluster OOF
    resid_c = y_log - oof_out["cluster__stack_rk"].to_numpy()
    smear = float(np.average(np.exp(resid_c), weights=w))
    oof_out.to_parquet(OUT / "cv_oof.parquet")

    # ---- metrics ----
    models = ["cams_raw", "cams_linear", "paperlike_gbdt"] + M.BASE_MODELS + ["stack", "stack_rk"]
    subs = subsets(df)
    for scheme in schemes:
        for mname in models:
            if mname == "cams_raw":
                pred = df.cams_pm25.to_numpy()
            else:
                pred = np.exp(oof_out[f"{scheme}__{mname}"].to_numpy()) * (smear if mname.startswith("stack") else 1.0)
            for sname, mask in subs.items():
                if mask.sum() < 30:
                    continue
                records.append(dict(scheme=scheme, model=mname, subset=sname, **M.metrics(df.pm25.to_numpy()[mask], pred[mask])))
    met = pd.DataFrame(records)
    met.to_csv(OUT / "cv_metrics.csv", index=False)
    log("\n" + met[met.subset.isin(["all", "ghana", "ghana_reference"])].pivot_table(index=["subset", "model"], columns="scheme", values="r2").round(3).to_string())

    # per-site metrics (cluster CV, final model)
    ps = []
    pred_lin = np.exp(oof_out["cluster__stack_rk"]) * smear
    for sid, g in df.groupby("site_id"):
        if len(g) >= 30:
            ps.append(dict(site_id=sid, name=g.name.iloc[0], country=g.country.iloc[0], network=g.network.iloc[0],
                           lat=g.lat.iloc[0], lon=g.lon.iloc[0], **M.metrics(g.pm25, pred_lin[g.index])))
    pd.DataFrame(ps).to_csv(OUT / "per_site_metrics.csv", index=False)

    # ---- uncertainty: cross-fitted scale model + nested conformal coverage ----
    ud = df.copy()
    ud["krige_var"] = oof_out["cluster__krige_var"]; ud["dist_nearest_km"] = oof_out["cluster__dist_nearest_km"]
    ud["n_obs_100km"] = oof_out["cluster__n_obs_100km"]
    abs_r = np.abs(resid_c)
    cfolds = oof_all["cluster"]["fold"].to_numpy()
    sigma = np.zeros(len(df))
    for k in np.unique(cfolds):
        tr, te = cfolds != k, cfolds == k
        sigma[te] = np.maximum(M.fit_scale_model(ud[tr], abs_r[tr]).predict(ud[te][M.SCALE_FEATURES].to_numpy(dtype="float32")), 0.03)
    scores = abs_r / sigma
    cov = []
    for alpha in (0.1, 0.32):
        hit = np.zeros(len(df), bool); width = np.zeros(len(df))
        for k in np.unique(cfolds):
            q = M.conformal_quantile(scores[cfolds != k], alpha)
            te = cfolds == k
            hit[te] = scores[te] <= q
            lo, hi = np.exp(oof_out["cluster__stack_rk"][te] - q * sigma[te]), np.exp(oof_out["cluster__stack_rk"][te] + q * sigma[te])
            width[te] = hi - lo
        for sname, mask in subs.items():
            if mask.sum() >= 30:
                cov.append(dict(nominal=1 - alpha, subset=sname, coverage=float(hit[mask].mean()), median_width=float(np.median(width[mask])), n=int(mask.sum())))
    pd.DataFrame(cov).to_csv(OUT / "uncertainty_coverage.csv", index=False)
    log("coverage:\n" + pd.DataFrame(cov).round(3).to_string())

    # ---- ablation (LightGBM, leave-cluster-out) ----
    groups = {"no_static": F.STATIC_COLS, "no_fire": F.FIRE_COLS, "no_cams": F.CAMS_COLS + ["log_cams_pm25", "cams_pm25_lag1", "cams_pm25_3d", "cams_dust_lag1", "dust_frac", "log_cams_dust"],
              "no_coords": F.COORD_COLS, "no_merra_met": F.MET_COLS + ["precip_3d", "precip_lag1", "log_blh", "ventilation", "rh_lag1", "ws10_lag1", "merra_aod_3d", "wind_dir_sin", "wind_dir_cos"]}
    abl = []
    ablation_items = [] if reuse else [("all_features", [])] + list(groups.items())
    for gname, drop in ablation_items:
        fs = [f for f in feats if f not in drop]
        if gname == "all_features":
            pred = oof_out["cluster__lgbm"].to_numpy()
        else:
            o = run_cv(df, fs, "cluster", which=["lgbm"])
            pred = o["lgbm"].to_numpy()
        for sname in ("all", "ghana"):
            mask = subs[sname]
            abl.append(dict(variant=gname, subset=sname, n_features=len(fs), **M.metrics(df.pm25.to_numpy()[mask], np.exp(pred[mask]))))
    if abl:
        pd.DataFrame(abl).to_csv(OUT / "ablation_metrics.csv", index=False)
        log("ablation:\n" + pd.DataFrame(abl).round(3).to_string())

    # ---- final fit ----
    base = M.BaseSet(feats, params=PARAMS).fit(df, w)
    scale_df = ud
    scale_model = M.fit_scale_model(scale_df, abs_r)
    q90 = M.conformal_quantile(scores, 0.1); q68 = M.conformal_quantile(scores, 0.32)
    imp = base.models["lgbm"].booster_.feature_importance("gain")
    top = np.argsort(imp)[::-1][:20]
    aoa = M.AOA.fit(df, [feats[i] for i in top], imp[top], cfolds)
    meta = dict(features=feats, n_train=len(df), n_sites=int(df.site_id.nunique()), period=[str(df.date.min().date()), str(df.date.max().date())],
                trained_at=pd.Timestamp.now().isoformat())
    meta["rk_lambda"] = rk_lambda
    sysm = M.GHPM25(base, stack, cg, scale_model, q90, q68, aoa, smear, meta, site_offsets=offsets.to_dict(), rk_lambda=rk_lambda)
    sysm.save()
    log(f"final model saved; q90={q90:.3f} q68={q68:.3f} smear={smear:.3f} AOA thr={aoa.threshold:.3f}")

    # ---- SHAP (LightGBM member) ----
    import shap
    samp = df.sample(min(6000, len(df)), random_state=1)
    expl = shap.TreeExplainer(base.models["lgbm"])
    sv = expl.shap_values(samp[feats].to_numpy(dtype="float32"))
    shap_imp = pd.DataFrame({"feature": feats, "mean_abs_shap": np.abs(sv).mean(0)}).sort_values("mean_abs_shap", ascending=False)
    shap_imp.to_csv(OUT / "shap_importance.csv", index=False)
    top_feats = shap_imp.feature.head(12).tolist()
    sdf = samp[top_feats].reset_index(drop=True).add_prefix("x__")
    sdf = pd.concat([sdf, pd.DataFrame(sv[:, [feats.index(f) for f in top_feats]], columns=[f"s__{f}" for f in top_feats])], axis=1)
    sdf.to_parquet(OUT / "shap_sample.parquet")
    (OUT / "run_summary.json").write_text(json.dumps(dict(stack=stack, correlogram=cg, q90=q90, q68=q68, smear=smear, **meta), indent=2, default=str))
    log("done")


if __name__ == "__main__":
    main()
