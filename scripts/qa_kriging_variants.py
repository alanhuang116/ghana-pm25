"""Diagnose Stage-2 residual kriging variants on saved out-of-fold predictions.

A  homoscedastic nugget, raw residuals (original)
B  heteroscedastic nugget: tau^2 + sigma_cal,i^2
C  remove shrunken persistent site offsets b_i before kriging (krige day-specific anomalies)
D  B + C
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ghana_pm25 import config  # noqa: E402
from ghana_pm25 import model as M  # noqa: E402
from ghana_pm25.calibration import haversine  # noqa: E402


def krige(tl, to, sl, so, r, s_extra, cg):
    from scipy.linalg import cho_factor, cho_solve
    sill = cg["var"]; s2 = cg["c0"] * sill; tau2 = (1 - cg["c0"]) * sill
    D = haversine(sl[:, None], so[:, None], sl[None, :], so[None, :])
    C = s2 * np.exp(-D / cg["L_km"]) + np.diag(tau2 + s_extra + 1e-6)
    cf = cho_factor(C, lower=True)
    d = haversine(tl[:, None], to[:, None], sl[None, :], so[None, :])
    c = s2 * np.exp(-d / cg["L_km"]) * (d < 300)
    return c @ cho_solve(cf, r)


def main():
    oof = pd.read_parquet(config.OUTPUTS / "cv_oof.parquet")
    tt = pd.read_parquet(config.PROCESSED / "training_table.parquet", columns=["site_id", "date", "cal_sigma_log"])
    oof = oof.merge(tt, on=["site_id", "date"], how="left").reset_index(drop=True)
    summ = json.loads((config.OUTPUTS / "run_summary.json").read_text())
    y = np.log(oof.pm25.clip(lower=1)).to_numpy()
    rows = []
    for scheme in ("cluster", "site", "random"):
        stk = oof[f"{scheme}__stack"].to_numpy()
        folds = pd.read_parquet(config.OUTPUTS / "cv_oof.parquet", columns=[])  # placeholder to keep API symmetric
        # reconstruct folds: rows whose stack_rk differs are held out together; use the fold ids saved in the oof? not saved -> recompute
        from sklearn.model_selection import GroupKFold, KFold
        if scheme == "cluster":
            split = GroupKFold(10).split(oof, groups=oof.cluster)
        elif scheme == "site":
            split = GroupKFold(10).split(oof, groups=oof.site_id)
        else:
            split = KFold(10, shuffle=True, random_state=42).split(oof)
        fold = np.zeros(len(oof), int)
        for k, (_, te) in enumerate(split):
            fold[te] = k
        r = y - stk
        # shrunken site offsets from OOF residuals (k = 15 pseudo-days)
        g = pd.DataFrame({"s": oof.site_id, "r": r}).groupby("s").r.agg(["sum", "count"])
        b_site = (g["sum"] / (g["count"] + 15)).reindex(oof.site_id).to_numpy()
        anomalies = r - b_site
        res_df = oof[["date", "lat", "lon"]].copy(); res_df["resid"] = anomalies
        cg_anom = M.fit_correlogram(res_df)
        cg_raw = summ["correlogram"]
        preds = {v: stk.copy() for v in "ABCD"}
        sig2 = oof.cal_sigma_log.fillna(0.3).to_numpy() ** 2
        for k in range(10):
            hold = fold == k
            for _, grp in oof[hold].groupby("date"):
                te = grp.index.to_numpy()
                same = oof.index[(oof.date == grp.date.iloc[0]) & ~hold].to_numpy()
                if len(same) == 0:
                    continue
                tl, to = oof.lat.to_numpy()[te], oof.lon.to_numpy()[te]
                sl, so = oof.lat.to_numpy()[same], oof.lon.to_numpy()[same]
                preds["A"][te] += krige(tl, to, sl, so, r[same], np.zeros(len(same)), cg_raw)
                preds["B"][te] += krige(tl, to, sl, so, r[same], sig2[same], cg_raw)
                preds["C"][te] += krige(tl, to, sl, so, anomalies[same], np.zeros(len(same)), cg_anom)
                preds["D"][te] += krige(tl, to, sl, so, anomalies[same], sig2[same], cg_anom)
        for v, p in [("stage1", stk)] + list(preds.items()):
            for sub, mask in (("all", np.ones(len(oof), bool)), ("ghana", (oof.country == "GH").to_numpy())):
                mt = M.metrics(oof.pm25.to_numpy()[mask], (np.exp(p) * summ["smear"])[mask])
                rows.append(dict(scheme=scheme, variant=v, subset=sub, r2=mt["r2"], rmse=mt["rmse"]))
        print(scheme, "anomaly correlogram:", {k: round(cg_anom[k], 3) for k in ("c0", "L_km", "var")}, flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(config.OUTPUTS / "qa_kriging_variants.csv", index=False)
    print(out.pivot_table(index=["subset", "variant"], columns="scheme", values="r2").round(3).to_string())


if __name__ == "__main__":
    main()
