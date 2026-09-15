"""GH-PM25 hybrid spatiotemporal model.

Stage 1  Stacked ensemble (log space) of heterogeneous learners
         M1 LightGBM        target ln(PM)
         M2 XGBoost         target ln(PM / CAMS)      (multiplicative bias-correction of CAMS)
         M3 CatBoost        target ln(PM)             (ordered boosting, symmetric trees)
         M4 ExtraTrees      target ln(PM)             (randomised, low-variance)
         Non-negative stacking weights learnt on spatially grouped out-of-fold predictions.
Stage 2  Residual simple kriging of same-day station residuals with an exponential
         correlogram + nugget fitted on pooled out-of-fold residual pairs.
Stage 3  Uncertainty: locally-adaptive (normalised) split-conformal prediction intervals,
         scale model sigma(x) fitted on cross-fitted absolute residuals.
         Area of applicability (Meyer & Pebesma 2021) dissimilarity index.
"""
from __future__ import annotations

import json
import pickle
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.optimize import curve_fit, nnls
from scipy.spatial import cKDTree

from . import config
from .calibration import haversine
from .features import FEATURES

BASE_MODELS = ["lgbm", "xgb_ratio", "catboost", "extratrees"]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def site_clusters(stations: pd.DataFrame, km: float = 30.0) -> pd.Series:
    """City-scale clusters (complete linkage on great-circle distance)."""
    xyz = np.c_[np.cos(np.radians(stations.lat)) * np.cos(np.radians(stations.lon)),
                np.cos(np.radians(stations.lat)) * np.sin(np.radians(stations.lon)), np.sin(np.radians(stations.lat))]
    Z = linkage(xyz * 6371.0, method="complete")
    lab = fcluster(Z, t=km, criterion="distance")
    return pd.Series(lab, index=stations.site_id.values, name="cluster")


def sample_weights(df: pd.DataFrame) -> np.ndarray:
    """Down-weight dense city clusters and noisier (calibrated LCS) observations."""
    n_sites = df.groupby("cluster").site_id.transform("nunique")
    w_cluster = 1.0 / np.sqrt(n_sites)
    q = 1.0 / (0.2 ** 2 + df["cal_sigma_log"] ** 2)
    q = q / (1.0 / (0.2 ** 2 + 0.1 ** 2))
    return (w_cluster * q).to_numpy()


def metrics(y, yhat) -> dict:
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    ok = np.isfinite(y) & np.isfinite(yhat)
    y, yhat = y[ok], yhat[ok]
    if len(y) < 3:
        return dict(n=int(len(y)))
    r = np.corrcoef(y, yhat)[0, 1]
    ss_res, ss_tot = np.sum((y - yhat) ** 2), np.sum((y - y.mean()) ** 2)
    slope = np.polyfit(y, yhat, 1)[0]
    return dict(n=int(len(y)), r=float(r), r2=float(1 - ss_res / ss_tot), rmse=float(np.sqrt(np.mean((yhat - y) ** 2))),
                mae=float(np.mean(np.abs(yhat - y))), mb=float(np.mean(yhat - y)),
                nmb=float(np.sum(yhat - y) / np.sum(y)), slope=float(slope), mean_obs=float(y.mean()))


# ---------------------------------------------------------------------------
# base learners
# ---------------------------------------------------------------------------
QUICK = bool(__import__("os").environ.get("GHPM25_QUICK"))


def _make(name: str, params: dict | None = None):
    p = dict(params or {})
    if QUICK:  # smoke-test mode
        p.update(n_estimators=60, iterations=60)
    if name == "lgbm":
        import lightgbm as lgb
        return lgb.LGBMRegressor(n_estimators=p.get("n_estimators", 1200), learning_rate=p.get("learning_rate", 0.03),
                                 num_leaves=p.get("num_leaves", 48), min_child_samples=p.get("min_child_samples", 40),
                                 subsample=0.8, subsample_freq=1, colsample_bytree=p.get("colsample_bytree", 0.6),
                                 reg_lambda=p.get("reg_lambda", 5.0), verbose=-1, n_jobs=-1)
    if name == "xgb_ratio":
        import xgboost as xgb
        return xgb.XGBRegressor(n_estimators=p.get("n_estimators", 1000), learning_rate=p.get("learning_rate", 0.03), max_depth=p.get("max_depth", 7),
                                min_child_weight=p.get("min_child_weight", 10), subsample=0.8, colsample_bytree=p.get("colsample_bytree", 0.6),
                                reg_lambda=p.get("reg_lambda", 5.0), tree_method="hist", n_jobs=-1)
    if name == "catboost":
        from catboost import CatBoostRegressor
        return CatBoostRegressor(iterations=p.get("iterations", 1500), learning_rate=p.get("learning_rate", 0.05), depth=p.get("depth", 8),
                                 l2_leaf_reg=p.get("l2_leaf_reg", 5), loss_function="RMSE", verbose=0, thread_count=-1)
    if name == "extratrees":
        from sklearn.ensemble import ExtraTreesRegressor
        return ExtraTreesRegressor(n_estimators=p.get("n_estimators", 300), min_samples_leaf=p.get("min_samples_leaf", 5),
                                   max_features=p.get("max_features", 0.5), n_jobs=-1, random_state=0)
    raise KeyError(name)


@dataclass
class BaseSet:
    features: list[str]
    models: dict = field(default_factory=dict)
    medians: dict = field(default_factory=dict)
    params: dict = field(default_factory=dict)

    def _X(self, df: pd.DataFrame, name: str) -> np.ndarray:
        X = df[self.features].to_numpy(dtype="float32")
        if name == "extratrees":
            med = np.array([self.medians[f] for f in self.features], dtype="float32")
            X = np.where(np.isnan(X), med, X)
        return X

    def fit(self, df: pd.DataFrame, w: np.ndarray, which=BASE_MODELS):
        self.medians = {f: float(np.nanmedian(df[f])) for f in self.features}
        y = np.log(df.pm25.clip(lower=1.0)).to_numpy()
        offset = np.log(df.cams_pm25.clip(lower=1.0)).to_numpy()
        for name in which:
            m = _make(name, self.params.get(name))
            target = y - offset if name == "xgb_ratio" else y
            m.fit(self._X(df, name), target, sample_weight=w)
            self.models[name] = m
        return self

    def predict(self, df: pd.DataFrame) -> dict[str, np.ndarray]:
        offset = np.log(df.cams_pm25.clip(lower=1.0)).to_numpy()
        out = {}
        for name, m in self.models.items():
            p = m.predict(self._X(df, name))
            out[name] = p + offset if name == "xgb_ratio" else p
        return out


# ---------------------------------------------------------------------------
# hyper-parameter search under spatial cross-validation
# ---------------------------------------------------------------------------
SEARCH_SPACE = {
    "lgbm": dict(num_leaves=[15, 31, 63, 127], min_child_samples=[20, 50, 100, 200], learning_rate=[0.02, 0.05],
                 n_estimators=[400, 800, 1500], colsample_bytree=[0.4, 0.6, 0.8], reg_lambda=[0.0, 5.0, 20.0]),
    "xgb_ratio": dict(max_depth=[4, 6, 8, 10], min_child_weight=[1, 5, 20, 50], learning_rate=[0.02, 0.05],
                      n_estimators=[400, 800, 1500], colsample_bytree=[0.4, 0.6, 0.8], reg_lambda=[1.0, 5.0, 20.0]),
    "catboost": dict(depth=[6, 8, 10], iterations=[800, 1500, 2500], learning_rate=[0.03, 0.06], l2_leaf_reg=[1, 5, 15]),
    "extratrees": dict(min_samples_leaf=[1, 3, 5, 10, 20], max_features=[0.3, 0.5, 0.7, 1.0], n_estimators=[300]),
}


def tune(df: pd.DataFrame, feats: list[str], w: np.ndarray, n_iter: int = 12, n_folds: int = 5, seed: int = 7, log=print) -> dict:
    """Random search per learner; objective = weighted RMSE (log space) under leave-city-cluster-out CV.
    The cluster-to-fold assignment is shuffled with a different seed from the evaluation folds."""
    rng = np.random.default_rng(seed)
    clusters = df.cluster.unique()
    rng.shuffle(clusters)
    fold_of = {c: i % n_folds for i, c in enumerate(clusters)}
    folds = df.cluster.map(fold_of).to_numpy()
    y = np.log(df.pm25.clip(lower=1.0)).to_numpy()
    best, history = {}, []
    for name, space in SEARCH_SPACE.items():
        keys = list(space)
        cands = [{}] + [{k: space[k][rng.integers(len(space[k]))] for k in keys} for _ in range(n_iter)]
        scores = []
        for cand in cands:
            pred = np.zeros(len(df))
            for k in range(n_folds):
                tr, te = folds != k, folds == k
                bs = BaseSet(feats, params={name: cand}).fit(df[tr], w[tr], [name])
                pred[te] = bs.predict(df[te])[name]
            rmse = float(np.sqrt(np.average((pred - y) ** 2, weights=w)))
            scores.append(rmse)
            history.append(dict(model=name, rmse_log=rmse, **{f"p_{k}": v for k, v in cand.items()}))
        i = int(np.argmin(scores))
        best[name] = cands[i]
        log(f"  tuned {name}: best log-RMSE {scores[i]:.4f} (default {scores[0]:.4f}) params={cands[i]}")
    pd.DataFrame(history).to_csv(config.OUTPUTS / "tuning_history.csv", index=False)
    (config.MODELS / "hyperparams.json").write_text(json.dumps(best, indent=2, default=float))
    return best


# ---------------------------------------------------------------------------
# stacking
# ---------------------------------------------------------------------------
def stack_weights(oof: pd.DataFrame, y: np.ndarray, w: np.ndarray) -> dict:
    P = oof[BASE_MODELS].to_numpy(dtype=float)
    ok = np.isfinite(P).all(1) & np.isfinite(y)
    sw = np.sqrt(w[ok])
    coef, _ = nnls(P[ok] * sw[:, None], y[ok] * sw)
    if not np.isfinite(coef).all() or coef.sum() <= 0:
        coef = np.ones(len(BASE_MODELS))
    coef = coef / coef.sum()
    bias = float(np.average(y[ok] - P[ok] @ coef, weights=w[ok]))
    return dict(weights=dict(zip(BASE_MODELS, coef.tolist())), bias=bias)


def nested_stack(oof: pd.DataFrame, y: np.ndarray, w: np.ndarray, folds: np.ndarray) -> np.ndarray:
    """Stacked out-of-fold prediction whose weights never see the held-out fold (honest evaluation)."""
    out = np.zeros(len(oof))
    for k in np.unique(folds):
        tr, te = folds != k, folds == k
        st = stack_weights(oof[tr], y[tr], w[tr])
        out[te] = apply_stack({m: oof[m].to_numpy()[te] for m in BASE_MODELS}, st)
    return out


def apply_stack(preds: dict, stack: dict) -> np.ndarray:
    return sum(stack["weights"][k] * preds[k] for k in BASE_MODELS) + stack["bias"]


# ---------------------------------------------------------------------------
# residual kriging
# ---------------------------------------------------------------------------
def _expo(d, c0, L):
    return c0 * np.exp(-d / L)


def fit_correlogram(res: pd.DataFrame, max_km: float = 400.0) -> dict:
    """Pooled same-day residual correlogram: corr(d) = c0 exp(-d/L)."""
    bins = np.array([0, 2, 5, 10, 20, 35, 50, 75, 100, 150, 200, 300, 400])
    sums = {k: [] for k in range(len(bins) - 1)}
    var = float(np.var(res.resid))
    for _, g in res.groupby("date"):
        if len(g) < 2:
            continue
        la, lo, r = g.lat.to_numpy(), g.lon.to_numpy(), g.resid.to_numpy()
        iu = np.triu_indices(len(g), 1)
        d = haversine(la[iu[0]], lo[iu[0]], la[iu[1]], lo[iu[1]])
        prod = r[iu[0]] * r[iu[1]]
        b = np.digitize(d, bins) - 1
        for k in range(len(bins) - 1):
            m = b == k
            if m.any():
                sums[k].append((prod[m].sum(), m.sum()))
    centers, corr, npairs = [], [], []
    for k, v in sums.items():
        if not v:
            continue
        s = sum(a for a, _ in v); n = sum(c for _, c in v)
        if n < 30:
            continue
        centers.append((bins[k] + bins[k + 1]) / 2)
        corr.append(float(s / n / var))
        npairs.append(int(n))
    centers, corr = np.array(centers), np.array(corr)
    (c0, L), _ = curve_fit(_expo, centers, corr, p0=(0.6, 50.0), bounds=([0.0, 1.0], [0.999, 2000.0]),
                           sigma=1 / np.sqrt(np.array(npairs)))
    return dict(c0=float(c0), L_km=float(L), var=var, bins_km=centers.tolist(), corr=corr.tolist(), npairs=npairs)


def krige_residuals(tgt_lat, tgt_lon, st_lat, st_lon, st_resid, cg: dict, max_km: float = 300.0):
    """Simple kriging (zero mean) of residuals to targets. Returns (mean, variance) in log space."""
    from scipy.linalg import cho_factor, cho_solve
    tgt_lat, tgt_lon = np.asarray(tgt_lat, float), np.asarray(tgt_lon, float)
    st_lat, st_lon, st_resid = (np.asarray(a, float) for a in (st_lat, st_lon, st_resid))
    ok = np.isfinite(st_resid) & np.isfinite(st_lat) & np.isfinite(st_lon)
    st_lat, st_lon, st_resid = st_lat[ok], st_lon[ok], st_resid[ok]
    n = len(st_resid)
    sill = cg["var"]
    s2 = cg["c0"] * sill            # structured variance
    tau2 = (1 - cg["c0"]) * sill    # nugget (micro-scale + sensor noise)
    if n == 0:
        return np.zeros(len(tgt_lat)), np.full(len(tgt_lat), sill)
    D = haversine(st_lat[:, None], st_lon[:, None], st_lat[None, :], st_lon[None, :])
    C = s2 * np.exp(-D / cg["L_km"]) + (tau2 + 1e-6 * sill) * np.eye(n)
    cf = cho_factor(C, lower=True)
    Cinv = cho_solve(cf, np.eye(n))
    alpha = cho_solve(cf, st_resid)
    mean = np.zeros(len(tgt_lat)); var = np.full(len(tgt_lat), sill)
    step = 20000
    for i in range(0, len(tgt_lat), step):
        d = haversine(tgt_lat[i:i + step, None], tgt_lon[i:i + step, None], st_lat[None, :], st_lon[None, :])
        c = s2 * np.exp(-d / cg["L_km"]) * (d <= max_km)
        mean[i:i + step] = c @ alpha
        var[i:i + step] = sill - np.einsum("ij,jk,ik->i", c, Cinv, c)
    return mean, np.maximum(var, tau2)


SITE_SHRINK = 15.0  # pseudo-days of shrinkage toward zero for persistent site offsets


def site_offsets(site_id: pd.Series, resid: np.ndarray) -> pd.Series:
    """Shrunken mean residual per site: persistent sensor/micro-environment offset b_i."""
    g = pd.DataFrame({"s": site_id.to_numpy(), "r": resid}).groupby("s").r.agg(["sum", "count"])
    return g["sum"] / (g["count"] + SITE_SHRINK)


def rk_loo_same_day(df: pd.DataFrame, pred_col: str, cg: dict, holdout_mask: np.ndarray,
                    offsets: pd.Series | None = None) -> tuple[np.ndarray, np.ndarray]:
    """For each held-out row, krige day-specific residual anomalies (residual minus the site's persistent
    offset) from same-day NON-held-out rows at other sites."""
    mean = np.zeros(len(df)); var = np.full(len(df), cg["var"])
    resid = np.log(df.pm25.clip(lower=1)).to_numpy() - df[pred_col].to_numpy()
    if offsets is not None:
        resid = resid - df.site_id.map(offsets).fillna(0.0).to_numpy()
    for _, g in df.groupby("date"):
        idx = g.index.to_numpy()
        tr = idx[~holdout_mask[idx]]
        te = idx[holdout_mask[idx]]
        if len(te) == 0 or len(tr) == 0:
            continue
        m, v = krige_residuals(df.lat.to_numpy()[te], df.lon.to_numpy()[te], df.lat.to_numpy()[tr], df.lon.to_numpy()[tr], resid[tr], cg)
        mean[te], var[te] = m, v
    return mean, var


# ---------------------------------------------------------------------------
# uncertainty
# ---------------------------------------------------------------------------
SCALE_FEATURES = ["krige_var", "dist_nearest_km", "n_obs_100km", "cams_pm25", "cams_dust", "dust_frac", "rh", "blh",
                  "lat", "doy_sin", "doy_cos", "frp_100km"]


class ScaleModel:
    """Smooth model of the expected absolute log-error: cubic B-splines on transformed predictors + ridge,
    fitted to ln(|r| + 0.02). Smoothness avoids the rings/bands a tree model would imprint on interval maps;
    conformal calibration keeps coverage valid regardless of how well sigma is estimated."""

    def __init__(self):
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import Ridge
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import SplineTransformer, StandardScaler
        self.pipe = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                                  SplineTransformer(n_knots=6, degree=3, extrapolation="constant"), Ridge(alpha=10.0))
        self.offset = 0.0
        self.cap = np.inf

    @staticmethod
    def _X(X) -> np.ndarray:
        X = np.asarray(X, dtype="float64").copy()
        c = {f: i for i, f in enumerate(SCALE_FEATURES)}
        for f in ("dist_nearest_km", "n_obs_100km", "cams_pm25", "cams_dust", "frp_100km"):
            X[:, c[f]] = np.log1p(np.clip(X[:, c[f]], 0, None))
        X[:, c["blh"]] = np.log(np.clip(X[:, c["blh"]], 30, None))
        return X

    def fit(self, X, abs_resid):
        z = np.log(np.asarray(abs_resid) + 0.02)
        self.pipe.fit(self._X(X), z)
        pred = np.exp(self.pipe.predict(self._X(X)))
        self.offset = float(np.mean(abs_resid) / np.mean(pred))  # re-centre (Jensen)
        self.cap = float(np.percentile(pred * self.offset, 99.5))  # no extrapolation beyond the training range
        return self

    def predict(self, X):
        return np.minimum(np.exp(self.pipe.predict(self._X(X))) * self.offset, self.cap)


def fit_scale_model(df: pd.DataFrame, abs_resid: np.ndarray):
    return ScaleModel().fit(df[SCALE_FEATURES].to_numpy(dtype="float64"), abs_resid)


def conformal_quantile(scores: np.ndarray, alpha: float = 0.1) -> float:
    n = len(scores)
    k = int(np.ceil((n + 1) * (1 - alpha)))
    return float(np.sort(scores)[min(k, n) - 1])


# ---------------------------------------------------------------------------
# area of applicability
# ---------------------------------------------------------------------------
@dataclass
class AOA:
    feats: list[str]
    weights: np.ndarray
    mean: np.ndarray
    sd: np.ndarray
    train: np.ndarray
    dbar: float
    threshold: float
    tree: cKDTree | None = None

    @classmethod
    def fit(cls, df: pd.DataFrame, feats: list[str], importance: np.ndarray, folds: np.ndarray, max_n: int = 6000, seed=0):
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(df), size=min(max_n, len(df)), replace=False)
        X = df[feats].to_numpy(dtype="float64")[idx]
        mean, sd = np.nanmean(X, 0), np.nanstd(X, 0) + 1e-9
        w = importance / importance.sum()
        Z = np.nan_to_num((X - mean) / sd) * w
        sub = rng.choice(len(Z), size=min(2000, len(Z)), replace=False)
        dd = np.sqrt(((Z[sub, None, :] - Z[None, sub, :]) ** 2).sum(-1))
        dbar = float(dd[np.triu_indices(len(sub), 1)].mean())
        f = folds[idx]
        di = np.empty(len(Z))
        for k in np.unique(f):
            tr, te = f != k, f == k
            di[te] = cKDTree(Z[tr]).query(Z[te], k=1)[0] / dbar
        q1, q3 = np.percentile(di, [25, 75])
        thr = float(q3 + 1.5 * (q3 - q1))
        return cls(feats, w, mean, sd, Z, dbar, thr)

    def di(self, df: pd.DataFrame) -> np.ndarray:
        if self.tree is None:
            self.tree = cKDTree(self.train)
        Z = np.nan_to_num((df[self.feats].to_numpy(dtype="float64") - self.mean) / self.sd) * self.weights
        return self.tree.query(Z, k=1)[0] / self.dbar


# ---------------------------------------------------------------------------
# full fitted system
# ---------------------------------------------------------------------------
@dataclass
class GHPM25:
    base: BaseSet
    stack: dict
    correlogram: dict
    scale_model: object
    q90: float
    q68: float
    aoa: AOA
    smearing: float
    meta: dict
    site_offsets: dict = field(default_factory=dict)
    rk_lambda: float = 1.0

    def save(self, path=None):
        path = path or config.MODELS / "ghpm25_model.pkl"
        with open(path, "wb") as f:
            pickle.dump(self, f)
        (config.MODELS / "ghpm25_model_meta.json").write_text(json.dumps(
            dict(stack=self.stack, correlogram=self.correlogram, q90=self.q90, q68=self.q68, smearing=self.smearing,
                 aoa_threshold=self.aoa.threshold, **self.meta), indent=2, default=str))

    @staticmethod
    def load(path=None) -> "GHPM25":
        with open(path or config.MODELS / "ghpm25_model.pkl", "rb") as f:
            return pickle.load(f)
