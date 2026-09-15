"""Daily 1-km PM2.5 mapping over Ghana with the fitted GH-PM25 system."""
from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr

from . import config, features as F
from .calibration import haversine
from .grid import admin_rasters
from .model import GHPM25, SCALE_FEATURES, apply_stack, krige_residuals
from .sources import firms

DAILY_DIR = config.OUTPUTS / "daily"


class GridContext:
    """Static part of the grid, loaded once."""

    def __init__(self):
        adm = admin_rasters()
        self.adm = adm
        self.mask = adm["mask"]
        self.lons, self.lats = adm["lons"], adm["lats"]
        LON, LAT = np.meshgrid(self.lons, self.lats)
        self.ii, self.jj = np.nonzero(self.mask)
        self.lat = LAT[self.mask].astype("float64")
        self.lon = LON[self.mask].astype("float64")
        st = xr.open_dataset(config.PROCESSED / "static" / "static_ghana.nc")
        st = st.reindex(lat=self.lats, lon=self.lons, method="nearest", tolerance=1e-4)
        self.static = F.transform_static(pd.DataFrame({v: st[v].values[self.mask] for v in st.data_vars}))
        self.pop_density = st["pop_density"].values[self.mask]
        cell_km2 = (config.RES * 111.32) ** 2 * np.cos(np.deg2rad(self.lat))
        self.pop = self.pop_density * cell_km2
        # coarse 0.05 deg lattice for fire predictors (smoke fields are smooth)
        self.fire_res = 0.05
        self.flons = np.arange(config.LON_MIN, config.LON_MAX + 1e-9, self.fire_res) + self.fire_res / 2
        self.flats = np.arange(config.LAT_MIN, config.LAT_MAX + 1e-9, self.fire_res) + self.fire_res / 2
        FL, FA = np.meshgrid(self.flons, self.flats)
        self.flon, self.flat = FL.ravel(), FA.ravel()
        self.fi = np.clip(((self.lat - config.LAT_MIN) / self.fire_res).astype(int), 0, len(self.flats) - 1)
        self.fj = np.clip(((self.lon - config.LON_MIN) / self.fire_res).astype(int), 0, len(self.flons) - 1)


def _fire_grid(ctx: GridContext, fires, date, cams: F.Lattice, met: F.Lattice) -> pd.DataFrame:
    d = np.full(len(ctx.flat), np.datetime64(pd.Timestamp(date)), dtype="datetime64[ns]")
    w = met.weights(ctx.flat, ctx.flon)
    t = met.t_index(pd.DatetimeIndex(d))
    u = met.sample("u10", t, w); v = met.sample("v10", t, w)
    f = firms.fire_features(fires, ctx.flon, ctx.flat, pd.Timestamp(date), np.nan_to_num(u), np.nan_to_num(v))
    nfj = len(ctx.flons)
    idx = ctx.fi * nfj + ctx.fj
    return pd.DataFrame({k: f[k][idx] for k in F.FIRE_COLS})


def predict_day(model: GHPM25, ctx: GridContext, date, cams: F.Lattice, met: F.Lattice, fires: pd.DataFrame,
                obs_day: pd.DataFrame | None, stations: pd.DataFrame | None = None, station_static: pd.DataFrame | None = None) -> xr.Dataset:
    date = pd.Timestamp(date).normalize()
    n = len(ctx.lat)
    dates = np.full(n, np.datetime64(date), dtype="datetime64[ns]")
    dyn = F.dynamic_features(ctx.lat, ctx.lon, dates, cams, met)
    fire = _fire_grid(ctx, fires, date, cams, met)
    X = pd.concat([dyn, fire, ctx.static.reset_index(drop=True)], axis=1)
    X["lat"], X["lon"] = ctx.lat, ctx.lon
    feats = model.base.features
    preds = model.base.predict(X)
    stage1 = apply_stack(preds, model.stack)

    # ---- stage 2: residual kriging with today's calibrated observations ----
    rk_mean = np.zeros(n); rk_var = np.full(n, model.correlogram["var"])
    dist_near = np.full(n, 2000.0); n100 = np.zeros(n)
    n_obs = 0
    if obs_day is not None and len(obs_day):
        o = obs_day.merge(stations[["site_id", "lat", "lon"]], on="site_id").merge(station_static, on="site_id", how="left")
        od = pd.DatetimeIndex(np.full(len(o), np.datetime64(date), dtype="datetime64[ns]"))
        odyn = F.dynamic_features(o.lat.to_numpy(), o.lon.to_numpy(), od, cams, met)
        ofire = F.fire_block(fires, o.lat.to_numpy(), o.lon.to_numpy(), od, odyn.u10.to_numpy(), odyn.v10.to_numpy())
        OX = pd.concat([odyn, ofire, F.transform_static(o[[c for c in F.STATIC_COLS]]).reset_index(drop=True)], axis=1)
        OX["lat"], OX["lon"] = o.lat.to_numpy(), o.lon.to_numpy()
        ok = OX.cams_pm25.notna().to_numpy()
        o, OX = o[ok].reset_index(drop=True), OX[ok].reset_index(drop=True)
        if len(o):
            ostage1 = apply_stack(model.base.predict(OX), model.stack)
            resid = np.log(o.pm25.clip(lower=1)).to_numpy() - ostage1
            resid = resid - o.site_id.map(model.site_offsets).fillna(0.0).to_numpy()  # krige day-specific anomalies only
            rk_mean, rk_var = krige_residuals(ctx.lat, ctx.lon, o.lat.to_numpy(), o.lon.to_numpy(), resid, model.correlogram)
            n_obs = len(o)
            for i in range(0, n, 50000):
                d = haversine(ctx.lat[i:i + 50000, None], ctx.lon[i:i + 50000, None], o.lat.to_numpy()[None, :], o.lon.to_numpy()[None, :])
                dist_near[i:i + 50000] = d.min(1); n100[i:i + 50000] = (d < 100).sum(1)
    logpred = stage1 + model.rk_lambda * rk_mean  # lambda selected by leave-site-out CV (0 => Stage 1 only)

    # ---- stage 3: uncertainty + AOA ----
    U = X.copy()
    U["krige_var"], U["dist_nearest_km"], U["n_obs_100km"] = rk_var, np.minimum(dist_near, 2000), n100
    sigma = np.maximum(model.scale_model.predict(U[SCALE_FEATURES].to_numpy(dtype="float32")), 0.03)
    mean = np.exp(logpred) * model.smearing
    lo90, hi90 = np.exp(logpred - model.q90 * sigma), np.exp(logpred + model.q90 * sigma)
    di = model.aoa.di(X)
    cams_raw = X["cams_pm25"].to_numpy()

    def grid(v, dtype="float32", fill=np.nan):
        a = np.full(ctx.mask.shape, fill, dtype=dtype)
        a[ctx.ii, ctx.jj] = v
        return a

    ds = xr.Dataset(
        {
            "pm25": (("lat", "lon"), grid(mean), dict(units="ug m-3", long_name="Daily mean PM2.5 (GH-PM25)")),
            "pm25_lower90": (("lat", "lon"), grid(lo90), dict(units="ug m-3", long_name="90% prediction interval lower bound")),
            "pm25_upper90": (("lat", "lon"), grid(hi90), dict(units="ug m-3", long_name="90% prediction interval upper bound")),
            "pm25_stage1": (("lat", "lon"), grid(np.exp(stage1) * model.smearing), dict(units="ug m-3", long_name="Stage-1 ensemble (no kriging)")),
            "cams_pm25": (("lat", "lon"), grid(cams_raw), dict(units="ug m-3", long_name="CAMS global PM2.5 (bilinear)")),
            "log_sigma": (("lat", "lon"), grid(sigma), dict(long_name="Predicted absolute log-error scale")),
            "aoa_di": (("lat", "lon"), grid(di), dict(long_name="Dissimilarity index (Meyer & Pebesma 2021)")),
            "inside_aoa": (("lat", "lon"), grid((di <= model.aoa.threshold).astype("int8"), "int8", -1)),
        },
        coords={"lat": ctx.lats, "lon": ctx.lons, "time": date},
        attrs=dict(title="GH-PM25 daily 1-km PM2.5 for Ghana", n_obs_used=n_obs, date=str(date.date()),
                   institution="GH-PM25 research system", conventions="CF-1.8"),
    )
    return ds


def region_stats(ds: xr.Dataset, ctx: GridContext, thresholds=(15.0, 35.0)) -> pd.DataFrame:
    pm = ds.pm25.values[ctx.mask]
    lo, hi = ds.pm25_lower90.values[ctx.mask], ds.pm25_upper90.values[ctx.mask]
    rows = []
    for level in ("adm1", "adm2"):
        idx = ctx.adm[level][ctx.mask]
        names = ctx.adm[f"{level}_names"]
        for k, name in enumerate(names):
            m = (idx == k) & np.isfinite(pm)
            if not m.any():
                continue
            p = ctx.pop[m]
            wsum = p.sum()
            rec = dict(level=level, id=k, name=name, area_mean=float(pm[m].mean()),
                       pop_weighted=float((pm[m] * p).sum() / wsum) if wsum > 0 else float(pm[m].mean()),
                       lower90=float(lo[m].mean()), upper90=float(hi[m].mean()), max=float(pm[m].max()),
                       population=float(wsum))
            for t in thresholds:
                rec[f"pop_frac_gt{int(t)}"] = float(p[pm[m] > t].sum() / wsum) if wsum > 0 else float((pm[m] > t).mean())
            rows.append(rec)
    p = ctx.pop
    ok = np.isfinite(pm)
    nat = dict(level="adm0", id=0, name="Ghana", area_mean=float(pm[ok].mean()), pop_weighted=float((pm[ok] * p[ok]).sum() / p[ok].sum()),
               lower90=float(lo[ok].mean()), upper90=float(hi[ok].mean()), max=float(pm[ok].max()), population=float(p[ok].sum()))
    for t in thresholds:
        nat[f"pop_frac_gt{int(t)}"] = float(p[ok][pm[ok] > t].sum() / p[ok].sum())
    rows.insert(0, nat)
    return pd.DataFrame(rows)
