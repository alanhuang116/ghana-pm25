"""Feature engineering shared by training (station-days) and mapping (1-km grid cells).

Dynamic predictors are bilinearly interpolated from their native lattices (CAMS 0.8 deg
sub-lattice, MERRA-2/POWER 0.5 x 0.625 deg) to target coordinates, then augmented with
temporal lags/accumulations and physically-motivated derived terms.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config
from .sources import firms

CAMS_COLS = ["cams_pm25", "cams_pm10", "cams_aod", "cams_dust", "cams_co", "cams_no2", "cams_pm25_max"]
MET_COLS = ["t2m", "t2m_range", "rh", "qv2m", "td2m", "precip", "ws10", "u10", "v10", "ps", "ssrd", "cloud", "merra_aod", "blh"]

STATIC_COLS = [
    "elev", "elev_sd", "elev_rel_10km", "lc_tree", "lc_shrub", "lc_grass", "lc_crop", "lc_built", "lc_bare", "lc_water",
    "log_pop", "log_ntl", "road_density", "dist_road", "dist_coast",
    "lc_built_s3km", "lc_built_s10km", "log_pop_s3km", "log_pop_s10km", "log_ntl_s3km", "log_ntl_s10km",
    "road_density_s3km", "road_density_s10km", "lc_crop_s10km", "lc_tree_s10km", "lc_bare_s10km",
]
DYNAMIC_DERIVED = [
    "log_cams_pm25", "cams_pm25_lag1", "cams_pm25_3d", "cams_dust_lag1", "dust_frac", "log_cams_dust",
    "precip_3d", "precip_lag1", "log_blh", "ventilation", "rh_lag1", "ws10_lag1", "merra_aod_3d",
    "wind_dir_sin", "wind_dir_cos",
]
FIRE_COLS = ["frp_25km", "frp_100km", "fire_upwind"]
TIME_COLS = ["doy_sin", "doy_cos", "dow"]
COORD_COLS = ["lat", "lon"]

FEATURES = CAMS_COLS + MET_COLS + DYNAMIC_DERIVED + FIRE_COLS + STATIC_COLS + TIME_COLS + COORD_COLS


# ---------------------------------------------------------------------------
# bilinear interpolation from a regular lattice stored in long format
# ---------------------------------------------------------------------------
class Lattice:
    """Regular lattice (lat0 + i*dlat, lon0 + j*dlon) with daily fields in long format."""

    def __init__(self, df: pd.DataFrame, cols: list[str], dlat: float, dlon: float, lat0: float, lon0: float):
        self.dlat, self.dlon, self.lat0, self.lon0 = dlat, dlon, lat0, lon0
        self.cols = cols
        df = df.copy()
        df["i"] = np.round((df.lat - lat0) / dlat).astype(int)
        df["j"] = np.round((df.lon - lon0) / dlon).astype(int)
        self.ni, self.nj = df.i.max() + 1, df.j.max() + 1
        self.dates = np.sort(df.date.unique())
        di = {d: k for k, d in enumerate(self.dates)}
        df["t"] = df.date.map(di)
        self.cube = {}
        for c in cols:
            a = np.full((len(self.dates), self.ni, self.nj), np.nan, dtype="float32")
            a[df.t.to_numpy(), df.i.to_numpy(), df.j.to_numpy()] = df[c].to_numpy(dtype="float32")
            self.cube[c] = a
        self.date_index = pd.DatetimeIndex(self.dates)

    def weights(self, lat: np.ndarray, lon: np.ndarray):
        fi = (lat - self.lat0) / self.dlat
        fj = (lon - self.lon0) / self.dlon
        i0 = np.clip(np.floor(fi).astype(int), 0, self.ni - 2)
        j0 = np.clip(np.floor(fj).astype(int), 0, self.nj - 2)
        wi, wj = np.clip(fi - i0, 0, 1), np.clip(fj - j0, 0, 1)
        return i0, j0, wi, wj

    def sample(self, col: str, t_idx: np.ndarray, w) -> np.ndarray:
        i0, j0, wi, wj = w
        a = self.cube[col]
        v00, v01 = a[t_idx, i0, j0], a[t_idx, i0, j0 + 1]
        v10, v11 = a[t_idx, i0 + 1, j0], a[t_idx, i0 + 1, j0 + 1]
        W = np.stack([(1 - wi) * (1 - wj), (1 - wi) * wj, wi * (1 - wj), wi * wj])
        V = np.stack([v00, v01, v10, v11])
        m = ~np.isnan(V)
        wsum = (W * m).sum(0)
        out = (np.nan_to_num(V) * W).sum(0) / np.where(wsum > 0, wsum, np.nan)
        return out.astype("float32")

    def t_index(self, dates: pd.DatetimeIndex | np.ndarray, lag: int = 0) -> np.ndarray:
        d = pd.DatetimeIndex(dates) - pd.Timedelta(days=lag)
        idx = self.date_index.get_indexer(d)
        return idx


def load_cams_lattice(path=None) -> Lattice:
    cams = pd.read_parquet(path or config.INTERIM / "cams_daily.parquet")
    nodes = pd.read_parquet(config.INTERIM / "cams_nodes.parquet") if path is None else None
    if "lat" not in cams:
        nodes = nodes if nodes is not None else pd.read_parquet(config.INTERIM / "cams_nodes.parquet")
        cams = cams.merge(nodes[["node", "lat", "lon"]], on="node")
    return cams_lattice_from_df(cams)


def cams_lattice_from_df(cams: pd.DataFrame) -> Lattice:
    if "lat" not in cams:
        ll = cams.node.str.split("_", expand=True).astype(float)
        cams = cams.assign(lat=ll[0], lon=ll[1])
    cams["date"] = pd.to_datetime(cams["date"]).dt.tz_localize(None)
    return Lattice(cams, CAMS_COLS, 0.8, 0.8, float(cams.lat.min()), float(cams.lon.min()))


def met_lattice_from_df(met: pd.DataFrame) -> Lattice:
    met = met.copy()
    met["date"] = pd.to_datetime(met["date"])
    return Lattice(met, MET_COLS, 0.5, 0.625, float(met.lat.min()), float(met.lon.min()))


def _lagged(lat_: Lattice, col: str, dates, w, lag: int):
    t = lat_.t_index(dates, lag)
    out = np.full(len(t), np.nan, dtype="float32")
    ok = t >= 0
    if ok.any():
        w_ok = tuple(x[ok] for x in w)
        out[ok] = lat_.sample(col, t[ok], w_ok)
    return out


def dynamic_features(lat: np.ndarray, lon: np.ndarray, dates: np.ndarray, cams: Lattice, met: Lattice) -> pd.DataFrame:
    """Dynamic predictors for arrays of (lat, lon, date) of equal length."""
    dates = pd.DatetimeIndex(dates)
    wc = cams.weights(lat, lon)
    wm = met.weights(lat, lon)
    out = {}
    for c in CAMS_COLS:
        out[c] = _lagged(cams, c, dates, wc, 0)
    for c in MET_COLS:
        out[c] = _lagged(met, c, dates, wm, 0)
    out["cams_pm25_lag1"] = _lagged(cams, "cams_pm25", dates, wc, 1)
    lag2 = _lagged(cams, "cams_pm25", dates, wc, 2)
    out["cams_pm25_3d"] = np.nanmean(np.stack([out["cams_pm25"], out["cams_pm25_lag1"], lag2]), axis=0)
    out["cams_dust_lag1"] = _lagged(cams, "cams_dust", dates, wc, 1)
    out["log_cams_pm25"] = np.log(np.maximum(out["cams_pm25"], 0.1))
    out["log_cams_dust"] = np.log1p(np.maximum(out["cams_dust"], 0))
    out["dust_frac"] = out["cams_dust"] / (out["cams_pm10"] + 1.0)
    p1 = _lagged(met, "precip", dates, wm, 1)
    p2 = _lagged(met, "precip", dates, wm, 2)
    out["precip_lag1"] = p1
    out["precip_3d"] = np.nansum(np.stack([out["precip"], p1, p2]), axis=0)
    out["log_blh"] = np.log(np.maximum(out["blh"], 30))
    out["ventilation"] = out["blh"] * out["ws10"]
    out["rh_lag1"] = _lagged(met, "rh", dates, wm, 1)
    out["ws10_lag1"] = _lagged(met, "ws10", dates, wm, 1)
    a1 = _lagged(met, "merra_aod", dates, wm, 1)
    a2 = _lagged(met, "merra_aod", dates, wm, 2)
    out["merra_aod_3d"] = np.nanmean(np.stack([out["merra_aod"], a1, a2]), axis=0)
    wdir = np.arctan2(-out["u10"], -out["v10"])  # direction wind blows from
    out["wind_dir_sin"], out["wind_dir_cos"] = np.sin(wdir), np.cos(wdir)
    doy = dates.dayofyear.to_numpy()
    out["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    out["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    out["dow"] = dates.dayofweek.to_numpy()
    return pd.DataFrame(out)


def fire_block(fires: pd.DataFrame, lat: np.ndarray, lon: np.ndarray, dates: np.ndarray,
               u: np.ndarray, v: np.ndarray) -> pd.DataFrame:
    dates = pd.DatetimeIndex(dates)
    res = {k: np.zeros(len(lat), dtype="float32") for k in FIRE_COLS}
    for d in np.unique(dates):
        m = np.asarray(dates == d)
        f = firms.fire_features(fires, lon[m], lat[m], pd.Timestamp(d), np.nan_to_num(u[m]), np.nan_to_num(v[m]))
        for k in FIRE_COLS:
            res[k][m] = f[k]
    return pd.DataFrame(res)


def transform_static(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["dist_road"] = np.log1p(df["dist_road"])
    df["dist_coast"] = np.log1p(df["dist_coast"])
    return df


def build_training_table() -> pd.DataFrame:
    stations = pd.read_parquet(config.PROCESSED / "stations.parquet")
    obs = pd.read_parquet(config.PROCESSED / "obs_daily_calibrated.parquet")
    st_static = transform_static(pd.read_parquet(config.PROCESSED / "station_static.parquet"))
    cams = load_cams_lattice()
    met = met_lattice_from_df(pd.read_parquet(config.INTERIM / "power_daily.parquet"))
    fires = pd.read_parquet(config.INTERIM / "firms_hist.parquet")

    df = obs.merge(stations[["site_id", "lat", "lon", "network", "country", "is_reference"]], on="site_id")
    df = df.sort_values(["date", "site_id"]).reset_index(drop=True)
    dyn = dynamic_features(df.lat.to_numpy(), df.lon.to_numpy(), df.date.to_numpy(), cams, met)
    fire = fire_block(fires, df.lat.to_numpy(), df.lon.to_numpy(), df.date.to_numpy(), dyn.u10.to_numpy(), dyn.v10.to_numpy())
    df = pd.concat([df, dyn, fire], axis=1).merge(st_static, on="site_id", how="left")
    return df
