"""VIIRS 375-m active-fire detections (keyless) and fire predictors.

A single sensor (Suomi-NPP VIIRS) is used for the whole record for consistency:

* 2022-08-04 .. 2024-12-31 : NASA FIRMS yearly country archives (VNP14IMG, standard processing)
* 2025-01-01 .. today       : NOAA Enterprise VIIRS I-band fire EDR (EFIRE-VIIRSI v1r3; the legacy
                              AF-Iband EDR ends Feb 2025) on the NOAA Open Data Dissemination bucket
                              s3://noaa-nesdis-snpp-pds/VIIRS_EFIRE_VIIRSI_EDR/ (~1-2 h latency).

Granule selection for the EDR uses a circular sun-synchronous orbit model: each day a sparse set
of probe granules is read, their embedded sampled geolocation gives nadir position and pass
direction, from which the orbit's ascending-node time and longitude are estimated; every granule
whose nadir track passes within one swath of West Africa is then read.
This is robust to the along-track timing drift that defeats fixed 16-day repeat templates.

Biomass-burning smoke is advected, so besides local FRP an upwind-weighted fire influence is
computed for each target location.
"""
from __future__ import annotations

import datetime as dt
import io
import re
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

from .. import config
from ..net import download, get

BUCKET = "https://noaa-nesdis-snpp-pds.s3.amazonaws.com"
PREFIX = "VIIRS_EFIRE_VIIRSI_EDR"
EDR_START = dt.date(2025, 1, 1)
CACHE = config.RAW / "fires"
FIRE_BBOX = dict(lon_min=-10.0, lon_max=10.0, lat_min=3.0, lat_max=15.5)
WINDOWS = [(-120, 300), (600, 1020)]  # UTC minutes containing the ~01:30 / ~13:30 LST overpasses of the region
NS = {"s": "http://s3.amazonaws.com/doc/2006-03-01/"}
FIRMS_COUNTRIES = ["Ghana", "Cote_d_Ivoire", "Togo", "Burkina_Faso", "Benin", "Nigeria", "Mali", "Niger", "Liberia", "Guinea"]

INCL = np.deg2rad(98.72)
PERIOD = 101.44              # minutes
EARTH_ROT = 360.0 / 1436.07  # deg / minute
SWATH_HALF_DEG = 13.6        # ~1500 km half swath

_SESSION = None


def _session():
    global _SESSION
    if _SESSION is None:
        import requests
        _SESSION = requests.Session()
        _SESSION.mount("https://", requests.adapters.HTTPAdapter(pool_connections=64, pool_maxsize=64))
    return _SESSION


# ---------------------------------------------------------------------------
# EDR access
# ---------------------------------------------------------------------------
def _list_day(day: dt.date) -> list[str]:
    keys, tok = [], None
    while True:
        p = {"list-type": "2", "prefix": f"{PREFIX}/{day:%Y/%m/%d}/"}
        if tok:
            p["continuation-token"] = tok
        root = ET.fromstring(get(BUCKET + "/", params=p, timeout=120).content)
        keys += [c.find("s:Key", NS).text for c in root.findall("s:Contents", NS)]
        if root.findtext("s:IsTruncated", namespaces=NS) != "true":
            return sorted(keys)
        tok = root.findtext("s:NextContinuationToken", namespaces=NS)


def _minute_of_day(key: str) -> float:
    m = re.search(r"_s\d{8}(\d{2})(\d{2})(\d{2})", key)
    return int(m.group(1)) * 60 + int(m.group(2)) + int(m.group(3)) / 60.0


def _in_windows(key: str) -> bool:
    t = _minute_of_day(key)
    return any(a <= t <= b or a <= t - 1440 <= b for a, b in WINDOWS)


def read_granule(key: str) -> dict | None:
    """Download a whole EFIRE granule (~1 MB; much faster than HTTP range reads) and parse in memory.

    Returns fire pixels plus the granule's nadir position and pass direction from the embedded
    24 x 100 sampled geolocation grid."""
    import h5py
    for _ in range(3):
        try:
            content = _session().get(f"{BUCKET}/{key}", timeout=60).content
            with h5py.File(io.BytesIO(content), "r") as f:
                df = pd.DataFrame({
                    "latitude": f["FP_Latitude"][:], "longitude": f["FP_Longitude"][:], "frp": f["FP_FireRadiativePower"][:],
                    "confidence": f["FP_FireMask"][:].astype("int16"), "pac": f["FP_PersistentAnomalyCategory"][:].astype("int16"),
                })
                slat, slon = f["Sampled_Latitude"][:], f["Sampled_Longitude"][:]
            mid_r, mid_c = slat.shape[0] // 2, slat.shape[1] // 2
            nlat, nlon = float(slat[mid_r, mid_c]), float(slon[mid_r, mid_c])
            asc = bool(slat[-1, mid_c] > slat[0, mid_c])
            ok = -90 <= nlat <= 90 and -180 <= nlon <= 180
            return dict(key=key, t=_minute_of_day(key) + 0.7, asc=asc, nadir=(nlat, nlon) if ok else None, pixels=df)
        except Exception:
            continue
    return None


def _regional(g: dict) -> pd.DataFrame:
    df, b = g["pixels"], FIRE_BBOX
    out = df[df.latitude.between(b["lat_min"], b["lat_max"]) & df.longitude.between(b["lon_min"], b["lon_max"])].copy()
    out["granule"] = g["key"].rsplit("/", 1)[-1]
    return out


# ---------------------------------------------------------------------------
# orbit model
# ---------------------------------------------------------------------------
def _node_from_granule(g: dict):
    if g.get("nadir") is None:
        return None
    lat, lon = g["nadir"]
    s = np.sin(np.deg2rad(lat)) / np.sin(INCL)
    if abs(s) > 1:
        return None
    u = np.rad2deg(np.arcsin(s))
    if not g["asc"]:
        u = 180.0 - u
    node_t = g["t"] - u / 360.0 * PERIOD
    ur = np.deg2rad(u)
    node_lon = lon - np.rad2deg(np.arctan2(np.cos(INCL) * np.sin(ur), np.cos(ur))) + EARTH_ROT * (g["t"] - node_t)
    return node_t, node_lon


def _nadir(t, node_t, node_lon):
    u = np.deg2rad(360.0 * (t - node_t) / PERIOD)
    lat = np.rad2deg(np.arcsin(np.sin(INCL) * np.sin(u)))
    lon = node_lon + np.rad2deg(np.arctan2(np.cos(INCL) * np.sin(u), np.cos(u))) - EARTH_ROT * (t - node_t)
    return lat, (lon + 180.0) % 360.0 - 180.0


def fires_for_day(day: dt.date, workers: int = 48, use_cache: bool = True, probe_stride: int = 50) -> tuple[pd.DataFrame, dict]:
    f = CACHE / "edr_daily" / f"{day:%Y%m%d}.parquet"
    if use_cache and f.exists():
        return pd.read_parquet(f), {"day": str(day), "cached": True}
    keys = _list_day(day)
    cand = [k for k in keys if _in_windows(k)]
    probes = cand[::probe_stride]
    with ThreadPoolExecutor(workers) as ex:
        got = {g["key"]: g for g in ex.map(read_granule, probes) if g is not None}
    nodes = [n for n in (_node_from_granule(g) for g in got.values()) if n is not None]
    diag = dict(day=str(day), n_granules=len(keys), n_probes=len(probes), n_geolocated=len(nodes))
    if len(nodes) >= 3:
        t_ref = nodes[0][0]
        k = np.array([round((t - t_ref) / PERIOD) for t, _ in nodes])
        tt = np.array([t for t, _ in nodes]) - k * PERIOD
        ll = np.array([lo for _, lo in nodes]) + k * EARTH_ROT * PERIOD
        node_t = float(np.median(tt))
        node_lon = float(np.rad2deg(np.arctan2(np.median(np.sin(np.deg2rad(ll))), np.median(np.cos(np.deg2rad(ll))))))
        b = FIRE_BBOX
        sel = []
        for key in cand:
            ts = _minute_of_day(key) + np.array([-0.3, 0.7, 1.7])
            la, lo = _nadir(ts, node_t, node_lon)
            dlon = np.maximum(0, np.maximum(b["lon_min"] - lo, lo - b["lon_max"]))
            if np.any((la > b["lat_min"] - 8) & (la < b["lat_max"] + 8) & (dlon < SWATH_HALF_DEG + 6)):
                sel.append(key)
        diag.update(mode="orbit", n_selected=len(sel), node_t_iqr_min=float(np.subtract(*np.percentile(tt, [75, 25]))))
    else:
        sel = cand
        diag.update(mode="window_fallback", n_selected=len(sel))
    todo = [k for k in sel if k not in got]
    with ThreadPoolExecutor(workers) as ex:
        for g in ex.map(read_granule, todo):
            if g is not None:
                got[g["key"]] = g
    parts = [_regional(got[k]) for k in sel if k in got]
    df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["latitude", "longitude", "frp", "confidence", "pac", "sample", "granule"])
    df["date"] = pd.Timestamp(day)
    if day < dt.date.today() - dt.timedelta(days=1):
        f.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(f)
    return df, diag


def clean_edr(df: pd.DataFrame) -> pd.DataFrame:
    """Fire-mask nominal/high confidence (8, 9), no persistent anomaly (gas flares, industry, volcanoes)."""
    df = df[(df.confidence >= 8) & (df.pac == 0) & (df.frp > 0) & (df.frp < 1e5)]  # FRP fill value is -9999
    df = df.drop_duplicates(["latitude", "longitude", "date", "frp"])
    return df[["date", "latitude", "longitude", "frp"]].astype({"frp": "float32"})


# ---------------------------------------------------------------------------
# FIRMS archive (2022-2024)
# ---------------------------------------------------------------------------
def firms_archive(year: int, start: str, end: str) -> pd.DataFrame:
    frames = []
    for c in FIRMS_COUNTRIES:
        url = f"https://firms.modaps.eosdis.nasa.gov/data/country/viirs-snpp/{year}/viirs-snpp_{year}_{c}.csv"
        dest = CACHE / "firms" / str(year) / f"{c}.csv"
        try:
            download(url, dest)
        except RuntimeError:
            continue
        df = pd.read_csv(dest, usecols=["latitude", "longitude", "acq_date", "frp", "confidence", "type"])
        df["date"] = pd.to_datetime(df.acq_date)
        df = df[(df.date >= start) & (df.date <= end)]
        conf = df.confidence.astype(str).str.lower().str[0]
        df = df[(conf != "l") & (df["type"] == 0)]  # type 0 = presumed vegetation fire
        b = FIRE_BBOX
        df = df[df.latitude.between(b["lat_min"], b["lat_max"]) & df.longitude.between(b["lon_min"], b["lon_max"])]
        frames.append(df[["date", "latitude", "longitude", "frp"]])
    return pd.concat(frames, ignore_index=True).drop_duplicates()


def load_history(start: str, end: str, day_workers: int = 2) -> pd.DataFrame:
    s, e = pd.Timestamp(start).date(), pd.Timestamp(end).date()
    frames = []
    for y in range(s.year, min(e.year, EDR_START.year - 1) + 1):
        a = max(s, dt.date(y, 1, 1)); b = min(e, dt.date(y, 12, 31))
        frames.append(firms_archive(y, str(a), str(b)))
        print(f"  FIRMS S-NPP archive {y}: {len(frames[-1])} detections", flush=True)
    days = [d.date() for d in pd.date_range(max(s, EDR_START), e)]
    diags = []
    with ThreadPoolExecutor(day_workers) as ex:
        for i, (df, dg) in enumerate(ex.map(lambda d: fires_for_day(d, workers=32), days)):
            if len(df):
                frames.append(clean_edr(df))
            diags.append(dg)
            if i % 25 == 0:
                print(f"  EDR {days[i]} ({i + 1}/{len(days)}) {dg}", flush=True)
    pd.DataFrame(diags).to_csv(config.OUTPUTS / "fire_edr_selection_log.csv", index=False)
    out = pd.concat(frames, ignore_index=True)
    out["frp"] = out["frp"].astype("float32")
    return out


def load_recent(days: int = 10) -> pd.DataFrame:
    today = dt.date.today()
    frames = []
    for i in range(days, -1, -1):
        df, _ = fires_for_day(today - dt.timedelta(days=i))
        if len(df):
            frames.append(clean_edr(df))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["date", "latitude", "longitude", "frp"])


# ---------------------------------------------------------------------------
# predictors
# ---------------------------------------------------------------------------
def fire_features(fires: pd.DataFrame, lon: np.ndarray, lat: np.ndarray, date: pd.Timestamp,
                  u: np.ndarray, v: np.ndarray) -> dict[str, np.ndarray]:
    """Fire predictors for target points on one day (fires of day t and t-1).

    frp_25km  : log1p sum FRP * exp(-d^2 / (2 * 10 km^2))   (local fires, Gaussian kernel; name kept for compatibility)
    frp_100km : log1p sum FRP * exp(-d^2 / (2 * 50 km^2))   (regional fires, Gaussian kernel)
    fire_upwind : log1p sum FRP * max(cos(theta), 0) * exp(-d / 150 km), d < 500 km,
                  theta = angle between fire->target vector and the mean 10-m wind.
    Smooth kernels avoid the disk-shaped artefacts that hard radii create in the maps.
    """
    sel = fires[(fires.date >= date - pd.Timedelta(days=1)) & (fires.date <= date)]
    n = len(lon)
    out = {k: np.zeros(n, dtype="float32") for k in ["frp_25km", "frp_100km", "fire_upwind"]}
    if sel.empty:
        return out
    agg = sel.assign(la=(sel.latitude / 0.05).round() * 0.05, lo=(sel.longitude / 0.05).round() * 0.05) \
             .groupby(["la", "lo"], as_index=False).frp.sum()
    flon, flat, frp = agg.lo.to_numpy(), agg.la.to_numpy(), agg.frp.to_numpy()
    km = 111.32
    wn = np.hypot(u, v) + 1e-6
    step = max(1, int(4e6 // max(len(agg), 1)))
    for i in range(0, n, step):
        sl = slice(i, i + step)
        coslat = np.cos(np.deg2rad(lat[sl]))[:, None]
        dx = (lon[sl, None] - flon[None, :]) * km * coslat
        dy = (lat[sl, None] - flat[None, :]) * km
        d = np.hypot(dx, dy) + 1e-3
        out["frp_25km"][sl] = np.log1p((frp * np.exp(-d ** 2 / (2 * 10.0 ** 2))).sum(1))
        out["frp_100km"][sl] = np.log1p((frp * np.exp(-d ** 2 / (2 * 50.0 ** 2))).sum(1))
        cos_t = (dx * u[sl, None] + dy * v[sl, None]) / (d * wn[sl, None])
        w = np.clip(cos_t, 0, None) * np.exp(-d / 150.0) * (d < 500)
        out["fire_upwind"][sl] = np.log1p((frp * w).sum(1))
    return out
