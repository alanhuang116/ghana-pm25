"""NASA POWER daily regional API (keyless): MERRA-2 / GEOS meteorology + aerosol.

Grid 0.5 deg x 0.625 deg. Near-real-time extension from GEOS FP-IT (1-3 day latency).
One parameter and <= 366 days per request; bounding boxes <= 10 deg are tiled.

Parameters used:
  T2M, T2M_RANGE, RH2M, QV2M, T2MDEW  near-surface temperature/humidity
  PRECTOTCORR                         bias-corrected precipitation (mm/day)
  WS10M, U10M, V10M                   10-m wind (m/s)
  PS, PBLTOP                          surface pressure / PBL-top pressure (kPa, Pa?) -> PBL depth
  ALLSKY_SFC_SW_DWN, CLOUD_AMT        radiation / cloud fraction
  AOD_55                              MERRA-2 total AOD at 550 nm
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .. import config
from ..net import get

URL = "https://power.larc.nasa.gov/api/temporal/daily/regional"
PARAMS = ["T2M", "T2M_RANGE", "RH2M", "QV2M", "T2MDEW", "PRECTOTCORR", "WS10M", "U10M", "V10M",
          "PS", "PBLTOP", "ALLSKY_SFC_SW_DWN", "CLOUD_AMT", "AOD_55"]
RENAME = {"T2M": "t2m", "T2M_RANGE": "t2m_range", "RH2M": "rh", "QV2M": "qv2m", "T2MDEW": "td2m",
          "PRECTOTCORR": "precip", "WS10M": "ws10", "U10M": "u10", "V10M": "v10", "PS": "ps",
          "PBLTOP": "pbltop", "ALLSKY_SFC_SW_DWN": "ssrd", "CLOUD_AMT": "cloud", "AOD_55": "merra_aod"}
CACHE = config.RAW / "power"


def _tiles(b: dict, size: float = 9.5):
    lo = b["lon_min"]
    while lo < b["lon_max"]:
        la = b["lat_min"]
        while la < b["lat_max"]:
            yield dict(lon_min=lo, lon_max=min(lo + size, b["lon_max"]), lat_min=la, lat_max=min(la + size, b["lat_max"]))
            la += size
        lo += size


def _fetch_one(param: str, tile: dict, start: pd.Timestamp, end: pd.Timestamp, refresh: bool) -> pd.DataFrame:
    tag = f"{param}_{tile['lon_min']:.2f}_{tile['lat_min']:.2f}_{tile['lon_max']:.2f}_{tile['lat_max']:.2f}_{start:%Y%m%d}_{end:%Y%m%d}"
    f = CACHE / f"{tag}.parquet"
    if f.exists() and not refresh:
        return pd.read_parquet(f)
    CACHE.mkdir(parents=True, exist_ok=True)
    # POWER rejects boxes smaller than 2 deg: pad
    lat0, lat1 = tile["lat_min"], max(tile["lat_max"], tile["lat_min"] + 2.0)
    lon0, lon1 = tile["lon_min"], max(tile["lon_max"], tile["lon_min"] + 2.0)
    r = get(URL, params={"parameters": param, "community": "RE", "latitude-min": lat0, "latitude-max": lat1,
                         "longitude-min": lon0, "longitude-max": lon1, "start": f"{start:%Y%m%d}",
                         "end": f"{end:%Y%m%d}", "format": "JSON"}, timeout=600)
    if r.status_code != 200:
        raise RuntimeError(f"POWER {param} failed {r.status_code}: {r.text[:300]}")
    rows = []
    for feat in r.json()["features"]:
        lon, lat = feat["geometry"]["coordinates"][:2]
        series = feat["properties"]["parameter"][param]
        for d, v in series.items():
            rows.append((lat, lon, d, v))
    df = pd.DataFrame(rows, columns=["lat", "lon", "date", param])
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")
    df.loc[df[param] <= -998, param] = np.nan
    df.to_parquet(f)
    return df


def fetch(bbox: dict, start: str, end: str, refresh_recent: bool = False) -> pd.DataFrame:
    from scipy.spatial import cKDTree

    s, e = pd.Timestamp(start), pd.Timestamp(end)
    out = None
    base_grid = None
    for p in PARAMS:
        frames = []
        cur = s
        while cur <= e:
            nxt = min(pd.Timestamp(year=cur.year, month=12, day=31), e)
            recent = (pd.Timestamp.today() - nxt).days < 30
            for t in _tiles(bbox):
                frames.append(_fetch_one(p, t, cur, nxt, refresh=refresh_recent and recent))
            cur = nxt + pd.Timedelta(days=1)
        df = pd.concat(frames).drop_duplicates(["lat", "lon", "date"])
        if out is None:
            out = df
            base_grid = df[["lat", "lon"]].drop_duplicates().reset_index(drop=True)
        else:
            grid = df[["lat", "lon"]].drop_duplicates().reset_index(drop=True)
            if len(grid) != len(base_grid) or not grid.merge(base_grid).shape[0] == len(base_grid):
                # different native grid (solar/cloud/aerosol at 1 deg): nearest-neighbour to the met grid
                _, idx = cKDTree(grid[["lat", "lon"]].to_numpy()).query(base_grid[["lat", "lon"]].to_numpy())
                mapping = base_grid.assign(src_lat=grid.lat.to_numpy()[idx], src_lon=grid.lon.to_numpy()[idx])
                df = mapping.merge(df.rename(columns={"lat": "src_lat", "lon": "src_lon"}), on=["src_lat", "src_lon"]) \
                            .drop(columns=["src_lat", "src_lon"])
            out = out.merge(df, on=["lat", "lon", "date"], how="left")
        print(f"  POWER {p}: {len(df)} rows", flush=True)
    out = out.rename(columns=RENAME)
    # PBL depth from pressure difference (hypsometric, scale height ~8.4 km in the tropics)
    ps_pa = out["ps"] * 1000.0  # kPa -> Pa
    pbl = out["pbltop"]
    pbl_pa = np.where(pbl < 200, pbl * 1000.0, pbl)  # tolerate kPa or Pa
    out["blh"] = 8400.0 * np.log(ps_pa / pbl_pa)
    out.loc[(out["blh"] < 0) | (out["blh"] > 6000), "blh"] = np.nan
    return out.drop(columns=["pbltop"])
