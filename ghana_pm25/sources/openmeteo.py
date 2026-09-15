"""CAMS global atmospheric composition via the Open-Meteo air-quality API (keyless).

CAMS global (ECMWF IFS-COMPO, 0.4 deg, 3-hourly, assimilates satellite AOD) provides
pm2_5, pm10, aerosol_optical_depth, dust, carbon_monoxide and nitrogen_dioxide.
History on Open-Meteo starts 2022-08-04.

The free tier is weighted (~ n_locations * n_days/14 per call, 5,000/h, 10,000/day),
so nodes are requested on a 0.8 deg lattice aligned with the native 0.4 deg CAMS grid
and requests are paced with a token bucket. Fields are later bilinearly interpolated.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import time

import numpy as np
import pandas as pd

from .. import config
from ..net import get

AQ_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"
CAMS_VARS = ["pm2_5", "pm10", "aerosol_optical_depth", "dust", "carbon_monoxide", "nitrogen_dioxide"]
CAMS_RES = 0.8
CACHE = config.RAW / "openmeteo"
HOURLY_BUDGET = 4200.0


class _Pacer:
    def __init__(self, per_hour: float):
        self.rate = per_hour / 3600.0
        self.tokens = per_hour * 0.5
        self.cap = per_hour * 0.5
        self.t = time.time()

    def take(self, w: float):
        while True:
            now = time.time()
            self.tokens = min(self.cap, self.tokens + (now - self.t) * self.rate)
            self.t = now
            if self.tokens >= w:
                self.tokens -= w
                return
            time.sleep(min(60.0, (w - self.tokens) / self.rate + 1))


PACER = _Pacer(HOURLY_BUDGET)


def node_id(lat: float, lon: float) -> str:
    return f"{lat:.2f}_{lon:.2f}"


def lattice_nodes(bbox: dict | None = None, res: float = CAMS_RES) -> pd.DataFrame:
    b = bbox or dict(lon_min=config.LON_MIN, lon_max=config.LON_MAX, lat_min=config.LAT_MIN, lat_max=config.LAT_MAX)
    lons = np.round(np.arange(np.floor(b["lon_min"] / res) * res, np.ceil(b["lon_max"] / res) * res + res / 2, res), 2)
    lats = np.round(np.arange(np.floor(b["lat_min"] / res) * res, np.ceil(b["lat_max"] / res) * res + res / 2, res), 2)
    LON, LAT = np.meshgrid(lons, lats)
    df = pd.DataFrame({"lon": LON.ravel(), "lat": LAT.ravel()})
    df["node"] = [node_id(a, o) for a, o in zip(df.lat, df.lon)]
    return df


def nodes_around(points: pd.DataFrame, res: float = CAMS_RES) -> pd.DataFrame:
    """Lattice nodes enclosing each point (4 corners) for bilinear interpolation."""
    rows = []
    for lo, la in zip(points["lon"], points["lat"]):
        x0, y0 = np.floor(lo / res) * res, np.floor(la / res) * res
        for dx in (0, res):
            for dy in (0, res):
                rows.append((round(x0 + dx, 2), round(y0 + dy, 2)))
    df = pd.DataFrame(rows, columns=["lon", "lat"]).drop_duplicates()
    df["node"] = [node_id(a, o) for a, o in zip(df.lat, df.lon)]
    return df.reset_index(drop=True)


def _daily(h: pd.DataFrame) -> pd.DataFrame:
    h = h.copy()
    h["date"] = h["time"].dt.floor("D")
    g = h.groupby(["node", "date"])
    out = g[CAMS_VARS].mean()
    out["cams_pm25_max"] = g["pm2_5"].max()
    out["cams_n"] = g["pm2_5"].count()
    out = out.rename(columns={
        "pm2_5": "cams_pm25", "pm10": "cams_pm10", "aerosol_optical_depth": "cams_aod",
        "dust": "cams_dust", "carbon_monoxide": "cams_co", "nitrogen_dioxide": "cams_no2"})
    return out.reset_index()


def _fetch(nodes: pd.DataFrame, start: str, end: str, tag: str, batch: int = 100, refresh: bool = False,
           wait_on_quota: bool = True) -> pd.DataFrame:
    out_dir = CACHE / "cams" / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    nodes = nodes.sort_values("node").reset_index(drop=True)
    ndays = (pd.Timestamp(end) - pd.Timestamp(start)).days + 1
    frames = []
    for i in range(0, len(nodes), batch):
        b = nodes.iloc[i:i + batch]
        key = hashlib.md5("|".join(b.node).encode()).hexdigest()[:10]
        f = out_dir / f"{start}_{end}_{key}.parquet"
        if f.exists() and not refresh:
            frames.append(pd.read_parquet(f))
            continue
        PACER.take(len(b) * max(1.0, ndays / 14.0))
        params = dict(latitude=",".join(map(str, b.lat)), longitude=",".join(map(str, b.lon)),
                      hourly=",".join(CAMS_VARS), start_date=start, end_date=end, timezone="GMT")
        for _attempt in range(200):  # free-tier hourly/daily quotas: wait them out
            try:
                r = get(AQ_URL, params=params, timeout=300, retries=2)
                break
            except RuntimeError as e:
                if not wait_on_quota or ("limit" not in str(e).lower() and "429" not in str(e)):
                    raise
                print(f"    quota reached, sleeping 15 min ({str(e)[:90]})", flush=True)
                time.sleep(900)
        if r.status_code != 200:
            raise RuntimeError(f"CAMS request failed {r.status_code}: {r.text[:300]}")
        js = r.json()
        js = js if isinstance(js, list) else [js]
        hs = []
        for (_, node), item in zip(b.iterrows(), js):
            h = item["hourly"]
            df = pd.DataFrame({v: h.get(v) for v in CAMS_VARS}, dtype="float64")
            df["time"] = pd.to_datetime(h["time"])
            df["node"] = node["node"]
            hs.append(df)
        d = _daily(pd.concat(hs, ignore_index=True))
        d.to_parquet(f)
        frames.append(d)
    return pd.concat(frames, ignore_index=True)


def fetch_history(nodes: pd.DataFrame, start: str, end: str, tag: str = "hist", chunk_days: int = 182) -> pd.DataFrame:
    s, e = pd.Timestamp(start), pd.Timestamp(end)
    chunks, cur = [], s
    while cur <= e:
        nxt = min(cur + pd.Timedelta(days=chunk_days - 1), e)
        chunks.append((cur, nxt))
        cur = nxt + pd.Timedelta(days=1)
    frames = []
    for cur, nxt in reversed(chunks):  # newest first: most ground observations are recent
        print(f"  CAMS {cur.date()} -> {nxt.date()} ({len(nodes)} nodes)", flush=True)
        frames.append(_fetch(nodes, cur.strftime("%Y-%m-%d"), nxt.strftime("%Y-%m-%d"), tag))
    return pd.concat(frames, ignore_index=True).drop_duplicates(["node", "date"], keep="last")


def fetch_recent(nodes: pd.DataFrame, past_days: int = 8, future_days: int = 1) -> pd.DataFrame:
    """Latest CAMS analyses/forecast for operations (includes today's forecast)."""
    today = dt.date.today()
    start = (today - dt.timedelta(days=past_days)).isoformat()
    end = (today + dt.timedelta(days=future_days)).isoformat()
    return _fetch(nodes, start, end, tag="nrt", refresh=True, wait_on_quota=False)  # operations: fail fast, use cache
