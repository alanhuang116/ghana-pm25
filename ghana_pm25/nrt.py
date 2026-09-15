"""Near-real-time data assimilation for the daily operational run.

Product tiers
  forecast  D+1         CAMS forecast + ECMWF IFS meteorology (bias-adjusted) + persistence fires, no obs
  nowcast   D0          CAMS analysis/forecast + IFS meteorology + same-day fires + any obs received
  nrt       D-1 .. D-6  inputs as available; re-run daily as late data (POWER, OpenAQ) arrives
  final     <= D-7      all inputs final (reprocessed weekly)
"""
from __future__ import annotations

import datetime as dt
import gzip
import io
import os

import numpy as np
import pandas as pd

from . import calibration, config, features as F
from .net import SESSION, get
from .sources import firms, ground, openmeteo, power

FC_URL = "https://api.open-meteo.com/v1/forecast"
NRT_DIR = config.INTERIM / "nrt"
NRT_DIR.mkdir(parents=True, exist_ok=True)


def tier(date: pd.Timestamp, today: pd.Timestamp) -> str:
    d = (today - date).days
    if d < 0:
        return "forecast"
    if d == 0:
        return "nowcast"
    if d < 7:
        return "nrt"
    return "final"


# ---------------------------------------------------------------------------
def update_cams(nodes: pd.DataFrame) -> pd.DataFrame:
    hist = pd.read_parquet(config.INTERIM / "cams_daily.parquet")
    recent = openmeteo.fetch_recent(nodes, past_days=10, future_days=1)
    recent = recent[recent.cams_n >= 6]  # at least 6 of 24 hourly steps (CAMS is 3-hourly)
    out = pd.concat([hist[~hist.date.isin(recent.date.unique())], recent], ignore_index=True)
    out.to_parquet(config.INTERIM / "cams_daily.parquet")
    return out


def update_power() -> pd.DataFrame:
    today = pd.Timestamp.today().normalize()
    start = f"{today.year}-01-01" if today.month > 1 else f"{today.year - 1}-01-01"
    hist = pd.read_parquet(config.INTERIM / "power_daily.parquet")
    new = power.fetch(ground.REGION, start, (today - pd.Timedelta(days=1)).strftime("%Y-%m-%d"), refresh_recent=True)
    new = new.dropna(subset=["t2m"])
    out = pd.concat([hist[hist.date < pd.Timestamp(start)], new], ignore_index=True).drop_duplicates(["lat", "lon", "date"], keep="last")
    out.to_parquet(config.INTERIM / "power_daily.parquet")
    return out


def ifs_fill(met: pd.DataFrame, days_ahead: int = 1) -> pd.DataFrame:
    """Extend MERRA-2/POWER meteorology to D+days_ahead with ECMWF IFS (Open-Meteo), bias-adjusted
    per node & variable by linear regression on the most recent overlapping 45 days."""
    b = dict(lat_min=config.LAT_MIN - 0.5, lat_max=config.LAT_MAX + 0.5, lon_min=config.LON_MIN - 0.7, lon_max=config.LON_MAX + 0.7)
    last = met.dropna(subset=["t2m"]).date.max()
    nodes = met[(met.date == last) & met.lat.between(b["lat_min"], b["lat_max"]) & met.lon.between(b["lon_min"], b["lon_max"])][["lat", "lon"]]
    today = pd.Timestamp.today().normalize()
    start = (last - pd.Timedelta(days=45)).strftime("%Y-%m-%d")
    end = (today + pd.Timedelta(days=days_ahead)).strftime("%Y-%m-%d")
    hourly = ["temperature_2m", "relative_humidity_2m", "dew_point_2m", "precipitation", "wind_speed_10m",
              "wind_direction_10m", "surface_pressure", "shortwave_radiation", "cloud_cover", "boundary_layer_height"]
    frames = []
    nodes = nodes.reset_index(drop=True)
    for i in range(0, len(nodes), 100):
        nb = nodes.iloc[i:i + 100]
        openmeteo.PACER.take(len(nb) * max(1.0, (pd.Timestamp(end) - pd.Timestamp(start)).days / 14))
        r = get(FC_URL, params=dict(latitude=",".join(map(str, nb.lat)), longitude=",".join(map(str, nb.lon)),
                                    hourly=",".join(hourly), start_date=start, end_date=end, timezone="GMT", models="ecmwf_ifs"), timeout=300, retries=2)  # 9-km IFS (includes PBL height)
        if r.status_code != 200:
            raise RuntimeError(f"IFS request failed {r.status_code}: {r.text[:200]}")
        js = r.json(); js = js if isinstance(js, list) else [js]
        for (_, node), item in zip(nb.iterrows(), js):
            h = pd.DataFrame(item["hourly"])
            for c in hourly:
                h[c] = pd.to_numeric(h[c], errors="coerce")
            h["time"] = pd.to_datetime(h["time"])
            h["date"] = h.time.dt.floor("D")
            ws = h.wind_speed_10m / 3.6; rad = np.deg2rad(h.wind_direction_10m)
            h["u"] = -ws * np.sin(rad); h["v"] = -ws * np.cos(rad); h["ws"] = ws
            g = h.groupby("date")
            d = pd.DataFrame({
                "t2m": g.temperature_2m.mean(), "t2m_range": g.temperature_2m.max() - g.temperature_2m.min(),
                "rh": g.relative_humidity_2m.mean(), "td2m": g.dew_point_2m.mean(), "precip": g.precipitation.sum(),
                "ws10": g.ws.mean(), "u10": g.u.mean(), "v10": g.v.mean(), "ps": g.surface_pressure.mean() / 10.0,
                "ssrd": g.shortwave_radiation.mean() * 24 / 1000.0, "cloud": g.cloud_cover.mean(), "blh": g.boundary_layer_height.mean(),
            }).reset_index()
            e = 6.112 * np.exp(17.67 * d.td2m / (d.td2m + 243.5))  # hPa
            d["qv2m"] = 622.0 * e / (d.ps * 10 - 0.378 * e)       # g/kg
            d["lat"], d["lon"] = node.lat, node.lon
            frames.append(d)
    ifs = pd.concat(frames, ignore_index=True)
    cols = ["t2m", "t2m_range", "rh", "td2m", "precip", "ws10", "u10", "v10", "ps", "ssrd", "cloud", "blh", "qv2m"]
    ov = ifs.merge(met, on=["lat", "lon", "date"], suffixes=("_ifs", ""))
    adj = []
    for (la, lo), g in ifs.groupby(["lat", "lon"]):
        o = ov[(ov.lat == la) & (ov.lon == lo)]
        gg = g[g.date > last].copy()
        for c in cols:
            x, y = o[f"{c}_ifs"].to_numpy(dtype=float), o[c].to_numpy(dtype=float)
            ok = np.isfinite(x) & np.isfinite(y)
            if ok.sum() >= 15 and np.std(x[ok]) > 1e-6 and c != "precip":
                a, b0 = np.polyfit(x[ok], y[ok], 1)
                gg[c] = a * gg[c] + b0
            elif ok.sum() >= 15 and c == "precip":
                gg[c] = gg[c] * (y[ok].sum() + 1) / (x[ok].sum() + 1)
        adj.append(gg)
    fill = pd.concat(adj, ignore_index=True)
    # MERRA-2 AOD not forecast: persistence of the last 3-day mean
    aod = met.dropna(subset=["merra_aod"]).sort_values("date").groupby(["lat", "lon"]).merra_aod \
             .apply(lambda s: s.tail(3).mean()).rename("merra_aod").reset_index()  # last 3 valid days
    fill = fill.merge(aod, on=["lat", "lon"], how="left")
    fill["source"] = "ifs_adjusted"
    return pd.concat([met.assign(source="merra2_power"), fill], ignore_index=True)


# ---------------------------------------------------------------------------
def update_obs(days: int = 12) -> pd.DataFrame:
    """Latest OpenAQ archive files (typically 1-3 day latency) -> QC -> daily."""
    locs = ground.openaq_locations()
    locs = locs[locs.country.isin(["GH", "CI", "TG", "BF", "NG", "ML"])]
    today = dt.date.today()
    since = (today - dt.timedelta(days=days)).isoformat()
    raw = ground.download_openaq(locs, start=since, workers=32)
    hq = ground.hourly_qc(raw)
    d = ground.daily(hq)
    d["date"] = pd.to_datetime(d["date"])
    return d


def live_openaq_latest(api_key: str | None = None) -> pd.DataFrame | None:
    """Optional real-time hourly readings via OpenAQ v3 (requires a free API key in OPENAQ_API_KEY)."""
    key = api_key or os.environ.get("OPENAQ_API_KEY")
    if not key:
        return None
    rows = []
    r = SESSION.get("https://api.openaq.org/v3/parameters/2/latest", params={"limit": 1000, "iso": "GH"},
                    headers={"X-API-Key": key}, timeout=60)
    if r.status_code != 200:
        return None
    for it in r.json().get("results", []):
        rows.append(dict(location_id=it.get("locationsId"), value=it.get("value"), time=it.get("datetime", {}).get("utc"),
                         lat=it.get("coordinates", {}).get("latitude"), lon=it.get("coordinates", {}).get("longitude")))
    return pd.DataFrame(rows)
