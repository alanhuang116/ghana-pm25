"""TAHMO automatic weather stations (Trans-African HydroMeteorological Observatory).

Access needs an account. Credentials are read from the environment, never from code:

    TAHMO_USERNAME, TAHMO_PASSWORD      (GitHub Actions: repository secrets of the same names)

Locally they can also be put in a git-ignored `.env` file at the repository root (KEY=value lines).

APIs (base https://datahub.tahmo.org, HTTP Basic auth)
  services/assets/v2/stations                                   station metadata
  services/assets/v2/variables                                  variable shortcodes and units
  services/measurements/v2/stations/{code}/measurements/{set}   5-min series, set = raw | controlled
                                                                 <= 365 days per request, UTC timestamps
Each value carries a quality flag (1 good, 2 suspect, 3 doubtful, 4 erroneous); only flag 1 is kept.

Daily means use the same 00-24 UTC day as GH-PM25 (= Ghana local time) and require 75 % of the
expected 5-min samples (as the ground PM2.5 rules require 18 of 24 hours).
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import requests

from .. import config
from ..net import get

BASE = "https://datahub.tahmo.org"
CACHE = config.RAW / "tahmo"
DAILY = config.INTERIM / "tahmo_daily.parquet"
STATIONS = config.INTERIM / "tahmo_stations.parquet"

# Greater Accra Region with a margin (stations here are used for the Accra experiments and portal view)
ACCRA_BBOX = dict(lon_min=-0.60, lon_max=0.75, lat_min=5.40, lat_max=6.15)

# TAHMO shortcode -> GH-PM25 meteorology column, daily aggregation
VARIABLES = {
    "te": ("t2m", "mean"),        # air temperature, degC
    "rh": ("rh", "mean"),         # relative humidity (TAHMO reports a 0-1 fraction; converted to %)
    "pr": ("precip", "sum"),      # precipitation, mm
    "ws": ("ws10", "mean"),       # wind speed, m/s (TAHMO sensors sit at ~2 m, not 10 m)
    "wd": ("wd", "circmean"),     # wind direction, degrees
    "wg": ("wgust", "max"),       # wind gust, m/s
    "ap": ("ps", "mean"),         # atmospheric pressure, kPa
    "ra": ("ssrd", "mean"),       # shortwave radiation, W/m2 -> MJ/m2/day
    "vp": ("vp", "mean"),         # vapour pressure, kPa
}
SAMPLES_PER_DAY = 288  # 5-min
MIN_COMPLETENESS = 0.75


class MissingCredentials(RuntimeError):
    pass


def _read_dotenv() -> dict:
    f = config.ROOT / ".env"
    out = {}
    if f.exists():
        for line in f.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                out[k.strip()] = v.strip().strip('"').strip("'")
    return out


def credentials() -> tuple[str, str]:
    env = {**_read_dotenv(), **os.environ}
    user, pw = env.get("TAHMO_USERNAME"), env.get("TAHMO_PASSWORD")
    if not user or not pw:
        raise MissingCredentials("set TAHMO_USERNAME and TAHMO_PASSWORD (environment, GitHub secrets or .env)")
    return user, pw


def available() -> bool:
    try:
        credentials()
        return True
    except MissingCredentials:
        return False


def _request(endpoint: str, params: dict | None = None) -> dict:
    r = get(f"{BASE}/{endpoint}", params=params or {}, auth=requests.auth.HTTPBasicAuth(*credentials()), timeout=300, retries=4)
    if r.status_code == 401:
        raise RuntimeError("TAHMO rejected the credentials (HTTP 401)")
    if r.status_code == 403:
        raise RuntimeError(f"TAHMO: no access to {endpoint} (HTTP 403)")
    if r.status_code != 200:
        raise RuntimeError(f"TAHMO {endpoint} failed {r.status_code}: {r.text[:200]}")
    return r.json()


# ---------------------------------------------------------------------------
# metadata
# ---------------------------------------------------------------------------
def parse_stations(js: dict) -> pd.DataFrame:
    rows = []
    for e in js.get("data", []):
        loc = e.get("location") or {}
        rows.append(dict(code=e.get("code"), name=loc.get("name") or e.get("name") or e.get("code"),
                         lat=loc.get("latitude"), lon=loc.get("longitude"), elev=loc.get("elevationmsl"),
                         country=loc.get("countrycode"), active=e.get("active", True),
                         installed=e.get("installationdate")))
    df = pd.DataFrame(rows, columns=["code", "name", "lat", "lon", "elev", "country", "active", "installed"])
    df[["lat", "lon", "elev"]] = df[["lat", "lon", "elev"]].apply(pd.to_numeric, errors="coerce")
    return df.dropna(subset=["lat", "lon"]).reset_index(drop=True)


def stations() -> pd.DataFrame:
    df = parse_stations(_request("services/assets/v2/stations", {"sort": "code"}))
    df.to_parquet(STATIONS)
    return df


def variables() -> pd.DataFrame:
    js = _request("services/assets/v2/variables")
    rows = [dict(shortcode=e["variable"].get("shortcode"), description=e["variable"].get("description"),
                 units=e["variable"].get("units")) for e in js.get("data", []) if "variable" in e]
    return pd.DataFrame(rows)


def in_bbox(df: pd.DataFrame, b: dict = ACCRA_BBOX) -> pd.DataFrame:
    return df[df.lat.between(b["lat_min"], b["lat_max"]) & df.lon.between(b["lon_min"], b["lon_max"])]


# ---------------------------------------------------------------------------
# measurements
# ---------------------------------------------------------------------------
def _empty_measurements() -> pd.DataFrame:
    return pd.DataFrame({"time": pd.Series(dtype="datetime64[ns]"), "variable": pd.Series(dtype=object),
                         "sensor": pd.Series(dtype=object), "value": pd.Series(dtype=float)})


def parse_measurements(js: dict) -> pd.DataFrame:
    """Long table time, variable, sensor, value (flag-1 values only; others set to NaN)."""
    frames = []
    for res in js.get("results", []):
        for s in res.get("series", []) or []:
            if not s.get("values"):
                continue
            d = pd.DataFrame(s["values"], columns=s["columns"])
            frames.append(d[[c for c in ("time", "variable", "sensor", "value", "quality") if c in d]])
    if not frames:
        return _empty_measurements()
    d = pd.concat(frames, ignore_index=True)
    d["time"] = pd.to_datetime(d["time"], utc=True).dt.tz_localize(None)
    d["value"] = pd.to_numeric(d["value"], errors="coerce")
    if "quality" in d:
        d.loc[d.quality != 1, "value"] = np.nan
        d = d.drop(columns="quality")
    return d


def measurements(code: str, start: str, end: str, dataset: str = "controlled", refresh: bool = False) -> pd.DataFrame:
    """5-min measurements, cached per station and calendar month."""
    CACHE.mkdir(parents=True, exist_ok=True)
    frames = []
    this_month = pd.Timestamp.now("UTC").tz_localize(None).to_period("M")
    for p in pd.period_range(pd.Timestamp(start), pd.Timestamp(end), freq="M"):
        f = CACHE / f"{code}_{dataset}_{p}.parquet"
        if f.exists() and not refresh and p < this_month - 1:
            frames.append(pd.read_parquet(f))
            continue
        s, e = p.start_time, p.end_time.floor("s")
        js = _request(f"services/measurements/v2/stations/{code}/measurements/{dataset}",
                      {"start": f"{s:%Y-%m-%dT%H:%M:%SZ}", "end": f"{e:%Y-%m-%dT%H:%M:%SZ}"})
        d = parse_measurements(js)
        d.to_parquet(f)
        frames.append(d)
    frames = [f for f in frames if len(f)]  # months without data would turn the time column into object dtype
    d = pd.concat(frames, ignore_index=True) if frames else _empty_measurements()
    d["time"] = pd.to_datetime(d["time"])
    return d[(d.time >= pd.Timestamp(start)) & (d.time < pd.Timestamp(end) + pd.Timedelta(days=1))]


def _circmean_deg(x: pd.Series) -> float:
    x = x.dropna()
    if x.empty:
        return np.nan
    r = np.deg2rad(x.to_numpy())
    return float(np.rad2deg(np.arctan2(np.sin(r).mean(), np.cos(r).mean())) % 360)


def daily(m: pd.DataFrame) -> pd.DataFrame:
    """5-min long table -> one row per day with GH-PM25 column names. Multi-sensor variables are averaged."""
    if m.empty:
        return pd.DataFrame(columns=["date"])
    m = m[m.variable.isin(VARIABLES)].copy()
    m["date"] = m.time.dt.floor("D")
    # one value per timestamp and variable (average duplicate sensors)
    agg_ts = m.groupby(["variable", "date", "time"]).value.mean().reset_index()
    out = []
    for var, g in agg_ts.groupby("variable"):
        col, how = VARIABLES[var]
        n = g.groupby("date").value.count()
        if how == "circmean":
            v = g.groupby("date").value.apply(_circmean_deg)
        elif how == "sum":
            # rainfall: scale up to a full day only when complete enough
            v = g.groupby("date").value.sum() * SAMPLES_PER_DAY / n.clip(lower=1)
        else:
            v = g.groupby("date").value.agg(how)
        v[n < MIN_COMPLETENESS * SAMPLES_PER_DAY] = np.nan
        out.append(v.rename(col))
        if var == "te":
            mx, mn = g.groupby("date").value.max(), g.groupby("date").value.min()
            r = (mx - mn).rename("t2m_range")
            r[n < MIN_COMPLETENESS * SAMPLES_PER_DAY] = np.nan
            out.append(r)
    d = pd.concat(out, axis=1).reset_index()
    if "rh" in d and d.rh.median() <= 1.5:   # fraction -> percent
        d["rh"] = d.rh * 100.0
    if "ssrd" in d:                          # W/m2 daily mean -> MJ/m2/day (as NASA POWER)
        d["ssrd"] = d.ssrd * 0.0864
    if {"ws10", "wd"} <= set(d.columns):
        rad = np.deg2rad(d.wd)
        d["u10"], d["v10"] = -d.ws10 * np.sin(rad), -d.ws10 * np.cos(rad)
    return d


def update(start: str, end: str | None = None, bbox: dict | None = ACCRA_BBOX, log=print) -> pd.DataFrame:
    """Refresh station list and daily series for all stations in `bbox` (None = every station the account can read)."""
    end = end or pd.Timestamp.now("UTC").strftime("%Y-%m-%d")
    st = stations()
    sel = in_bbox(st, bbox) if bbox else st
    log(f"TAHMO: account sees {len(st)} stations, {len(sel)} in the selected area")
    hist = pd.read_parquet(DAILY) if DAILY.exists() else pd.DataFrame(columns=["code", "date"])
    frames = []
    for code in sel.code:
        try:
            d = daily(measurements(code, start, end))
        except RuntimeError as e:  # e.g. 403 on a station outside the grant
            log(f"  {code}: {e}")
            continue
        if not d.empty:
            frames.append(d.assign(code=code))
            log(f"  {code}: {d.date.min():%Y-%m-%d}..{d.date.max():%Y-%m-%d}, {len(d)} days")
    new = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["code", "date"])
    out = pd.concat([hist, new], ignore_index=True).drop_duplicates(["code", "date"], keep="last")
    out = out.sort_values(["code", "date"]).reset_index(drop=True)
    out.to_parquet(DAILY)
    return out
