"""Ground PM2.5 observations (keyless).

1. OpenAQ public S3 archive (records/csv.gz/locationid=<ID>/year=YYYY/month=MM/...):
   Clarity (Breathe Accra, Ghana AQ, Clarity CI), AirGradient (UG, Afri-SET, EPA Ghana,
   LAMATA), Data354, AirQo (Nigeria), IQAir, Miri Africa, HabitatMap, Aurassure, and the
   US State Department BAM-1020 reference monitors (StateAir Accra/Abidjan/Bamako, AirNow Abuja/Lagos).
2. AirNow "EmbassyHistorical" files (files.airnowtech.org) for State Department reference
   monitors not present in OpenAQ (Lome, Ouagadougou) and to cross-check Accra/Abidjan.

Hourly QC -> daily means (>= 18 valid hours, i.e. 75 % completeness).
"""
from __future__ import annotations

import gzip
import io
import re
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd

from .. import config
from ..net import SESSION, get

S3 = "https://openaq-data-archive.s3.amazonaws.com"
AIRNOW_S3 = "https://s3-us-west-1.amazonaws.com/files.airnowtech.org"
CACHE = config.RAW / "ground"
META = config.RAW / "metadata" / "openaq_westafrica_locations.tsv"

# Regional training domain (Ghana + neighbours within ~800 km)
REGION = dict(lon_min=-9.0, lon_max=9.0, lat_min=4.0, lat_max=13.5)

# Networks excluded after screening: Aurassure Nigeria (daily means up to ~900 ug/m3, median site
# mean 245 ug/m3, physically implausible vs co-located networks); Miri Africa / AirQo NG / HabitatMap
# report too few hours per day to form daily means.
EXCLUDED = {("NG", "Aurassure")}

REFERENCE_PROVIDERS = {"StateAir Accra", "StateAir Abidjan", "StateAir Bamako", "AirNow", "US Embassy (AirNow)"}

EMBASSY = {  # AirNow EmbassyHistorical city -> (lat, lon, country)
    "Accra": (5.580642, -0.170724, "GH"),
    "Abidjan": (5.360000, -4.008300, "CI"),
    "Lome": (6.185167, 1.214166, "TG"),
    "Ouagadougou": (12.304907, -1.497184, "BF"),
    "Abuja": (9.041648, 7.4735, "NG"),
    "Lagos": (6.440483, 3.4213, "NG"),
    "Bamako": (12.632524, -8.036911, "ML"),
}


# ---------------------------------------------------------------------------
# registry
# ---------------------------------------------------------------------------
def openaq_locations(min_days: int = 20) -> pd.DataFrame:
    d = pd.read_csv(META, sep="\t")
    d["last"] = pd.to_datetime(d.s3_last_day.astype(str), errors="coerce")
    d = d[(d["last"] >= config.START_DATE) & (d.s3_n_days >= min_days)]
    m = d.lat.between(REGION["lat_min"], REGION["lat_max"]) & d.lon.between(REGION["lon_min"], REGION["lon_max"])
    d = d[m].copy()
    d = d[~d.apply(lambda r: (r.country, r.provider) in EXCLUDED, axis=1)]
    # explorer pages missing: legacy Clarity devices re-registered later (367215-367219 -> 947129...)
    legacy = d.location_id.isin([367215, 367216, 367217, 367218, 367219])
    d.loc[legacy, "provider"] = "Clarity (legacy 2022-23)"  # older hardware/firmware: calibrated separately
    unk = d.country.astype(str).str.startswith("unknown")
    in_gh = d.lat.between(4.5, 11.3) & d.lon.between(-3.3, 1.3)
    d.loc[unk & in_gh, "country"] = "GH"
    d["site_id"] = "oaq_" + d.location_id.astype(str)
    d["network"] = d.provider.fillna("Unknown")
    d["is_reference"] = d.provider.isin(REFERENCE_PROVIDERS)
    d["name"] = d.name_s3.str.replace(r"-\d+$", "", regex=True)
    return d[["site_id", "location_id", "name", "country", "network", "owner", "lat", "lon", "is_reference"]]


# ---------------------------------------------------------------------------
# OpenAQ archive
# ---------------------------------------------------------------------------
def _list_keys(prefix: str) -> list[str]:
    keys, token = [], None
    ns = {"s": "http://s3.amazonaws.com/doc/2006-03-01/"}
    while True:
        params = {"list-type": "2", "prefix": prefix, "max-keys": "1000"}
        if token:
            params["continuation-token"] = token
        r = get(S3 + "/", params=params, timeout=120)
        root = ET.fromstring(r.content)
        keys += [c.find("s:Key", ns).text for c in root.findall("s:Contents", ns)]
        if root.findtext("s:IsTruncated", namespaces=ns) != "true":
            return keys
        token = root.findtext("s:NextContinuationToken", namespaces=ns)


def _fetch_key(key: str) -> bytes | None:
    dest = CACHE / "openaq" / key.replace("records/csv.gz/", "")
    recent = _date_of(key) >= (pd.Timestamp.today() - pd.Timedelta(days=4)).strftime("%Y%m%d")
    if dest.exists() and not recent:  # files of the last few days may still be growing
        return dest.read_bytes()
    for _ in range(4):
        try:
            r = SESSION.get(f"{S3}/{key}", timeout=60)
            if r.status_code == 200:
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(r.content)
                return r.content
        except Exception:
            pass
    return None


def _date_of(key: str) -> str:
    m = re.search(r"-(\d{8})\.csv\.gz$", key)
    return m.group(1) if m else ""


def download_openaq(locs: pd.DataFrame, start: str = config.START_DATE, workers: int = 32) -> pd.DataFrame:
    s = start.replace("-", "")
    all_keys = []
    months = pd.period_range(pd.Timestamp(start), pd.Timestamp.today(), freq="M")
    if len(months) <= 3:  # incremental update: list only the relevant month prefixes
        prefixes = [f"records/csv.gz/locationid={lid}/year={p.year}/month={p.month:02d}/" for lid in locs.location_id for p in months]
    else:
        prefixes = [f"records/csv.gz/locationid={lid}/" for lid in locs.location_id]
    with ThreadPoolExecutor(16) as ex:
        futs = {ex.submit(_list_keys, pre): pre for pre in prefixes}
        for f in as_completed(futs):
            all_keys += [k for k in f.result() if _date_of(k) >= s]
    print(f"  OpenAQ: {len(all_keys)} daily files for {len(locs)} locations", flush=True)
    frames = []
    with ThreadPoolExecutor(workers) as ex:
        for i, content in enumerate(ex.map(_fetch_key, all_keys)):
            if content:
                try:
                    df = pd.read_csv(io.BytesIO(gzip.decompress(content)), dtype=str, on_bad_lines="skip")
                    frames.append(df[df["parameter"].isin(["pm25", "relativehumidity", "temperature", "pm10", "pm1"])])
                except Exception:
                    pass
            if i % 5000 == 0:
                print(f"    {i}/{len(all_keys)}", flush=True)
    raw = pd.concat(frames, ignore_index=True)
    raw["value"] = pd.to_numeric(raw["value"], errors="coerce")
    # datetime strings carry local offsets; Ghana == UTC, neighbours UTC+0/+1. Use UTC throughout.
    raw["time"] = pd.to_datetime(raw["datetime"], utc=True, errors="coerce").dt.tz_convert(None)
    raw["site_id"] = "oaq_" + raw["location_id"].astype(str)
    return raw[["site_id", "sensors_id", "time", "parameter", "value"]].dropna(subset=["time"])


# ---------------------------------------------------------------------------
# AirNow embassy historical
# ---------------------------------------------------------------------------
def download_embassy(cities=("Lome", "Ouagadougou", "Accra", "Abidjan"), start_year: int = 2022) -> pd.DataFrame:
    ns = {"s": "http://s3.amazonaws.com/doc/2006-03-01/"}
    frames = []
    for city in cities:
        r = get(AIRNOW_S3 + "/", params={"list-type": "2", "prefix": f"airnow/EmbassyHistorical/{city}/"})
        keys = [c.find("s:Key", ns).text for c in ET.fromstring(r.content).findall("s:Contents", ns)]
        keys = [k for k in keys if "PM2.5" in k and k.endswith(".csv")]
        by_year: dict[str, list[str]] = {}
        for k in keys:
            y = re.search(r"_(\d{4})_", k)
            if y and int(y.group(1)) >= start_year:
                by_year.setdefault(y.group(1), []).append(k)
        for y, ks in by_year.items():
            ytd = [k for k in ks if "YTD" in k]
            use = ytd if ytd else ks
            for k in use:
                dest = CACHE / "airnow" / k.split("EmbassyHistorical/")[1]
                if not dest.exists() or int(y) >= pd.Timestamp.today().year - 1:
                    rr = get(f"{AIRNOW_S3}/{k}", timeout=120)
                    if rr.status_code != 200:
                        continue
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    dest.write_bytes(rr.content)
                df = pd.read_csv(dest)
                df.columns = [c.strip() for c in df.columns]
                df = df[df["QC Name"].astype(str).str.strip().str.lower() == "valid"]
                t = pd.to_datetime(df[["Year", "Month", "Day", "Hour"]].astype(int).rename(columns=str.lower))
                frames.append(pd.DataFrame({"site_id": f"emb_{city.lower()}", "sensors_id": "bam", "time": t,
                                            "parameter": "pm25", "value": pd.to_numeric(df["Raw Conc."], errors="coerce")}))
    out = pd.concat(frames, ignore_index=True).drop_duplicates(["site_id", "time"])
    # AirNow "Date (LT)" is local time; Lome/Accra/Abidjan/Ouaga are UTC+0 -> no shift needed.
    return out


def embassy_registry() -> pd.DataFrame:
    rows = [dict(site_id=f"emb_{c.lower()}", location_id=-1, name=f"US Embassy {c} (BAM-1020)", country=v[2],
                 network="US Embassy (AirNow)", owner="US Department of State", lat=v[0], lon=v[1], is_reference=True)
            for c, v in EMBASSY.items() if c in ("Lome", "Ouagadougou", "Accra", "Abidjan")]
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# QC + daily aggregation
# ---------------------------------------------------------------------------
def hourly_qc(raw: pd.DataFrame) -> pd.DataFrame:
    """Hourly PM2.5 (+RH/T) per site with automated quality control.

    * physical range 0 < PM2.5 <= 1000 ug/m3
    * sub-hourly readings averaged to hourly
    * stuck-sensor flag: >= 6 identical consecutive hourly values
    * Hampel filter: |x - rolling median(25 h)| > max(6 * MAD, 30 ug/m3) removed as spike
    """
    pm = raw[raw.parameter == "pm25"].copy()
    pm = pm[(pm.value > 0) & (pm.value <= 1000)]
    pm["hour"] = pm.time.dt.floor("h")
    h = pm.groupby(["site_id", "hour"], as_index=False)["value"].mean()
    out = []
    for sid, g in h.groupby("site_id"):
        g = g.set_index("hour").sort_index()
        s = g["value"]
        same = s.diff().eq(0)
        run = same.groupby((~same).cumsum()).cumsum()
        stuck = run >= 5
        med = s.rolling("25h", center=True, min_periods=6).median()
        mad = (s - med).abs().rolling("25h", center=True, min_periods=6).median()
        spike = (s - med).abs() > np.maximum(6 * 1.4826 * mad, 30)
        keep = ~(stuck | spike)
        gg = g.loc[keep].reset_index()
        gg["site_id"] = sid
        out.append(gg)
    hq = pd.concat(out, ignore_index=True).rename(columns={"value": "pm25"})
    # co-reported humidity / temperature (AirGradient, some Clarity)
    for p, col in (("relativehumidity", "sensor_rh"), ("temperature", "sensor_t")):
        x = raw[raw.parameter == p].copy()
        if len(x):
            x["hour"] = x.time.dt.floor("h")
            x = x.groupby(["site_id", "hour"], as_index=False)["value"].mean().rename(columns={"value": col})
            hq = hq.merge(x, on=["site_id", "hour"], how="left")
    return hq


def daily(hq: pd.DataFrame, min_hours: int = 12, min_per_block: int = 2) -> pd.DataFrame:
    """Daily means requiring >= `min_hours` valid hours and >= `min_per_block` hours in each
    6-h block (00-06, 06-12, 12-18, 18-24 UTC) so that the diurnal cycle is represented."""
    hq = hq.copy()
    hq["date"] = hq.hour.dt.floor("D")
    hq["block"] = hq.hour.dt.hour // 6
    agg = {"pm25": ["mean", "count", "std"]}
    for c in ("sensor_rh", "sensor_t"):
        if c in hq:
            agg[c] = ["mean"]
    d = hq.groupby(["site_id", "date"]).agg(agg)
    d.columns = ["pm25", "n_hours", "pm25_hourly_sd"] + [c for c in ("sensor_rh", "sensor_t") if c in hq]
    blocks = hq.groupby(["site_id", "date", "block"]).size().unstack(fill_value=0).reindex(columns=range(4), fill_value=0)
    d["min_block_hours"] = blocks.min(axis=1)
    d = d.reset_index()
    return d[(d.n_hours >= min_hours) & (d.min_block_hours >= min_per_block)]
