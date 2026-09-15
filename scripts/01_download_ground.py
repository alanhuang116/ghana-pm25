"""Download and QC ground PM2.5 observations; writes stations.parquet + obs_daily.parquet."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ghana_pm25 import config  # noqa: E402
from ghana_pm25.sources import ground  # noqa: E402


def main():
    locs = ground.openaq_locations()
    emb = ground.embassy_registry()
    # AirNow embassy files duplicate OpenAQ StateAir Accra/Abidjan; keep AirNow only for Lome/Ouagadougou,
    # and use AirNow Accra/Abidjan only after the StateAir feed ends in OpenAQ.
    stations = pd.concat([locs, emb], ignore_index=True)
    print(stations.groupby(["country", "network"]).size().to_string())

    if "--skip-download" in sys.argv:
        raw = pd.read_parquet(config.INTERIM / "openaq_raw.parquet")
        emb_raw = pd.read_parquet(config.INTERIM / "airnow_embassy_raw.parquet")
    else:
        raw = ground.download_openaq(locs)
        raw.to_parquet(config.INTERIM / "openaq_raw.parquet")
        emb_raw = ground.download_embassy()
        emb_raw.to_parquet(config.INTERIM / "airnow_embassy_raw.parquet")
    raw = raw[raw.site_id.isin(stations.site_id)]

    hq = ground.hourly_qc(pd.concat([raw, emb_raw], ignore_index=True))
    hq.to_parquet(config.INTERIM / "obs_hourly_qc.parquet")
    d = ground.daily(hq)

    # Accra/Abidjan: OpenAQ StateAir and AirNow files are the same instrument -> merge by date
    for city, oaq in (("accra", "oaq_9764"), ("abidjan", "oaq_9762")):
        e = d[d.site_id == f"emb_{city}"]
        o = d[d.site_id == oaq]
        extra = e[~e.date.isin(o.date)].copy()
        extra["site_id"] = oaq
        d = pd.concat([d[d.site_id != f"emb_{city}"], extra], ignore_index=True)
    stations = stations[~stations.site_id.isin(["emb_accra", "emb_abidjan"])]

    d = d[d.date >= config.START_DATE]
    counts = d.groupby("site_id").size().rename("n_days")
    stations = stations.merge(counts, left_on="site_id", right_index=True, how="inner")
    stations.to_parquet(config.PROCESSED / "stations.parquet")
    d.to_parquet(config.PROCESSED / "obs_daily.parquet")
    print(f"stations with data: {len(stations)}; station-days: {len(d)}")
    print(stations.groupby(["country", "network"]).n_days.agg(["size", "sum"]).to_string())


if __name__ == "__main__":
    main()
