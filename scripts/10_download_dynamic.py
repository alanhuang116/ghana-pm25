"""Download daily dynamic predictors for the whole study period.

* CAMS composition (Open-Meteo) on the 0.8 deg lattice over Ghana + lattice cells enclosing
  every training station in neighbouring countries.
* NASA POWER MERRA-2/GEOS meteorology + AOD over the regional training domain.
* NASA FIRMS VIIRS fire detections.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ghana_pm25 import config  # noqa: E402
from ghana_pm25.sources import firms, ground, openmeteo, power  # noqa: E402


def cams_nodes() -> pd.DataFrame:
    stations = pd.concat([ground.openaq_locations(), ground.embassy_registry()])
    nodes = pd.concat([openmeteo.lattice_nodes(), openmeteo.nodes_around(stations)]).drop_duplicates("node")
    return nodes.reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default=config.START_DATE)
    ap.add_argument("--end", default=(pd.Timestamp.today().normalize() - pd.Timedelta(days=2)).strftime("%Y-%m-%d"))
    ap.add_argument("--what", default="power,firms,cams")
    args = ap.parse_args()
    what = args.what.split(",")

    if "power" in what:
        met = power.fetch(ground.REGION, args.start, args.end)
        met.to_parquet(config.INTERIM / "power_daily.parquet")
        print("POWER", met.shape, met.date.min(), met.date.max())
    if "firms" in what:
        f = firms.load_history(args.start, args.end)
        f.to_parquet(config.INTERIM / "firms_hist.parquet")
        print("FIRMS", f.shape, f.date.min(), f.date.max())
    if "cams" in what:
        nodes = cams_nodes()
        print(f"CAMS nodes: {len(nodes)}")
        cams = openmeteo.fetch_history(nodes, args.start, args.end)
        cams.to_parquet(config.INTERIM / "cams_daily.parquet")
        nodes.to_parquet(config.INTERIM / "cams_nodes.parquet")
        print("CAMS", cams.shape)


if __name__ == "__main__":
    main()
