"""Assemble cams_daily.parquet / cams_nodes.parquet from whatever CAMS chunks are already cached
(lets training proceed while older chunks wait for the Open-Meteo daily quota)."""
from __future__ import annotations

import glob
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ghana_pm25 import config  # noqa: E402
from ghana_pm25.sources import ground, openmeteo  # noqa: E402

files = glob.glob(str(config.RAW / "openmeteo" / "cams" / "hist" / "*.parquet"))
cams = pd.concat([pd.read_parquet(f) for f in files]).drop_duplicates(["node", "date"])
stations = pd.concat([ground.openaq_locations(), ground.embassy_registry()])
nodes = pd.concat([openmeteo.lattice_nodes(), openmeteo.nodes_around(stations)]).drop_duplicates("node")
# keep only dates where every node is present
cnt = cams.groupby("date").node.nunique()
full_dates = cnt[cnt >= len(nodes)].index
cams = cams[cams.date.isin(full_dates)]
cams.to_parquet(config.INTERIM / "cams_daily.parquet")
nodes.to_parquet(config.INTERIM / "cams_nodes.parquet")
print(f"CAMS {cams.date.min().date()} -> {cams.date.max().date()}, {cams.date.nunique()} complete days, {len(nodes)} nodes")
