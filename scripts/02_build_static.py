"""Build static covariate rasters (Ghana grid + boxes around out-of-country stations)."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ghana_pm25 import config  # noqa: E402
from ghana_pm25.sources import static  # noqa: E402


def main():
    only = sys.argv[1] if len(sys.argv) > 1 else None
    stations = pd.read_parquet(config.PROCESSED / "stations.parquet")
    boxes = [("ghana", config.LON_MIN, config.LON_MAX, config.LAT_MIN, config.LAT_MAX)] + static.station_boxes(stations)
    print(f"{len(boxes)} boxes")
    for b in boxes:
        if only and b[0] != only:
            continue
        print("box", b, flush=True)
        static.build_box(*b)
    if not only:
        s = static.sample(stations)
        s.to_parquet(config.PROCESSED / "station_static.parquet")
        print("station static", s.shape, "missing stations:", set(stations.site_id) - set(s.site_id))


if __name__ == "__main__":
    main()
