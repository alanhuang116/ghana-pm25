"""Central configuration: paths, spatial domain, analysis grid, and time span."""
from __future__ import annotations

from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RAW = DATA / "raw"
INTERIM = DATA / "interim"
PROCESSED = DATA / "processed"
MODELS = ROOT / "models"
OUTPUTS = ROOT / "outputs"
PORTAL = ROOT / "portal"
PORTAL_DATA = PORTAL / "data"
REPORT = ROOT / "report"

for _p in (RAW, INTERIM, PROCESSED, MODELS, OUTPUTS, PORTAL_DATA, REPORT):
    _p.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Spatial domain
# ---------------------------------------------------------------------------
# Ghana extent is roughly lon -3.26..1.20, lat 4.73..11.17. The analysis grid is
# a regular WGS84 lattice of 0.01 deg (~1.1 km) cell centres.
RES = 0.01
LON_MIN, LON_MAX = -3.30, 1.25
LAT_MIN, LAT_MAX = 4.70, 11.20

# Wider "training region" so stations in neighbouring countries can inform the
# model about Sahelian dust / Gulf-of-Guinea gradients that Ghana alone samples poorly.
TRAIN_BBOX = dict(lon_min=-8.6, lon_max=4.0, lat_min=4.2, lat_max=15.2)

# Coarse lattice used to pull gridded reanalysis/forecast fields (CAMS ~0.4 deg,
# ERA5 0.25 deg). 0.25 deg spacing over the training region.
COARSE_RES = 0.25


def grid_axes() -> tuple[np.ndarray, np.ndarray]:
    """Return (lons, lats) cell-centre vectors of the 0.01 deg analysis grid.

    lats run north -> south so arrays are in image (row, col) order.
    """
    lons = np.round(np.arange(LON_MIN + RES / 2, LON_MAX, RES), 5)
    lats = np.round(np.arange(LAT_MAX - RES / 2, LAT_MIN, -RES), 5)
    return lons, lats


def coarse_axes(bbox: dict | None = None, res: float = COARSE_RES) -> tuple[np.ndarray, np.ndarray]:
    b = bbox or dict(lon_min=LON_MIN, lon_max=LON_MAX, lat_min=LAT_MIN, lat_max=LAT_MAX)
    lons = np.round(np.arange(np.floor(b["lon_min"] / res) * res, b["lon_max"] + res, res), 4)
    lats = np.round(np.arange(np.floor(b["lat_min"] / res) * res, b["lat_max"] + res, res), 4)
    return lons, lats


# ---------------------------------------------------------------------------
# Time span
# ---------------------------------------------------------------------------
# CAMS global atmospheric-composition fields via Open-Meteo start 2022-08-04.
START_DATE = "2022-08-04"

# Air-quality reference values (ug/m3)
WHO_24H = 15.0
WHO_ANNUAL = 5.0
