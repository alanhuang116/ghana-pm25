"""Analysis grid helpers: Ghana mask, administrative index rasters, boundaries."""
from __future__ import annotations

import json
from functools import lru_cache

import numpy as np
import rasterio.features
import xarray as xr
from affine import Affine

from . import config
from .net import download

GB = "https://media.githubusercontent.com/media/wmgeolab/geoBoundaries/9469f09592ced973a3448cf66b6100b741b64c0d/releaseData/gbOpen/GHA"
BOUNDARY_DIR = config.RAW / "boundaries"


def boundaries(level: int, simplified: bool = True) -> dict:
    name = f"geoBoundaries-GHA-ADM{level}{'_simplified' if simplified else ''}.geojson"
    dest = download(f"{GB}/ADM{level}/{name}", BOUNDARY_DIR / name)
    return json.loads(dest.read_text(encoding="utf-8"))


def transform() -> Affine:
    lons, lats = config.grid_axes()
    return Affine(config.RES, 0, lons[0] - config.RES / 2, 0, -config.RES, lats[0] + config.RES / 2)


@lru_cache(1)
def admin_rasters() -> dict:
    """mask (bool), adm1 index (int16, -1 outside), adm2 index, names."""
    f = config.PROCESSED / "grid_admin.nc"
    lons, lats = config.grid_axes()
    shape = (len(lats), len(lons))
    if not f.exists():
        out = {}
        for lvl in (0, 1, 2):
            gj = boundaries(lvl)
            shapes = [(feat["geometry"], i) for i, feat in enumerate(gj["features"])]
            r = rasterio.features.rasterize(shapes, out_shape=shape, transform=transform(), fill=-1, dtype="int16", all_touched=False)
            out[f"adm{lvl}"] = (("lat", "lon"), r)
        ds = xr.Dataset(out, coords={"lat": lats, "lon": lons})
        ds.attrs["adm1_names"] = json.dumps([ft["properties"]["shapeName"] for ft in boundaries(1)["features"]])
        ds.attrs["adm2_names"] = json.dumps([ft["properties"]["shapeName"] for ft in boundaries(2)["features"]])
        ds.to_netcdf(f)
    ds = xr.open_dataset(f)
    return dict(mask=ds.adm0.values >= 0, adm1=ds.adm1.values, adm2=ds.adm2.values,
                adm1_names=json.loads(ds.attrs["adm1_names"]), adm2_names=json.loads(ds.attrs["adm2_names"]),
                lons=lons, lats=lats)
