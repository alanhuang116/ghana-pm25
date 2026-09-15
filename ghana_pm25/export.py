"""Export model products to the static web portal (portal/data) and to GIS formats.

Grid encoding (lossless PNG, decoded in the browser via canvas):
  main PNG : R*256+G = round(PM2.5 * 10)  (0.1 ug/m3 precision, max 6553.5)
             B = 0 outside Ghana, 1 outside area-of-applicability, 2 inside AoA
  aux PNG  : R = round(lower90 / mean * 250)
             G = round(upper90 / mean * 40)      (ratio up to 6.375)
             B = min(255, round(CAMS PM2.5))
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from PIL import Image

from . import config
from .grid import boundaries

P = config.PORTAL_DATA


def _png(arr_rgb: np.ndarray, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr_rgb, mode="RGB").save(path, optimize=True)


def encode_day(ds: xr.Dataset) -> tuple[np.ndarray, np.ndarray]:
    pm = ds.pm25.values
    valid = np.isfinite(pm)
    v = np.where(valid, np.clip(np.round(pm * 10), 0, 65535), 0).astype(np.uint16)
    code = np.where(~valid, 0, np.where(ds.inside_aoa.values == 1, 2, 1)).astype(np.uint8)
    main = np.dstack([(v >> 8).astype(np.uint8), (v & 255).astype(np.uint8), code])
    safe = np.where(valid, pm, 1.0)
    lo = np.where(valid, np.clip(np.round(ds.pm25_lower90.values / safe * 250), 0, 255), 0).astype(np.uint8)
    hi = np.where(valid, np.clip(np.round(ds.pm25_upper90.values / safe * 40), 0, 255), 0).astype(np.uint8)
    cams = np.where(valid, np.clip(np.round(np.nan_to_num(ds.cams_pm25.values)), 0, 255), 0).astype(np.uint8)
    aux = np.dstack([lo, hi, cams])
    return main, aux


def write_day(ds: xr.Dataset):
    date = pd.Timestamp(ds.time.values)
    main, aux = encode_day(ds)
    _png(main, P / "grids" / f"{date:%Y}" / f"{date:%Y%m%d}_pm25.png")
    _png(aux, P / "grids" / f"{date:%Y}" / f"{date:%Y%m%d}_aux.png")


def write_netcdf(ds: xr.Dataset):
    date = pd.Timestamp(ds.time.values)
    f = config.OUTPUTS / "daily" / f"{date:%Y}" / f"GH-PM25_1km_{date:%Y%m%d}.nc"
    f.parent.mkdir(parents=True, exist_ok=True)
    enc = {v: dict(zlib=True, complevel=4) for v in ds.data_vars}
    for v in ("pm25", "pm25_lower90", "pm25_upper90", "pm25_stage1", "cams_pm25"):
        enc[v].update(dtype="int16", scale_factor=0.1, _FillValue=-32768)
    ds.to_netcdf(f, encoding=enc)
    return f


def coarse(ds: xr.Dataset, factor: int = 5) -> np.ndarray:
    a = ds.pm25.values
    ny, nx = a.shape[0] // factor, a.shape[1] // factor
    with np.errstate(all="ignore"):
        return np.nanmean(a[: ny * factor, : nx * factor].reshape(ny, factor, nx, factor), axis=(1, 3))


def write_cube(year: int, dates: list[pd.Timestamp], arrays: list[np.ndarray]):
    """0.05 deg cube for click time series: header json + little-endian uint16 (pm*10, 65535 = nodata)."""
    stack = np.stack(arrays)
    data = np.where(np.isfinite(stack), np.clip(np.round(stack * 10), 0, 65534), 65535).astype("<u2")
    (P / "cube").mkdir(parents=True, exist_ok=True)
    data.tofile(P / "cube" / f"pm25_005_{year}.bin")
    lons, lats = config.grid_axes()
    hdr = dict(year=year, dates=[d.strftime("%Y-%m-%d") for d in dates], ny=int(stack.shape[1]), nx=int(stack.shape[2]),
               lon0=float(lons[0] - config.RES / 2), lat0=float(lats[0] + config.RES / 2), res=0.05, scale=0.1, nodata=65535)
    (P / "cube" / f"pm25_005_{year}.json").write_text(json.dumps(hdr))


def update_cube(year: int, updates: dict):
    """Insert/replace days in an existing yearly cube."""
    jf, bf = P / "cube" / f"pm25_005_{year}.json", P / "cube" / f"pm25_005_{year}.bin"
    days = {}
    if jf.exists() and bf.exists():
        hdr = json.loads(jf.read_text())
        data = np.fromfile(bf, dtype="<u2").reshape(len(hdr["dates"]), hdr["ny"], hdr["nx"])
        for d, arr in zip(hdr["dates"], data):
            days[pd.Timestamp(d)] = np.where(arr == 65535, np.nan, arr * 0.1)
    days.update({pd.Timestamp(k): v for k, v in updates.items()})
    ds = sorted(days)
    write_cube(year, ds, [days[d] for d in ds])


def write_boundaries(tolerance: float = 0.004):
    import shapely
    from shapely.geometry import mapping, shape
    for lvl in (0, 1, 2):
        gj = boundaries(lvl)
        feats = []
        for i, ft in enumerate(gj["features"]):
            g = shapely.simplify(shape(ft["geometry"]), tolerance if lvl < 2 else tolerance * 1.5, preserve_topology=True)
            g = shapely.set_precision(g, 1e-4)
            feats.append(dict(type="Feature", id=i, properties=dict(id=i, name=ft["properties"]["shapeName"]), geometry=mapping(g)))
        (P / "boundaries").mkdir(parents=True, exist_ok=True)
        (P / "boundaries" / f"adm{lvl}.geojson").write_text(json.dumps(dict(type="FeatureCollection", features=feats), separators=(",", ":")))


def write_region_series(stats: pd.DataFrame):
    """stats: rows (date, level, id, name, pop_weighted, lower90, upper90, pop_frac_gt15, pop_frac_gt35, area_mean, max)."""
    stats = stats.sort_values("date")
    dates = sorted(stats.date.unique())
    out = {"dates": [pd.Timestamp(d).strftime("%Y-%m-%d") for d in dates]}
    for level in ("adm0", "adm1", "adm2"):
        s = stats[stats.level == level]
        names = s.drop_duplicates("id").sort_values("id")[["id", "name"]]
        pop = s.groupby("id").population.first().reindex(names.id)
        block = {"names": names.name.tolist(), "population": [round(float(p)) for p in pop.fillna(0)]}
        for col, nd in (("pop_weighted", 1), ("lower90", 1), ("upper90", 1), ("pop_frac_gt15", 3), ("pop_frac_gt35", 3), ("max", 1)):
            piv = s.pivot_table(index="date", columns="id", values=col).reindex(dates)
            block[col] = [[None if not np.isfinite(x) else round(float(x), nd) for x in piv[c].to_numpy()] for c in names.id]
        out[level] = block
    (P / "series").mkdir(parents=True, exist_ok=True)
    (P / "series" / "regions.json").write_text(json.dumps(out, separators=(",", ":")))


def write_station_data(stations: pd.DataFrame, obs: pd.DataFrame, oof: pd.DataFrame | None):
    st = stations.copy()
    st = st[["site_id", "name", "country", "network", "lat", "lon", "is_reference", "n_days"]]
    o = obs[["site_id", "date", "pm25", "pm25_raw"]].copy()
    if oof is not None:
        o = o.merge(oof[["site_id", "date", "cv_pred", "stage1_pred"]], on=["site_id", "date"], how="left")
    o["date"] = pd.to_datetime(o.date).dt.strftime("%Y-%m-%d")
    series = {}
    for sid, g in o.groupby("site_id"):
        series[sid] = {c: [None if (isinstance(x, float) and not np.isfinite(x)) else (round(x, 1) if isinstance(x, float) else x)
                           for x in g[c].tolist()] for c in g.columns if c != "site_id"}
    (P / "stations").mkdir(parents=True, exist_ok=True)
    (P / "stations" / "stations.json").write_text(json.dumps(json.loads(st.to_json(orient="records")), separators=(",", ":")))
    (P / "stations" / "series.json").write_text(json.dumps(series, separators=(",", ":")))


def write_json(name: str, obj):
    f = P / name
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps(obj, separators=(",", ":"), default=str))


def write_geotiff(ds: xr.Dataset, var: str, path: Path):
    import rasterio
    from .grid import transform
    a = ds[var].values.astype("float32")
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", driver="GTiff", height=a.shape[0], width=a.shape[1], count=1, dtype="float32",
                       crs="EPSG:4326", transform=transform(), nodata=np.nan, compress="deflate") as dst:
        dst.write(a, 1)
        dst.update_tags(units="ug m-3", product="GH-PM25", variable=var)
