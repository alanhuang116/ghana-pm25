"""Static km-scale covariates on the global-aligned 0.01 deg lattice (keyless sources).

Layer                     Source (resolution)                                  Access
------------------------  ---------------------------------------------------  ---------------------------
elev, elev_sd             Copernicus DEM GLO-90 (90 m)                         AWS COG, windowed reads
lc_* fractions            ESA WorldCover v200 2021 (10 m)                       AWS COG overviews (~100 m)
pop_density               WorldPop R2025A 2024 constrained, UN-adjusted (1 km)  data.worldpop.org
ntl                       VIIRS DNB monthly composites (World Bank LEN, 15")    AWS COG, median of 4 months
road_density, dist_road   OpenStreetMap motorway/trunk/primary/secondary        Overpass API
dist_coast                Natural Earth 1:10m coastline                         GitHub raw GeoJSON

Neighbourhood (Gaussian-smoothed, sigma 3 km / 10 km) versions are added for urban layers.
Each layer is built for a padded box then cropped, so smoothing is consistent everywhere.
"""
from __future__ import annotations

import json
import math
import time

import numpy as np
import pandas as pd
import rasterio
import xarray as xr
from rasterio.enums import Resampling
from rasterio.windows import from_bounds
from scipy import ndimage

from .. import config
from ..net import SESSION, download

CACHE = config.RAW / "static"
OUT = config.PROCESSED / "static"
RES = config.RES
PAD = 0.2  # deg padding for neighbourhood operations

GDAL_ENV = dict(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif",
                GDAL_HTTP_MULTIRANGE="YES", VSI_CACHE="TRUE", GDAL_HTTP_MAX_RETRY="5", GDAL_HTTP_RETRY_DELAY="3")

WC_CLASSES = {10: "tree", 20: "shrub", 30: "grass", 40: "crop", 50: "built", 60: "bare", 80: "water", 90: "wetland", 95: "mangrove"}
WORLDPOP_ISO = ["GHA", "CIV", "TGO", "BFA", "NGA", "MLI", "BEN", "GIN", "LBR", "NER"]
NTL_MONTHS = ["202311", "202312", "202401", "202402"]
OVERPASS = ["https://maps.mail.ru/osm/tools/overpass/api/interpreter", "https://overpass.kumi.systems/api/interpreter",
            "https://overpass-api.de/api/interpreter"]


def box_axes(lon_min, lon_max, lat_min, lat_max):
    """Cell centres aligned to the global x.xx5 lattice; lats north->south."""
    lons = np.round(np.arange(math.floor(lon_min / RES) * RES + RES / 2, lon_max, RES), 5)
    lats = np.round(np.arange(math.ceil(lat_max / RES) * RES - RES / 2, lat_min, -RES), 5)
    return lons, lats


def _read_window(url: str, bounds, shape, resampling=Resampling.average, fill=np.nan) -> np.ndarray:
    """Read `bounds` (w,s,e,n) of a remote COG resampled to `shape`; areas outside the raster are `fill`."""
    out = np.full(shape, fill, dtype="float32")
    with rasterio.Env(**GDAL_ENV):
        try:
            with rasterio.open("/vsicurl/" + url) as ds:
                l, b, r, t = ds.bounds
                w, s, e, n = bounds
                iw, is_, ie, in_ = max(w, l), max(s, b), min(e, r), min(n, t)
                if ie <= iw or in_ <= is_:
                    return out
                ny, nx = shape
                c0 = int(round((iw - w) / (e - w) * nx)); c1 = int(round((ie - w) / (e - w) * nx))
                r0 = int(round((n - in_) / (n - s) * ny)); r1 = int(round((n - is_) / (n - s) * ny))
                if c1 <= c0 or r1 <= r0:
                    return out
                win = from_bounds(iw, is_, ie, in_, transform=ds.transform)
                a = ds.read(1, window=win, out_shape=(r1 - r0, c1 - c0), resampling=resampling, masked=True)
                out[r0:r1, c0:c1] = a.filled(fill).astype("float32")
        except rasterio.errors.RasterioIOError:
            pass
    return out


def _block_reduce(a: np.ndarray, f: int, fn) -> np.ndarray:
    ny, nx = a.shape[0] // f, a.shape[1] // f
    return fn(a[: ny * f, : nx * f].reshape(ny, f, nx, f), axis=(1, 3))


# ---------------------------------------------------------------------------
# individual layers
# ---------------------------------------------------------------------------
def dem(bounds, shape, sub=6):
    w, s, e, n = bounds
    hi = (shape[0] * sub, shape[1] * sub)
    acc = np.full(hi, np.nan, dtype="float32")
    for la in range(math.floor(s), math.ceil(n)):
        for lo in range(math.floor(w), math.ceil(e)):
            ns_, ew = ("N" if la >= 0 else "S"), ("E" if lo >= 0 else "W")
            name = f"Copernicus_DSM_COG_30_{ns_}{abs(la):02d}_00_{ew}{abs(lo):03d}_00_DEM"
            url = f"https://copernicus-dem-90m.s3.amazonaws.com/{name}/{name}.tif"
            a = _read_window(url, bounds, hi, Resampling.average)
            acc = np.where(np.isnan(acc), a, acc)
    acc = np.nan_to_num(acc, nan=0.0)  # ocean tiles are absent
    return _block_reduce(acc, sub, np.mean), _block_reduce(acc, sub, np.std)


def worldcover(bounds, shape, sub=10):
    w, s, e, n = bounds
    hi = (shape[0] * sub, shape[1] * sub)
    acc = np.zeros(hi, dtype="uint8")
    for la in range(math.floor(s / 3) * 3, math.ceil(n / 3) * 3, 3):
        for lo in range(math.floor(w / 3) * 3, math.ceil(e / 3) * 3, 3):
            ns_, ew = ("N" if la >= 0 else "S"), ("E" if lo >= 0 else "W")
            url = (f"https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map/"
                   f"ESA_WorldCover_10m_2021_v200_{ns_}{abs(la):02d}{ew}{abs(lo):03d}_Map.tif")
            a = _read_window(url, bounds, hi, Resampling.nearest, fill=0)
            acc = np.where(acc == 0, a.astype("uint8"), acc)
    out = {}
    valid = _block_reduce((acc > 0).astype("float32"), sub, np.sum)
    for code, name in WC_CLASSES.items():
        cnt = _block_reduce((acc == code).astype("float32"), sub, np.sum)
        out[f"lc_{name}"] = np.where(valid > 0, cnt / np.maximum(valid, 1), 0).astype("float32")
    # unclassified (0) inside ocean -> treat as water
    out["lc_water"] = np.where(valid == 0, 1.0, out["lc_water"]).astype("float32")
    return out


def population(bounds, shape, sub=3):
    w, s, e, n = bounds
    hi = (shape[0] * sub, shape[1] * sub)
    acc = np.full(hi, np.nan, dtype="float32")
    for iso in WORLDPOP_ISO:
        url = (f"https://data.worldpop.org/GIS/Population/Global_2015_2030/R2025A/2024/{iso}/v1/1km_ua/constrained/"
               f"{iso.lower()}_pop_2024_CN_1km_R2025A_UA_v1.tif")
        dest = CACHE / "worldpop" / url.rsplit("/", 1)[-1]
        try:
            download(url, dest)
        except RuntimeError:
            continue
        with rasterio.open(dest) as ds:
            l, b, r, t = ds.bounds
            if r <= w or l >= e or t <= s or b >= n:
                continue
        # nearest at 3x oversampling of 0.01 deg, then density (people/km2) from 30" cells
        a = _read_local(dest, bounds, hi)
        acc = np.where(np.isnan(acc), a, np.where(np.isnan(a), acc, np.maximum(acc, a)))
    cell_km2 = (30 / 3600 * 111.32) ** 2 * np.cos(np.deg2rad((s + n) / 2))
    dens = np.nan_to_num(acc, nan=0.0) / cell_km2
    return _block_reduce(dens, sub, np.mean)


def _read_local(path, bounds, shape):
    out = np.full(shape, np.nan, dtype="float32")
    with rasterio.open(path) as ds:
        w, s, e, n = bounds
        l, b, r, t = ds.bounds
        iw, is_, ie, in_ = max(w, l), max(s, b), min(e, r), min(n, t)
        if ie <= iw or in_ <= is_:
            return out
        ny, nx = shape
        c0 = int(round((iw - w) / (e - w) * nx)); c1 = int(round((ie - w) / (e - w) * nx))
        r0 = int(round((n - in_) / (n - s) * ny)); r1 = int(round((n - is_) / (n - s) * ny))
        win = from_bounds(iw, is_, ie, in_, transform=ds.transform)
        a = ds.read(1, window=win, out_shape=(r1 - r0, c1 - c0), resampling=Resampling.nearest, masked=True, boundless=True)
        out[r0:r1, c0:c1] = a.filled(np.nan)
    return out


def nightlights(bounds, shape, sub=2):
    hi = (shape[0] * sub, shape[1] * sub)
    stack = []
    for ym in NTL_MONTHS:
        y, m = ym[:4], ym[4:]
        last = pd.Timestamp(f"{y}-{m}-01") + pd.offsets.MonthEnd(0)
        url = (f"https://globalnightlight.s3.amazonaws.com/composites/npp_{ym}_ops/"
               f"DNB_npp_{y}{m}01-{y}{m}{last.day:02d}_global_ecm-slcorr_v10_ops.avg_rade9.tif")
        stack.append(_read_window(url, bounds, hi, Resampling.average))
    a = np.nanmedian(np.stack(stack), axis=0)
    a = np.clip(np.nan_to_num(a, nan=0.0), 0, None)
    return _block_reduce(a, sub, np.mean)


def roads(bounds, shape, tile=1.0):
    """Major-road length density (km per km2) and distance to nearest major road (km)."""
    w, s, e, n = bounds
    ny, nx = shape
    length = np.zeros(shape, dtype="float64")
    for la in np.arange(math.floor(s), math.ceil(n), tile):
        for lo in np.arange(math.floor(w), math.ceil(e), tile):
            tb = (max(la, s), max(lo, w), min(la + tile, n), min(lo + tile, e))
            if tb[2] <= tb[0] or tb[3] <= tb[1]:
                continue
            ways = [np.asarray(c) for c in _overpass_ways(tb) if len(c) >= 2]
            if not ways:
                continue
            p0 = np.concatenate([c[:-1] for c in ways])
            seg = np.concatenate([np.diff(c, axis=0) for c in ways])
            seglen = np.hypot(seg[:, 0] * 111.32 * np.cos(np.deg2rad(p0[:, 1])), seg[:, 1] * 111.32)
            # densify every segment to ~50 m sub-segments and bin their lengths into cells
            k = np.maximum(1, (seglen / 0.05).astype(int))
            idx = np.repeat(np.arange(len(k)), k)
            offs = np.arange(len(idx)) - np.repeat(np.cumsum(k) - k, k)
            t = (offs + 0.5) / k[idx]
            xs = p0[idx, 0] + seg[idx, 0] * t
            ys = p0[idx, 1] + seg[idx, 1] * t
            cols = np.floor((xs - w) / RES).astype(int)
            rows = np.floor((n - ys) / RES).astype(int)
            # ways crossing tile edges are returned by both neighbouring queries: count only in-tile points
            in_tile = (xs >= tb[1]) & (xs < tb[3]) & (ys >= tb[0]) & (ys < tb[2])
            ok = in_tile & (cols >= 0) & (cols < nx) & (rows >= 0) & (rows < ny)
            np.add.at(length, (rows[ok], cols[ok]), (seglen[idx] / k[idx])[ok])
    cell_km2 = (RES * 111.32) ** 2 * np.cos(np.deg2rad((s + n) / 2))
    dens = (length / cell_km2).astype("float32")
    dist = ndimage.distance_transform_edt(length == 0) * RES * 111.32
    return dens, dist.astype("float32")


def _overpass_ways(tb):
    key = "_".join(f"{v:.2f}" for v in tb)
    f = CACHE / "osm" / f"roads_{key}.json"
    if f.exists():
        return json.loads(f.read_text())
    q = (f'[out:json][timeout:300];way["highway"~"^(motorway|motorway_link|trunk|trunk_link|primary|primary_link|secondary)$"]'
         f'({tb[0]},{tb[1]},{tb[2]},{tb[3]});out geom;')
    for attempt in range(6):
        for url in OVERPASS:
            try:
                r = SESSION.get(url, params={"data": q}, timeout=320)
                if r.status_code == 200:
                    ways = [[(p["lon"], p["lat"]) for p in el.get("geometry", [])] for el in r.json().get("elements", [])]
                    f.parent.mkdir(parents=True, exist_ok=True)
                    f.write_text(json.dumps(ways))
                    return ways
            except Exception:
                pass
        time.sleep(20 * (attempt + 1))
    raise RuntimeError(f"Overpass failed for {tb}")


_COAST = None


def dist_coast(lons: np.ndarray, lats: np.ndarray) -> np.ndarray:
    global _COAST
    import shapely
    from shapely.geometry import box, shape as shp
    if _COAST is None:
        url = "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_10m_coastline.geojson"
        dest = download(url, CACHE / "ne_10m_coastline.geojson")
        gj = json.loads(dest.read_text(encoding="utf-8"))
        region = box(-25, -5, 25, 25)
        geoms = [shp(ft["geometry"]).intersection(region) for ft in gj["features"]]
        _COAST = shapely.union_all([g for g in geoms if not g.is_empty])
    LON, LAT = np.meshgrid(lons, lats)
    # local equirectangular km coordinates
    k = np.cos(np.deg2rad(np.mean(lats)))
    coast_scaled = shapely.transform(_COAST, lambda xy: np.c_[xy[:, 0] * k, xy[:, 1]])
    pts = shapely.points(LON.ravel() * k, LAT.ravel())
    shapely.prepare(coast_scaled)
    d = shapely.distance(pts, coast_scaled) * 111.32
    return d.reshape(LON.shape).astype("float32")


# ---------------------------------------------------------------------------
# assemble
# ---------------------------------------------------------------------------
def _smooth(a, sigma_km):
    return ndimage.gaussian_filter(a, sigma=sigma_km / (RES * 111.32), mode="nearest")


def build_box(name: str, lon_min, lon_max, lat_min, lat_max, overwrite=False) -> xr.Dataset:
    f = OUT / f"static_{name}.nc"
    if f.exists() and not overwrite:
        return xr.open_dataset(f)
    OUT.mkdir(parents=True, exist_ok=True)
    lons, lats = box_axes(lon_min, lon_max, lat_min, lat_max)
    plons, plats = box_axes(lon_min - PAD, lon_max + PAD, lat_min - PAD, lat_max + PAD)
    bounds = (plons[0] - RES / 2, plats[-1] - RES / 2, plons[-1] + RES / 2, plats[0] + RES / 2)
    shape = (len(plats), len(plons))
    t = time.time()
    layers = {}
    layers["elev"], layers["elev_sd"] = dem(bounds, shape)
    print(f"    [{name}] dem {time.time()-t:.0f}s", flush=True)
    layers.update(worldcover(bounds, shape))
    print(f"    [{name}] worldcover {time.time()-t:.0f}s", flush=True)
    layers["pop_density"] = population(bounds, shape)
    layers["ntl"] = nightlights(bounds, shape)
    print(f"    [{name}] pop+ntl {time.time()-t:.0f}s", flush=True)
    layers["road_density"], layers["dist_road"] = roads(bounds, shape)
    print(f"    [{name}] roads {time.time()-t:.0f}s", flush=True)

    # neighbourhood context
    layers["log_pop"] = np.log1p(layers["pop_density"])
    layers["log_ntl"] = np.log1p(layers["ntl"])
    for base in ("lc_built", "log_pop", "log_ntl", "road_density", "lc_crop", "lc_tree", "lc_bare"):
        for sig in (3, 10):
            layers[f"{base}_s{sig}km"] = _smooth(layers[base], sig).astype("float32")
    layers["elev_rel_10km"] = layers["elev"] - _smooth(layers["elev"], 10)
    layers["dist_road"] = np.minimum(layers["dist_road"], 100)

    # crop padding
    i0 = int(round((plats[0] - lats[0]) / RES)); j0 = int(round((lons[0] - plons[0]) / RES))
    crop = {k: v[i0:i0 + len(lats), j0:j0 + len(lons)] for k, v in layers.items()}
    crop["dist_coast"] = dist_coast(lons, lats)
    ds = xr.Dataset({k: (("lat", "lon"), v.astype("float32")) for k, v in crop.items()}, coords={"lat": lats, "lon": lons})
    ds.attrs["description"] = __doc__
    ds.to_netcdf(f)
    print(f"    [{name}] done {ds.sizes} in {time.time()-t:.0f}s", flush=True)
    return ds


def station_boxes(stations: pd.DataFrame, cell: float = 0.5, margin: float = 0.15) -> list[tuple]:
    """Boxes (name, lon_min, lon_max, lat_min, lat_max) covering stations outside the Ghana grid."""
    g = config
    inside = stations.lon.between(g.LON_MIN + margin, g.LON_MAX - margin) & stations.lat.between(g.LAT_MIN + margin, g.LAT_MAX - margin)
    out = stations[~inside]
    tiles = sorted({(math.floor(lo / cell), math.floor(la / cell)) for lo, la in zip(out.lon, out.lat)})
    boxes = []
    for tx, ty in tiles:
        boxes.append((f"tile_{tx}_{ty}", tx * cell - margin, (tx + 1) * cell + margin, ty * cell - margin, (ty + 1) * cell + margin))
    return boxes


def sample(stations: pd.DataFrame) -> pd.DataFrame:
    """Static covariates at station cells (from the Ghana grid or station boxes)."""
    rows = []
    datasets = {"ghana": xr.open_dataset(OUT / "static_ghana.nc")}
    for b in station_boxes(stations):
        datasets[b[0]] = xr.open_dataset(OUT / f"static_{b[0]}.nc")
    for _, st in stations.iterrows():
        rec = None
        for name, ds in datasets.items():
            if ds.lon.min() <= st.lon <= ds.lon.max() and ds.lat.min() <= st.lat <= ds.lat.max():
                v = ds.sel(lon=st.lon, lat=st.lat, method="nearest")
                cand = {k: float(v[k]) for k in ds.data_vars}
                # prefer the box where the station is farthest from the edge
                edge = min(st.lon - float(ds.lon.min()), float(ds.lon.max()) - st.lon, st.lat - float(ds.lat.min()), float(ds.lat.max()) - st.lat)
                if rec is None or edge > rec[0]:
                    rec = (edge, cand)
        if rec is not None:
            rows.append({"site_id": st.site_id, **rec[1]})
    return pd.DataFrame(rows)
