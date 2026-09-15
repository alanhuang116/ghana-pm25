"""Produce daily 1-km maps for the full record and export portal assets.

    python scripts/04_predict_history.py [--start 2022-08-06] [--end 2026-09-10] [--workers 6]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ghana_pm25 import config, export, features as F  # noqa: E402
from ghana_pm25.model import GHPM25  # noqa: E402

_W = {}


def _init_worker(threads: int):
    if _W:
        return _W
    from ghana_pm25.predict import GridContext
    m = GHPM25.load()
    for name, est in m.base.models.items():
        try:
            if name == "catboost":
                est.set_params(thread_count=threads)
            else:
                est.set_params(n_jobs=threads)
        except Exception:
            pass
    _W.update(
        model=m, ctx=GridContext(), cams=F.load_cams_lattice(),
        met=F.met_lattice_from_df(pd.read_parquet(config.INTERIM / "power_daily.parquet")),
        fires=pd.read_parquet(config.INTERIM / "firms_hist.parquet").assign(date=lambda d: pd.to_datetime(d.date)),
        obs=pd.read_parquet(config.PROCESSED / "training_table.parquet", columns=["site_id", "date", "pm25"]),
        stations=pd.read_parquet(config.PROCESSED / "stations.parquet"),
        sstatic=pd.read_parquet(config.PROCESSED / "station_static.parquet"),
    )
    return _W


def run_day(date: pd.Timestamp, threads: int):
    from ghana_pm25.predict import predict_day, region_stats
    w = _init_worker(threads)
    t = time.time()
    obs_day = w["obs"][w["obs"].date == date]
    ds = predict_day(w["model"], w["ctx"], date, w["cams"], w["met"], w["fires"], obs_day, w["stations"], w["sstatic"])
    export.write_netcdf(ds)
    export.write_day(ds)
    rs = region_stats(ds, w["ctx"])
    rs["date"] = date
    rs["n_obs"] = int(ds.attrs["n_obs_used"])
    return date, export.coarse(ds), ds.pm25.values.astype("float32"), rs, time.time() - t, int(ds.attrs["n_obs_used"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default=None)
    ap.add_argument("--end", default=None)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    cams_dates = pd.read_parquet(config.INTERIM / "cams_daily.parquet", columns=["date"]).date
    met_dates = pd.read_parquet(config.INTERIM / "power_daily.parquet", columns=["date"]).date
    start = pd.Timestamp(args.start) if args.start else max(cams_dates.min(), met_dates.min()) + pd.Timedelta(days=2)
    end = pd.Timestamp(args.end) if args.end else min(cams_dates.max(), met_dates.max())
    dates = list(pd.date_range(start, end))
    threads = max(1, 30 // args.workers)
    from ghana_pm25.grid import admin_rasters
    admin_rasters()  # build shared admin grids once before forking workers
    print(f"{len(dates)} days {start.date()} -> {end.date()} with {args.workers} workers x {threads} threads", flush=True)

    cube, stats, monthly = {}, [], {}
    t0 = time.time()
    gen = Parallel(n_jobs=args.workers, backend="loky", return_as="generator_unordered")(delayed(run_day)(d, threads) for d in dates)
    for i, (date, co, full, rs, secs, nobs) in enumerate(gen):
        cube.setdefault(date.year, {})[date] = co
        stats.append(rs)
        key = date.strftime("%Y-%m")
        s, n = monthly.get(key, (np.zeros_like(full, dtype="float64"), 0))
        monthly[key] = (s + np.nan_to_num(full), n + 1)
        if i % 20 == 0:
            el = time.time() - t0
            print(f"  {i + 1}/{len(dates)} {date.date()} obs={nobs} {secs:.1f}s/day; elapsed {el / 60:.1f} min, eta {el / (i + 1) * (len(dates) - i - 1) / 60:.0f} min", flush=True)

    # cubes per year
    for y, dd in cube.items():
        ds_sorted = sorted(dd)
        export.write_cube(y, ds_sorted, [dd[d] for d in ds_sorted])
    allstats = pd.concat(stats, ignore_index=True)
    allstats.to_parquet(config.OUTPUTS / "region_daily_stats.parquet")
    export.write_region_series(allstats)

    # monthly / annual means (GeoTIFF + portal PNG)
    import xarray as xr
    from ghana_pm25.grid import admin_rasters
    adm = admin_rasters()
    lons, lats = adm["lons"], adm["lats"]
    years = {}
    for key, (s, n) in sorted(monthly.items()):
        m = np.where(adm["mask"], s / n, np.nan).astype("float32")
        ds = xr.Dataset({"pm25": (("lat", "lon"), m)}, coords={"lat": lats, "lon": lons})
        export.write_geotiff(ds, "pm25", config.OUTPUTS / "monthly" / f"GH-PM25_1km_monthly_{key.replace('-', '')}.tif")
        years.setdefault(key[:4], []).append((s, n))
    export.write_json("aggregates/months.json", sorted(monthly))
    for y, parts in years.items():
        s = sum(p[0] for p in parts); n = sum(p[1] for p in parts)
        m = np.where(adm["mask"], s / n, np.nan).astype("float32")
        ds = xr.Dataset({"pm25": (("lat", "lon"), m)}, coords={"lat": lats, "lon": lons})
        export.write_geotiff(ds, "pm25", config.OUTPUTS / "annual" / f"GH-PM25_1km_annual_{y}_n{n}days.tif")
    print(f"done in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
