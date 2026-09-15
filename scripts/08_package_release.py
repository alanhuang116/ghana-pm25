"""Package the 1-km dataset for distribution on GitHub Releases and write portal/data/downloads.json.

Assets (release tag `data`):
  GH-PM25_1km_daily_YYYY-MM.zip          daily NetCDF files of one month
  GH-PM25_1km_monthly_mean_YYYY-MM.tif   monthly mean GeoTIFF
  GH-PM25_1km_annual_mean_YYYY.tif       annual mean GeoTIFF (partial years flagged in README)
  GH-PM25_region_daily_statistics.csv    national / regional / district daily statistics
  GH-PM25_DATA_README.txt                data dictionary, grid, citation

    python scripts/08_package_release.py --all --upload          # first publication
    python scripts/08_package_release.py --recent 10 --upload    # daily: months touched in the last 10 days
"""
from __future__ import annotations

import argparse
import datetime as dt
import glob
import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ghana_pm25 import __version__, config  # noqa: E402

TAG = "data"
DIST = config.ROOT / "dist"
DAILY = config.OUTPUTS / "daily"


def repo_slug() -> str:
    if os.environ.get("GITHUB_REPOSITORY"):
        return os.environ["GITHUB_REPOSITORY"]
    url = subprocess.run(["git", "config", "--get", "remote.origin.url"], capture_output=True, text=True, cwd=config.ROOT).stdout.strip()
    return url.rstrip("/").removesuffix(".git").split("github.com")[-1].lstrip(":/")


def gh(*args, check=True) -> subprocess.CompletedProcess:
    return subprocess.run(["gh", *args], capture_output=True, text=True, check=check, cwd=config.ROOT)


def ensure_release(repo: str):
    if gh("release", "view", TAG, "-R", repo, check=False).returncode != 0:
        gh("release", "create", TAG, "-R", repo, "--title", "GH-PM25 1-km dataset",
           "--notes", "Daily 1-km PM2.5 for Ghana (GH-PM25). Monthly zips of daily NetCDF files, mean-map GeoTIFFs and regional statistics. "
                      "Updated automatically every day. See GH-PM25_DATA_README.txt.")


def remote_assets(repo: str) -> list[dict]:
    r = gh("release", "view", TAG, "-R", repo, "--json", "assets", check=False)
    return json.loads(r.stdout)["assets"] if r.returncode == 0 else []


def month_files(month: str) -> list[Path]:
    y, m = month.split("-")
    return sorted(Path(p) for p in glob.glob(str(DAILY / y / f"GH-PM25_1km_{y}{m}*.nc")))


def merge_remote_month(repo: str, month: str, names: set[str]):
    """On a fresh runner only recent days exist locally: pull the published zip and restore missing days."""
    name = f"GH-PM25_1km_daily_{month}.zip"
    if name not in names:
        return
    DIST.mkdir(exist_ok=True)
    tmp = DIST / f"remote_{name}"
    gh("release", "download", TAG, "-R", repo, "-p", name, "-O", str(tmp), "--clobber")
    with zipfile.ZipFile(tmp) as z:
        for member in z.namelist():
            dest = DAILY / member
            if not dest.exists():
                dest.parent.mkdir(parents=True, exist_ok=True)
                with z.open(member) as src, open(dest, "wb") as out:
                    out.write(src.read())
    tmp.unlink()


def build_month(month: str) -> list[Path]:
    import rasterio
    import xarray as xr
    from ghana_pm25.export import write_geotiff
    files = month_files(month)
    if not files:
        return []
    DIST.mkdir(exist_ok=True)
    zpath = DIST / f"GH-PM25_1km_daily_{month}.zip"
    with zipfile.ZipFile(zpath, "w", compression=zipfile.ZIP_STORED) as z:  # NetCDF is already deflated
        for f in files:
            z.write(f, arcname=f"{f.parent.name}/{f.name}")
    acc, n = None, 0
    for f in files:
        with xr.open_dataset(f) as ds:
            a = ds.pm25.values.astype("float64")
        acc = np.nan_to_num(a) if acc is None else acc + np.nan_to_num(a)
        mask = np.isfinite(a)
        n += 1
    mean = np.where(mask, acc / n, np.nan).astype("float32")
    lons, lats = config.grid_axes()
    tif = DIST / f"GH-PM25_1km_monthly_mean_{month}.tif"
    write_geotiff(xr.Dataset({"pm25": (("lat", "lon"), mean)}, coords={"lat": lats, "lon": lons}), "pm25", tif)
    with rasterio.open(tif, "r+") as dst:
        dst.update_tags(days=str(n), period=month)
    return [zpath, tif]


def build_annual() -> list[Path]:
    import xarray as xr
    from ghana_pm25.export import write_geotiff
    out = []
    lons, lats = config.grid_axes()
    for ydir in sorted(p for p in DAILY.iterdir() if p.is_dir()):
        files = sorted(ydir.glob("*.nc"))
        acc, n, mask = None, 0, None
        for f in files:
            with xr.open_dataset(f) as ds:
                a = ds.pm25.values.astype("float64")
            mask = np.isfinite(a)
            acc = np.nan_to_num(a) if acc is None else acc + np.nan_to_num(a)
            n += 1
        if n:
            tif = DIST / f"GH-PM25_1km_annual_mean_{ydir.name}.tif"
            write_geotiff(xr.Dataset({"pm25": (("lat", "lon"), np.where(mask, acc / n, np.nan).astype("float32"))}, coords={"lat": lats, "lon": lons}), "pm25", tif)
            out.append(tif)
    return out


def build_stats_and_readme() -> list[Path]:
    DIST.mkdir(exist_ok=True)
    st = pd.read_parquet(config.OUTPUTS / "region_daily_stats.parquet").sort_values(["date", "level", "id"])
    st["date"] = pd.to_datetime(st.date).dt.strftime("%Y-%m-%d")
    csv = DIST / "GH-PM25_region_daily_statistics.csv"
    st.round(4).to_csv(csv, index=False)
    days = sorted(p.stem.split("_")[-1] for p in DAILY.rglob("*.nc"))
    summ = json.loads((config.OUTPUTS / "run_summary.json").read_text())
    readme = DIST / "GH-PM25_DATA_README.txt"
    readme.write_text(f"""GH-PM25 - daily 1-km PM2.5 for Ghana (product version {__version__})
Generated {dt.datetime.utcnow():%Y-%m-%d %H:%M} UTC. Coverage {days[0]} to {days[-1]} ({len(days)} days).

GRID      regular latitude-longitude (WGS84, EPSG:4326), 0.01 degree (~1.1 km), 650 rows x 455 columns
          cell centres 3.295W..1.245E, 11.195N..4.705N; cells outside Ghana are missing
DAY       00-24 UTC (= Ghana local time)

DAILY NETCDF (GH-PM25_1km_YYYYMMDD.nc, CF-1.8, zlib)
  pm25            daily mean PM2.5 (ug/m3), final product
  pm25_lower90    lower bound of the 90% conformal prediction interval (ug/m3)
  pm25_upper90    upper bound of the 90% conformal prediction interval (ug/m3)
  pm25_stage1     stage-1 ensemble prediction (ug/m3)
  cams_pm25       CAMS global PM2.5 input, bilinear (ug/m3)
  log_sigma       predicted absolute log-error scale
  aoa_di          dissimilarity index (Meyer & Pebesma 2021)
  inside_aoa      1 inside / 0 outside the area of applicability, -1 no data
  attribute n_obs_used: ground monitors assimilated that day
  PM2.5 fields are int16 with scale_factor 0.1 (decoded automatically by xarray/netCDF4)

PRODUCT TIERS  forecast (D+1) / nowcast (D0) / nrt (D-1..D-6, re-processed daily) / final (<= D-7)
               Days in the current and previous month may be revised by later runs.

MEAN GEOTIFFS  GH-PM25_1km_monthly_mean_YYYY-MM.tif, GH-PM25_1km_annual_mean_YYYY.tif (partial years = mean of available days)
STATISTICS     GH-PM25_region_daily_statistics.csv: level (adm0/adm1/adm2), id, name, area_mean, pop_weighted,
               lower90, upper90, max, population, pop_frac_gt15, pop_frac_gt35, date, n_obs

ACCURACY (leave-city-cluster-out cross-validation, {summ['n_train']:,} station-days, {summ['n_sites']} monitors)
  see the technical report in the portal (report/GH-PM25_Technical_Report.html)

SOURCES  OpenAQ; US Department of State AirNow; Copernicus CAMS (via Open-Meteo); NASA MERRA-2 / POWER;
         ECMWF IFS (via Open-Meteo); NASA FIRMS and NOAA VIIRS EFIRE; ESA WorldCover; Copernicus DEM;
         WorldPop; OpenStreetMap contributors; Natural Earth; geoBoundaries. Acknowledge these when using the data.
""", encoding="utf-8")
    return [csv, readme]


def classify(name: str) -> dict:
    if name.startswith("GH-PM25_1km_daily_"):
        return dict(kind="daily_month", period=name[len("GH-PM25_1km_daily_"):-4])
    if name.startswith("GH-PM25_1km_monthly_mean_"):
        return dict(kind="monthly_mean", period=name[len("GH-PM25_1km_monthly_mean_"):-4])
    if name.startswith("GH-PM25_1km_annual_mean_"):
        return dict(kind="annual_mean", period=name[len("GH-PM25_1km_annual_mean_"):-4])
    if name.endswith("statistics.csv"):
        return dict(kind="region_stats", period="daily")
    if name.endswith("README.txt"):
        return dict(kind="readme", period="")
    return dict(kind="other", period="")


def write_downloads_json(repo: str):
    assets = remote_assets(repo)
    items = []
    for a in assets:
        c = classify(a["name"])
        if c["kind"] == "other":
            continue
        rec = dict(name=a["name"], size=a["size"], url=f"https://github.com/{repo}/releases/download/{TAG}/{a['name']}", **c)
        if c["kind"] == "daily_month":
            rec["days"] = len(month_files(c["period"])) or None
        items.append(rec)
    (config.PORTAL_DATA / "downloads.json").write_text(json.dumps(dict(
        release_url=f"https://github.com/{repo}/releases/tag/{TAG}", updated=dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S"), assets=items), indent=1))
    print(f"downloads.json: {len(items)} assets")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true", help="package every month")
    ap.add_argument("--recent", type=int, default=10, help="package months touched in the last N days")
    ap.add_argument("--upload", action="store_true")
    args = ap.parse_args()
    repo = repo_slug()
    if args.upload:
        ensure_release(repo)
    names = {a["name"] for a in remote_assets(repo)} if args.upload else set()
    if args.all:
        months = sorted({f"{p.stem[-8:-4]}-{p.stem[-4:-2]}" for p in DAILY.rglob("*.nc")})
    else:
        today = pd.Timestamp.today().normalize()
        months = sorted({d.strftime("%Y-%m") for d in pd.date_range(today - pd.Timedelta(days=args.recent), today + pd.Timedelta(days=1))})
    built = []
    for month in months:
        if args.upload:
            merge_remote_month(repo, month, names)
        built += build_month(month)
        print("packaged", month, flush=True)
    built += build_annual() if args.all else [p for p in build_annual() if p.stem.endswith(str(pd.Timestamp.today().year))]
    built += build_stats_and_readme()
    if args.upload:
        for i in range(0, len(built), 10):
            gh("release", "upload", TAG, "-R", repo, "--clobber", *map(str, built[i:i + 10]))
            print("uploaded", [p.name for p in built[i:i + 10]], flush=True)
        write_downloads_json(repo)


if __name__ == "__main__":
    main()
