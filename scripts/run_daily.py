"""Operational daily run of the GH-PM25 monitoring system.

    python scripts/run_daily.py                 # refresh inputs, re-map D-7..D+1, update portal
    python scripts/run_daily.py --retrain       # additionally re-download ground data and retrain (monthly)

Schedule once per day (e.g. 06:30 UTC) with Windows Task Scheduler (scripts/schedule_windows_task.ps1),
cron, or the GitHub Actions workflow in .github/workflows/daily.yml.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ghana_pm25 import calibration, config, export, features as F, nrt  # noqa: E402
from ghana_pm25.model import GHPM25  # noqa: E402
from ghana_pm25.predict import GridContext, predict_day, region_stats  # noqa: E402
from ghana_pm25.sources import firms  # noqa: E402

LOG = config.OUTPUTS / "logs"
LOG.mkdir(parents=True, exist_ok=True)


def log(msg):
    line = f"{dt.datetime.utcnow():%Y-%m-%d %H:%M:%S}Z  {msg}"
    print(line, flush=True)
    with open(LOG / f"run_{dt.date.today():%Y%m%d}.log", "a", encoding="utf-8") as f:
        f.write(line + "\n")


def step(name, fn, freshness, *a, **k):
    t = time.time()
    try:
        out = fn(*a, **k)
        log(f"OK   {name} ({time.time() - t:.0f}s)")
        return out
    except Exception as e:  # keep the system running on partial input failure
        log(f"FAIL {name}: {e}\n{traceback.format_exc()}")
        freshness[f"{name}_error"] = str(e)[:200]
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--retrain", action="store_true")
    ap.add_argument("--back", type=int, default=7, help="days before today to re-map")
    ap.add_argument("--ahead", type=int, default=1, help="forecast days")
    args = ap.parse_args()
    py = sys.executable
    freshness = {}
    today = pd.Timestamp.today().normalize()

    if args.retrain:
        subprocess.run([py, str(ROOT / "scripts" / "01_download_ground.py")], check=True)
        subprocess.run([py, str(ROOT / "scripts" / "03_train_evaluate.py")], check=True)

    nodes = pd.read_parquet(config.INTERIM / "cams_nodes.parquet")
    cams_df = step("cams", nrt.update_cams, freshness, nodes)
    if cams_df is None:
        cams_df = pd.read_parquet(config.INTERIM / "cams_daily.parquet")
    freshness["CAMS"] = str(pd.to_datetime(cams_df.date).max().date())

    met = step("merra2_power", nrt.update_power, freshness)
    if met is None:
        met = pd.read_parquet(config.INTERIM / "power_daily.parquet")
    freshness["MERRA-2 (POWER)"] = str(met.dropna(subset=["t2m"]).date.max().date())
    met_ext = step("ifs_fill", nrt.ifs_fill, freshness, met, args.ahead)
    met_ext = met if met_ext is None else met_ext.drop(columns=["source"])

    fires = pd.read_parquet(config.INTERIM / "firms_hist.parquet")
    fires["date"] = pd.to_datetime(fires.date)
    recent = step("viirs_fires", firms.load_recent, freshness, 10)
    if recent is not None and len(recent):
        recent["date"] = pd.to_datetime(recent.date)
        fires = pd.concat([fires[fires.date < recent.date.min()], recent], ignore_index=True)
        fires.to_parquet(config.INTERIM / "firms_hist.parquet")
    freshness["VIIRS fires"] = str(fires.date.max().date())

    cams = F.cams_lattice_from_df(cams_df)
    metl = F.met_lattice_from_df(met_ext)
    stations = pd.read_parquet(config.PROCESSED / "stations.parquet")
    sstatic = pd.read_parquet(config.PROCESSED / "station_static.parquet")

    # ---- observations ----
    obs_hist = pd.read_parquet(config.PROCESSED / "training_table.parquet", columns=["site_id", "date", "pm25", "pm25_raw"])
    new = step("openaq_obs", nrt.update_obs, freshness, 12)
    obs = obs_hist
    if new is not None and len(new):
        new = new[new.site_id.isin(stations.site_id)]
        o = new.merge(stations[["site_id", "lat", "lon"]], on="site_id")
        dyn = F.dynamic_features(o.lat.to_numpy(), o.lon.to_numpy(), o.date.to_numpy(), cams, metl)
        env = pd.concat([o[["site_id", "date"]].reset_index(drop=True), dyn[["rh", "dust_frac"]]], axis=1)
        models = json.loads((config.MODELS / "lcs_calibration.json").read_text())
        cal = calibration.apply(new[["site_id", "date", "pm25"]], stations, env, models)
        (config.INTERIM / "nrt").mkdir(exist_ok=True)
        prev = config.INTERIM / "nrt" / "obs_daily_calibrated.parquet"
        allnrt = pd.concat([pd.read_parquet(prev), cal]) if prev.exists() else cal
        allnrt = allnrt.drop_duplicates(["site_id", "date"], keep="last")
        allnrt.to_parquet(prev)
        obs = pd.concat([obs_hist, allnrt[["site_id", "date", "pm25", "pm25_raw"]]]).drop_duplicates(["site_id", "date"], keep="last")
        freshness["Ground monitors"] = str(pd.to_datetime(new.date).max().date())

    # ---- mapping ----
    model = GHPM25.load()
    ctx = GridContext()
    last_cams = pd.to_datetime(cams_df.date).max()
    last_met = pd.to_datetime(met_ext.dropna(subset=["t2m"]).date).max()
    dates = [d for d in pd.date_range(today - pd.Timedelta(days=args.back), today + pd.Timedelta(days=args.ahead)) if d <= min(last_cams, last_met)]
    tiers_f = config.OUTPUTS / "tiers.json"
    tiers = json.loads(tiers_f.read_text()) if tiers_f.exists() else {}
    stats, cube_upd = [], {}
    for d in dates:
        t = time.time()
        od = obs[pd.to_datetime(obs.date) == d]
        ds = predict_day(model, ctx, d, cams, metl, fires, od, stations, sstatic)
        export.write_netcdf(ds)
        export.write_day(ds)
        rs = region_stats(ds, ctx); rs["date"] = d; rs["n_obs"] = int(ds.attrs["n_obs_used"])
        stats.append(rs)
        cube_upd.setdefault(d.year, {})[d] = export.coarse(ds)
        tiers[d.strftime("%Y-%m-%d")] = nrt.tier(d, today)
        log(f"map {d.date()} tier={tiers[d.strftime('%Y-%m-%d')]} obs={rs.n_obs.iloc[0]} ({time.time() - t:.0f}s)")
    for y, upd in cube_upd.items():
        export.update_cube(y, upd)
    tiers_f.write_text(json.dumps(tiers))
    if stats:
        s = pd.concat(stats, ignore_index=True)
        allstats = pd.read_parquet(config.OUTPUTS / "region_daily_stats.parquet")
        allstats = pd.concat([allstats[~allstats.date.isin(s.date.unique())], s], ignore_index=True)
        allstats.to_parquet(config.OUTPUTS / "region_daily_stats.parquet")
        export.write_region_series(allstats)
        s.to_csv(config.OUTPUTS / f"region_stats_latest.csv", index=False)
    freshness["last_run_utc"] = dt.datetime.utcnow().strftime("%Y-%m-%d %H:%M")
    (config.OUTPUTS / "freshness.json").write_text(json.dumps(freshness, indent=2))
    subprocess.run([py, str(ROOT / "scripts" / "05_export_portal.py")], check=True)
    log("daily run complete")


if __name__ == "__main__":
    main()
