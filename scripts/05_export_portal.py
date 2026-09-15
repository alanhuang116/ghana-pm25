"""Assemble portal/data from model outputs (manifest, validation, stations, boundaries)."""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ghana_pm25 import config, export  # noqa: E402
from ghana_pm25 import __version__  # noqa: E402

OUT = config.OUTPUTS

SOURCES = [
    dict(name="OpenAQ archive — Clarity, AirGradient, Data354, IQAir and other networks", role="Ground truth (calibrated low-cost sensors)", resolution="point, hourly → daily", access="s3://openaq-data-archive (keyless)"),
    dict(name="US Dept. of State BAM-1020 (Accra, Abidjan, Lomé, Ouagadougou, Bamako, Abuja, Lagos)", role="Reference ground truth & sensor calibration", resolution="point, hourly → daily", access="OpenAQ / AirNow files (keyless)"),
    dict(name="CAMS global atmospheric composition (ECMWF IFS-COMPO)", role="PM2.5, PM10, AOD, dust, CO, NO₂ predictors", resolution="0.4°, 3-hourly", access="Open-Meteo air-quality API"),
    dict(name="MERRA-2 / GEOS FP-IT via NASA POWER", role="T, RH, humidity, precipitation, wind, pressure, PBL, radiation, cloud, AOD", resolution="0.5° × 0.625°, daily", access="NASA POWER regional API"),
    dict(name="ECMWF IFS 0.25° (Open-Meteo)", role="Meteorology for nowcast/forecast days (bias-adjusted to MERRA-2)", resolution="0.25°, hourly", access="Open-Meteo forecast API"),
    dict(name="NOAA-20 VIIRS I-band Active Fire EDR (+ FIRMS S-NPP for Aug–Oct 2022)", role="Fire radiative power: local and upwind-weighted smoke predictors", resolution="375 m, 2 overpasses/day", access="NOAA Open Data Dissemination (AWS)"),
    dict(name="ESA WorldCover v200 (2021)", role="Land-cover fractions (tree, crop, built, bare, water…)", resolution="10 m → 0.01°", access="AWS COG"),
    dict(name="Copernicus DEM GLO-90", role="Elevation, roughness, relative relief", resolution="90 m → 0.01°", access="AWS COG"),
    dict(name="WorldPop R2025A 2024 (constrained, UN-adjusted)", role="Population density; exposure weighting", resolution="30″ (~1 km)", access="data.worldpop.org"),
    dict(name="VIIRS DNB monthly composites (World Bank Light Every Night)", role="Night-time lights (urban activity)", resolution="15″ → 0.01°", access="AWS COG"),
    dict(name="OpenStreetMap major roads", role="Road density, distance to road", resolution="vector → 0.01°", access="Overpass API"),
    dict(name="Natural Earth coastline; geoBoundaries GHA ADM0–2", role="Distance to coast; regional statistics", resolution="vector", access="GitHub (keyless)"),
]


def scatter_bins(obs, pred, n=40, lo=2.0, hi=400.0):
    ok = np.isfinite(obs) & np.isfinite(pred) & (obs > 0) & (pred > 0)
    lx = np.clip(np.log10(obs[ok]), np.log10(lo), np.log10(hi) - 1e-9)
    ly = np.clip(np.log10(pred[ok]), np.log10(lo), np.log10(hi) - 1e-9)
    H, _, _ = np.histogram2d(lx, ly, bins=n, range=[[np.log10(lo), np.log10(hi)], [np.log10(lo), np.log10(hi)]])
    cells = [[int(i), int(j), int(H[i, j])] for i, j in zip(*np.nonzero(H))]
    return dict(min=lo, max=hi, n=n, cells=cells)


def main():
    P = config.PORTAL_DATA
    summary = json.loads((OUT / "run_summary.json").read_text())
    smear = summary["smear"]
    stations = pd.read_parquet(config.PROCESSED / "stations.parquet")
    table = pd.read_parquet(config.PROCESSED / "training_table.parquet", columns=["site_id", "date", "pm25", "pm25_raw"])
    oof = pd.read_parquet(OUT / "cv_oof.parquet")

    # ---- validation bundle ----
    met = pd.read_csv(OUT / "cv_metrics.csv")
    val = dict(
        metrics=json.loads(met.to_json(orient="records")),
        scatter={s: scatter_bins(oof.pm25.to_numpy(), np.exp(oof[f"{s}__stack_rk"].to_numpy()) * smear) for s in ["cluster", "site", "random", "temporal"]},
        shap=json.loads(pd.read_csv(OUT / "shap_importance.csv").to_json(orient="records")),
        correlogram=summary["correlogram"],
        coverage=json.loads(pd.read_csv(OUT / "uncertainty_coverage.csv").to_json(orient="records")),
        calibration=json.loads(pd.read_csv(OUT / "calibration_report.csv").to_json(orient="records")),
        ablation=json.loads(pd.read_csv(OUT / "ablation_metrics.csv").to_json(orient="records")),
        per_site=json.loads(pd.read_csv(OUT / "per_site_metrics.csv").to_json(orient="records")),
        stack=summary["stack"],
    )
    export.write_json("validation/validation.json", val)

    # ---- stations ----
    o = oof[["site_id", "date"]].copy()
    o["cv_pred"] = np.exp(oof["cluster__stack_rk"]) * smear
    o["stage1_pred"] = np.exp(oof["cluster__stack"]) * smear
    obs_all = table.copy()
    # include the latest NRT observations when present
    nrt_obs = config.INTERIM / "nrt" / "obs_daily_calibrated.parquet"
    if nrt_obs.exists():
        n = pd.read_parquet(nrt_obs, columns=["site_id", "date", "pm25", "pm25_raw"])
        obs_all = pd.concat([obs_all, n[~n.set_index(["site_id", "date"]).index.isin(obs_all.set_index(["site_id", "date"]).index)]])
    export.write_station_data(stations, obs_all.sort_values(["site_id", "date"]), o)
    export.write_boundaries()

    # ---- manifest ----
    stats = pd.read_parquet(OUT / "region_daily_stats.parquet")
    nrt_stats = OUT / "region_daily_stats_nrt.parquet"
    if nrt_stats.exists():
        s2 = pd.read_parquet(nrt_stats)
        stats = pd.concat([stats[~stats.date.isin(s2.date.unique())], s2], ignore_index=True)
        export.write_region_series(stats)
    dates = sorted(pd.to_datetime(stats.date.unique()))
    tiers_f = OUT / "tiers.json"
    tiers = json.loads(tiers_f.read_text()) if tiers_f.exists() else {}
    n_obs = stats[stats.level == "adm0"].set_index("date")["n_obs"] if "n_obs" in stats else pd.Series(dtype=int)
    lons, lats = config.grid_axes()
    fin = met[(met.scheme == "cluster") & (met.model == "stack_rk")].set_index("subset")
    cams = met[(met.scheme == "cluster") & (met.model == "cams_raw")].set_index("subset")
    hl = ""
    if "all" in fin.index:
        hl = (f"Across all {int(fin.loc['all','n']):,} station-days, the final model reaches R² = {fin.loc['all','r2']:.2f}, "
              f"RMSE = {fin.loc['all','rmse']:.1f} µg/m³ and MAE = {fin.loc['all','mae']:.1f} µg/m³, against R² = {cams.loc['all','r2']:.2f} and "
              f"RMSE = {cams.loc['all','rmse']:.1f} µg/m³ for raw CAMS.")
        if "ghana" in fin.index:
            hl += f" For Ghanaian stations: R² = {fin.loc['ghana','r2']:.2f}, RMSE = {fin.loc['ghana','rmse']:.1f} µg/m³."
    freshness = json.loads((OUT / "freshness.json").read_text()) if (OUT / "freshness.json").exists() else {}
    manifest = dict(
        product="GH-PM25", version=__version__, generated_at=pd.Timestamp.utcnow().strftime("%Y-%m-%dT%H:%M:%S"),
        dates=[d.strftime("%Y-%m-%d") for d in dates], latest=max(d for d in dates if tiers.get(d.strftime("%Y-%m-%d")) != "forecast").strftime("%Y-%m-%d"),
        tiers=tiers, n_obs={d.strftime("%Y-%m-%d"): int(v) for d, v in n_obs.items()},
        grid=dict(w=len(lons), h=len(lats), res=config.RES, lon0=config.LON_MIN, lon1=config.LON_MAX, lat0=config.LAT_MIN, lat1=config.LAT_MAX),
        training=dict(n_train=summary["n_train"], n_sites=summary["n_sites"], period=summary["period"]),
        headline=dict(text=hl), sources=SOURCES, freshness=freshness,
    )
    export.write_json("manifest.json", manifest)
    # technical report alongside the portal
    rep = config.REPORT / "GH-PM25_Technical_Report.html"
    if rep.exists():
        (config.PORTAL / "report").mkdir(exist_ok=True)
        shutil.copy(rep, config.PORTAL / "report" / rep.name)
        if (config.REPORT / "figures").exists():
            shutil.copytree(config.REPORT / "figures", config.PORTAL / "report" / "figures", dirs_exist_ok=True)
    print("portal data written:", len(dates), "days; latest", manifest["latest"])


if __name__ == "__main__":
    main()
