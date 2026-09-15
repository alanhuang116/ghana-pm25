# GH-PM25: Daily 1-km PM2.5 for Ghana

GH-PM25 produces a daily map of fine particulate matter (PM2.5) for all of Ghana on a 0.01° (~1.1 km) grid. It combines ground monitors with CAMS atmospheric composition, MERRA-2 weather, VIIRS fire detections and kilometre-scale land-surface data. A hybrid machine-learning and geostatistical model does the downscaling, and every pixel gets a calibrated 90% prediction interval. The system runs once a day and publishes to a static web portal.

* **Live portal:** https://alanhuang116.github.io/ghana-pm25/ (updated automatically every day at 06:30 UTC)
* **Download the 1-km dataset:**
  * From the **Download data** tab in the portal. It builds a GeoTIFF or CSV for any day in your browser.
  * From **[GitHub Releases → `data`](https://github.com/alanhuang116/ghana-pm25/releases/tag/data)**, which holds:
    * monthly zips of daily NetCDF-4 files;
    * monthly and annual mean GeoTIFFs;
    * regional and district daily statistics;
    * a data dictionary.
* **Technical report:** https://alanhuang116.github.io/ghana-pm25/report/GH-PM25_Technical_Report.html. The source is in `report/`.
* **Local run:** `python -m http.server 8765 --directory portal`. Daily NetCDF files are written to `outputs/daily/YYYY/GH-PM25_1km_YYYYMMDD.nc`.

### Permanent operation on GitHub

The `.github/workflows/daily.yml` job runs every day on GitHub Actions. It needs no local machine.

1. Restore the pipeline state (fitted model, static grids, input caches) from the `pipeline-state` release.
2. Refresh the inputs, then re-map D−7 to D+1 (`scripts/run_daily.py`).
3. Publish the updated months to the `data` release (`scripts/08_package_release.py`).
4. Save the pipeline state.
5. Redeploy the portal to the `gh-pages` branch.

A weekly status commit keeps the schedule from being paused for inactivity. To start a run by hand, open **Actions → GH-PM25 daily run → Run workflow**.

## Method in brief

1. **Ground truth.** Daily PM2.5 comes from the OpenAQ S3 archive (Clarity, AirGradient, Data354, IQAir and others) and from US State Department BAM-1020 monitors (via AirNow). It goes through hourly quality control and daily completeness rules. Low-cost sensors are then harmonised to the BAM reference with network-specific corrections chosen by cross-validation.
2. **Predictors.** There are about 70 in total:
   * CAMS PM2.5, PM10, dust, AOD, CO and NO₂;
   * MERRA-2 meteorology, boundary-layer depth and AOD;
   * VIIRS fire radiative power, local and upwind-weighted;
   * WorldCover land-cover fractions, Copernicus DEM, WorldPop population, VIIRS night lights and OSM roads, each with neighbourhood smoothing;
   * calendar terms.
3. **Stage 1.** Four learners: LightGBM, XGBoost (on the log-ratio to CAMS), CatBoost and ExtraTrees.
   * Hyperparameters are tuned under leave-city-cluster-out CV.
   * The ensemble is chosen from an NNLS stack and all equal-weight subsets, using a fixed multi-design criterion. The current choice is CatBoost + ExtraTrees.
4. **Stage 2.** Same-day kriging of residual anomalies after removing each site's offset.
   * Its strength λ is selected by leave-site-out CV.
   * It is currently λ = 0, because kriging propagated low-cost-sensor noise.
5. **Stage 3.** Locally adaptive split-conformal 90% intervals, plus an area-of-applicability index (Meyer & Pebesma 2021).
6. **Validation.** Four designs: random, leave-month-out, leave-site-out and leave-city-cluster-out. The last is the primary one.

## Setup

```powershell
uv venv .venv --python 3.12
uv pip install --python .venv\Scripts\python.exe -r requirements.txt
```

No API keys are required. Optionally, set `OPENAQ_API_KEY` to enable live hourly station readings.

## Full rebuild

```powershell
.venv\Scripts\python.exe scripts\01_download_ground.py         # OpenAQ + AirNow, QC, daily means
.venv\Scripts\python.exe scripts\02_build_static.py            # static 1-km covariates
.venv\Scripts\python.exe scripts\10_download_dynamic.py        # CAMS, MERRA-2 (POWER), VIIRS fires
.venv\Scripts\python.exe scripts\03_train_evaluate.py          # calibration, CV, final model
.venv\Scripts\python.exe scripts\04_predict_history.py         # daily maps for the record
.venv\Scripts\python.exe scripts\05_export_portal.py           # portal data
.venv\Scripts\python.exe scripts\06_make_figures.py            # report figures
.venv\Scripts\python.exe scripts\07_build_report.py            # technical report
```

The CAMS download is paced to stay within Open-Meteo's free quota. A first full pull takes a few hours; later runs reuse the cache.

## Daily operations

```powershell
.venv\Scripts\python.exe scripts\run_daily.py            # refresh inputs, map D-7..D+1, update portal
.venv\Scripts\python.exe scripts\run_daily.py --retrain  # also refresh ground data and retrain (monthly)
```

Each day in the portal is tagged with one of these product tiers:

| Tier | Days | What it means |
|---|---|---|
| forecast | D+1 | No observations yet |
| nowcast | D0 | Today's inputs |
| nrt | D-1 to D-6 | Re-mapped each day as late data arrive |
| final | D-7 and older | All inputs complete |

To schedule the run on Windows, use `scripts/schedule_windows_task.ps1`. To schedule it in the cloud, use `.github/workflows/daily.yml` (GitHub Actions, which also publishes to GitHub Pages).

## Repository layout

```
ghana_pm25/      package (sources/, calibration, features, model, predict, nrt, export)
scripts/         numbered pipeline steps, run_daily, QA scripts
portal/          static web portal (MapLibre GL JS + d3), data in portal/data
report/          technical report, figures, tables
data/            raw / interim / processed (cached downloads)
models/          fitted model, calibration models
outputs/         daily NetCDF, GeoTIFF aggregates, CV results, logs
```

## Limitations

* **Calibration coverage.** Ghana's only open reference monitor (US Embassy Accra) stopped regular reporting in March 2025, and there has never been one in the north. Sensor calibration therefore rests on coastal co-location, so uncertainty is larger in northern Ghana and during the Harmattan. The prediction intervals reflect this.
* **Record length.** The record starts in August 2022, when CAMS data via Open-Meteo become available.
* **Satellite AOD.** No 1-km satellite AOD is used, because MAIAC requires an Earthdata account. It can be added as an optional predictor.
