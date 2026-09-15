"""Save / restore the operational pipeline state as a GitHub Release asset (tag `pipeline-state`).

The daily GitHub Actions job starts from a clean runner: it restores the fitted model, static grids,
input time series and run bookkeeping, runs scripts/run_daily.py, then saves the updated state.

    python scripts/09_pipeline_state.py save [--upload]
    python scripts/09_pipeline_state.py restore
"""
from __future__ import annotations

import os
import subprocess
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TAG = "pipeline-state"
ASSET = "pipeline-state.tar.gz"
DIST = ROOT / "dist"

PATHS = [
    "models/ghpm25_model.pkl", "models/ghpm25_model_meta.json", "models/lcs_calibration.json", "models/hyperparams.json",
    "data/processed/stations.parquet", "data/processed/station_static.parquet", "data/processed/training_table.parquet",
    "data/processed/grid_admin.nc", "data/processed/static/static_ghana.nc",
    "data/interim/cams_daily.parquet", "data/interim/cams_nodes.parquet", "data/interim/power_daily.parquet",
    "data/interim/firms_hist.parquet", "data/interim/nrt/obs_daily_calibrated.parquet",
    "data/raw/boundaries", "data/raw/metadata/openaq_westafrica_locations.tsv",
    "outputs/region_daily_stats.parquet", "outputs/tiers.json", "outputs/freshness.json", "outputs/run_summary.json",
    "outputs/cv_oof.parquet", "outputs/cv_metrics.csv", "outputs/calibration_report.csv", "outputs/uncertainty_coverage.csv",
    "outputs/ablation_metrics.csv", "outputs/shap_importance.csv", "outputs/per_site_metrics.csv",
]


def repo_slug() -> str:
    if os.environ.get("GITHUB_REPOSITORY"):
        return os.environ["GITHUB_REPOSITORY"]
    url = subprocess.run(["git", "config", "--get", "remote.origin.url"], capture_output=True, text=True, cwd=ROOT).stdout.strip()
    return url.rstrip("/").removesuffix(".git").split("github.com")[-1].lstrip(":/")


def gh(*args, check=True):
    return subprocess.run(["gh", *args], capture_output=True, text=True, check=check, cwd=ROOT)


def save(upload: bool):
    DIST.mkdir(exist_ok=True)
    out = DIST / ASSET
    with tarfile.open(out, "w:gz", compresslevel=6) as tar:
        for p in PATHS:
            if (ROOT / p).exists():
                tar.add(ROOT / p, arcname=p)
            else:
                print("  (missing, skipped)", p)
    print(f"state bundle {out.stat().st_size / 1e6:.0f} MB")
    if upload:
        repo = repo_slug()
        if gh("release", "view", TAG, "-R", repo, check=False).returncode != 0:
            gh("release", "create", TAG, "-R", repo, "--title", "Pipeline state (internal)",
               "--notes", "Operational state used by the daily GitHub Actions run (model, grids, input caches). Not a data product.", "--prerelease")
        gh("release", "upload", TAG, "-R", repo, "--clobber", str(out))
        print("uploaded", ASSET)


def restore():
    DIST.mkdir(exist_ok=True)
    out = DIST / ASSET
    gh("release", "download", TAG, "-R", repo_slug(), "-p", ASSET, "-O", str(out), "--clobber")
    with tarfile.open(out, "r:gz") as tar:
        tar.extractall(ROOT, filter="data")
    out.unlink()
    print("state restored")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "save":
        save("--upload" in sys.argv)
    elif cmd == "restore":
        restore()
    else:
        print(__doc__)
