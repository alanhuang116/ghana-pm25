"""QA of the fire record.

1. Orbit-model granule selection vs a full scan of the overpass windows (EFIRE, S-NPP) on
   independent days across seasons: fraction of detections / FRP captured.
2. Product consistency in the 2024 overlap: daily regional FRP from EFIRE vs NASA FIRMS VNP14IMG.
"""
from __future__ import annotations

import datetime as dt
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ghana_pm25 import config  # noqa: E402
from ghana_pm25.sources import firms  # noqa: E402

QA_DAYS = [dt.date(2025, 1, 9), dt.date(2025, 2, 21), dt.date(2025, 4, 2), dt.date(2025, 7, 15), dt.date(2025, 11, 30),
           dt.date(2025, 12, 23), dt.date(2026, 1, 17), dt.date(2026, 2, 8), dt.date(2026, 3, 26), dt.date(2026, 8, 20)]
OVERLAP = [d.date() for d in pd.date_range("2024-11-20", "2024-12-31", freq="3D")] + [d.date() for d in pd.date_range("2024-03-01", "2024-03-30", freq="5D")]


def selection_qa():
    rows = []
    for day in QA_DAYS:
        sel_df, diag = firms.fires_for_day(day, use_cache=False)
        keys = [k for k in firms._list_day(day) if firms._in_windows(k)]
        with ThreadPoolExecutor(48) as ex:
            full = [firms._regional(g) for g in ex.map(firms.read_granule, keys) if g is not None]
        if not full:
            continue
        full = pd.concat(full, ignore_index=True); full["date"] = pd.Timestamp(day)
        fc, sc = firms.clean_edr(full), firms.clean_edr(sel_df) if len(sel_df) else full.iloc[:0]
        rows.append(dict(date=str(day), mode=diag.get("mode"), granules_window=len(keys), granules_selected=diag.get("n_selected"),
                         fires_full=len(fc), fires_selected=len(sc), frp_full=round(float(fc.frp.sum()), 1), frp_selected=round(float(sc.frp.sum()), 1),
                         capture_count=len(sc) / max(len(fc), 1), capture_frp=float(sc.frp.sum()) / max(float(fc.frp.sum()), 1e-6)))
        print(rows[-1], flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(config.OUTPUTS / "qa_fire_selection.csv", index=False)
    return df


def overlap_qa():
    f24 = firms.firms_archive(2024, "2024-01-01", "2024-12-31")
    rows = []
    for day in OVERLAP:
        e, _ = firms.fires_for_day(day)
        e = firms.clean_edr(e) if len(e) else e
        fm = f24[f24.date == pd.Timestamp(day)]
        rows.append(dict(date=str(day), efire_n=len(e), firms_n=len(fm), efire_frp=float(e.frp.sum()) if len(e) else 0.0, firms_frp=float(fm.frp.sum())))
        print(rows[-1], flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(config.OUTPUTS / "qa_fire_efire_vs_firms.csv", index=False)
    ok = (df.firms_frp > 0) & (df.efire_frp > 0)
    r = np.corrcoef(np.log(df.firms_frp[ok]), np.log(df.efire_frp[ok]))[0, 1]
    print(f"log-FRP correlation {r:.3f}; median EFIRE/FIRMS FRP ratio {np.median(df.efire_frp[ok] / df.firms_frp[ok]):.3f}; "
          f"median count ratio {np.median(df.efire_n[ok] / np.maximum(df.firms_n[ok], 1)):.3f}")
    return df


if __name__ == "__main__":
    print(selection_qa().round(3).to_string())
    overlap_qa()
