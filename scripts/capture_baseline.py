#!/usr/bin/env python3
"""Freeze a fixed-parameter run of the model as a regression baseline (AC-2).

Runs ``run_global_analysis`` at the canonical fixed parameters on the full
5 arc-minute dataset (``data/GIS/updated_GIS_output_cleaned.csv``) and writes the
country- and district-level frames. The committed copy under
``results/regression_baseline/`` is the PRE-refactor reference that Gate 2
(AC-5) diffs against, so do not regenerate it after touching the solver -- point
``--out`` somewhere else instead.

    python scripts/capture_baseline.py                      # writes the baseline
    python scripts/capture_baseline.py --out results/regression_new
"""

import argparse
import json
import resource
import sys
import time
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
import src.gis_global_module as gis  # noqa: E402
from src.gis_global_module import run_global_analysis  # noqa: E402

# Fixed parameters, verbatim from AC-2. Changing any of these invalidates the
# committed baseline.
BASELINE_PARAMETERS = dict(
    crr_adjustment=0,
    time_gathering_water=2.84,
    practical_limit_bicycle=40,
    practical_limit_buckets=20,
    met=4,
    watts=40,
    hill_polarity="downhill_uphill",
    urban_adjustment=1.3,
    rural_adjustment=1.4,
    calculate_distance=True,
    plot=False,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=project_root / "results" / "regression_baseline",
        help="directory to write countries.csv and districts.csv into",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help=(
            "override the zone frame (CSV or Parquet). Defaults to the full "
            "5 arc-minute CSV the committed baseline was captured from."
        ),
    )
    args = parser.parse_args()

    out_dir = args.out if args.out.is_absolute() else Path.cwd() / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.input is not None:
        gis.URB_DATA_FILE = args.input

    print(f"Running run_global_analysis with {BASELINE_PARAMETERS}")
    started = time.perf_counter()
    df_countries, df_districts, _df_zones = run_global_analysis(**BASELINE_PARAMETERS)
    total_seconds = time.perf_counter() - started
    load_seconds = gis.LAST_LOAD_SECONDS

    df_countries.to_csv(out_dir / "countries.csv", index=False)
    df_districts.to_csv(out_dir / "districts.csv", index=False)

    # AC-7 measures the run excluding the initial data load, so record both.
    # Peak RSS is recorded too: it is the 5-arcmin datum Phase D extrapolates
    # from for the AC-6 30-arcsecond budget.
    peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform != "darwin":  # Linux reports kilobytes, macOS reports bytes
        peak_rss *= 1024
    timing = {
        "input": str(gis.URB_DATA_FILE),
        "total_seconds": round(total_seconds, 3),
        "load_seconds": round(load_seconds, 3),
        "seconds_excluding_load": round(total_seconds - load_seconds, 3),
        "peak_rss_gb": round(peak_rss / 1e9, 3),
        "modelled_cells": int(len(_df_zones)) if len(_df_zones.columns) else None,
    }
    (out_dir / "timing.json").write_text(json.dumps(timing, indent=2) + "\n")

    print(f"Wrote {len(df_countries)} countries to {out_dir / 'countries.csv'}")
    print(f"Wrote {len(df_districts)} districts to {out_dir / 'districts.csv'}")
    print(
        f"Total {total_seconds:.1f} s "
        f"(data load {load_seconds:.1f} s, "
        f"excluding load {total_seconds - load_seconds:.1f} s)"
    )


if __name__ == "__main__":
    main()
