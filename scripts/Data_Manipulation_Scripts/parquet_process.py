"""Pick MC draws closest to the median / 5th / 95th of global population
without water.

Selection uses the GLOBAL row of ``countries_simulation_result_*.parquet``
(equivalent to summing ``zone_pop_*`` over a zone file).

The Monte Carlo run only stores country and district aggregates, so by
default those three draws are re-run locally (with the parameters stored in
``checkpoint.json``) to write their per-cell zone Parquet, which is copied to
the names expected by the export/rasterise scripts. Pass
``--no-rerun-zones-locally`` to only identify the median, 5th, and 95th percentile runs.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import pandas as pd

data_script_dir = Path(__file__).resolve().parent
repo_root = data_script_dir.parent.parent
sys.path.insert(0, str(repo_root))

parquet_data_dir = repo_root / "results" / "parquet_files"
checkpoint_path = parquet_data_dir / "checkpoint.json"

LABEL_MEDIAN = "median"
LABEL_5TH = "5th_percentile"
LABEL_95TH = "95th_percentile"


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--rerun-zones-locally",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="after selecting draws, re-run them locally with zone Parquet "
        "(default: on; use --no-rerun-zones-locally for selection only)",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="with zone re-run, overwrite existing zone_simulation_result_<id>.parquet",
    )
    return p.parse_args(argv)


def select_percentile_runs() -> pd.DataFrame:
    sums_list = []
    for path in sorted(parquet_data_dir.glob("countries_simulation_result_*.parquet")):
        run_id = int(path.stem.rsplit("_", 1)[-1])
        df = pd.read_parquet(
            path,
            columns=["ISOCODE", "country_pop_with_water", "country_pop_without_water"],
        )
        global_row = df.loc[df["ISOCODE"] == "GLOBAL"]
        if global_row.empty:
            print(f"No GLOBAL row in {path.name}; skipping.")
            continue
        row = global_row.iloc[0]
        sums_list.append(
            {
                "run_id": run_id,
                "sum_zone_pop_with_water": float(row["country_pop_with_water"]),
                "sum_zone_pop_without_water": float(row["country_pop_without_water"]),
            }
        )

    if not sums_list:
        raise SystemExit(
            f"No countries_simulation_result_*.parquet files in {parquet_data_dir}"
        )

    sums_df = pd.DataFrame(sums_list).sort_values("run_id").reset_index(drop=True)
    sums_df.to_csv(parquet_data_dir / "sums_of_zone_populations.csv", index=False)

    median_without = sums_df["sum_zone_pop_without_water"].median()
    p5 = sums_df["sum_zone_pop_without_water"].quantile(0.05)
    p95 = sums_df["sum_zone_pop_without_water"].quantile(0.95)

    sums_df["diff_median"] = (
        sums_df["sum_zone_pop_without_water"] - median_without
    ).abs()
    sums_df["diff_5"] = (sums_df["sum_zone_pop_without_water"] - p5).abs()
    sums_df["diff_95"] = (sums_df["sum_zone_pop_without_water"] - p95).abs()

    median_run = int(sums_df.loc[sums_df["diff_median"].idxmin(), "run_id"])
    run_5th = int(sums_df.loc[sums_df["diff_5"].idxmin(), "run_id"])
    run_95th = int(sums_df.loc[sums_df["diff_95"].idxmin(), "run_id"])

    print(f"Runs found: {len(sums_df)}")
    print(f"Median pop without water: {median_without:,.0f}")
    print(f"  median draw: {median_run}")
    print(f"  5th percentile draw: {run_5th}")
    print(f"  95th percentile draw: {run_95th}")

    selected = pd.DataFrame(
        [
            {
                "run_median_without_water": median_run,
                "run_5th_without_water": run_5th,
                "run_95th_without_water": run_95th,
            }
        ]
    )
    out = parquet_data_dir / "percentile_runs_of_zone_populations.csv"
    selected.to_csv(out, index=False)
    print(f"Saved {out}")
    return selected


def _job_for_draw(run_id: int, parameters: dict, run_config: dict, chunk_plan):
    return {
        "index": run_id,
        "zone_file": str(run_config["zone_file"]),
        "crr_adjustment": parameters["crr_adjustments"][run_id],
        "time_gathering_water": parameters["time_gatherings"][run_id],
        "practical_limit_bicycle": parameters["practical_limits_bicycle"][run_id],
        "practical_limit_buckets": parameters["practical_limits_buckets"][run_id],
        "met": parameters["mets"][run_id],
        "watts": parameters["watts_values"][run_id],
        "hill_polarity": parameters["hill_polarities"][run_id],
        "urban_adjustment": parameters["urban_adjustments"][run_id],
        "rural_adjustment": parameters["rural_adjustments"][run_id],
        "urban_borehole_lose_fraction": parameters["urban_borehole_lose_fractions"][
            run_id
        ],
        "rural_borehole_lose_fraction": parameters["rural_borehole_lose_fractions"][
            run_id
        ],
        "unallocated_vulnerable_fraction": parameters[
            "unallocated_vulnerable_fractions"
        ][run_id],
        "save_zone_parquet": True,
        "chunk_rows": run_config.get("chunk_rows") or None,
        "chunk_plan": chunk_plan,
    }


def rerun_zone_draws_locally(selected: pd.DataFrame, *, force: bool = False) -> None:
    if not checkpoint_path.exists():
        raise SystemExit(
            f"Missing {checkpoint_path}; need checkpoint parameters to re-run "
            "the selected draws."
        )

    checkpoint = json.loads(checkpoint_path.read_text())
    parameters = checkpoint["parameters"]
    run_config = checkpoint["run_config"]
    zone_file = Path(run_config["zone_file"])
    if not zone_file.is_absolute():
        zone_file = repo_root / zone_file
    if not zone_file.exists():
        raise SystemExit(f"Zone input not found: {zone_file}")

    import src.chunked_analysis as chunked
    import src.gis_global_module as gis
    import scripts.run_monte_carlo_gcp as gcp_mc

    gis.URB_DATA_FILE = zone_file
    gcp_mc.RESULTS_PATH = parquet_data_dir.parent
    gcp_mc.PARQUET_PATH = parquet_data_dir
    gcp_mc.CHECKPOINT_FILE = checkpoint_path

    chunk_rows = int(run_config.get("chunk_rows") or 0)
    chunk_plan = None
    if chunk_rows:
        print(f"Planning chunks (budget {chunk_rows:,}) ...", flush=True)
        chunk_plan = chunked.plan_chunks(zone_file, max_rows_per_chunk=chunk_rows)
        print(f"  {chunk_plan.describe()}", flush=True)

    label_by_id = {}
    for run_id, label in (
        (int(selected.at[0, "run_median_without_water"]), LABEL_MEDIAN),
        (int(selected.at[0, "run_5th_without_water"]), LABEL_5TH),
        (int(selected.at[0, "run_95th_without_water"]), LABEL_95TH),
    ):
        label_by_id.setdefault(run_id, []).append(label)

    for run_id, labels in label_by_id.items():
        zone_out = parquet_data_dir / f"zone_simulation_result_{run_id}.parquet"
        if zone_out.exists() and not force:
            print(f"Zone file already exists for draw {run_id}; skipping re-run.")
        else:
            if zone_out.exists() and force:
                zone_out.unlink()
            print(
                f"\nRe-running draw {run_id} with zone Parquet "
                f"({', '.join(labels)}) ..."
            )
            job = _job_for_draw(
                run_id,
                parameters,
                {**run_config, "zone_file": zone_file},
                chunk_plan,
            )
            gcp_mc.run_and_save_simulation(job)
            if not zone_out.exists():
                raise SystemExit(
                    f"Zone re-run finished but {zone_out} was not written."
                )
            print(f"  wrote {zone_out}")

        for label in labels:
            dest = parquet_data_dir / f"zone_simulation_result_{label}.parquet"
            shutil.copy2(zone_out, dest)
            print(f"  copied -> {dest.name}")


def main(argv=None) -> None:
    args = parse_args(argv)
    selected = select_percentile_runs()
    if args.rerun_zones_locally:
        rerun_zone_draws_locally(selected, force=args.force)
        print(
            "\nDone. Next: "
            "python scripts/Data_Manipulation_Scripts/export_results_with_centroids.py "
            "&& python scripts/Data_Manipulation_Scripts/rasterise_results.py"
        )
    else:
        print(
            "\nSelection only (--no-rerun-zones-locally). To re-run zone maps:\n"
            "  python scripts/Data_Manipulation_Scripts/parquet_process.py"
        )


if __name__ == "__main__":
    main()
