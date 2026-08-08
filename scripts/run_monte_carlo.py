"""Local Monte Carlo driver.

Defaults reproduce the historical behaviour exactly: 1000 iterations against the
5 arc-minute CSV, per-iteration zone/district/country Parquet dumps, and the two
pickles the notebooks read.

``--resolution 30as`` switches to the ADR-5 profile: aggregate-only persistence
(no per-cell Parquet, no pickles -- at ~140M cells those are terabytes and
tens of GB respectively), and the per-cell frame is trimmed to the three columns
anything downstream actually uses before it ever leaves the worker.

    python scripts/run_monte_carlo.py
    python scripts/run_monte_carlo.py --iterations 10 --workers 4
    python scripts/run_monte_carlo.py --resolution 30as --iterations 10 \
        --input data/GIS/extracted/master_30as.parquet
"""

import argparse
import concurrent.futures
from pathlib import Path
import sys
from tqdm import tqdm
import time

# Resolve project root and update sys.path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
import src.gis_global_module as gis  # noqa
import src.gis_monte_carlo as mc  # noqa
from src.monte_carlo_constants import sample_parameters  # noqa

# Results path
RESULTS_PATH = project_root / "results"
PARQUET_PATH = RESULTS_PATH / "parquet_files"

# -------------------------------------------------------------------------------
# DEFINE MONTE CARLO SIMULATION PARAMETERS
# -------------------------------------------------------------------------------

# Define the number of simulations to run
# Expect ~15-20 minutes for one simulation,
# but multiprocessing will speed up large batches signimficantly
NUM_ITERATIONS = 1000

# Define maximum simultaneous processes to run for multiprocessing
# 15 was the most that could run on a 32 core hyperthreaded machine
MAX_WORKERS = 15

# At 30 arc-seconds each worker holds a ~140M-row frame, so the parallelism that
# is free at 5 arc-minutes is what blows the 24 GB budget (ADR-5). Run one draw
# at a time unless the operator says otherwise.
MAX_WORKERS_30AS = 1

# The only per-cell columns anything downstream reads. Everything else is
# dropped inside the worker so it never crosses a process boundary.
ZONE_RESULT_COLUMNS = ["fid", "zone_pop_with_water", "zone_pop_without_water"]

# -------------------------------------------------------------------------------

# The walking and cycling sampling parameters live in
# src/monte_carlo_constants.py so that this script, run_monte_carlo_gcp.py and
# run_monte_carlo_test.py can never disagree about them again (ADR-7).

# -------------------------------------------------------------------------------


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--resolution",
        choices=("5min", "30as"),
        default="5min",
        help=(
            "grid resolution of the input. '5min' (default) keeps the historical "
            "behaviour; '30as' switches to aggregate-only persistence per ADR-5."
        ),
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=NUM_ITERATIONS,
        help=f"number of Monte Carlo draws (default {NUM_ITERATIONS})",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help=(
            "zone frame to run against: a CSV, a .parquet file, or a directory "
            "holding a partitioned Parquet dataset. Defaults to the 5 arc-minute "
            "CSV the model has always used."
        ),
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help=(
            f"simultaneous worker processes (default {MAX_WORKERS} at 5min, "
            f"{MAX_WORKERS_30AS} at 30as -- a 30as worker holds a ~140M-row frame)"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=RESULTS_PATH,
        help="directory for the aggregate CSVs (default results/)",
    )
    parser.add_argument(
        "--save-zone-parquet",
        dest="save_zone_parquet",
        action="store_true",
        default=None,
        help="force per-iteration per-cell Parquet dumps on (default: on at 5min)",
    )
    parser.add_argument(
        "--no-save-zone-parquet",
        dest="save_zone_parquet",
        action="store_false",
        help="force them off (default: off at 30as, where they would be terabytes)",
    )
    parser.add_argument(
        "--plot",
        dest="plot",
        action="store_true",
        default=None,
        help="show the country choropleths at the end (default: on at 5min)",
    )
    parser.add_argument(
        "--no-plot",
        dest="plot",
        action="store_false",
        help="suppress them (default at 30as, and what you want on a headless box)",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    is_30as = args.resolution == "30as"
    workers = (
        args.workers
        if args.workers is not None
        else (MAX_WORKERS_30AS if is_30as else MAX_WORKERS)
    )
    save_zone_parquet = (
        args.save_zone_parquet if args.save_zone_parquet is not None else not is_30as
    )
    # ADR-5 / spec: the pickles are read by two notebooks only. Keep them at
    # 5 arc-minutes, skip them at 30 arc-seconds.
    write_pickle = not is_30as
    plot = args.plot if args.plot is not None else not is_30as

    if args.input is not None:
        gis.URB_DATA_FILE = args.input
    print(f"Resolution: {args.resolution}")
    print(f"Zone frame: {gis.URB_DATA_FILE}")
    if is_30as:
        print(
            "30as profile: persisting per-draw country/district aggregates only "
            "(no per-cell Parquet, no simulation pickles)."
        )

    output_dir = args.output_dir
    # Per-draw aggregates live beside the summary CSVs, so --output-dir moves the
    # whole run rather than half of it.
    parquet_path = output_dir / PARQUET_PATH.name
    output_dir.mkdir(parents=True, exist_ok=True)
    parquet_path.mkdir(parents=True, exist_ok=True)

    # Monte Carlo parameters (canonical distributions, see ADR-7)
    parameters = sample_parameters(args.iterations)
    crr_adjustments = parameters["crr_adjustments"]
    time_gatherings = parameters["time_gatherings"]
    practical_limits_bicycle = parameters["practical_limits_bicycle"]
    practical_limits_buckets = parameters["practical_limits_buckets"]
    mets = parameters["mets"]
    watts_values = parameters["watts_values"]
    hill_polarities = parameters["hill_polarities"]
    urban_adjustments = parameters["urban_adjustments"]
    rural_adjustments = parameters["rural_adjustments"]

    print(crr_adjustments)
    print(time_gatherings)
    print(practical_limits_bicycle)
    print(mets)
    print(watts_values)
    print(hill_polarities)
    print(urban_adjustments)
    print(rural_adjustments)

    # Initialize lists to store results from each output
    districts_simulation_results = []
    countries_simulation_results = []

    # Record the start time
    start_time = time.time()
    print("Starting Monte Carlo simulations...")
    print(f"Running {args.iterations} simulations...")
    print(f"Running {workers} simulations concurrently...")
    print("Start time:", time.strftime("%H:%M:%S", time.localtime()))
    print("\n\n")

    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
        # Submit all simulations to the executor
        futures = [
            executor.submit(
                mc.run_simulation,
                crr_adjustment,
                time_gathering_water,
                practical_limit_bicycle,
                practical_limit_buckets,
                met,
                watts,
                hill_polarity,
                urban_adjustment,
                rural_adjustment,
                use_sample_data=False,  # Enable sample data
                # ADR-5: trim the per-cell frame inside the worker, so a
                # multi-GB zone frame is never pickled back to the parent.
                zone_columns=ZONE_RESULT_COLUMNS if save_zone_parquet else [],
            )
            for (
                crr_adjustment,
                time_gathering_water,
                practical_limit_bicycle,
                practical_limit_buckets,
                met,
                watts,
                hill_polarity,
                urban_adjustment,
                rural_adjustment,
            ) in zip(
                crr_adjustments,
                time_gatherings,
                practical_limits_bicycle,
                practical_limits_buckets,
                mets,
                watts_values,
                hill_polarities,
                urban_adjustments,
                rural_adjustments,  # type: ignore
            )
        ]

        # Initialize tqdm progress bar
        futures_progress = tqdm(
            concurrent.futures.as_completed(futures),
            total=len(futures),
            desc="Simulating",
        )

        for i, future in enumerate(concurrent.futures.as_completed(futures)):
            countries_result, district_result, zone_result = future.result()

            # save the results
            districts_simulation_results.append(district_result)
            countries_simulation_results.append(countries_result)

            # Save results to Parquet files for interruption resilience
            if save_zone_parquet:
                output_file = parquet_path / f"zone_simulation_result_{i}.parquet"
                zone_result[ZONE_RESULT_COLUMNS].to_parquet(output_file, index=False)

            # Always save district and countries results progressively: these
            # are the ~3k-row aggregates ADR-5 says to persist.
            district_file = parquet_path / f"district_simulation_result_{i}.parquet"
            district_result.to_parquet(district_file, index=False)

            countries_file = parquet_path / f"countries_simulation_result_{i}.parquet"
            countries_result.to_parquet(countries_file, index=False)

            del zone_result
            futures_progress.update()  # Update the progress bar

    futures_progress.close()  # Close the progress bar

    mc.process_mc_results(
        countries_simulation_results,
        plot=plot,
        output_dir=str(output_dir),
        write_pickle=write_pickle,
    )
    mc.process_districts_results(
        districts_simulation_results,
        output_dir=str(output_dir),
        write_pickle=write_pickle,
    )

    # Record the end time
    end_time = time.time()

    # Calculate and print the time taken by the simulations in minutes and
    # hours
    time_taken = end_time - start_time
    print(f"Time taken: {time_taken / 60:.2f} minutes")
    print(f"Time taken: {time_taken / 3600:.2f} hours")


if __name__ == "__main__":
    main()
