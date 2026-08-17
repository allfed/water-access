"""Local Monte Carlo driver.

Defaults reproduce the historical *harness*: 1000 iterations against the
5 arc-minute CSV, per-iteration zone/district/country Parquet dumps, and the two
pickles the notebooks read. The one deliberate model change since is the
cos-latitude population split, on by default since 2026-08-09 --
``--no-area-weighted-population`` restores the pre-correction placement.

``--resolution 30as`` switches to the 30 arc-second profile: aggregate-only persistence
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

import numpy as np

# Resolve project root and update sys.path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
import src.gis_global_module as gis  # noqa
import src.gis_monte_carlo as mc  # noqa
from src.monte_carlo_parameters import PARAMETER_NAMES, sample_parameters  # noqa

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
# is free at 5 arc-minutes is what blows the 24 GB budget. Run one draw
# at a time unless the operator says otherwise.
MAX_WORKERS_30AS = 1

# Country-group chunk budget at 30 arc-seconds. See
# src/chunked_analysis.DEFAULT_CHUNK_ROWS for the calibration behind the number:
# peak RSS ~= 1.0 + 1.2 GB per million modelled cells in the largest chunk, so
# 12M projects to ~15 GB against the 24 GB memory budget.
CHUNK_ROWS_30AS = 12_000_000

# The only per-cell columns anything downstream reads. Everything else is
# dropped inside the worker so it never crosses a process boundary.
ZONE_RESULT_COLUMNS = ["fid", "zone_pop_with_water", "zone_pop_without_water"]

# -------------------------------------------------------------------------------

# The walking and cycling sampling parameters live in
# src/monte_carlo_parameters.py so that this script, run_monte_carlo_gcp.py and
# run_monte_carlo_test.py can never disagree about them again.

# -------------------------------------------------------------------------------


def _init_worker(zone_file):
    """Point a freshly spawned worker at the right zone frame.

    ``--input`` rebinds a module-level constant, and on macOS (and any Python
    3.14+ default) ProcessPoolExecutor *spawns* workers, so they re-import the
    module and get the packaged default rather than the parent's override.
    Without this the parent plans against one file and every worker silently
    analyses another.
    """
    if zone_file is not None:
        import src.gis_global_module as worker_gis

        worker_gis.URB_DATA_FILE = zone_file


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
            "behaviour; '30as' switches to aggregate-only persistence."
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
        "--chunk-rows",
        type=int,
        default=None,
        help=(
            "run each draw one group of whole countries at a time, at most this "
            "many modelled cells per group. Peak memory then follows this "
            f"number, not the grid size. Default: {CHUNK_ROWS_30AS:,} at 30as, "
            "off at 5min. 0 disables it."
        ),
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
    parser.add_argument(
        "--area-weighted-population",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "split each country's population by pop_density x cos(latitude) "
            "rather than pop_density alone, correcting the lat/long grid's "
            "poleward cell-area shrinkage. Default: on (2026-08-09). "
            "--no-area-weighted-population reproduces the pre-correction "
            "behaviour every earlier published result was produced under."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help=(
            "seed the Monte Carlo parameter draws, so that two runs consume "
            "IDENTICAL parameter tuples -- what the 30as/5min twin comparison "
            "needs. Default: unseeded, exactly as before."
        ),
    )
    return parser.parse_args(argv)


def format_parameter_draws(parameters, num_iterations):
    """Render one ``PARAM_DRAW`` line per draw, for log-to-log diffing.

    The 30as/5min twin comparison compares a 30 arc-second run against a 5 arc-minute twin and needs
    evidence that the only difference between them was the resolution. Both run
    logs carry these lines, so ``diff <(grep PARAM_DRAW a.log) <(grep PARAM_DRAW
    b.log)`` is the proof.
    """
    lines = []
    columns = [parameters[name] for name in PARAMETER_NAMES]
    for i in range(num_iterations):
        cells = []
        for name, column in zip(PARAMETER_NAMES, columns):
            value = column[i]
            # ``hill_polarities`` is a string; everything else is numeric and
            # printed at enough precision to round-trip a comparison.
            cells.append(
                f"{name}={value}"
                if isinstance(value, str)
                else f"{name}={float(value):.12g}"
            )
        lines.append(f"PARAM_DRAW {i:03d} " + " ".join(cells))
    return lines


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
    # The pickles are read by two notebooks only. Keep them at
    # 5 arc-minutes, skip them at 30 arc-seconds.
    write_pickle = not is_30as
    plot = args.plot if args.plot is not None else not is_30as

    if args.input is not None:
        gis.URB_DATA_FILE = args.input
    print(f"Resolution: {args.resolution}")
    print(f"Zone frame: {gis.URB_DATA_FILE}")
    print(
        "Population split: "
        + (
            "pop_density x cos(latitude) (area weighted)"
            if args.area_weighted_population
            else "pop_density only (historical)"
        )
    )

    chunk_rows = (
        args.chunk_rows
        if args.chunk_rows is not None
        else (CHUNK_ROWS_30AS if is_30as else 0)
    )
    chunk_plan = None
    if chunk_rows:
        # Plan once, here, and hand the same plan to every draw: it depends only
        # on the input file, and building it re-reads three columns of the whole
        # grid.
        import src.chunked_analysis as chunked

        print(f"Planning country chunks (budget {chunk_rows:,} cells) ...", flush=True)
        chunk_plan = chunked.plan_chunks(
            gis.URB_DATA_FILE,
            max_rows_per_chunk=chunk_rows,
            area_weighted_population=args.area_weighted_population,
        )
        print(f"  {chunk_plan.describe()}", flush=True)
        largest_district = chunk_plan.notes.get("largest_district_rows", 0)
        if largest_district > chunk_rows:
            print(
                f"  WARNING: one district holds {largest_district:,} cells, more "
                f"than the {chunk_rows:,} budget. A district is the smallest work "
                "unit this scheme can make, so that chunk will overrun.",
                flush=True,
            )

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

    # Monte Carlo parameters (canonical distributions; see monte_carlo_parameters.py)
    if args.seed is not None:
        # sample_parameters draws through the legacy global RNG (numpy's, which
        # is also what scipy's ``.rvs`` uses by default), so seeding it here
        # fixes every one of the nine parameters. Same trick as
        # scripts/validate_lut.py.
        np.random.seed(args.seed)
        print(f"Parameter seed: {args.seed}")
    else:
        print("Parameter seed: none (unseeded draws)")
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
    urban_borehole_lose_fractions = parameters["urban_borehole_lose_fractions"]
    rural_borehole_lose_fractions = parameters["rural_borehole_lose_fractions"]
    unallocated_vulnerable_fractions = parameters["unallocated_vulnerable_fractions"]

    # One line per draw, all parameters, so two run logs can be diffed directly.
    for line in format_parameter_draws(parameters, args.iterations):
        print(line)

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

    with concurrent.futures.ProcessPoolExecutor(
        max_workers=workers,
        initializer=_init_worker,
        initargs=(gis.URB_DATA_FILE,),
    ) as executor:
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
                urban_borehole_lose_fraction,
                rural_borehole_lose_fraction,
                unallocated_vulnerable_fraction,
                use_sample_data=False,  # Enable sample data
                # Trim the per-cell frame inside the worker so a multi-GB zone
                # frame is never pickled back to the parent.
                zone_columns=ZONE_RESULT_COLUMNS if save_zone_parquet else [],
                chunk_rows=chunk_rows or None,
                chunk_plan=chunk_plan,
                area_weighted_population=args.area_weighted_population,
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
                urban_borehole_lose_fraction,
                rural_borehole_lose_fraction,
                unallocated_vulnerable_fraction,
            ) in zip(
                crr_adjustments,
                time_gatherings,
                practical_limits_bicycle,
                practical_limits_buckets,
                mets,
                watts_values,
                hill_polarities,
                urban_adjustments,
                rural_adjustments,
                urban_borehole_lose_fractions,
                rural_borehole_lose_fractions,
                unallocated_vulnerable_fractions,
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
            # are the ~3k-row aggregates to persist.
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
