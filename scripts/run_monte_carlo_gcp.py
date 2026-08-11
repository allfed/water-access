"""Monte Carlo driver for the GCP Spot VM (launched by gcp/deploy-spot.sh).

This is the PRODUCTION driver: checkpoint/resume across preemptions, per-worker
disk writes, and no large frames retained in the parent process — the only
setup that survives a 1000-iteration run. Its sibling
scripts/run_monte_carlo.py is the local/CLI harness (resolution flag, chunked
execution, seedable draws) and scripts/run_monte_carlo_test.py is a
two-iteration smoke test. All three draw their parameters from the SAME module,
src/monte_carlo_constants.py (ADR-7), so their ranges cannot drift apart again.
For a small local test run, set NUM_ITERATIONS/MAX_WORKERS below.

SCENARIO RUNS (deterministic, no sampling): set
UNALLOCATED_VULNERABLE_FRACTION_FIXED in src/monte_carlo_constants.py to hold
the unallocated-allocation parameter g constant across every iteration.
  * "prorata" — the recommended CENTRAL scenario: the unallocated population
    mirrors each stratum's known source mix (≈ +2.5% affected).
  * 0.0 — lower bracket, all unallocated people retain access (the pre-2026-08
    behaviour).
  * 1.0 — upper bracket, all unallocated people are grid-vulnerable
    (≈ +5.8% affected).
Leave it as None for the default uncertainty run, which samples
g ~ Uniform(0, 1) per iteration. See UNALLOCATED_GAP_INVESTIGATION.md §6. The
borehole lose fractions are scenario parameters too, over the HEMP ranges in
that same module (ELECTRIC_BOREHOLE_SHARE_ESTIMATE.md).

Every sampled parameter is recorded per iteration in checkpoint.json and in
sampled_parameters.csv (one row per iteration, written next to the parquet
files); iteration k's inputs are row k.
"""

import csv
import multiprocessing
import numpy as np
from pathlib import Path
import sys
from tqdm import tqdm
import time
import json
import signal
import os

# Resolve project root and update sys.path.
# Derived from __file__, not the CWD: the previous `Path().resolve().parent`
# only landed on the right directory if the script was launched from a checkout
# literally named "water-access", and silently pointed elsewhere otherwise.
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
import src.gis_monte_carlo as mc  # noqa
from src.monte_carlo_constants import sample_parameters  # noqa

# Results path
RESULTS_PATH = project_root / "results"
PARQUET_PATH = RESULTS_PATH / "parquet_files"
CHECKPOINT_FILE = PARQUET_PATH / "checkpoint.json"
SAMPLED_PARAMETERS_FILE = PARQUET_PATH / "sampled_parameters.csv"

# -------------------------------------------------------------------------------
# DEFINE MONTE CARLO SIMULATION PARAMETERS
# -------------------------------------------------------------------------------

# Define the number of simulations to run
# Expect ~15-20 minutes for one simulation,
# but multiprocessing will speed up large batches signimficantly
NUM_ITERATIONS = 1000

# Define maximum simultaneous processes to run for multiprocessing
# Each worker peaks at roughly 3 GB, so this is limited by RAM as well as
# vCPUs: 24 for n2-standard-32 (32 vCPUs / 128 GB), 4 for n2-standard-8
# (8 vCPUs / 32 GB). Reduce if the VM runs out of memory.
MAX_WORKERS = 24

# Give up on a single iteration if no result arrives within this many seconds
# (protects the run from hanging if a worker process is killed, e.g. by the
# kernel OOM killer). A normal iteration takes ~15-30 minutes.
ITERATION_TIMEOUT = 2 * 60 * 60

# -------------------------------------------------------------------------------

# Every sampling parameter -- the walking/cycling ranges, the borehole lose
# fractions and the unallocated-allocation parameter g -- lives in
# src/monte_carlo_constants.py so that this script, run_monte_carlo.py and
# run_monte_carlo_test.py can never disagree about them again (ADR-7). Edit
# ranges and scenario switches there, not here.

# -------------------------------------------------------------------------------

# Global flag for graceful shutdown
shutdown_requested = False


def handle_preemption(signum, frame):
    """Handle Spot VM preemption signal (30 second warning)"""
    global shutdown_requested
    print("\n⚠️  Preemption signal received! Saving checkpoint...")
    shutdown_requested = True


def save_sampled_parameters(parameters):
    """Write the per-iteration sampled parameters to a CSV alongside the parquets.

    checkpoint.json is the machine-readable record used for resume; this CSV is
    the human/analysis-readable one. Row k holds the inputs of iteration k, so
    it joins directly onto *_simulation_result_k.parquet.
    """
    names = list(parameters.keys())
    n_rows = min(len(parameters[name]) for name in names)
    with open(SAMPLED_PARAMETERS_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["iteration"] + names)
        for i in range(n_rows):
            writer.writerow([i] + [parameters[name][i] for name in names])


def save_checkpoint(completed_iterations, parameters):
    """Save checkpoint with completed iterations and parameters"""
    checkpoint = {
        "completed_iterations": completed_iterations,
        "total_iterations": NUM_ITERATIONS,
        "parameters": parameters,
        "timestamp": time.time(),
    }
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump(checkpoint, f, indent=2)
    save_sampled_parameters(parameters)
    print(f"✅ Checkpoint saved: {len(completed_iterations)} iterations completed")


def load_checkpoint():
    """Load checkpoint if it exists.

    ⚠️ RESUME HAZARD: a surviving checkpoint.json restores ALL parameter
    arrays from the moment the run started — every constant in
    src/monte_carlo_constants.py, including the scenario switch
    UNALLOCATED_VULNERABLE_FRACTION_FIXED, is silently ignored on resume.
    Editing parameter ranges and redeploying has NO effect unless the
    checkpoint (and the copies deploy-spot.sh pulls back from the GCS
    bucket) are deleted first. Before
    any fresh production run: wipe checkpoints/ AND results/parquet_files/
    in the bucket, or the run will resume the old parameters and/or skip
    "completed" iterations from the previous run (see
    get_completed_iterations, which trusts file presence with no provenance
    check).
    """
    if CHECKPOINT_FILE.exists():
        with open(CHECKPOINT_FILE, "r") as f:
            checkpoint = json.load(f)
        print(
            f"📂 Resuming from checkpoint: {len(checkpoint['completed_iterations'])} iterations already completed"
        )
        return checkpoint
    return None


def get_completed_iterations():
    """Check which iterations have been completed based on saved files"""
    completed = []
    for i in range(NUM_ITERATIONS):
        zone_file = PARQUET_PATH / f"zone_simulation_result_{i}.parquet"
        district_file = PARQUET_PATH / f"district_simulation_result_{i}.parquet"
        countries_file = PARQUET_PATH / f"countries_simulation_result_{i}.parquet"

        # Only consider iteration complete if all three files exist
        if zone_file.exists() and district_file.exists() and countries_file.exists():
            completed.append(i)

    return completed


def process_saved_results():
    """Process all saved parquet files"""
    print("\n📊 Processing saved results...")

    # Load all saved district results
    districts_simulation_results = []
    countries_simulation_results = []

    completed = get_completed_iterations()

    for i in completed:
        district_file = PARQUET_PATH / f"district_simulation_result_{i}.parquet"
        countries_file = PARQUET_PATH / f"countries_simulation_result_{i}.parquet"

        if district_file.exists():
            import pandas as pd

            districts_simulation_results.append(pd.read_parquet(district_file))

        if countries_file.exists():
            import pandas as pd

            countries_simulation_results.append(pd.read_parquet(countries_file))

    if districts_simulation_results and countries_simulation_results:
        # plot=False: plotting calls fig.show(), which needs a browser and
        # crashes on the headless VM before the CSVs get written
        mc.process_mc_results(countries_simulation_results, plot=False)
        mc.process_districts_results(districts_simulation_results)
        print(f"✅ Processed {len(completed)} simulation results")
    else:
        print("⚠️  No results to process yet")


def run_and_save_simulation(
    i,
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
):
    """Run one simulation in a worker process and write its results to disk.

    The full zone result is ~2.2M rows (roughly 1 GB in memory). Writing it
    here and returning only the iteration index keeps that DataFrame out of
    the parent process. Previously each result was pickled back to the parent
    and retained by its future object, growing the parent by ~1 GB per
    completed iteration until the OOM killer terminated it mid-run.
    """
    countries_result, district_result, zone_result = mc.run_simulation(
        crr_adjustment,
        time_gathering_water,
        practical_limit_bicycle,
        practical_limit_buckets,
        met,
        watts,
        hill_polarity,
        urban_adjustment,
        rural_adjustment,
        urban_borehole_lose_fraction=urban_borehole_lose_fraction,
        rural_borehole_lose_fraction=rural_borehole_lose_fraction,
        unallocated_vulnerable_fraction=unallocated_vulnerable_fraction,
        use_sample_data=False,
    )

    # Keep only the columns needed for the zone results
    filtered_zone_result = zone_result[
        ["fid", "zone_pop_with_water", "zone_pop_without_water"]
    ]

    district_result.to_parquet(
        PARQUET_PATH / f"district_simulation_result_{i}.parquet", index=False
    )
    countries_result.to_parquet(
        PARQUET_PATH / f"countries_simulation_result_{i}.parquet", index=False
    )
    filtered_zone_result.to_parquet(
        PARQUET_PATH / f"zone_simulation_result_{i}.parquet", index=False
    )

    return i


if __name__ == "__main__":
    # Set up signal handler for Spot VM preemption
    signal.signal(signal.SIGTERM, handle_preemption)

    # Create parquet directory if it doesn't exist
    PARQUET_PATH.mkdir(parents=True, exist_ok=True)

    # Check for existing completed iterations
    completed_iterations = get_completed_iterations()

    # Load checkpoint if it exists
    checkpoint = load_checkpoint()

    if completed_iterations:
        print(f"🔍 Found {len(completed_iterations)} completed iterations")

        # If we've completed all iterations, just process and exit
        if len(completed_iterations) >= NUM_ITERATIONS:
            print("✅ All iterations complete! Processing results...")
            process_saved_results()
            sys.exit(0)

    # Generate or restore parameters (see the resume hazard note on
    # load_checkpoint: a checkpoint beats every constant in this file)
    if checkpoint and "parameters" in checkpoint:
        # Restore parameters from checkpoint
        params = checkpoint["parameters"]
        if "urban_borehole_lose_fractions" not in params:
            sys.exit(
                "❌ checkpoint.json predates the borehole lose-fraction "
                "sampling — resuming it would run the wrong scenario. "
                "Delete the stale checkpoint (and the bucket copies "
                "deploy-spot.sh restores) and start fresh."
            )
        if "unallocated_vulnerable_fractions" not in params:
            sys.exit(
                "❌ checkpoint.json predates the unallocated-allocation "
                "parameter — resuming it would run the wrong scenario. "
                "Delete the stale checkpoint (and the bucket copies "
                "deploy-spot.sh restores) and start fresh."
            )
        if checkpoint.get("total_iterations") != NUM_ITERATIONS:
            sys.exit(
                f"❌ checkpoint.json was written for "
                f"{checkpoint.get('total_iterations')} iterations but "
                f"NUM_ITERATIONS is {NUM_ITERATIONS}. Delete the stale "
                "checkpoint (and bucket copies) and start fresh."
            )
        # Keep `parameters` bound so the periodic and final save_checkpoint
        # calls below still work on a resumed run. Without this it is only
        # defined in the else-branch, so any resume (i.e. every Spot VM
        # preemption) raised NameError at the first 10-iteration checkpoint.
        parameters = params
        print("📂 Parameters restored from checkpoint")
    else:
        # Generate new parameters (canonical distributions, see ADR-7).
        # .tolist() rather than list(): checkpoint.json and
        # sampled_parameters.csv need plain Python scalars, and json.dump
        # refuses numpy ones. It also turns the object array a "prorata"
        # scenario run produces back into plain strings.
        drawn = sample_parameters(NUM_ITERATIONS)
        parameters = {
            name: np.asarray(values).tolist() for name, values in drawn.items()
        }
        save_checkpoint(completed_iterations, parameters)

    # Element k of each list is the input of iteration k (the parquet files are
    # named by the same index).
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

    # Determine which iterations to run
    iterations_to_run = [
        i for i in range(NUM_ITERATIONS) if i not in completed_iterations
    ]

    if not iterations_to_run:
        print("✅ All iterations already complete! Processing results...")
        process_saved_results()
        sys.exit(0)

    # Record the start time
    start_time = time.time()
    print(f"\n🚀 Starting Monte Carlo simulations...")
    print(
        f"📊 Running {len(iterations_to_run)} remaining iterations (out of {NUM_ITERATIONS} total)"
    )
    print(f"⚡ Running {MAX_WORKERS} simulations concurrently...")
    print(f"🕐 Start time: {time.strftime('%H:%M:%S', time.localtime())}")
    print("\n")

    # Workers write their own results to disk and return only the iteration
    # index, so no large DataFrames are transferred to (or retained by) the
    # parent process. maxtasksperchild=1 replaces each worker after a single
    # iteration, so memory that pandas/numpy don't return to the OS can't
    # accumulate across iterations inside long-lived workers.
    pool = multiprocessing.Pool(processes=MAX_WORKERS, maxtasksperchild=1)

    pending = []
    for idx in iterations_to_run:
        async_result = pool.apply_async(
            run_and_save_simulation,
            (
                idx,
                crr_adjustments[idx],
                time_gatherings[idx],
                practical_limits_bicycle[idx],
                practical_limits_buckets[idx],
                mets[idx],
                watts_values[idx],
                hill_polarities[idx],
                urban_adjustments[idx],
                rural_adjustments[idx],
                urban_borehole_lose_fractions[idx],
                rural_borehole_lose_fractions[idx],
                unallocated_vulnerable_fractions[idx],
            ),
        )
        pending.append((idx, async_result))
    pool.close()

    # Initialize tqdm progress bar
    futures_progress = tqdm(
        total=len(iterations_to_run),
        initial=0,
        desc="Simulating",
    )

    for idx, async_result in pending:
        if shutdown_requested:
            print("\n⚠️  Shutting down gracefully...")
            pool.terminate()
            break

        try:
            completed_index = async_result.get(timeout=ITERATION_TIMEOUT)

            # Update completed iterations
            completed_iterations.append(completed_index)

            # Save checkpoint periodically (every 10 iterations)
            if len(completed_iterations) % 10 == 0:
                save_checkpoint(completed_iterations, parameters)

            futures_progress.update()

        except multiprocessing.TimeoutError:
            print(
                f"\n❌ Iteration {idx}: no result within {ITERATION_TIMEOUT}s "
                "(worker process may have been killed); continuing"
            )
        except Exception as e:
            print(f"\n❌ Error in iteration {idx}: {e}")

    futures_progress.close()
    pool.terminate()
    pool.join()

    # Save final checkpoint
    if not shutdown_requested:
        save_checkpoint(completed_iterations, parameters)

    # Process all results (including previously saved ones)
    process_saved_results()

    # Record the end time
    end_time = time.time()

    # Calculate and print the time taken
    time_taken = end_time - start_time
    print(
        f"\n⏱️  Session time: {time_taken / 60:.2f} minutes ({time_taken / 3600:.2f} hours)"
    )
    print(f"✅ Completed {len(completed_iterations)} total iterations")

    if shutdown_requested:
        print("\n⚠️  Script interrupted but progress saved. Run again to continue.")
        sys.exit(1)
