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

# The walking and cycling sampling parameters live in
# src/monte_carlo_constants.py so that this script, run_monte_carlo.py and
# run_monte_carlo_test.py can never disagree about them again (ADR-7). The
# values there are the ones this script used to define locally -- they are
# canonical.

# -------------------------------------------------------------------------------

# Global flag for graceful shutdown
shutdown_requested = False


def handle_preemption(signum, frame):
    """Handle Spot VM preemption signal (30 second warning)"""
    global shutdown_requested
    print("\n⚠️  Preemption signal received! Saving checkpoint...")
    shutdown_requested = True


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
    print(f"✅ Checkpoint saved: {len(completed_iterations)} iterations completed")


def load_checkpoint():
    """Load checkpoint if it exists"""
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

    # Generate or restore parameters
    if checkpoint and "parameters" in checkpoint:
        # Restore parameters from checkpoint
        params = checkpoint["parameters"]
        crr_adjustments = np.array(params["crr_adjustments"])
        time_gatherings = np.array(params["time_gatherings"])
        practical_limits_bicycle = np.array(params["practical_limits_bicycle"])
        practical_limits_buckets = np.array(params["practical_limits_buckets"])
        mets = np.array(params["mets"])
        watts_values = np.array(params["watts_values"])
        hill_polarities = np.array(params["hill_polarities"])
        urban_adjustments = np.array(params["urban_adjustments"])
        rural_adjustments = np.array(params["rural_adjustments"])
        print("📂 Parameters restored from checkpoint")
    else:
        # Generate new parameters (canonical distributions, see ADR-7)
        drawn = sample_parameters(NUM_ITERATIONS)
        crr_adjustments = drawn["crr_adjustments"]
        time_gatherings = drawn["time_gatherings"]
        practical_limits_bicycle = drawn["practical_limits_bicycle"]
        practical_limits_buckets = drawn["practical_limits_buckets"]
        mets = drawn["mets"]
        watts_values = drawn["watts_values"]
        hill_polarities = drawn["hill_polarities"]
        urban_adjustments = drawn["urban_adjustments"]
        rural_adjustments = drawn["rural_adjustments"]

        # Save parameters for potential resume
        parameters = {name: values.tolist() for name, values in drawn.items()}
        save_checkpoint(completed_iterations, parameters)

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
