import concurrent.futures
import numpy as np
from pathlib import Path
import sys
from tqdm import tqdm
import time
import json
import signal
import os

# Resolve project root and update sys.path
project_root = Path().resolve().parent
sys.path.append(str(project_root))
import src.gis_monte_carlo as mc  # noqa


# Results path
RESULTS_PATH = project_root / "water-access/results"
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
# 15 was the most that could run on a 32 core hyperthreaded machine
MAX_WORKERS = 15

# -------------------------------------------------------------------------------


# -------------------------------------------------------------------------------
# DEFINE WALKING AND CYCLING PARAMETERS FOR MONTE CARLO SIMULATIONS
# -------------------------------------------------------------------------------

# CRR adjustments.
# 1 means one road type better, -1 means one road type worse
CRR_LOWER_ESTIMATE = -1
CRR_UPPER_ESTIMATE = 1

# Time gathering water in hours
TIME_GATHERING_LOWER_ESTIMATE = 4
TIME_GATHERING_UPPER_ESTIMATE = 7

# Practical load limits for cycling in kg
PRACTICAL_LIMITS_BICYCLE_LOWER_ESTIMATE = 30
PRACTICAL_LIMITS_BICYCLE_UPPER_ESTIMATE = 45

# Practical load limits for walking with buckets in kg
PRACTICAL_LIMITS_BUCKET_LOWER_ESTIMATE = 15
PRACTICAL_LIMITS_BUCKET_UPPER_ESTIMATE = 25

# Average METS available for walking with buckets to and from water source
METS_LOWER_ESTIMATE = 3
METS_UPPER_ESTIMATE = 6

# Average watts available for cycling to and from water source
WATTS_LOWER_ESTIMATE = 20
WATTS_UPPER_ESTIMATE = 80

# Polarity options (randomly chosen from list each simulation run)
# The first word defines the trip to the water source
# The second word defines the trip from the water source
# Options to include: "uphill_downhill", "downhill_uphill", "uphill_flat",
# "flat_uphill", "downhill_flat", "flat_downhill", "flat_flat"

POLARITY_OPTIONS = [
    "uphill_downhill",
    "uphill_flat",
    "flat_uphill",
    "downhill_uphill",
]

# Adjustments for euclidean distance to account for paths taken to water
# not being straight lines
URBAN_ADJUSTMENT_LOWER_ESTIMATE = 1.2
URBAN_ADJUSTMENT_UPPER_ESTIMATE = 1.5

# Set the parameters for the GPD distribution for rural adjustments.
# Shape, scale, and loc
# These values were obtained from the scripts/create_pareto_distribution.py
RURAL_PDR_PARETO_SHAPE = 0.20007812499999994
RURAL_PDR_PARETO_SCALE = 0.19953125000000005
RURAL_PDR_PARETO_LOC = 1.0

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
        "timestamp": time.time()
    }
    with open(CHECKPOINT_FILE, 'w') as f:
        json.dump(checkpoint, f, indent=2)
    print(f"✅ Checkpoint saved: {len(completed_iterations)} iterations completed")


def load_checkpoint():
    """Load checkpoint if it exists"""
    if CHECKPOINT_FILE.exists():
        with open(CHECKPOINT_FILE, 'r') as f:
            checkpoint = json.load(f)
        print(f"📂 Resuming from checkpoint: {len(checkpoint['completed_iterations'])} iterations already completed")
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
        mc.process_mc_results(countries_simulation_results)
        mc.process_districts_results(districts_simulation_results)
        print(f"✅ Processed {len(completed)} simulation results")
    else:
        print("⚠️  No results to process yet")


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
        # Generate new parameters
        crr_adjustments = np.random.randint(
            CRR_LOWER_ESTIMATE, CRR_UPPER_ESTIMATE + 1, size=NUM_ITERATIONS
        )
        time_gatherings = mc.sample_normal(
            TIME_GATHERING_LOWER_ESTIMATE,
            TIME_GATHERING_UPPER_ESTIMATE,
            NUM_ITERATIONS,
        )
        practical_limits_bicycle = mc.sample_normal(
            PRACTICAL_LIMITS_BICYCLE_LOWER_ESTIMATE,
            PRACTICAL_LIMITS_BICYCLE_UPPER_ESTIMATE,
            NUM_ITERATIONS,
        )
        practical_limits_buckets = mc.sample_normal(
            PRACTICAL_LIMITS_BUCKET_LOWER_ESTIMATE,
            PRACTICAL_LIMITS_BUCKET_UPPER_ESTIMATE,
            NUM_ITERATIONS,
        )
        mets = mc.sample_normal(METS_LOWER_ESTIMATE, METS_UPPER_ESTIMATE, NUM_ITERATIONS)
        watts_values = mc.sample_normal(
            WATTS_LOWER_ESTIMATE, WATTS_UPPER_ESTIMATE, NUM_ITERATIONS
        )
        hill_polarities = np.random.choice(POLARITY_OPTIONS, NUM_ITERATIONS)
        urban_adjustments = mc.sample_normal(
            URBAN_ADJUSTMENT_LOWER_ESTIMATE,
            URBAN_ADJUSTMENT_UPPER_ESTIMATE,
            NUM_ITERATIONS,
        )
        rural_adjustments = mc.sample_gpd(
            RURAL_PDR_PARETO_SHAPE,
            RURAL_PDR_PARETO_SCALE,
            RURAL_PDR_PARETO_LOC,
            NUM_ITERATIONS,
        )
        
        # Save parameters for potential resume
        parameters = {
            "crr_adjustments": crr_adjustments.tolist(),
            "time_gatherings": time_gatherings.tolist(),
            "practical_limits_bicycle": practical_limits_bicycle.tolist(),
            "practical_limits_buckets": practical_limits_buckets.tolist(),
            "mets": mets.tolist(),
            "watts_values": watts_values.tolist(),
            "hill_polarities": hill_polarities.tolist(),
            "urban_adjustments": urban_adjustments.tolist(),
            "rural_adjustments": rural_adjustments.tolist()
        }
        save_checkpoint(completed_iterations, parameters)

    # Determine which iterations to run
    iterations_to_run = [i for i in range(NUM_ITERATIONS) if i not in completed_iterations]
    
    if not iterations_to_run:
        print("✅ All iterations already complete! Processing results...")
        process_saved_results()
        sys.exit(0)

    # Initialize lists to store results from each output
    districts_simulation_results = []
    countries_simulation_results = []

    # Record the start time
    start_time = time.time()
    print(f"\n🚀 Starting Monte Carlo simulations...")
    print(f"📊 Running {len(iterations_to_run)} remaining iterations (out of {NUM_ITERATIONS} total)")
    print(f"⚡ Running {MAX_WORKERS} simulations concurrently...")
    print(f"🕐 Start time: {time.strftime('%H:%M:%S', time.localtime())}")
    print("\n")

    with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Submit only the remaining simulations
        futures = {}
        for idx in iterations_to_run:
            if shutdown_requested:
                break
                
            future = executor.submit(
                mc.run_simulation,
                crr_adjustments[idx],
                time_gatherings[idx],
                practical_limits_bicycle[idx],
                practical_limits_buckets[idx],
                mets[idx],
                watts_values[idx],
                hill_polarities[idx],
                urban_adjustments[idx],
                rural_adjustments[idx],
                use_sample_data=False,
            )
            futures[future] = idx

        # Initialize tqdm progress bar
        futures_progress = tqdm(
            total=len(iterations_to_run),
            initial=0,
            desc="Simulating",
        )

        for future in concurrent.futures.as_completed(futures):
            if shutdown_requested:
                print("\n⚠️  Shutting down gracefully...")
                executor.shutdown(wait=False)
                break
                
            i = futures[future]
            
            try:
                countries_result, district_result, zone_result = future.result()

                # Keep results in memory for final processing
                districts_simulation_results.append(district_result)
                countries_simulation_results.append(countries_result)

                # Filter zone results
                filtered_zone_result = zone_result[
                    ["fid", "zone_pop_with_water", "zone_pop_without_water"]
                ]

                # Save all results to Parquet files
                output_file = PARQUET_PATH / f"zone_simulation_result_{i}.parquet"
                filtered_zone_result.to_parquet(output_file, index=False)
                
                district_file = PARQUET_PATH / f"district_simulation_result_{i}.parquet"
                district_result.to_parquet(district_file, index=False)
                
                countries_file = PARQUET_PATH / f"countries_simulation_result_{i}.parquet"
                countries_result.to_parquet(countries_file, index=False)

                # Update completed iterations
                completed_iterations.append(i)
                
                # Save checkpoint periodically (every 10 iterations)
                if len(completed_iterations) % 10 == 0:
                    save_checkpoint(completed_iterations, parameters)

                futures_progress.update()
                
            except Exception as e:
                print(f"\n❌ Error in iteration {i}: {e}")
                continue

    futures_progress.close()

    # Save final checkpoint
    if not shutdown_requested:
        save_checkpoint(completed_iterations, parameters)

    # Process all results (including previously saved ones)
    process_saved_results()

    # Record the end time
    end_time = time.time()

    # Calculate and print the time taken
    time_taken = end_time - start_time
    print(f"\n⏱️  Session time: {time_taken / 60:.2f} minutes ({time_taken / 3600:.2f} hours)")
    print(f"✅ Completed {len(completed_iterations)} total iterations")
    
    if shutdown_requested:
        print("\n⚠️  Script interrupted but progress saved. Run again to continue.")
        sys.exit(1)