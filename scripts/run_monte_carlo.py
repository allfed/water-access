import concurrent.futures
from pathlib import Path
import sys
from tqdm import tqdm
import time

# Resolve project root and update sys.path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
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

# -------------------------------------------------------------------------------

# The walking and cycling sampling parameters live in
# src/monte_carlo_constants.py so that this script, run_monte_carlo_gcp.py and
# run_monte_carlo_test.py can never disagree about them again (ADR-7).

# -------------------------------------------------------------------------------


if __name__ == "__main__":
    # Monte Carlo parameters (canonical distributions, see ADR-7)
    parameters = sample_parameters(NUM_ITERATIONS)
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
    zone_simulation_results = []

    # Record the start time
    start_time = time.time()
    print("Starting Monte Carlo simulations...")
    print(f"Running {NUM_ITERATIONS} simulations...")
    print(f"Running {MAX_WORKERS} simulations concurrently...")
    print("Start time:", time.strftime("%H:%M:%S", time.localtime()))
    print("\n\n")

    with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
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

            # keep only the columns needed for the zone results
            filtered_zone_result = zone_result[
                ["fid", "zone_pop_with_water", "zone_pop_without_water"]
            ]

            # Save all results to Parquet files for interruption resilience
            output_file = PARQUET_PATH / f"zone_simulation_result_{i}.parquet"
            filtered_zone_result.to_parquet(output_file, index=False)

            # Also save district and countries results progressively
            district_file = PARQUET_PATH / f"district_simulation_result_{i}.parquet"
            district_result.to_parquet(district_file, index=False)

            countries_file = PARQUET_PATH / f"countries_simulation_result_{i}.parquet"
            countries_result.to_parquet(countries_file, index=False)

            futures_progress.update()  # Update the progress bar

    futures_progress.close()  # Close the progress bar

    mc.process_mc_results(countries_simulation_results)
    mc.process_districts_results(districts_simulation_results)

    # Record the end time
    end_time = time.time()

    # Calculate and print the time taken by the simulations in minutes and
    # hours
    time_taken = end_time - start_time
    print(f"Time taken: {time_taken / 60:.2f} minutes")
    print(f"Time taken: {time_taken / 3600:.2f} hours")
