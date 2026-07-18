import concurrent.futures
import numpy as np
from pathlib import Path
import sys
from tqdm import tqdm
import time

# Resolve project root and update sys.path
project_root = Path().resolve().parent
sys.path.append(str(project_root))
import src.gis_monte_carlo as mc  # noqa
import src.monte_carlo_config as cfg  # noqa


# Results path
RESULTS_PATH = project_root / "water-access/results"
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

# Walking/cycling sensitivity parameters are defined once in
# src/monte_carlo_config.py and shared across all three run scripts so they can
# never drift apart. See that module for the PROVISIONAL parameter values and
# the checkpoint warning.


if __name__ == "__main__":
    # Monte Carlo parameters
    crr_adjustments = np.random.randint(
        cfg.CRR_LOWER_ESTIMATE, cfg.CRR_UPPER_ESTIMATE + 1, size=NUM_ITERATIONS
    )
    # Time gathering water is sampled LOGNORMAL (not normal) -- see config.
    time_gatherings = mc.sample_lognormal(
        cfg.TIME_GATHERING_LOWER_ESTIMATE,
        cfg.TIME_GATHERING_UPPER_ESTIMATE,
        NUM_ITERATIONS,
    )
    practical_limits_bicycle = mc.sample_normal(
        cfg.PRACTICAL_LIMITS_BICYCLE_LOWER_ESTIMATE,
        cfg.PRACTICAL_LIMITS_BICYCLE_UPPER_ESTIMATE,
        NUM_ITERATIONS,
    )
    practical_limits_buckets = mc.sample_normal(
        cfg.PRACTICAL_LIMITS_BUCKET_LOWER_ESTIMATE,
        cfg.PRACTICAL_LIMITS_BUCKET_UPPER_ESTIMATE,
        NUM_ITERATIONS,
    )
    mets = mc.sample_normal(
        cfg.METS_LOWER_ESTIMATE, cfg.METS_UPPER_ESTIMATE, NUM_ITERATIONS
    )
    # Mechanical cycling watts are DERIVED from the sampled METs via the ACSM
    # cycle-ergometry inversion (62 kg reference), not sampled independently.
    # PROVISIONAL -- see config.derive_watts_from_mets.
    watts_values = cfg.derive_watts_from_mets(mets)

    hill_polarities = np.random.choice(cfg.POLARITY_OPTIONS, NUM_ITERATIONS)

    urban_adjustments = mc.sample_normal(
        cfg.URBAN_ADJUSTMENT_LOWER_ESTIMATE,
        cfg.URBAN_ADJUSTMENT_UPPER_ESTIMATE,
        NUM_ITERATIONS,
    )
    rural_adjustments = mc.sample_gpd(
        cfg.RURAL_PDR_PARETO_SHAPE,
        cfg.RURAL_PDR_PARETO_SCALE,
        cfg.RURAL_PDR_PARETO_LOC,
        NUM_ITERATIONS,
    )

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
