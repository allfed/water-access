#!/usr/bin/env python3
"""
Test version of Monte Carlo simulation - runs only 2 iterations for testing GCP setup
This allows testing the full pipeline with minimal cost (~$0.01-0.02)
"""

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
PARQUET_PATH = RESULTS_PATH / "parquet_files_test"  # Different folder for test

# -------------------------------------------------------------------------------
# TEST CONFIGURATION - ONLY 2 ITERATIONS!
# -------------------------------------------------------------------------------

# Define the number of simulations to run
NUM_ITERATIONS = 2  # ONLY 2 FOR TESTING!

# Define maximum simultaneous processes
MAX_WORKERS = 2  # Only 2 for testing

# -------------------------------------------------------------------------------

# Uses the same shared walking/cycling parameters as production, imported from
# src/monte_carlo_config.py (only NUM_ITERATIONS / MAX_WORKERS differ here).


if __name__ == "__main__":
    print("=" * 60)
    print("RUNNING TEST VERSION - ONLY 2 ITERATIONS")
    print("This is for testing GCP setup, not production runs")
    print("=" * 60)
    print()

    # Create test results directory
    PARQUET_PATH.mkdir(parents=True, exist_ok=True)

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

    print("Parameters generated:")
    print(f"  CRR adjustments: {crr_adjustments}")
    print(f"  Time gathering: {time_gatherings}")
    print(f"  Practical limits (bicycle): {practical_limits_bicycle}")
    print(f"  Practical limits (buckets): {practical_limits_buckets}")
    print(f"  METS: {mets}")
    print(f"  Watts: {watts_values}")
    print(f"  Hill polarities: {hill_polarities}")
    print(f"  Urban adjustments: {urban_adjustments}")
    print(f"  Rural adjustments: {rural_adjustments}")
    print()

    # Initialize lists to store results
    districts_simulation_results = []
    countries_simulation_results = []
    zone_simulation_results = []

    # Record the start time
    start_time = time.time()
    print("Starting TEST Monte Carlo simulations...")
    print(f"Running {NUM_ITERATIONS} TEST iterations...")
    print(f"Running {MAX_WORKERS} simulations concurrently...")
    print("Start time:", time.strftime("%H:%M:%S", time.localtime()))
    print("\n")

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
                use_sample_data=False,
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
                rural_adjustments,
            )
        ]

        # Initialize tqdm progress bar
        futures_progress = tqdm(
            concurrent.futures.as_completed(futures),
            total=len(futures),
            desc="Test Simulating",
        )

        for i, future in enumerate(concurrent.futures.as_completed(futures)):
            countries_result, district_result, zone_result = future.result()

            # Save the results
            districts_simulation_results.append(district_result)
            countries_simulation_results.append(countries_result)

            # Filter zone results
            filtered_zone_result = zone_result[
                ["fid", "zone_pop_with_water", "zone_pop_without_water"]
            ]

            # Save all results to Parquet files (testing progressive save)
            output_file = PARQUET_PATH / f"zone_simulation_result_{i}.parquet"
            filtered_zone_result.to_parquet(output_file, index=False)

            district_file = PARQUET_PATH / f"district_simulation_result_{i}.parquet"
            district_result.to_parquet(district_file, index=False)

            countries_file = PARQUET_PATH / f"countries_simulation_result_{i}.parquet"
            countries_result.to_parquet(countries_file, index=False)

            print(f"✅ Saved test results for iteration {i}")

            futures_progress.update()

    futures_progress.close()

    # Process results
    print("\nProcessing test results...")
    mc.process_mc_results(countries_simulation_results)
    mc.process_districts_results(districts_simulation_results)

    # Record the end time
    end_time = time.time()

    # Calculate and print the time taken
    time_taken = end_time - start_time
    print(f"\n{'='*60}")
    print(f"TEST COMPLETE!")
    print(f"Time taken: {time_taken / 60:.2f} minutes")
    print(f"Results saved in: {PARQUET_PATH}")
    print(f"{'='*60}")

    # List saved files
    print("\nSaved test files:")
    for f in PARQUET_PATH.glob("*.parquet"):
        print(f"  - {f.name}")

    print("\n✅ Test simulation successful! GCP setup is working correctly.")
