import concurrent.futures
import numpy as np
from pathlib import Path
import sys
from tqdm import tqdm
import time

# Resolve project root and update sys.path
project_root = Path().resolve().parent
sys.path.append(str(project_root))
import src.gis_monte_carlo as mc

# -------------------------------------------------------------------------------
# DEFINE MONTE CARLO SIMULATION PARAMETERS
# -------------------------------------------------------------------------------

# Define the number of simulations to run
# Expect ~15-20 minutes for one simulation,
# but multiprocessing will speed up large batches significantly
NUM_ITERATIONS = 15

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
WATTS_LOWER_ESTIMATE = 50
WATTS_UPPER_ESTIMATE = 100

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

# -------------------------------------------------------------------------------


if __name__ == "__main__":
    # Monte Carlo parameters
    crr_adjustments = np.random.randint(
        CRR_LOWER_ESTIMATE, CRR_UPPER_ESTIMATE + 1, size=NUM_ITERATIONS
    )
    time_gatherings = mc.sample_normal(
        TIME_GATHERING_LOWER_ESTIMATE, TIME_GATHERING_UPPER_ESTIMATE, NUM_ITERATIONS
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

    print(crr_adjustments)
    print(time_gatherings)
    print(practical_limits_bicycle)
    print(mets)
    print(watts_values)
    print(hill_polarities)

    # Initialize lists to store results from each output
    districts_simulation_results = []
    countries_simulation_results = []

    # Record the start time
    start_time = time.time()

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
            )
            for crr_adjustment, time_gathering_water, practical_limit_bicycle, practical_limit_buckets, met, watts, hill_polarity in zip(
                crr_adjustments,
                time_gatherings,
                practical_limits_bicycle,
                practical_limits_buckets,
                mets,
                watts_values,
                hill_polarities,
            )
        ]

        # Initialize tqdm progress bar
        futures_progress = tqdm(
            concurrent.futures.as_completed(futures),
            total=len(futures),
            desc="Simulating",
        )

        # Collect results as they complete
        for future in concurrent.futures.as_completed(futures):
            countries_result, district_result = future.result()  # Unpack the results
            districts_simulation_results.append(district_result)
            countries_simulation_results.append(countries_result)
            futures_progress.update()  # Update the progress bar

    futures_progress.close()  # Close the progress bar

    mc.process_mc_results(countries_simulation_results)
    mc.process_districts_results(districts_simulation_results)

    # Record the end time
    end_time = time.time()
    # Calculate and print the time taken by the simulations
    print(f"The simulations took {end_time - start_time} seconds.")
