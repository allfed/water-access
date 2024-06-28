import concurrent.futures
import numpy as np
from scipy.stats import norm, lognorm
import pandas as pd
from pathlib import Path
import sys
from tqdm import tqdm
import pickle
import numpy as np
from scipy.stats import norm
import time

# # Resolve project root and update sys.path
project_root = Path().resolve().parent
sys.path.append(str(project_root))
import src.gis_global_module as gis
import src.gis_monte_carlo as mc


if __name__ == "__main__":
    # Monte Carlo parameters
    num_iterations = 50  # Number of simulations to run

    crr_adjustments = np.random.randint(-1, 2, size=num_iterations)
    time_gatherings = mc.sample_normal(4, 8, num_iterations)
    practical_limits_bicycle = mc.sample_normal(30, 45, num_iterations)
    practical_limits_buckets = mc.sample_normal(15, 25, num_iterations)
    mets = mc.sample_normal(3, 6, num_iterations)
    watts_values = mc.sample_normal(50, 100, num_iterations)

    polarity_options = [
        "uphill_downhill",
        "uphill_flat",
        "flat_uphill",
        "downhill_uphill",
    ]
    hill_polarities = np.random.choice(polarity_options, num_iterations)

    print(crr_adjustments)
    print(time_gatherings)
    print(practical_limits_bicycle)
    print(mets)
    print(watts_values)
    print(hill_polarities)

    simulation_results = []

    # Record the start time
    start_time = time.time()

    with concurrent.futures.ProcessPoolExecutor(max_workers=40) as executor:
        # Submit all simulations to the executor
        # TODO multiple outputs??
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
            simulation_results.append(future.result())
            futures_progress.update()  # Update the progress bar

    futures_progress.close()  # Close the progress bar

    #TODO check if it's just here i need to add zones results
    # mc.process_mc_results(simulation_results)
    mc.process_zones_results(simulation_results)

    # Record the end time
    end_time = time.time()
    # Calculate and print the time taken by the simulations
    print(f"The simulations took {end_time - start_time} seconds.")
