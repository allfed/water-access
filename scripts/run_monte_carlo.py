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

# # Resolve project root and update sys.path
project_root = Path().resolve().parent
sys.path.append(str(project_root))
import src.gis_global_module as gis
import src.gis_monte_carlo as mc


if __name__ == '__main__':
    # Monte Carlo parameters
    num_iterations = 5  # Number of simulations to run
    crr_adjustments = np.random.randint(-1, 2, size=num_iterations)
    time_gatherings = mc.sample_normal(4, 10, num_iterations)
    practical_limits_bicycle = mc.sample_normal(30, 45, num_iterations)
    practical_limits_buckets = mc.sample_normal(15, 25, num_iterations)
    mets = mc.sample_normal(2.5, 5, num_iterations)
    # max_loads = mc.sample_normal(2.5, 5, num_iterations)

    print(crr_adjustments)
    print(time_gatherings)
    print(practical_limits_bicycle)
    print(mets)

    simulation_results = []

    with concurrent.futures.ProcessPoolExecutor(max_workers=2) as executor:
        # Submit all simulations to the executor
        futures = [
            executor.submit(
                mc.run_simulation,
                crr_adjustment,
                time_gathering_water,
                practical_limit_bicycle,
                practical_limit_buckets,
                met,
            )
            for crr_adjustment, time_gathering_water, practical_limit_bicycle, practical_limit_buckets, met in zip(
                crr_adjustments,
                time_gatherings,
                practical_limits_bicycle,
                practical_limits_buckets,
                mets,
            )
        ]

        # Initialize tqdm progress bar
        futures_progress = tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Simulating")

        # Collect results as they complete
        for future in concurrent.futures.as_completed(futures):
            simulation_results.append(future.result())
            futures_progress.update()  # Update the progress bar

    futures_progress.close()  # Close the progress bar
    mc.process_mc_results(simulation_results)
