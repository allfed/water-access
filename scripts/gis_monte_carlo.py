import concurrent.futures
import numpy as np
from scipy.stats import norm, lognorm
import pandas as pd
from pathlib import Path
import sys

# # Resolve project root and update sys.path
project_root = Path().resolve().parent
sys.path.append(str(project_root))
import src.gis_global_module as gis


def sample_normal(low, high, n):
    mean = np.mean([high, low])
    stdev = (high - mean) / 1.645
    samples = np.abs(norm.rvs(loc=mean, scale=stdev, size=n))

    return samples


def sample_lognormal(low, high, n):
    logHigh = np.log(high)
    logLow = np.log(low)

    mean = np.mean([logHigh, logLow])
    stdev = (logHigh - logLow) / (2 * 1.645)
    scale = np.exp(mean)
    samples = np.abs(lognorm.rvs(s=stdev, scale=scale, size=n))

    return samples


def run_simulation(
    crr_adjustment,
    time_gathering_water,
    practical_limit_bicycle,
    practical_limit_buckets,
    met,
):
    # This function should contain the logic that was previously in your loop
    # It should return whatever metrics you're interested in (e.g., velocities, max load)
    # For demonstration, it returns a placeholder result
    result = gis.run_global_analysis(
        crr_adjustment=crr_adjustment,
        time_gathering_water=time_gathering_water,
        practical_limit_bicycle=practical_limit_bicycle,
        practical_limit_buckets=practical_limit_buckets,
        met=met,
        plot=False,
    )
    return result


def process_mc_results(simulation_results):

    # Step 1: Calculate the median of "population_piped_with_access" for each DataFrame
    medians = [df["zone_pop_with_water"].median() for df in simulation_results]

    # Step 2: Identify the DataFrames for max, min, and mean values
    max_df = simulation_results[medians.index(max(medians))]
    min_df = simulation_results[medians.index(min(medians))]

    # For median, since it's a middle value, we sort the medians and find the median position
    sorted_medians_indices = np.argsort(medians)
    median_index = sorted_medians_indices[len(sorted_medians_indices) // 2]
    median_df = simulation_results[np.round(median_index)]

    gis.plot_chloropleth(max_df)
    gis.plot_chloropleth(median_df)
    gis.plot_chloropleth(min_df)


# Monte Carlo parameters
num_iterations = 3  # Number of simulations to run
crr_adjustments = np.random.randint(-1, 2, size=num_iterations)
time_gatherings = sample_normal(4, 10, num_iterations)
practical_limits_bicycle = sample_normal(30, 45, num_iterations)
practical_limits_buckets = sample_lognormal(15, 25, num_iterations)
mets = sample_normal(2.5, 5, num_iterations)
# max_loads = sample_normal(2.5, 5, num_iterations)

simulation_results = []

with concurrent.futures.ProcessPoolExecutor() as executor:
    # Submit all simulations to the executor
    futures = [
        executor.submit(
            run_simulation,
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

    # Collect results as they complete
    for future in concurrent.futures.as_completed(futures):
        simulation_results.append(future.result())

process_mc_results(simulation_results)
