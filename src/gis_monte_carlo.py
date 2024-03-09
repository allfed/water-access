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

def sample_normal(low, high, n):
    """
    Generate random samples from a normal distribution. 
    Based off of Guesstimate's implementation, translated from Javascript to Python.

    Parameters:
    low (float): The lower bound of the distribution.
    high (float): The upper bound of the distribution.
    n (int): The number of samples to generate.

    Returns:
    numpy.ndarray: An array of random samples from the normal distribution.
    """
    mean = np.mean([high, low])
    stdev = (high - mean) / 1.645
    samples = np.abs(norm.rvs(loc=mean, scale=stdev, size=n))

    return samples


def sample_lognormal(low, high, n):
    """
    Generate random samples from a lognormal distribution.

    Parameters:
    - low (float): The lower bound of the lognormal distribution.
    - high (float): The upper bound of the lognormal distribution.
    - n (int): The number of samples to generate.

    Returns:
    - samples (ndarray): An array of random samples from the lognormal distribution.
    """
    assert low > 0, "Low must be greater than 0 for lognormal distributions."
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
    calculate_distance=True,
):
    """
    Run a simulation to analyze global water access from walking or cycling based on sensitivity parameters..

    Parameters:
    - crr_adjustment (int): The adjustment factor for the coefficient of rolling resistance.
    - time_gathering_water (int or float): The time taken to gather water in minutes.
    - practical_limit_bicycle (int or float): The practical limit of water transportation using a bicycle.
    - practical_limit_buckets (int or float): The practical limit of water transportation using buckets.
    - met (int or float): The metabolic equivalent of task (MET) value.
    - calculate_distance (bool, optional): Whether to calculate the distance during the simulation. 
    Defaults to True, must be True to run a true simulation with the given parameters.

    Returns:
    - result: The result of the global analysis.

    Raises:
    - AssertionError: If any of the input parameters are of incorrect type.

    """
    assert isinstance(crr_adjustment, int), "CRR adjustment must be an integer."
    assert isinstance(time_gathering_water, (int, float)), "Time gathering water must be a number."
    assert isinstance(practical_limit_bicycle, (int, float)), "Practical limit bicycle must be a number."
    assert isinstance(practical_limit_buckets, (int, float)), "Practical limit buckets must be a number."
    assert isinstance(met, (int, float)), "MET must be a number."

    result = gis.run_global_analysis(
        crr_adjustment=crr_adjustment,
        time_gathering_water=time_gathering_water,
        practical_limit_bicycle=practical_limit_bicycle,
        practical_limit_buckets=practical_limit_buckets,
        met=met,
        calculate_distance=True,
        plot=False,
    )
    return result


def process_mc_results(simulation_results, plot=False):
    """
    Process the Monte Carlo simulation results. Calculate the median, mean, 90th percentile, 5th percentile, max, and min values and plot the results.

    Args:
        simulation_results (list): A list of DataFrames containing simulation results.
        plot (bool, optional): Whether to plot the chloropleth maps. Defaults to True.

    Returns:
        None
    """

    # Step 1: Calculate the median, mean, 90th percentile, and 5th percentile of "percent_with_water" for each DataFrame
    medians = [df["percent_with_water"].median() for df in simulation_results]
    means = [df["percent_with_water"].mean() for df in simulation_results]
    percentile_90 = [df["percent_with_water"].quantile(0.9) for df in simulation_results]
    percentile_5 = [df["percent_with_water"].quantile(0.05) for df in simulation_results]

    # Step 2: Identify the DataFrames for max, min, median, mean, 90th percentile, and 5th percentile values
    max_df = simulation_results[medians.index(max(medians))]
    min_df = simulation_results[medians.index(min(medians))]
    median_df = simulation_results[medians.index(np.median(medians))]
    mean_df = simulation_results[means.index(np.mean(means))]
    percentile_90_df = simulation_results[percentile_90.index(np.percentile(percentile_90, 90))]
    percentile_5_df = simulation_results[percentile_5.index(np.percentile(percentile_5, 5))]

    # Step 3: Plot the chloropleth maps for max, min, median, mean, 90th percentile, and 5th percentile if plot argument is True
    if plot:
        gis.plot_chloropleth(max_df)
        gis.plot_chloropleth(min_df)
        gis.plot_chloropleth(median_df)
        gis.plot_chloropleth(mean_df)
        gis.plot_chloropleth(percentile_90_df)
        gis.plot_chloropleth(percentile_5_df)

    # Step 4: Save the results to the results folder
    median_df.to_csv("results/median_results.csv")
    mean_df.to_csv("results/mean_results.csv")
    min_df.to_csv("results/min_results.csv")
    max_df.to_csv("results/max_results.csv")
    percentile_90_df.to_csv("results/90th_percentile_results.csv")
    percentile_5_df.to_csv("results/5th_percentile_results.csv")

    # Step 5: Pickle the simulation results
    with open('results/simulation_results.pkl', 'wb') as f:
        pickle.dump(simulation_results, f)

    print("Simulation results have been processed and saved to the results folder.")

