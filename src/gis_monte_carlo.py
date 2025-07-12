import numpy as np
from scipy.stats import norm, lognorm, genpareto
import pandas as pd
from pathlib import Path
import sys
import pickle
import os

# Resolve project root and update sys.path
project_root = Path().resolve().parent
sys.path.append(str(project_root))
import src.gis_global_module as gis  # noqa


def sample_normal(low, high, n, confidence=90):
    """Generate random samples from a normal distribution. Based off of
    Guesstimate's implementation, translated from Javascript to Python.

    Parameters:
    low (float): The lower bound of the distribution.
    high (float): The upper bound of the distribution.
    n (int): The number of samples to generate.
    confidence (int): The confidence level of the distribution. Must
                       be 90, 95, or 99.

    Returns:
    numpy.ndarray: An array of random samples from the normal distribution.
    """
    if confidence == 90:
        z = 1.645
    elif confidence == 95:
        z = 1.96
    elif confidence == 99:
        z = 2.575
    else:
        raise ValueError("Confidence level must be 90, 95, or 99")

    mean = np.mean([high, low])
    stdev = (high - mean) / z
    samples = np.abs(norm.rvs(loc=mean, scale=stdev, size=n))

    return samples


def sample_lognormal(low, high, n, confidence=90):
    """Generate random samples from a lognormal distribution.

    Parameters:
    - low (float): The lower bound of the lognormal distribution.
    - high (float): The upper bound of the lognormal distribution.
    - n (int): The number of samples to generate.
    - confidence (int): The confidence level of the distribution. Must be 90,
    95, or 99.

    Returns:
    - samples (ndarray): An array of random samples from the lognormal
    distribution.
    """
    assert low > 0, "Low must be greater than 0 for lognormal distributions."

    if confidence == 90:
        z = 1.645
    elif confidence == 95:
        z = 1.96
    elif confidence == 99:
        z = 2.575
    else:
        raise ValueError("Confidence level must be 90, 95, or 99")
    logHigh = np.log(high)
    logLow = np.log(low)

    mean = np.mean([logHigh, logLow])
    stdev = (logHigh - logLow) / (2 * z)
    scale = np.exp(mean)
    samples = np.abs(lognorm.rvs(s=stdev, scale=scale, size=n))

    return samples


def sample_gpd(shape_param, scale_param, loc_param=1.0, n=1000):
    """Generate random samples from a Generalized Pareto Distribution (GPD).

    Parameters:
    - shape_param (float): The shape parameter of the GPD.
    - scale_param (float): The scale parameter of the GPD.
    - loc_param (float): The location parameter of the GPD (default is 1.0).
    - n (int): The number of samples to generate.

    Returns:
    - samples (ndarray): An array of random samples from the GPD.
    """
    samples = genpareto.rvs(
        c=shape_param, loc=loc_param, scale=scale_param, size=n
    )
    return samples


def run_simulation(
    crr_adjustment,
    time_gathering_water,
    practical_limit_bicycle,
    practical_limit_buckets,
    met,
    watts,
    hill_polarity,
    urban_adjustment,
    rural_adjustment,
    calculate_distance=True,
    use_sample_data=False,
):
    """Run a simulation to analyze global water access from walking or cycling
    based on sensitivity parameters.

    Parameters:
    - crr_adjustment (int): The adjustment factor for the coefficient of
                            rolling resistance.
    - time_gathering_water (int or float): The time taken to gather water in
                                            minutes.
    - practical_limit_bicycle (int or float): The practical limit of water
                                              transportation using a bicycle.
    - practical_limit_buckets (int or float): The practical limit of water
                                              transportation using buckets.
    - met (int or float): The metabolic equivalent of task (MET) value.
    - calculate_distance (bool, optional): Whether to calculate the distance
                                           during the simulation.
    - use_sample_data (bool, optional): Whether to use sample data. Defaults
                                        to False, must be False to run a true
                                        simulation with the given parameters.

    Returns:
    - result: The result of the global analysis.

    Raises:
    - AssertionError: If any of the input parameters are of incorrect type.
    """
    # print type of crr_adjustment
    print(type(crr_adjustment))
    assert isinstance(
        crr_adjustment, (int, np.integer)
    ), "CRR adjustment must be an integer."
    assert isinstance(
        time_gathering_water, (int, float)
    ), "Time gathering water must be a number."
    assert isinstance(
        practical_limit_bicycle, (int, float)
    ), "Practical limit bicycle must be a number."
    assert isinstance(
        practical_limit_buckets, (int, float)
    ), "Practical limit buckets must be a number."
    assert isinstance(met, (int, float)), "MET must be a number."
    assert isinstance(watts, (int, float)), "Watts must be a number."
    assert isinstance(hill_polarity, str), "Hill polarity must be a string."
    assert isinstance(
        urban_adjustment, (int, float)
    ), "Urban adjustment must be a number."
    assert isinstance(
        rural_adjustment, (int, float)
    ), "Rural adjustment must be a number."

    df_countries, df_districts, df_zones = gis.run_global_analysis(
        crr_adjustment=crr_adjustment,
        time_gathering_water=time_gathering_water,
        practical_limit_bicycle=practical_limit_bicycle,
        practical_limit_buckets=practical_limit_buckets,
        met=met,
        watts=watts,
        hill_polarity=hill_polarity,
        urban_adjustment=urban_adjustment,
        rural_adjustment=rural_adjustment,
        calculate_distance=calculate_distance,
        plot=False,
        use_sample_data=use_sample_data,
    )
    return df_countries, df_districts, df_zones


def process_mc_results(
    countries_simulation_results, plot=True, output_dir="results"
):
    """Process the Monte Carlo simulation results. Calculate the median, 95th
    percentile, 5th percentile, max, and min values and plot the results.

    Args:
        simulation_results (list): A list of DataFrames containing simulation
        results.
        plot (bool, optional): Whether to plot the chloropleth maps. Defaults
        to True.
        output_dir (str, optional): The directory to save the results. Defaults
        to "results".

    Returns:
        None
    """

    # Calculate the median, 95th percentile, and 5th percentile of
    # "percent_with_water" for each DataFrame
    ordered_results = sorted(
        countries_simulation_results,
        key=lambda df: df["percent_with_water"].median(),
    )

    # Extract non-numeric columns
    non_numeric_cols = (
        pd.concat(ordered_results)
        .groupby("ISOCODE")
        .first()
        .reset_index()[["ISOCODE", "Entity", "region", "subregion"]]
    )  # noqa

    # Calculate the mean results for each country for all cols
    all_means = (
        pd.concat(ordered_results).groupby("ISOCODE").mean().reset_index()
    )
    # Calculate the median results for each country for all cols
    all_medians = (
        pd.concat(ordered_results).groupby("ISOCODE").median().reset_index()
    )
    # Calculate the 95th percentile results for each country for all cols
    all_percentile_95s = (
        pd.concat(ordered_results)
        .groupby("ISOCODE")
        .quantile(0.95)
        .reset_index()
    )
    # Calculate the 5th percentile results for each country for all cols
    all_percentile_5s = (
        pd.concat(ordered_results)
        .groupby("ISOCODE")
        .quantile(0.05)
        .reset_index()
    )

    # Merge with non-numeric
    all_means = pd.merge(all_means, non_numeric_cols, on="ISOCODE")
    all_medians = pd.merge(all_medians, non_numeric_cols, on="ISOCODE")
    all_percentile_95s = pd.merge(
        all_percentile_95s, non_numeric_cols, on="ISOCODE"
    )
    all_percentile_5s = pd.merge(
        all_percentile_5s, non_numeric_cols, on="ISOCODE"
    )

    # Step 2: Plot the chloropleth maps for max, min, median, 95th percentile,
    # and 5th percentile if plot argument is True
    if plot:
        gis.plot_chloropleth(all_means)
        gis.plot_chloropleth(all_medians)
        gis.plot_chloropleth(all_percentile_95s)
        gis.plot_chloropleth(all_percentile_5s)

    # Step 3: Save the results to the results folder
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # save all-column results
    all_medians.to_csv(os.path.join(output_dir, "country_median_results.csv"))
    all_means.to_csv(os.path.join(output_dir, "country_mean_results.csv"))
    all_percentile_95s.to_csv(
        os.path.join(output_dir, "country_95th_percentile_results.csv")
    )
    all_percentile_5s.to_csv(
        os.path.join(output_dir, "country_5th_percentile_results.csv")
    )

    # Step 4: pickle the simulation results
    with open(
        os.path.join(output_dir, "countries_simulation_results.pkl"), "wb"
    ) as f:
        pickle.dump(countries_simulation_results, f)

    print(
        "Country simulation results have been processed and saved to the "
        "results folder."
    )


def process_districts_results(
    districts_simulation_results, output_dir="results"
):
    """Process the Monte Carlo simulation results. Calculate the median, 95th
    percentile, 5th percentile, max, and min values and plot the results.

    Args:
        simulation_results (list): A list of DataFrames containing simulation
        results.
        plot (bool, optional): Whether to plot the chloropleth maps. Defaults
        to True.
        output_dir (str, optional): The directory to save the results. Defaults
        to "results".

    Returns:
        None
    """

    # Extract non-numeric columns (assuming 'Entity' and 'region' are the
    # non-numeric columns you mentioned)
    non_numeric_cols = (
        pd.concat(districts_simulation_results)
        .groupby("shapeName")
        .first()
        .reset_index()[
            ["shapeName", "ISOCODE", "Entity", "region", "subregion"]
        ]
    )

    # Calculate the mean results for each country for all cols
    all_means = (
        pd.concat(districts_simulation_results)
        .groupby("shapeName")
        .mean()
        .reset_index()
    )
    # Calculate the median results for each country for all cols
    all_medians = (
        pd.concat(districts_simulation_results)
        .groupby("shapeName")
        .median()
        .reset_index()
    )
    # Calculate the 95th percentile results for each country for all cols
    all_percentile_95s = (
        pd.concat(districts_simulation_results)
        .groupby("shapeName")
        .quantile(0.95)
        .reset_index()
    )
    # Calculate the 5th percentile results for each country for all cols
    all_percentile_5s = (
        pd.concat(districts_simulation_results)
        .groupby("shapeName")
        .quantile(0.05)
        .reset_index()
    )

    # Merge with non-numeric
    all_means = pd.merge(all_means, non_numeric_cols, on="shapeName")
    all_medians = pd.merge(all_medians, non_numeric_cols, on="shapeName")
    all_percentile_95s = pd.merge(
        all_percentile_95s, non_numeric_cols, on="shapeName"
    )
    all_percentile_5s = pd.merge(
        all_percentile_5s, non_numeric_cols, on="shapeName"
    )

    # Step 2: Save the results to the results folder
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Use os.path.join to create the full file paths and save

    # save all-column results
    all_medians.to_csv(
        os.path.join(output_dir, "districts_median_results.csv")
    )
    all_means.to_csv(os.path.join(output_dir, "districts_mean_results.csv"))
    all_percentile_95s.to_csv(
        os.path.join(output_dir, "districts_95th_percentile_results.csv")
    )
    all_percentile_5s.to_csv(
        os.path.join(output_dir, "districts_5th_percentile_results.csv")
    )

    # Step 4: pickle the simulation results
    with open(
        os.path.join(output_dir, "districts_simulation_results.pkl"), "wb"
    ) as f:
        pickle.dump(districts_simulation_results, f)

    print(
        "Districts simulation results have been processed and saved to the "
        "results folder."
    )
