import pathlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px

# import weightedstats as ws
import sys
from pathlib import Path
import pdb
import os
import warnings


# ## Import Data from CSVs.
# CSVs created in previous script, which did the cycling mobility on a per country basis
#
# ### GIS Data From QGIS Export
# https://ghsl.jrc.ec.europa.eu/download.php?ds=smod <- urbanisation
#
# https://www.earthenv.org/topography <- topography
#
# https://www.globio.info/download-grip-dataset <- roads


##############################################################################################################################
#
# PRE-PROCESSING
#
##############################################################################################################################

# Assuming the script is in the src directory, set the script_dir accordingly
script_dir = Path(__file__).resolve().parent
repo_root = script_dir.parent
results_dir = repo_root / "results"

# Define paths relative to the src directory
URB_DATA_FILE = repo_root / "data" / "GIS" / "updated_GIS_output_cleaned.csv"
URB_DATA_FILE_SAMPLE = (
    repo_root / "data" / "GIS" / "GIS_data_zones_sample_updated_stripped.csv"
)
# COUNTRY_DATA_FILE = (
#     repo_root / "data" / "processed" / "country_data_master_interpolated.csv"
# )
COUNTRY_DATA_FILE = repo_root / "data" / "processed" / "merged_data.csv"
EXPORT_FILE_LOCATION = repo_root / "data" / "processed"
CRR_FILE = repo_root / "data" / "lookup tables" / "Crr.csv"
FILE_PATH_PARAMS = (
    repo_root / "data" / "lookup tables" / "mobility-model-parameters.csv"
)

# Results File Path
DISTRICT_RESULTS_FILE_PATH = results_dir / "district_results_single_run.csv"
COUNTRY_RESULTS_FILE_PATH = results_dir / "country_results_single_run.csv"


def weighted_mean(var, wts):
    if len(var) == 0 or len(wts) == 0:
        return 0
    elif len(var) != len(wts):
        raise TypeError("Shapes of var and wts must be the same.")
    elif np.sum(wts) == 0:
        raise ZeroDivisionError("Weights sum to zero, can't be normalized.")
    else:
        return np.average(var, weights=wts)


def weighted_median_series(val, weight):
    """Calculates the weighted median
    ArithmeticError
    If the sum of the weights is zero, or if the weights are not positive.
    """
    try:
        df = pd.DataFrame({"val": val, "weight": weight})
        df_sorted = df.sort_values("val")
        cumsum = df_sorted["weight"].cumsum()
        cutoff = df_sorted["weight"].sum() / 2.0
        result = df_sorted[cumsum >= cutoff]["val"].iloc[0]
        # return just the value
    except:
        result = np.nan
    return result


def weighted_median(df, val_column, weight_column):
    """Calculates the weighted median
    ArithmeticError
    If the sum of the weights is zero, or if the weights are not positive.
    """
    df_sorted = df.sort_values(val_column)
    cumsum = df_sorted[weight_column].cumsum()
    cutoff = df_sorted[weight_column].sum() / 2.0
    return df_sorted[cumsum >= cutoff][val_column].iloc[0]


def run_weighted_median_on_grouped_df(df, groupby_column, value_column, weight_column):
    """Calculate the weighted median of a dataframe grouped by a column.
    Args:
        df (pandas.DataFrame): DataFrame to calculate weighted median on.
        groupby_column (str): Column to group by.
        value_column (str): Column to calculate weighted median on.
        weight_column (str): Column to use as weight.
    Returns:
        pandas.DataFrame: DataFrame with weighted median for each group.
    """
    # initialize empty list
    d = []
    # loop through each group
    for i in df[groupby_column].unique():
        df_group = df[df[groupby_column] == i]
        # if rows in dataframe are more than 1, calculate weighted median
        if len(df_group) > 1:
            median = weighted_median(df_group, value_column, weight_column)
        else:
            median = df_group[value_column].values[0]
        d.append(
            {
                groupby_column: i,
                "median": median,
            }
        )
    return pd.DataFrame(d)


def load_data(urb_data_file, country_data_file):
    """
    Load data from CSV files.

    Parameters:
    - urb_data_file (str): The file path of the urban data CSV file.
    - country_data_file (str): The file path of the country data CSV file.

    Returns:
    - df_zones_input (pandas.DataFrame): The loaded urban data as a DataFrame.
    - df_input (pandas.DataFrame): The loaded country data as a DataFrame.
    """
    try:
        df_zones_input = pd.read_csv(urb_data_file)
        df_input = pd.read_csv(country_data_file)
    except:
        df_zones_input = pd.read_csv("." + urb_data_file)
        df_input = pd.read_csv("." + country_data_file)

    return df_zones_input, df_input


# Function for managing urban/rural data
def manage_urban_rural(df_zones_input):
    """
    Converts the 'dtw_1' column from meters to kilometers and creates a new binary column
    'urban_rural' based on the 'GHS_SMOD' column. Rural values are below 15, while urban values
    are above 15.

    Args:
        df_zones_input (pandas.DataFrame): Input DataFrame containing the 'dtw_1' and 'GHS_SMOD' columns.

    Returns:
        pandas.DataFrame: DataFrame with the 'dtw_1' column converted to kilometers and a new 'urban_rural'
        column indicating whether a zone is urban (1) or rural (0).
    """
    # Convert dtw_1 from meters to kilometers
    df_zones_input["dtw_1"] /= 1000
    # Manage Urban / Rural Data
    # Use the GHS_SMOD_E2020_GLOBE_R2023A_54009_1000_V1_0 dataset.
    # from here: https://ghsl.jrc.ec.europa.eu/download.php?ds=smod
    # create new binary column for urban / rural. Rural is below 15, Urban above 15
    df_zones_input["urban_rural"] = np.where(df_zones_input["GHS_SMOD"] > 15, 1, 0)
    return df_zones_input


def adjust_euclidean(df_zones_input, urban_adjustment, rural_adjustment):
    """
    Adjusts distance to water to account for people having to travel on roads and paths instead of
    in straight lines (Euclidean distance), with different adjustments for urban and rural areas.

    Parameters:
    - df_zones_input (pd.DataFrame): DataFrame of GIS zones data
    - urban_adjustment (float): Adjustment ratio for urban areas
    - rural_adjustment (float): Adjustment ratio for rural areas
    """
    # Boolean indexing for urban areas
    urban_mask = df_zones_input["urban_rural"] == 1
    df_zones_input.loc[urban_mask, "dtw_1"] *= urban_adjustment

    # Boolean indexing for rural areas
    rural_mask = df_zones_input["urban_rural"] == 0
    df_zones_input.loc[rural_mask, "dtw_1"] *= rural_adjustment

    # Optionally, raise an error if there are invalid values
    if not df_zones_input["urban_rural"].isin([0, 1]).all():
        raise ValueError("Invalid value (not 0 or 1) in 'urban_rural' column")

    return df_zones_input


# Function for managing slope
def manage_slope(df_zones_input):
    """
    Perform slope management on the input DataFrame.

    Args:
        df_zones_input (pandas.DataFrame): The input DataFrame containing the slope data.

    Returns:
        pandas.DataFrame: The modified DataFrame after performing slope management.
    """
    df_zones_input["slope_1"].hist(bins=100, log=False)
    df_zones_input["slope_1"].quantile(np.arange(0, 1, 0.05))
    return df_zones_input


# Function for merging dataframes and adjusting populations
def merge_and_adjust_population(df_zones_input, df_input):
    """
    Merge and adjust population data based on zone and country information.

    Args:
        df_zones_input (DataFrame): Input DataFrame containing zone information.
        df_input (DataFrame): Input DataFrame containing country information.

    Returns:
        DataFrame: DataFrame with merged and adjusted population data.

    Raises:
        None
    """
    assert not df_zones_input.empty, "df_zones_input is empty"
    assert not df_input.empty, "df_input is empty"
    # this analysis loses some data as the overlap between the rasters is not perfect. To reduce this error, use the 30 arc second data. Too much heavy lifting for my computer to do this at the moment.
    # merge df_input and df_zones on ISO_CC. This assigns all the country data to each zone.
    # join inner will remove some of the data that is not in both datasets
    df_zones = df_zones_input.merge(
        df_input, left_on="ISOCODE", right_on="alpha3", how="inner"
    )

    # convert population density to percent of national population on a per country basis, grouped by ISO_CC
    df_zones["pop_density_perc"] = df_zones.groupby("ISOCODE")["pop_density"].apply(
        lambda x: x / x.sum()
    )
    # multiply population density by population on a per country basis
    df_zones["pop_zone"] = df_zones["pop_density_perc"] * df_zones["Population"]

    # sum the population in each zone
    df_zones["country_pop_raw"] = df_zones.groupby("ISOCODE")["pop_zone"].transform(
        "sum"
    )

    # trim the dataframe to only include rows where there is a population
    df_zones["any_pop"] = df_zones["pop_zone"].apply(lambda x: 1 if x > 10 else 0)
    df_zones = df_zones[df_zones["any_pop"] == 1]

    return df_zones


# Function for road analysis
# def road_analysis_old(df_zones):
#     """
#     Perform road analysis on the given DataFrame.

#     Args:
#         df_zones (pandas.DataFrame): The DataFrame containing the zones data.

#     Returns:
#         pandas.DataFrame: The DataFrame with additional columns for road analysis.

#     """
#     # Define the columns to work with for GRIP data
#     columns = ["grip_1_1", "grip_2_1", "grip_3_1", "grip_4_1", "grip_5_1"]

#     # Create a numpy array from the DataFrame for faster processing
#     data = df_zones[columns].to_numpy()

#     # Find the index of the first non-zero value in each row
#     non_zero_indices = np.argmax(data != 0, axis=1)

#     # Handle rows with all zeros
#     all_zeros = np.all(data == 0, axis=1)
#     non_zero_indices[all_zeros] = -1  # Set a distinct value for rows with all zeros

#     # Map indices to road types
#     road_types = np.array(
#         [
#             "No Roads",
#             "Highways",
#             "Primary Roads",
#             "Secondary Roads",
#             "Tertiary Roads",
#             "Local Roads",
#         ]
#     )

#     # Assign the dominant road type
#     df_zones["dominant_road_type"] = road_types[
#         non_zero_indices + 1
#     ]  # +1 to adjust for 'No Roads'

#     # Load the Crr mapping table
#     df_crr = pd.read_csv(CRR_FILE)

#     # Create a dictionary from the DataFrame
#     crr_mapping = pd.Series(
#         df_crr.Crr.values, index=df_crr["Assigned Zone Surface"]
#     ).to_dict()

#     # Map the road types in df_zones to Crr values
#     df_zones["Crr"] = df_zones["dominant_road_type"].map(crr_mapping)

#     return df_zones


def crr_add_uncertainty(road_type, adjustment):
    """
    The road type is adjusted by the given integer amount, unless the change would go out of bounds.
    In these cases, the adjustment stops at the best or worst road type.

    Parameters:
    road_type (str): The current road type.
    adjustment (int): The amount of adjustment to be made.

    Returns:
    str: The adjusted road type.

    Raises:
    None
    """

    # Define the road types in their logical order from best to worst
    road_types_logical_order = [
        "Highways",
        "Primary Roads",
        "Secondary Roads",
        "Tertiary Roads",
        "Local Roads",
        "No Roads",
    ]

    # Find the current index of the road type
    current_index = road_types_logical_order.index(road_type)

    # Adjust the index within the bounds
    new_index = np.clip(
        current_index + adjustment, 0, len(road_types_logical_order) - 1
    )

    # Return the adjusted road type
    return road_types_logical_order[new_index]


def road_analysis(df_zones, crr_adjustment=0):
    """
    Maps CRR values to road types in df_zones. Adjusts the CRR up or down to add uncertainty.

    Args:
        df_zones (pandas.DataFrame): The DataFrame containing zone data.
        crr_adjustment (int, optional): The adjustment value for Crr mapping. Defaults to 0.

    Returns:
        pandas.DataFrame: The DataFrame with additional columns for dominant road type and Crr values.
    """

    # Define the columns to work with for GRIP data
    columns = ["grip_1_1", "grip_2_1", "grip_3_1", "grip_4_1", "grip_5_1"]

    # Create a numpy array from the DataFrame for faster processing
    data = df_zones[columns].to_numpy()

    # Find the index of the first non-zero value in each row
    non_zero_indices = np.argmax(data != 0, axis=1)

    # Handle rows with all zeros
    all_zeros = np.all(data == 0, axis=1)
    non_zero_indices[all_zeros] = -1  # Set a distinct value for rows with all zeros

    # Map indices to road types
    road_types = np.array(
        [
            "No Roads",
            "Highways",
            "Primary Roads",
            "Secondary Roads",
            "Tertiary Roads",
            "Local Roads",
        ]
    )

    # Assign the dominant road type
    df_zones["dominant_road_type"] = road_types[
        non_zero_indices + 1
    ]  # +1 to adjust for 'No Roads'

    # Load the Crr mapping table
    df_crr = pd.read_csv(CRR_FILE)

    # Add uncertainty (make road type 1 better or 1 worse, unless road type is best (highway) or worst (no roads))
    df_crr["Crr"] = df_crr["Crr"].shift(-crr_adjustment)
    df_crr["Crr"] = df_crr["Crr"].ffill().bfill()

    # Create a series from the DataFrame
    crr_mapping = pd.Series(df_crr.Crr.values, index=df_crr["Assigned Zone Surface"])

    # Create a dictionary from the series
    crr_mapping = crr_mapping.to_dict()

    # Map the road types in df_zones to Crr values
    df_zones["Crr"] = df_zones["dominant_road_type"].map(crr_mapping)

    return df_zones


def calculate_nat_piped(df_zones):
    """
    Calculates National piped using the formula
    [ URBANPiped ×‘% urban’ ÷100+(100−‘% urban’ )÷100×RURALPiped ]

    """
    df_zones["NATPiped"] = (
        df_zones["URBANPiped"] * df_zones["% urban"] / 100
        + (100 - df_zones["% urban"]) / 100 * df_zones["RURALPiped"]
    )
    return df_zones


# Main function to run all steps
def preprocess_data(
    crr_adjustment, urban_adjustment, rural_adjustment, use_sample_data=False
):
    """
    Preprocesses data by loading, managing, merging, and adjusting population data.

    Args:
        crr_adjustment (float): The adjustment factor for road analysis.

    Returns:
        pandas.DataFrame: The preprocessed data.

    """

    if use_sample_data:
        urban_data_file = URB_DATA_FILE_SAMPLE

        warnings.warn(
            "Using sample data. This should only be done for testing, and not for generating real model results!"
        )
    else:
        urban_data_file = URB_DATA_FILE

    df_zones_input, df_input = load_data(urban_data_file, COUNTRY_DATA_FILE)
    df_zones_input = manage_urban_rural(df_zones_input)
    df_zones_input = adjust_euclidean(
        df_zones_input,
        urban_adjustment=urban_adjustment,
        rural_adjustment=rural_adjustment,
    )
    df_zones_input = manage_slope(df_zones_input)
    df_zones = merge_and_adjust_population(df_zones_input, df_input)
    df_zones = road_analysis(df_zones, crr_adjustment=crr_adjustment)
    df_zones = calculate_nat_piped(df_zones)
    return df_zones


##############################################################################################################################
#
# ANALYSIS I: Models
#
##############################################################################################################################


def load_hpv_parameters(file_path_params, hpv_name):
    """
    Load HPV parameters from a CSV file and return the subset of parameters for a specific HPV name.

    Parameters:
    - file_path_params (str): The file path to the CSV file containing the HPV parameters.
    - hpv_name (str): The name of the HPV for which to retrieve the parameters.

    Returns:
    - pandas.DataFrame: A DataFrame containing the subset of parameters for the specified HPV name.
    """
    allHPV_param_df = pd.read_csv(file_path_params)
    return allHPV_param_df[allHPV_param_df["Name"] == hpv_name]


def extract_slope_crr(df_zones):
    """
    Extracts slope and Crr values from the given DataFrame.

    Args:
        df_zones (pandas.DataFrame): The DataFrame containing slope and Crr values.

    Returns:
        tuple: A tuple containing two pandas.Series objects - slope_zones and Crr_values.
               slope_zones: The series containing slope values.
               Crr_values: The series containing Crr values.
    """
    df_zones["Crr"] = df_zones["Crr"].astype(float)
    # TODO update test to use country specific body mass
    country_average_weights = df_zones["Average Weight"]
    slope_zones = df_zones["slope_1"]
    Crr_values = df_zones["Crr"]
    return slope_zones, Crr_values, country_average_weights


def run_bicycle_model(
    mv, mo, hpv, slope_zones, Crr_values, country_average_weights, load_attempt
):
    """
    Runs a bicycle model for different slope zones and Crr values.

    Args:
        mv: Model variables.
        mo: Model options.
        hpv (object): Object representing the human-powered vehicle.
        slope_zones (list): List of slope zones.
        Crr_values (list): List of Crr values.
        load_attempt (int): Load attempt number.

    Returns:
        numpy.ndarray: Array of results for each slope zone and Crr value.
    """
    # Adjust the project root and import mobility module as needed
    project_root = Path().resolve().parent
    sys.path.append(str(project_root))
    import src.mobility_module as mm

    n_runs = len(slope_zones)
    results = np.zeros((n_runs, 3))
    # TODO update test to include country_average_weight
    for i, (slope, crr, country_average_weight) in enumerate(
        zip(slope_zones, Crr_values, country_average_weights)
    ):
        hpv.Crr = crr
        mv.m1 = country_average_weight
        results[i] = mm.mobility_models.single_bike_run(
            # need to modify single_bike_run function to take country_average_weight instead of m1
            # (done by updated mv to take country_average_weight as m1)
            mv,
            mo,
            hpv,
            slope,
            load_attempt,
        )
    # for i, (slope, crr) in enumerate(zip(slope_zones, Crr_values)):
    #     hpv.Crr = crr
    #     results[i] = mm.mobility_models.single_bike_run(
    #         mv, mo, hpv, slope, load_attempt
    #     )
    return results


def process_and_save_results(
    df_zones, results, export_file_location, velocity_type, save_csv=False
):
    """
    Process the results and optionally save them to a DataFrame.
    The results should only be saved for single runs, as saving all monte carlo csvs will be too large.

    Args:
        df_zones (pandas.DataFrame): The DataFrame containing the zones data.
        results (numpy.ndarray): The results array containing the velocity vectors.
        export_file_location (str): The file location where the output CSV will be saved.
        velocity_type (str): The type of velocity (e.g., walk/bicycle).
        save_csv (bool, optional): Whether to save the results as a CSV file. Defaults to False.

    Returns:
        pandas.DataFrame: The updated DataFrame with the new columns.

    """
    # Unpack results
    loaded_velocity_vec, unloaded_velocity_vec, max_load_vec = results.T

    # Average velocity between loaded and unloaded
    average_velocity = (loaded_velocity_vec + unloaded_velocity_vec) / 2

    # Customizing column names based on the type of velocity
    loaded_velocity_col = f"loaded_velocity_{velocity_type}"
    unloaded_velocity_col = f"unloaded_velocity_{velocity_type}"
    average_velocity_col = f"average_velocity_{velocity_type}"
    max_load_col = f"max_load_{velocity_type}"

    # Updating the DataFrame with the new columns
    df_zones[loaded_velocity_col] = loaded_velocity_vec
    df_zones[unloaded_velocity_col] = unloaded_velocity_vec
    df_zones[average_velocity_col] = average_velocity
    df_zones[max_load_col] = max_load_vec

    if save_csv:
        # Customizing the output CSV filename based on the type of velocity
        output_csv_filename = os.path.join(
            export_file_location, f"{velocity_type}_velocity_by_zone.csv"
        )

        # only save the newly created cols in the csv
        df_zones[
            [
                "fid",
                loaded_velocity_col,
                unloaded_velocity_col,
                average_velocity_col,
                max_load_col,
            ]
        ].to_csv(output_csv_filename)

    return df_zones


# function to map hill polarity strings to integers for lhillpo and ulhillpo, using a dictionary for the mapping
def map_hill_polarity(hill_polarity):
    """
    Maps the hill polarity string to integers for the unloaded trip (ulhillpo) and the loaded trip back (lhillpo).

    Args:
        hill_polarity (str): The hill polarity string.

    Returns:
        int, int: The integer values for ulhillpo and lhillpo.
    """
    hill_polarity_mapping = {
        "uphill_downhill": (1, -1),
        "downhill_uphill": (-1, 1),
        "uphill_flat": (1, 0),
        "flat_uphill": (0, 1),
        "downhill_flat": (-1, 0),
        "flat_downhill": (0, -1),
        "flat_flat": (0, 0),
    }
    ulhillpo, lhillpo = hill_polarity_mapping[hill_polarity]

    return ulhillpo, lhillpo


def calculate_and_merge_bicycle_distance(
    df_zones,
    calculate_distance,
    export_file_location,
    practical_limit_bicycle=40,
    watts=75,
    human_mass=62,
    hill_polarity="flat_uphill",
):
    """
    Calculates and merges bicycle distance for each zone in the given dataframe.

    Args:
        df_zones (pandas.DataFrame): The dataframe containing zone information.
        calculate_distance (bool): Flag indicating whether to calculate the distance or not.
        export_file_location (str): The file location to export the results.
        practical_limit_bicycle (int, optional): The practical limit for bicycle distance in kg. Defaults to 40.

    Returns:
        pandas.DataFrame: The dataframe with bicycle distance merged.

    Raises:
        FileNotFoundError: If the mobility model parameters file is not found.
    """
    if calculate_distance:
        # Adjust the project root and import mobility module as needed
        project_root = Path().resolve().parent
        sys.path.append(str(project_root))
        import src.mobility_module as mm

        file_path_params = FILE_PATH_PARAMS
        param_df = load_hpv_parameters(file_path_params, "Bicycle")

        # Overwrite practical limit with the function inputs
        param_df["PracticalLimit"] = practical_limit_bicycle

        # Get integer values for hill polarity
        ulhillpo, lhillpo = map_hill_polarity(hill_polarity)

        # Initialise mobility model options, using updated hill polarities
        mo = mm.model_options(ulhillpo=ulhillpo, lhillpo=lhillpo)
        mo.model_selection = 2  # Cycling model

        # Extract values, initalise vars, and run model
        slope_zones, Crr_values, country_average_weights = extract_slope_crr(df_zones)
        mv = mm.model_variables(P_t=watts)
        hpv = mm.HPV_variables(param_df, mv)

        results = run_bicycle_model(
            mv,
            mo,
            hpv,
            slope_zones,
            Crr_values,
            country_average_weights,
            load_attempt=15,
        )
        process_and_save_results(df_zones, results, export_file_location, "bicycle")
    else:
        df_zones_bicycle = pd.read_csv(
            export_file_location + "bicycle_velocity_by_zone.csv"
        )
        df_zones = df_zones.merge(df_zones_bicycle, on="fid", how="left")
    return df_zones


def run_walking_model(
    mv, mo, met, hpv, slope_zones, country_average_weights, load_attempt
):
    """
    Run the walking model for multiple slope zones.

    Args:
        mv: Model variables.
        mo: Model options.
        met: Metabolic equivalent of task.
        hpv (object): Object representing the human-powered vehicle.
        slope_zones (list): List of slope zones.
        Crr_values (list): List of Crr values.
        load_attempt (float): Load attempt number.

    Returns:
        numpy.ndarray: An array of results for each slope zone.

    """
    # Adjust the project root and import mobility module as needed
    project_root = Path().resolve().parent
    sys.path.append(str(project_root))
    import src.mobility_module as mm

    n_runs = len(slope_zones)
    results = np.zeros((n_runs, 3))

    # Run model for each zone with country-specific weights
    for i, (slope, country_average_weight) in enumerate(
        zip(slope_zones, country_average_weights)
    ):
        mv.m1 = country_average_weight
        met_values = mm.MET_values(
            mv,
            country_weight=country_average_weight,
            met=met,
            use_country_specific_weights=True,
        )
        results[i] = mm.mobility_models.single_lankford_run(
            mv, mo, met_values, hpv, slope, load_attempt
        )
    return results


def calculate_and_merge_walking_distance(
    df_zones,
    calculate_distance,
    export_file_location,
    practical_limit_buckets=20,
    met=4.5,
    human_mass=62,
    hill_polarity="flat_uphill",
):
    """
    Calculate and merge walking distance for zones.

    Args:
        df_zones (pandas.DataFrame): The input DataFrame containing zone data.
        calculate_distance (bool): Flag indicating whether to calculate the walking distance.
        export_file_location (str): The file location to export the results.
        practical_limit_buckets (int, optional): The practical limit buckets. Defaults to 20.
        met (float, optional): The MET value. Defaults to 3.3.

    Returns:
        pandas.DataFrame: The updated DataFrame with walking distance information.
    """
    if calculate_distance:
        # Adjust the project root and import mobility module as needed
        project_root = Path().resolve().parent
        sys.path.append(str(project_root))
        import src.mobility_module as mm

        file_path_params = FILE_PATH_PARAMS
        param_df = load_hpv_parameters(file_path_params, "Buckets")

        # Overwrite practical limit with the function inputs
        param_df["PracticalLimit"] = practical_limit_buckets

        # Get integer values for hill polarity
        ulhillpo, lhillpo = map_hill_polarity(hill_polarity)

        # Initialise mobility model options, using updated hill polarities
        mo = mm.model_options(ulhillpo=ulhillpo, lhillpo=lhillpo)
        mo.model_selection = 3  # Lankford model
        mv = mm.model_variables(m1=human_mass)
        hpv = mm.HPV_variables(param_df, mv)

        # Run model for each zone with country-specific weights
        slope_zones, Crr_values, country_average_weights = extract_slope_crr(df_zones)
        results = run_walking_model(
            mv, mo, met, hpv, slope_zones, country_average_weights, load_attempt=15
        )
        process_and_save_results(df_zones, results, export_file_location, "walk")
    else:
        df_zones_walking = pd.read_csv(
            export_file_location + "walk_velocity_by_zone.csv"
        )
        df_zones = df_zones.merge(df_zones_walking, on="fid", how="left")
    return df_zones


##############################################################################################################################
#
# ANALYSIS II: Population Water Access
#
##############################################################################################################################


def calculate_max_distances(df_zones, time_gathering_water):
    """
    Calculate the maximum distances achievable for gathering water in each zone.

    Args:
        df_zones (pandas.DataFrame): DataFrame containing zone information.
        time_gathering_water (float): Time taken to gather water in hours.

    Returns:
        pandas.DataFrame: DataFrame with additional columns for max distances and water ration.

    """
    # Max distance achievable (not round trip, just the distance from home to water source)
    df_zones["max distance cycling"] = (
        df_zones["average_velocity_bicycle"] * time_gathering_water / 2
    )
    df_zones["max distance walking"] = (
        df_zones["average_velocity_walk"] * time_gathering_water / 2
    )
    # Use water_ration_kms to calculate the water ration achievable per bike per zone
    df_zones["water_ration_kms"] = (
        df_zones["max distance cycling"] * df_zones["max_load_bicycle"]
    )
    return df_zones


def calculate_population_water_access(df_zones):
    """
    Calculates the population with and without access to water for each zone.

    Args:
        df_zones (pandas.DataFrame): DataFrame containing zone information.

    Returns:
        pandas.DataFrame: DataFrame with additional columns representing the population
        with and without access to water for each zone.
    """
    # Set df_zones["zone_pop_piped"] to 0 for all zones to begin with
    df_zones["zone_pop_piped"] = 0

    # If urban use urban piped and unpiped, if rural use rural piped and unpiped
    # Use the urban_rural column to do this
    df_zones["zone_pop_piped"] = (
        df_zones["pop_zone"] * df_zones["urban_rural"] * df_zones["URBANPiped"] / 100
        + df_zones["pop_zone"]
        * (1 - df_zones["urban_rural"])
        * df_zones["RURALPiped"]
        / 100
    )
    df_zones["zone_pop_unpiped"] = (
        df_zones["pop_zone"]
        * df_zones["urban_rural"]
        * (100 - df_zones["URBANPiped"])
        / 100
        + df_zones["pop_zone"]
        * (1 - df_zones["urban_rural"])
        * (100 - df_zones["RURALPiped"])
        / 100
    )

    # Determine if it's possible to reach water with walking/cycling
    df_zones["zone_cycling_okay"] = (
        df_zones["dtw_1"] < df_zones["max distance cycling"]
    ) * 1
    df_zones["zone_walking_okay"] = (
        df_zones["dtw_1"] < df_zones["max distance walking"]
    ) * 1

    # Calculate the fraction of the zone that can collect water by cycling
    df_zones["fraction_of_zone_with_cycling_access"] = df_zones["zone_cycling_okay"] * (
        df_zones["PBO"] / 100
    )
    df_zones["fraction_of_zone_with_walking_access"] = df_zones["zone_walking_okay"] * 1

    # Calculate the population that can access piped water by cycling and walking
    df_zones["population_piped_with_cycling_access"] = (
        df_zones["fraction_of_zone_with_cycling_access"] * df_zones["zone_pop_piped"]
    )
    df_zones["population_piped_with_walking_access"] = (
        df_zones["fraction_of_zone_with_walking_access"] * df_zones["zone_pop_piped"]
    )
    # Select the maximum between the two, if walkable, max will always be walking
    df_zones["population_piped_with_access"] = df_zones[
        ["population_piped_with_cycling_access", "population_piped_with_walking_access"]
    ].max(axis=1)

    # Calculate the population that can access water only by cycling
    # Equals population_piped_with_cycling_access if zone_walking_okay is 0
    df_zones["population_piped_with_only_cycling_access"] = df_zones[
        "population_piped_with_cycling_access"
    ] * (1 - df_zones["zone_walking_okay"])

    # Calculate zone population with and without water access
    df_zones["zone_pop_with_water"] = (
        df_zones["population_piped_with_access"] + df_zones["zone_pop_unpiped"]
    )
    df_zones["zone_pop_without_water"] = (
        df_zones["pop_zone"] - df_zones["zone_pop_with_water"]
    )
    return df_zones


def calculate_water_rations(df_zones):
    """
    Calculate the water rations achievable utilizing all the bikes in each zone.

    Parameters:
    - df_zones (pandas.DataFrame): A DataFrame containing the zone data.

    Returns:
    - df_zones (pandas.DataFrame): The input DataFrame with additional columns for water rations.

    The function calculates the water rations achievable by dividing the water ration distance (water_ration_kms)
    by the distance to water source (dtw_1) for each zone. It then calculates the number of bikes in each zone
    by dividing the population of the zone by the average household size and multiplying it by the PBO (Percent Bike Ownership) factor.
    Finally, it calculates the total water rations achievable in each zone by multiplying the number of bikes in the zone
    by the water rations per bike.

    Example usage:
    >>> df = calculate_water_rations(df_zones)
    """

    df_zones["water_rations_per_bike"] = (
        df_zones["water_ration_kms"] / df_zones["dtw_1"]
    )
    df_zones["bikes_in_zone"] = (
        df_zones["pop_zone"]
        / df_zones["Household_Size"]
        * df_zones["PBO"]
        / 100  # must divide by 100 as PBO is percentage.
    )
    df_zones["water_rations_achievable"] = (
        df_zones["bikes_in_zone"] * df_zones["water_rations_per_bike"]
    )
    return df_zones


def process_zones_for_water_access(df_zones, time_gathering_water=16):
    df_zones = calculate_max_distances(df_zones, time_gathering_water)
    df_zones = calculate_population_water_access(df_zones)
    df_zones = calculate_water_rations(df_zones)
    return df_zones


##############################################################################################################################
#
# ANALYSIS III: Aggregating Country Data
#
##############################################################################################################################


def aggregate_country_level_data(df_zones):
    """
    Aggregate zone data into country level summaries.

    Parameters:
        df_zones (DataFrame): The input DataFrame containing zone-level data.

    Returns:
        df_countries (DataFrame): The aggregated DataFrame with country-level summaries.
    """
    df_countries = (
        df_zones.groupby("ISOCODE")
        .agg(
            {
                "Entity": "first",
                "country_pop_raw": "first",
                "zone_pop_with_water": "sum",
                "zone_pop_without_water": "sum",
                "population_piped_with_access": "sum",
                "population_piped_with_cycling_access": "sum",
                "population_piped_with_walking_access": "sum",
                "population_piped_with_only_cycling_access": "sum",
                "NATPiped": "first",
                "region": "first",
                "subregion": "first",
                # Additional aggregations can be added here
            }
        )
        .reset_index()
    )
    # Rename the columns
    df_countries.columns = [
        "ISOCODE",
        "Entity",
        "country_pop_raw",
        "zone_pop_with_water",
        "zone_pop_without_water",
        "population_piped_with_access",
        "population_piped_with_cycling_access",
        "population_piped_with_walking_access",
        "population_piped_with_only_cycling_access",
        "NATPiped",
        "region",
        "subregion",
    ]

    return df_countries


def weighted_percentile(df, val_column, weight_column, percentile):
    """Calculates the weighted percentile
    ArithmeticError
    If the sum of the weights is zero, or if the weights are not positive.
    """
    df_sorted = df.sort_values(val_column)
    cumsum = df_sorted[weight_column].cumsum()
    cutoff = df_sorted[weight_column].sum() * percentile / 100.0
    return df_sorted[cumsum >= cutoff][val_column].iloc[0]


def calculate_weighted_results(df_zones):
    """
    Calculate the weighted median for each country group.
    """
    df_median_group = (
        df_zones.groupby("ISOCODE")
        .apply(
            lambda x: pd.Series(
                {
                    "weighted_med": weighted_median(x, "dtw_1", "pop_zone"),
                    "weighted_med_cycling": weighted_percentile(
                        x, "max distance cycling", "pop_zone", 50
                    ),
                    "weighted_5th_cycling": weighted_percentile(
                        x, "max distance cycling", "pop_zone", 5
                    ),
                    "weighted_95th_cycling": weighted_percentile(
                        x, "max distance cycling", "pop_zone", 95
                    ),
                    "weighted_med_walking": weighted_percentile(
                        x, "max distance walking", "pop_zone", 50
                    ),
                    "weighted_5th_walking": weighted_percentile(
                        x, "max distance walking", "pop_zone", 5
                    ),
                    "weighted_95th_walking": weighted_percentile(
                        x, "max distance walking", "pop_zone", 95
                    ),
                }
            )
        )
        .reset_index()
    )

    return df_median_group


def aggregate_global(df_zones):
    """
    Get global data to be added to the country-level data.

    Parameters:
        df_zones (DataFrame): The input DataFrame containing zone-level data.

    Returns:
        df_global (DataFrame): One row of global data
    """
    # Calculate the sum for the specified columns
    sum_data = {
        "country_pop_raw": df_zones["pop_zone"].sum(),
        "zone_pop_with_water": df_zones["zone_pop_with_water"].sum(),
        "zone_pop_without_water": df_zones["zone_pop_without_water"].sum(),
        "population_piped_with_access": df_zones["population_piped_with_access"].sum(),
        "population_piped_with_cycling_access": df_zones[
            "population_piped_with_cycling_access"
        ].sum(),
        "population_piped_with_walking_access": df_zones[
            "population_piped_with_walking_access"
        ].sum(),
        "population_piped_with_only_cycling_access": df_zones[
            "population_piped_with_only_cycling_access"
        ].sum(),
    }

    # Calculate the weighted median and the 5th and 95th percentiles
    weighted_data = {
        "weighted_med": weighted_median(df_zones, "dtw_1", "pop_zone"),
        "weighted_med_cycling": weighted_median(
            df_zones, "max distance cycling", "pop_zone"
        ),
        "weighted_5th_cycling": weighted_percentile(
            df_zones, "max distance cycling", "pop_zone", 5
        ),
        "weighted_95th_cycling": weighted_percentile(
            df_zones, "max distance cycling", "pop_zone", 95
        ),
        "weighted_med_walking": weighted_median(
            df_zones, "max distance walking", "pop_zone"
        ),
        "weighted_5th_walking": weighted_percentile(
            df_zones, "max distance walking", "pop_zone", 5
        ),
        "weighted_95th_walking": weighted_percentile(
            df_zones, "max distance walking", "pop_zone", 95
        ),
    }

    # Create a new DataFrame with the global row
    df_global = pd.DataFrame(
        [{"Entity": "Global", "ISOCODE": "GLOBAL", **sum_data, **weighted_data}]
    )

    return df_global


def clean_up_data(df_countries):
    """
    Clean up data from spurious country values.

    Parameters:
    - df_countries (pandas.DataFrame): The input dataframe containing country data.

    Returns:
    - df_countries (pandas.DataFrame): The cleaned dataframe with spurious country values removed.
    - countries_further_than_libya (pandas.DataFrame): The dataframe containing countries with distances greater than the maximum distance to water.
    - list_of_countries_to_remove (list): The list of specific countries manually removed from the dataframe.
    """
    # df_countries = df_countries.dropna()  # Remove any NaN rows

    # Remove outliers based on max possible distance to water
    libya_weighted_med = df_countries.loc[
        df_countries["ISOCODE"] == "LBY", "weighted_med"
    ].values

    if len(libya_weighted_med) > 0:
        max_distance = libya_weighted_med[0] + 1
        countries_further_than_libya = df_countries[
            df_countries["weighted_med"] > max_distance
        ]
        df_countries = df_countries[df_countries["weighted_med"] < max_distance]
    else:
        countries_further_than_libya = pd.DataFrame()
        print(
            "Warning: No data for Libya (ISOCODE 'LBY'). Skipping outlier removal based on Libya's weighted_med."
        )

    # Manually remove specific countries
    list_of_countries_to_remove = [
        "GUM",
        "ASM",
        "TON",
        "MNP",
        "ATG",
        "DMA",
        "ABW",
        "BRB",
    ]
    df_countries = df_countries[
        ~df_countries["ISOCODE"].isin(list_of_countries_to_remove)
    ]

    # Return the cleaned dataframe and lists of removed countries for logging or review
    return df_countries, countries_further_than_libya, list_of_countries_to_remove


def process_country_data(df_zones):
    """
    Orchestrate the processing of zone data into country-level summaries and cleanup.

    Args:
        df_zones (pandas.DataFrame): The input dataframe containing zone-level data.

    Returns:
        pandas.DataFrame: The processed dataframe containing country-level summaries.
    """
    assert not df_zones.empty, "Input dataframe is empty"
    warnings.warn("Input dataframe contains NaN values")

    df_countries = aggregate_country_level_data(df_zones)
    assert not df_countries.empty, "Country-level dataframe is empty"
    # TODO check this out (assert raises error, but probably not an issue as some NaNs are expected?)
    # assert not df_countries.isnull().values.any(), "Country-level dataframe contains NaN values"

    df_median_group = calculate_weighted_results(df_zones)
    assert not df_median_group.empty, "Weighted median dataframe is empty"
    # TODO check this out (assert raises error, but probably not an issue as some NaNs are expected?)
    # assert not df_median_group.isnull().values.any(), "Weighted median dataframe contains NaN values"

    df_countries = df_countries.merge(
        df_median_group, on="ISOCODE"
    )  # Merge weighted median

    # Add global row
    df_global = aggregate_global(df_zones)
    # replace country_pop_raw with sum of countries
    df_global["country_pop_raw"] = df_countries["country_pop_raw"].sum()

    df_countries = pd.concat([df_countries, df_global], ignore_index=True)

    # drop rows from the dataframe that have Nan in pop_zone and dtw_1
    df_zones = df_zones.dropna(subset=["pop_zone", "dtw_1"])

    # Rename and create percentage columns
    df_countries = df_countries.rename(
        columns={
            "zone_pop_with_water": "country_pop_with_water",
            "zone_pop_without_water": "country_pop_without_water",
        }
    )

    # Check if 'country_pop_with_water' exists in df_countries
    if "country_pop_with_water" not in df_countries.columns:
        raise KeyError(
            "Column 'country_pop_with_water' does not exist in df_countries. Available columns: {}".format(
                df_countries.columns
            )
        )

    df_countries["percent_with_water"] = (
        df_countries["country_pop_with_water"] / df_countries["country_pop_raw"] * 100
    )
    df_countries["percent_without_water"] = (
        df_countries["country_pop_without_water"]
        / df_countries["country_pop_raw"]
        * 100
    )

    # add percentage columns for cycling and water access
    df_countries["percent_piped_with_cycling_access"] = (
        df_countries["population_piped_with_cycling_access"]
        / df_countries["country_pop_raw"]
        * 100
    )

    df_countries["percent_piped_with_walking_access"] = (
        df_countries["population_piped_with_walking_access"]
        / df_countries["country_pop_raw"]
        * 100
    )

    df_countries["proportion_piped_access_from_cycling"] = (
        df_countries["population_piped_with_only_cycling_access"]
        / df_countries["population_piped_with_access"]
        * 100
    )

    df_countries["percent_with_only_cycling_access"] = (
        df_countries["population_piped_with_only_cycling_access"]
        / df_countries["country_pop_raw"]
        * 100
    )

    df_countries, removed_further_than_libya, removed_countries_list = clean_up_data(
        df_countries
    )

    # Log or print summary of removed countries
    if not removed_further_than_libya.empty:
        print(
            "Countries removed from analysis due to being further than Libya's median:",
            removed_further_than_libya["Entity"].tolist(),
        )
    print("Countries removed manually:", removed_countries_list)

    return df_countries


##############################################################################################################################
#
# ANALYSIS IV: Aggregating District Data
#
##############################################################################################################################


def aggregate_district_level_data(df_zones):
    """
    Aggregate zone data into district level summaries.

    Parameters:
        df_zones (DataFrame): The input DataFrame containing zone-level data.

    Returns:
        df_countries (DataFrame): The aggregated DataFrame with district-level summaries.
    """

    df_zones["district_pop_raw"] = df_zones.groupby("shapeID")["pop_zone"].transform(
        "sum"
    )

    df_districts = (
        df_zones.groupby("shapeID")
        .agg(
            {
                "Entity": "first",
                "ISOCODE": "first",
                "shapeName": "first",
                "district_pop_raw": "first",
                "zone_pop_with_water": "sum",
                "zone_pop_without_water": "sum",
                "population_piped_with_access": "sum",
                "population_piped_with_cycling_access": "sum",
                "population_piped_with_walking_access": "sum",
                "population_piped_with_only_cycling_access": "sum",
                "NATPiped": "first",
                "region": "first",
                "subregion": "first",
            }
        )
        .reset_index()
    )
    # Rename the columns
    df_districts.columns = [
        "shapeID",
        "Entity",
        "ISOCODE",
        "shapeName",
        "district_pop_raw",
        "zone_pop_with_water",
        "zone_pop_without_water",
        "population_piped_with_access",
        "population_piped_with_cycling_access",
        "population_piped_with_walking_access",
        "population_piped_with_only_cycling_access",
        "NATPiped",
        "region",
        "subregion",
    ]

    return df_districts


def process_district_data(df_zones):
    """
    Orchestrate the processing of zone data into district-level summaries and cleanup.

    Args:
        df_zones (pandas.DataFrame): The input dataframe containing zone-level data.

    Returns:
        pandas.DataFrame: The processed dataframe containing district-level summaries.
    """
    assert not df_zones.empty, "Input dataframe is empty"

    df_districts = aggregate_district_level_data(df_zones)

    # use groupby to create weighted median, needs to be speerate from the above groupby as it uses apply, which can't be used in the same groupby
    # needs to use apply because the function required two columns as input
    df_median_group = df_zones.groupby(["shapeID"]).apply(
        lambda x: pd.Series({"weighted_med": weighted_median(x, "dtw_1", "pop_zone")})
    )

    # merge the weighted median back into the df_districts dataframe
    df_districts = df_districts.merge(df_median_group, on="shapeID")

    # drop rows from the dataframe that have Nan in pop_zone and dtw_1
    df_zones = df_zones.dropna(subset=["pop_zone", "dtw_1"])

    # create summary columns
    # rename zone columns to country
    df_districts = df_districts.rename(
        columns={
            "zone_pop_with_water": "district_pop_with_water",
            "zone_pop_without_water": "district_pop_without_water",
        }
    )

    # create percent
    df_districts["percent_with_water"] = (
        df_districts["district_pop_with_water"] / df_districts["district_pop_raw"] * 100
    )
    df_districts["percent_without_water"] = (
        df_districts["district_pop_without_water"]
        / df_districts["district_pop_raw"]
        * 100
    )

    # add percentage columns for cycling and water access
    df_districts["percent_piped_with_cycling_access"] = (
        df_districts["population_piped_with_cycling_access"]
        / df_districts["district_pop_raw"]
        * 100
    )

    df_districts["percent_piped_with_walking_access"] = (
        df_districts["population_piped_with_walking_access"]
        / df_districts["district_pop_raw"]
        * 100
    )

    df_districts["proportion_piped_access_from_cycling"] = (
        df_districts["population_piped_with_only_cycling_access"]
        / df_districts["population_piped_with_access"]
        * 100
    )

    df_districts["percent_with_only_cycling_access"] = (
        df_districts["population_piped_with_only_cycling_access"]
        / df_districts["district_pop_raw"]
        * 100
    )

    list_of_countries_to_remove = [
        "GUM",
        "ASM",
        "TON",
        "MNP",
        "ATG",
        "DMA",
        "ABW",
        "BRB",
    ]
    df_districts = df_districts[
        ~df_districts["ISOCODE"].isin(list_of_countries_to_remove)
    ]

    return df_districts


##############################################################################################################################
#
# VISUALISATIONS
#
##############################################################################################################################


def plot_chloropleth(df_countries):
    hover_data_list = [
        "ISOCODE",
        "country_pop_raw",
        "country_pop_with_water",
        "country_pop_without_water",
        "population_piped_with_access",
        "population_piped_with_cycling_access",
        "population_piped_with_walking_access",
        "percent_without_water",
        "percent_with_water",
        "NATPiped",
        "region",
        "subregion",
        "weighted_med",
    ]

    choro = px.choropleth(
        title="Percent of Population Has to Relocate",
        data_frame=df_countries,
        locations="ISOCODE",
        height=600,
        color="percent_without_water",
        # use constant colorbar grading (not relative)
        color_continuous_scale="ylorbr",
        # color_continuous_scale="YlGnBu_r",
        # color_continuous_scale="PuRd",
        # color_continuous_scale="Greys",
        range_color=(0, 100),
        scope="world",
        hover_name="ISOCODE",
        hover_data=hover_data_list,
    )
    choro.layout.coloraxis.colorbar.title = ""
    choro.show()


##############################################################################################################################
#
# MAIN Function
#
##############################################################################################################################


def run_global_analysis(
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
    plot=False,
    human_mass=62,  # gets overridden by country specific weight
    use_sample_data=False,  # only change to test functionality
):
    """
    Runs one run of the global analysis for water access.

    Args:
        crr_adjustment (float): The adjustment factor for calculating the Coefficient of Rolling Resistance (CRR).
        time_gathering_water (float): The time taken to gather water in minutes.
        practical_limit_bicycle (float): The practical limit for distance traveled by bicycle in kilometers.
        practical_limit_buckets (float): The practical limit for distance traveled by carrying buckets in kilometers.
        met (str): metabolic equivalent of task (energy for walking).
        watts (float): The power output in watts (energy for cycling).
        calculate_distance (bool, optional): Whether to calculate distance or not. Defaults to True, otherwise new results are not generated.
        human_mass (float): The mass of an average human carrying water in kilograms.
        plot (bool, optional): Whether to plot the chloropleth map or not. Defaults to False.
        use_sample_data (bool, optional): Whether to use sample data or not (only for testing). Defaults to False.

    Returns:
        pandas.DataFrame: The processed data for each country.
    """
    df_zones = preprocess_data(
        crr_adjustment=crr_adjustment,
        urban_adjustment=urban_adjustment,
        rural_adjustment=rural_adjustment,
        use_sample_data=use_sample_data,
    )
    df_zones = calculate_and_merge_bicycle_distance(
        df_zones,
        calculate_distance=calculate_distance,
        export_file_location=EXPORT_FILE_LOCATION,
        practical_limit_bicycle=practical_limit_bicycle,
        watts=watts,
        human_mass=human_mass,
        hill_polarity=hill_polarity,
    )
    df_zones = calculate_and_merge_walking_distance(
        df_zones,
        calculate_distance=calculate_distance,
        export_file_location=EXPORT_FILE_LOCATION,
        practical_limit_buckets=practical_limit_buckets,
        met=met,
        human_mass=human_mass,
        hill_polarity=hill_polarity,
    )
    df_zones = process_zones_for_water_access(
        df_zones, time_gathering_water=time_gathering_water
    )

    # print("\n\n 1 \n\n")
    # print(len(df_zones["ISOCODE"].unique()))

    # add df_districts
    df_zones_districts = df_zones.copy()
    df_districts = process_district_data(df_zones_districts)
    df_countries = process_country_data(df_zones)

    # drop countries in df_distciricts if they don't appear in df_copuntries
    df_districts = df_districts[df_districts["ISOCODE"].isin(df_countries["ISOCODE"])]

    if plot:
        plot_chloropleth(df_countries)

    # print("\n\n 5 \n\n")
    # print(len(df_districts["ISOCODE"].unique()))
    # print(len(df_countries["ISOCODE"].unique()))

    # TODO consider retuyrning df_zones here as well
    return df_countries, df_districts, df_zones


if __name__ == "__main__":

    df_countries, df_districts, zones_results = run_global_analysis(
        crr_adjustment=0,
        time_gathering_water=5.5,
        practical_limit_bicycle=40,
        practical_limit_buckets=20,
        met=4.5,
        watts=75,
        hill_polarity="downhill_uphill",
        urban_adjustment=1.3,
        rural_adjustment=1.4,
        calculate_distance=True,
        plot=True,
        human_mass=62,  # gets overridden by country specific weight
        use_sample_data=True,
    )

    df_countries.to_csv(COUNTRY_RESULTS_FILE_PATH, index=False)
    df_districts.to_csv(DISTRICT_RESULTS_FILE_PATH, index=False)
