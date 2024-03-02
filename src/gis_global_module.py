import pathlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import weightedstats as ws
import sys
from pathlib import Path


def weighted_mean(var, wts):
    """Calculates the weighted mean"""
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

# Constants
URB_DATA_FILE = "./data/GIS/GIS_data_zones.csv"
COUNTRY_DATA_FILE = "./data/processed/country_data_master_interpolated.csv"
EXPORT_FILE_LOCATION = "./data/processed/"
CRR_FILE = "./data/lookup tables/Crr.csv"


# Function to load data
def load_data(urb_data_file, country_data_file):
    try:
        df_zones_input = pd.read_csv(urb_data_file)
        df_input = pd.read_csv(country_data_file)
    except:
        df_zones_input = pd.read_csv("." + urb_data_file)
        df_input = pd.read_csv("." + country_data_file)
    return df_zones_input, df_input


# Function for managing urban/rural data
def manage_urban_rural(df_zones_input):
    # Convert dtw_1 from meters to kilometers
    df_zones_input["dtw_1"] /= 1000
    # Manage Urban / Rural Data
    # Use the GHS_SMOD_E2020_GLOBE_R2023A_54009_1000_V1_0 dataset.
    # from here: https://ghsl.jrc.ec.europa.eu/download.php?ds=smod
    # create new binary column for urban / rural. Rural is below 15, Urban above 15
    df_zones_input["urban_rural"] = np.where(df_zones_input["URBAN_1"] > 15, 1, 0)
    return df_zones_input


# Function for managing slope
def manage_slope(df_zones_input):
    # Degrees from earthenv dataset
    df_zones_input["slope_1"].hist(bins=100, log=False)
    df_zones_input["slope_1"].quantile(np.arange(0, 1, 0.05))
    return df_zones_input


# Function for merging dataframes and adjusting populations
def merge_and_adjust_population(df_zones_input, df_input):
    # this analysis loses some data as the overlap between the rasters is not perfect. To reduce this error, use the 30 arc second data. Too much heavy lifting for my computer to do this at the moment.
    # merge df_input and df_zones on ISO_CC. This assigns all the country data to each zone.
    # join inner will remove some of the data that is not in both datasets
    df_zones = df_zones_input.merge(
        df_input, left_on="ISOCODE", right_on="alpha3", how="inner"
    )
    # adjust population to account for 9 values per raster point (2.5 to 5 arc min resoltuions. 9 values per point)
    df_zones["AdjPopFloat"] = df_zones["pop_count_15_1"] / 9

    # convert population density to percent of national population on a per country basis, grouped by ISO_CC
    df_zones["pop_density_perc"] = df_zones.groupby("ISOCODE")["AdjPopFloat"].apply(
        lambda x: x / x.sum()
    )
    # multiply population density by population on a per country basis
    df_zones["pop_zone"] = df_zones["pop_density_perc"] * df_zones["Population"]

    # sum the population in each zone
    df_zones["country_pop_raw"] = df_zones.groupby("ISOCODE")["pop_zone"].transform(
        "sum"
    )
    df_zones["country_pop_ratio"] = df_zones.groupby("ISOCODE")[
        "AdjPopFloat"
    ].transform("sum")

    # trim the dataframe to only include rows where there is a population
    # find non zero values AdjPopFloat
    df_zones["any_pop"] = df_zones["AdjPopFloat"].apply(lambda x: 1 if x > 10 else 0)
    df_zones = df_zones[df_zones["any_pop"] == 1]

    return df_zones


# Function for road analysis
def road_analysis_old(df_zones):
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

    # Create a dictionary from the DataFrame
    crr_mapping = pd.Series(
        df_crr.Crr.values, index=df_crr["Assigned Zone Surface"]
    ).to_dict()

    # Map the road types in df_zones to Crr values
    df_zones["Crr"] = df_zones["dominant_road_type"].map(crr_mapping)

    return df_zones


def crr_add_uncertainty(road_type, adjustment):
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

    # Create a series from the DataFrame
    crr_mapping = pd.Series(df_crr.Crr.values, index=df_crr["Assigned Zone Surface"])

    # Add uncertainty (make road type 1 better or 1 worse, unless road type is best (highway) or worst (no roads))
    crr_mapping = crr_mapping.shift(crr_adjustment)
    crr_mapping = crr_mapping.ffill()
    crr_mapping = crr_mapping.bfill()

    # Create a dictionary from the series
    crr_mapping = crr_mapping.to_dict()

    # Map the road types in df_zones to Crr values
    df_zones["Crr"] = df_zones["dominant_road_type"].map(crr_mapping)

    return df_zones


# Main function to run all steps
def preprocess_data(crr_adjustment):
    df_zones_input, df_input = load_data(URB_DATA_FILE, COUNTRY_DATA_FILE)
    df_zones_input = manage_urban_rural(df_zones_input)
    df_zones_input = manage_slope(df_zones_input)
    df_zones = merge_and_adjust_population(df_zones_input, df_input)
    df_zones = road_analysis(df_zones, crr_adjustment=crr_adjustment)
    return df_zones


##############################################################################################################################
#
# ANALYSIS I: Models
#
##############################################################################################################################


def load_hpv_parameters(file_path_params, hpv_name):
    allHPV_param_df = pd.read_csv(file_path_params)
    return allHPV_param_df[allHPV_param_df["Name"] == hpv_name]


def extract_slope_crr(df_zones):
    df_zones["Crr"] = df_zones["Crr"].astype(float)
    slope_zones = df_zones["slope_1"]
    Crr_values = df_zones["Crr"]
    return slope_zones, Crr_values


def run_bicycle_model(mv, mo, hpv, slope_zones, Crr_values, load_attempt):
    n_runs = len(slope_zones)
    results = np.zeros((n_runs, 3))
    for i, (slope, crr) in enumerate(zip(slope_zones, Crr_values)):
        hpv.Crr = crr
        results[i] = mm.mobility_models.single_bike_run(
            mv, mo, hpv, slope, load_attempt
        )
    return results


def process_and_save_results(df_zones, results, export_file_location, velocity_type):
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

    # Customizing the output CSV filename based on the type of velocity
    output_csv_filename = f"{export_file_location}{velocity_type}_velocity_by_zone.csv"

    # only save the newly created cols in the csv
    df_zones[
        [
            "fid",
            "loaded_velocity_bicycle",
            "unloaded_velocity_bicycle",
            "average_velocity_bicycle",
            "max_load_bicycle",
        ]
    ].to_csv(output_csv_filename)

    return df_zones


def calculate_and_merge_bicycle_distance(
    df_zones, calculate_distance, export_file_location, practical_limit_bicycle=40
):
    if calculate_distance:
        # Adjust the project root and import mobility module as needed
        project_root = Path().resolve().parent
        sys.path.append(str(project_root))
        import src.mobility_module as mm

        file_path_params = "./data/lookup tables/mobility-model-parameters.csv"
        param_df = load_hpv_parameters(file_path_params, "Bicycle")

        param_df["PracticalLimit"] = practical_limit_bicycle

        mo = mm.model_options()
        mo.model_selection = 2  # Cycling model
        mv = mm.model_variables()
        met = mm.MET_values(mv)
        hpv = mm.HPV_variables(param_df, mv)

        slope_zones, Crr_values = extract_slope_crr(df_zones)
        results = run_bicycle_model(
            mv, mo, hpv, slope_zones, Crr_values, load_attempt=25
        )
        process_and_save_results(df_zones, results, export_file_location, "bicycle")
    else:
        df_zones_bicycle = pd.read_csv(
            export_file_location + "bicycle_velocity_by_zone.csv"
        )
        df_zones = df_zones.merge(df_zones_bicycle, on="fid", how="left")
    return df_zones


def run_walking_model(mv, mo, met, hpv, slope_zones, load_attempt):
    n_runs = len(slope_zones)
    results = np.zeros((n_runs, 3))
    for i, slope in enumerate(slope_zones):
        results[i] = mm.mobility_models.single_lankford_run(
            mv, mo, met, hpv, slope, load_attempt
        )
    return results


def calculate_and_merge_walking_distance(
    df_zones,
    calculate_distance,
    export_file_location,
    practical_limit_buckets=20,
    met=3.3,
):
    if calculate_distance:
        # Adjust the project root and import mobility module as needed
        project_root = Path().resolve().parent
        sys.path.append(str(project_root))
        import src.mobility_module as mm

        file_path_params = "./data/lookup tables/mobility-model-parameters.csv"
        param_df = load_hpv_parameters(file_path_params, "Buckets")
        param_df["PracticalLimit"] = practical_limit_buckets

        mo = mm.model_options()
        mo.model_selection = 3  # Lankford model
        mv = mm.model_variables()
        met = mm.MET_values(mv, met=met)
        hpv = mm.HPV_variables(param_df, mv)

        slope_zones, Crr_values = extract_slope_crr(df_zones)
        results = run_walking_model(mv, mo, met, hpv, slope_zones, load_attempt=20)
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
        * df_zones["URBANNon-piped"]
        / 100
        + df_zones["pop_zone"]
        * (1 - df_zones["urban_rural"])
        * df_zones["RURALNon-piped"]
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

    # Calculate zone population with and without water access
    df_zones["zone_pop_with_water"] = (
        df_zones["population_piped_with_access"] + df_zones["zone_pop_unpiped"]
    )
    df_zones["zone_pop_without_water"] = (
        df_zones["pop_zone"] - df_zones["zone_pop_with_water"]
    )
    return df_zones


def calculate_water_rations(df_zones):
    # Calculate water rations achievable utilizing all the bikes in the zone
    df_zones["water_rations_per_bike"] = (
        df_zones["water_ration_kms"] / df_zones["dtw_1"]
    )
    # Number of bikes per zone is population divided by household number multiplied by PBO
    df_zones["bikes_in_zone"] = (
        df_zones["pop_zone"]
        / df_zones["Average household size (number of members)"]
        * df_zones["PBO"]
    )
    # Water rations achievable utilizing all the bikes in the zone
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
                "Nat Piped": "first",
                "region": "first",
                "subregion": "first",
                # Additional aggregations can be added here
            }
        )
        .reset_index()
    )
    return df_countries


def calculate_weighted_median(df_zones):
    """
    Calculate the weighted median for each country group.
    """
    df_median_group = df_zones.groupby("ISOCODE").apply(
        lambda x: pd.Series({"weighted_med": weighted_median(x, "dtw_1", "pop_zone")})
    )
    return df_median_group


def clean_up_data(df_countries):
    """
    Clean up data from spurious country values.
    """
    df_countries = df_countries.dropna()  # Remove any NaN rows

    # Remove outliers based on max possible distance to water
    max_distance = (
        df_countries.loc[df_countries["ISOCODE"] == "LBY", "weighted_med"].values[0] + 1
    )
    countries_further_than_libya = df_countries[
        df_countries["weighted_med"] > max_distance
    ]
    df_countries = df_countries[df_countries["weighted_med"] < max_distance]

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
    """
    df_countries = aggregate_country_level_data(df_zones)
    df_median_group = calculate_weighted_median(df_zones)
    df_countries = df_countries.merge(
        df_median_group, on="ISOCODE"
    )  # Merge weighted median

    # drop rows from the dataframe that have Nan in pop_zone and dtw_1
    df_zones = df_zones.dropna(subset=["pop_zone", "dtw_1"])

    # Rename and create percentage columns
    df_countries = df_countries.rename(
        columns={
            "zone_pop_with_water": "country_pop_with_water",
            "zone_pop_without_water": "country_pop_without_water",
        }
    )
    df_countries["percent_with_water"] = (
        df_countries["country_pop_with_water"] / df_countries["country_pop_raw"] * 100
    )
    df_countries["percent_without_water"] = (
        df_countries["country_pop_without_water"]
        / df_countries["country_pop_raw"]
        * 100
    )

    df_countries, removed_further_than_libya, removed_countries_list = clean_up_data(
        df_countries
    )

    # Log or print summary of removed countries
    print(
        "Countries removed from analysis due to being further than Libya's median:",
        removed_further_than_libya["Entity"].tolist(),
    )
    print("Countries removed manually:", removed_countries_list)

    return df_countries


##############################################################################################################################
#
# VISUALISATIONS
#
##############################################################################################################################


def plot_chloropleth(df_countries):
    hover_data_list = [
        "Entity",
        "country_pop_raw",
        "country_pop_with_water",
        "country_pop_without_water",
        "population_piped_with_access",
        "population_piped_with_cycling_access",
        "population_piped_with_walking_access",
        "percent_without_water",
        "percent_with_water",
        "Nat Piped",
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
        # color_continuous_scale="ylorbr",
        # color_continuous_scale="YlGnBu_r",
        # color_continuous_scale="PuRd",
        color_continuous_scale="Greys",
        range_color=(0, 100),
        scope="world",
        hover_name="Entity",
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
    plot=False,
):
    df_zones = preprocess_data(crr_adjustment=crr_adjustment)
    df_zones = calculate_and_merge_bicycle_distance(
        df_zones,
        calculate_distance=False,
        export_file_location=EXPORT_FILE_LOCATION,
        practical_limit_bicycle=practical_limit_bicycle,
    )
    df_zones = calculate_and_merge_walking_distance(
        df_zones,
        calculate_distance=False,
        export_file_location=EXPORT_FILE_LOCATION,
        practical_limit_buckets=practical_limit_buckets,
        met=met,
    )
    df_zones = process_zones_for_water_access(
        df_zones, time_gathering_water=time_gathering_water
    )
    df_countries = process_country_data(df_zones)
    if plot:
        plot_chloropleth(df_countries)

    return df_countries


if __name__ == "__main__":
    df_countries = run_global_analysis(
        crr_adjustment=0,
        time_gathering_water=6,
        practical_limit_bicycle=40,
        practical_limit_buckets=20,
        met=3.3,
        plot=True,
    )
