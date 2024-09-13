# Standard library imports
import os
from pathlib import Path

# Third-party library imports
import pandas as pd
import numpy as np
import pycountry
from sklearn.linear_model import LinearRegression


"""
This script processes and interpolates various datasets related to water access, GDP, bicycle ownership, 
and BMI/weight to generate a merged dataset with imputed values. The script includes the following functionalities:

1. **File Path Setup**:
   - Automatically determines file paths relative to the repository structure.
   - Ensures that data files are correctly located in the `data/original_data` directory.

2. **Data Processing**:
   - Processes WHO household water data, including handling special cases like Japan.
   - Adds country codes (alpha2, alpha3) and merges additional datasets including GDP, bicycle ownership, and BMI data.
   - Preprocesses and appends country information, including continent and neighboring countries.
   - Note special treatment of Japan due to small amounts of missing data. Assinging assumed values deemed more accurate than imputation.

3. **Imputation and Interpolation**:
   - Performs GDP-based regression imputation for missing values (Optional).
   - Conducts spatial interpolation using data from neighboring countries and continents.
   - Handles special cases such as small island nations and countries without continent data.

4. **Output**:
   - Generates final datasets with interpolated values and saves them to the `data/processed/semi-processed` directory.
   - Tracks the imputation methods used for transparency.

**Usage**:
- Ensure all required data files are placed in the `data/original_data` directory.
- Adjust any file paths or settings as necessary within the script.
- Run the script from the `scripts` directory to process the data and generate the output.

**Modules Used**:
- `pandas` for data manipulation.
- `numpy` for numerical operations.
- `pycountry` for country code management.
- `sklearn` for regression imputation.

This script is designed to be adaptable across different environments by using relative file paths and modular functions.
"""


# Define the root of the repository by going up one level from the script directory
data_script_dir = Path(__file__).resolve().parent
script_dir = data_script_dir.parent
repo_root = script_dir.parent

# Define the data directory relative to the repo root
data_dir = repo_root / "data" / "original_data"
semi_processed_dir = repo_root / "data" / "processed" / "semi-processed"

# Define paths to the files
water_JMP_file_path = data_dir / "WHO Household Water Data - 2023 Data.csv"
bicycle_file_path = data_dir / "global-bike-ownership.csv"
gdp_per_capita_file_path = data_dir / "gdp_data.csv"
bmi_women_file_path = data_dir / "Mean BMI Women 2016.csv"
bmi_men_file_path = data_dir / "Mean BMI Men 2016.csv"
height_file_path = data_dir / "Human Height by Birth Year.csv"
population_file_path = data_dir / "population.csv"  # OWID
household_size_file_path = data_dir / "undesa_pd_2022_hh-size-composition.xlsx"
country_regions_file_path = data_dir / "ISO-3166 Countries.csv"


## http://download.geonames.org/export/dump/countryInfo.txt
country_info_csv_path = data_dir / "countryInfo.txt"
country_info_output_file_path = semi_processed_dir / "preprocessed_countryInfo.txt"

# https://data.worldbank.org/indicator/NY.GDP.PCAP.CD
# https://ourworldindata.org/grapher/mean-body-mass-index


def add_alpha2_from_alpha3(df, col):
    """
    Adds a column of alpha2 codes to a dataframe with country name in column 'col'
    """
    input_countries = df[col]
    countries = []
    for input_country in input_countries:
        try:
            country = pycountry.countries.get(alpha_3=input_country)
            alpha2 = country.alpha_2
        except:
            alpha2 = "unk_" + str(input_country)
        countries.append(alpha2)
    df["alpha2"] = countries
    return df


def add_alpha3_from_ISO_numeric(df, col):
    """
    Adds a column of alpha3 codes to a dataframe with ISO numeric codes in the specified column.

    Parameters:
    df (pd.DataFrame): The DataFrame containing the ISO numeric codes.
    col (str): The column name in df containing ISO numeric codes.

    Returns:
    pd.DataFrame: The DataFrame with an additional 'alpha3' column containing the corresponding alpha-3 codes.
    """
    countries = []
    for input_country in df[col]:
        try:
            # Convert numeric code to a zero-padded string of length 3
            numeric_str = f"{int(input_country):03}"
            country = pycountry.countries.get(numeric=numeric_str)
            alpha3 = country.alpha_3 if country else f"unk_{input_country}"
        except Exception as e:
            alpha3 = f"unk_{input_country}"
        countries.append(alpha3)

    df["alpha3"] = countries
    return df


def add_alpha_codes(df, col):
    """
    adds a column of alpha3 codes to a dataframe with country name in column 'col'
    uses fuzzy logic to match countries
    """
    input_countries = df[col]
    countries = []
    for input_country in input_countries:
        # remove any text between brackets in input country
        input_country = input_country.split("(")[0]
        try:
            country = pycountry.countries.search_fuzzy(input_country)
            alpha3 = country[0].alpha_3
        except:
            alpha3 = "unk_" + input_country
            print("Couldn't match alpha3 code for", input_country)
        countries.append(alpha3)
    df["alpha3"] = countries
    return df


def load_gdp_per_capita(file_path):
    """
    Loads the World Bank GDP per capita data from a file and returns a DataFrame
    with only the most recent GDP per capita value for each country.
    """
    # Load the data, specifying the columns to keep only relevant data
    df = pd.read_csv(file_path, skiprows=4)

    # Set the country name and code as index
    df = df.set_index(["Country Name", "Country Code"])

    # Drop unnecessary columns (Indicator Name and Indicator Code)
    df = df.drop(columns=["Indicator Name", "Indicator Code"])

    # Convert all year columns to numeric, which will make non-numeric entries NaN
    df = df.apply(pd.to_numeric, errors="coerce")

    # Get the most recent year that has data for each country
    most_recent_year = df.apply(
        lambda row: row.dropna().index[-1] if row.dropna().any() else None, axis=1
    )
    most_recent_gdp = df.apply(
        lambda row: row.dropna().iloc[-1] if row.dropna().any() else None, axis=1
    )

    # Combine the most recent year and GDP into a DataFrame
    df_recent_gdp = pd.DataFrame(
        {
            "Most Recent Year": most_recent_year,
            "GDP per Capita (current US$)": most_recent_gdp,
        }
    )

    # Reset index to have a cleaner DataFrame
    df_recent_gdp = df_recent_gdp.reset_index()

    return df_recent_gdp


def spatial_imputation(df, list_of_vars, alpha2_col="alpha2", alpha3_col="alpha3"):
    """
    Interpolates missing values in the dataframe using bordering countries and continent.
    Prioritizes original data and avoids cascading imputations.
    Maintains alpha3 codes in the tracking DataFrame.
    Returns only the columns in list_of_vars and alpha3.
    """
    df_output = df.copy()

    # Initialize df_interp_track with an additional column for alpha3 codes
    df_interp_track = pd.DataFrame(index=df.index, columns=[alpha3_col] + list_of_vars)
    df_interp_track[alpha3_col] = df[
        alpha3_col
    ]  # Copy alpha3 codes into df_interp_track

    for variable in list_of_vars:
        df[variable] = df[variable].astype(float)

        # First pass: Impute using neighboring countries' original data
        for idx in df[df[variable].isna()].index:
            country = df.loc[idx, alpha2_col]
            neighbors = df.loc[idx, "neighbours"]

            if isinstance(neighbors, str):
                # Ensure both neighbor codes and alpha2_col are consistently uppercase
                neighbor_list = [n.strip().upper() for n in neighbors.split(",")]
                alpha2_values = df[alpha2_col].str.upper().values

                # Filter valid neighbors by checking against the existing country codes
                valid_neighbors = [n for n in neighbor_list if n in alpha2_values]

                if valid_neighbors:  # Check if there are any valid neighbors
                    neighbor_values = df.loc[
                        df[alpha2_col].str.upper().isin(valid_neighbors), variable
                    ].dropna()
                    if not neighbor_values.empty:
                        df_output.at[idx, variable] = neighbor_values.mean()
                        df_interp_track.at[idx, variable] = "neighbor interpolated"

        # Second pass: Impute using continent's original data
        for idx in df_output[df_output[variable].isna()].index:
            country = df_output.loc[idx, alpha2_col]
            continent = df_output.loc[idx, "Continent"]
            continent_values = df_output[df_output["Continent"] == continent][
                variable
            ].dropna()
            if not continent_values.empty:
                df_output.at[idx, variable] = continent_values.mean()
                df_interp_track.at[idx, variable] = "continent interpolated"

    # Return only the list_of_vars and the alpha3 column
    # BUG there are multiple entities for a single alpha3 code (e.g., sint maarten and netherlands for NLD)
    df_output = df_output[[alpha3_col] + list_of_vars]
    return df_output, df_interp_track


def drop_small_island_nations(df, alpha3_col="alpha3"):
    """
    Drops specified small island nations from the DataFrame based on their alpha3 codes.
    """
    # List of alpha3 codes to drop
    small_island_nations = ["MAF", "TCA", "VGB", "VIR", "CHI", "BLM", "AND"]

    # Drop rows where the alpha3 code is in the list
    df_cleaned = df[~df[alpha3_col].isin(small_island_nations)].copy()

    return df_cleaned


def drop_countries_without_continent(df, continent_col="Continent"):
    """
    Drops countries from the dataframe that do not have continent data.
    """
    # Drop rows where the continent column is NaN or empty
    df_cleaned = df.dropna(subset=[continent_col]).copy()
    df_cleaned = df_cleaned[df_cleaned[continent_col].str.strip() != ""]

    return df_cleaned


def import_and_select_latest_household_size(
    file_path: Path, sheet_name: str = "HH size and composition 2022"
) -> pd.DataFrame:
    """
    Imports an Excel file containing household size data, filters out non-numeric household sizes,
    and selects the latest household size value for each country based on the available data.

    Parameters:
    file_path (Path): The Path to the Excel file.
    sheet_name (str): The sheet name in the Excel file (default is 'HH size and composition 2022').

    Returns:
    pd.DataFrame: A DataFrame with the latest household size values for each country.
    """
    # Load the Excel file, skipping the first 4 rows to align with the correct header
    df = pd.read_excel(file_path, sheet_name=sheet_name, skiprows=4)

    # Select the necessary columns
    df = df[
        [
            "Country or area",
            "ISO Code",
            "Average household size (number of members)",
            "Reference date (dd/mm/yyyy)",
        ]
    ]

    # Rename columns for consistency
    df.rename(
        columns={
            "Country or area": "Country",
            "ISO Code": "ISO",
            "Average household size (number of members)": "Household_Size",
            "Reference date (dd/mm/yyyy)": "Year",
        },
        inplace=True,
    )

    # Filter out non-numeric household sizes (e.g., '..')
    df = df[pd.to_numeric(df["Household_Size"], errors="coerce").notna()]

    # Convert 'Household_Size' to a numeric type
    df["Household_Size"] = pd.to_numeric(df["Household_Size"])

    # Convert 'Year' column to datetime and extract the year
    df["Year"] = pd.to_datetime(df["Year"], errors="coerce").dt.year

    # Sort the DataFrame by 'ISO' and 'Year' in descending order
    df.sort_values(by=["ISO", "Year"], ascending=[True, False], inplace=True)

    # Drop duplicates to keep the latest 'Household_Size' value for each 'ISO'
    latest_household_size_df = df.drop_duplicates(subset=["ISO"], keep="first")

    # Select only the 'ISO' and 'Household_Size' columns
    latest_household_size_df = latest_household_size_df[["ISO", "Household_Size"]]

    # Return the resulting DataFrame
    return latest_household_size_df


def bicycle_data_manual_assumption(df):
    """
    Manually assigns bicycle ownership data for the following regions:
    - Macao MAC
    - Hong Kong HKG
    - Taiwan TWN
    Currently don't have data and they are being assigned through spatial interpolation
    This is a worse assumnption than assigning them Mainland China values
    """
    manual_PBO_value = 62.97
    df.loc[df["ISO"] == "MAC", "Bicycle Ownership"] = manual_PBO_value
    df.loc[df["ISO"] == "HKG", "Bicycle Ownership"] = manual_PBO_value
    df.loc[df["ISO"] == "TWN", "Bicycle Ownership"] = manual_PBO_value
    return df
    
    

def import_and_select_latest_pbo(csv_path):
    """
    Imports a CSV file containing 'ISO', 'Year', and 'PBO' columns,
    and selects the latest 'PBO' value for each 'ISO' by 'Year'.

    Parameters:
    csv_path (str): The file path to the CSV.

    Returns:
    pd.DataFrame: A DataFrame with the latest 'PBO' values for each 'ISO'.
    """
    # Import the CSV into a DataFrame
    df = pd.read_csv(csv_path)

    # Sort the DataFrame by 'ISO' and 'Year' in descending order
    df.sort_values(by=["ISO", "Year"], ascending=[True, False], inplace=True)

    # Drop duplicates to keep the latest 'PBO' value for each 'ISO'
    latest_pbo_df = df.drop_duplicates(subset=["ISO"], keep="first")

    # drop othe columns (keep PBO and alpha3)
    latest_pbo_df = latest_pbo_df[["ISO", "PBO"]]

    # Return the resulting DataFrame
    return latest_pbo_df


def handle_japan_case(row):
    """
    Handles missing data for Japan by assuming 100% URBANPiped and
    calculating RURALPiped based on the TOTALPiped and % urban.
    As Japan is highgly developed, it is assumed it shares approximately 100% urban piped water.
    Leaving this to spatial imputation will assign Asian average values which is a worse assumption than 100%
    """
    if (
        row["Country"] == "Japan"
        and pd.isna(row["URBANPiped"])
        and pd.isna(row["RURALPiped"])
    ):
        row["URBANPiped"] = 100.0
        row["RURALPiped"] = (
            row["TOTALPiped"] - (row["% urban"] * row["URBANPiped"] / 100)
        ) / ((100 - row["% urban"]) / 100)
    return row

def handle_taiwan_case(df):
    """
    Handles missing data for Taiwan by creating a new row with assumed values.
    Data sources:
    - 80.1% urban from https://www.cia.gov/the-world-factbook/field/urbanization/
    - 94.91% overall piped from https://www.water.gov.tw/en/Contents?nodeId=4875#
    """
    taiwan_data = {
        "Country": "Taiwan",
        "% urban": 80.1,
        "TOTALPiped": 94.91,
        "URBANPiped": 100.0,
        "RURALPiped": (94.91 - (80.1 * 100 / 100)) / ((100 - 80.1) / 100)
    }
    
    # Create a new row for Taiwan and append it to the dataframe
    taiwan_row = pd.DataFrame([taiwan_data])
    df = pd.concat([df, taiwan_row], ignore_index=True)
    
    return df


def handle_seychelles_case(row):
    """
    Handles missing data for Seychelles by assuming 100% URBANPiped and RURALPiped.
    Seycheles is a small island nation with high urbanization and piped water access.
    Kevin is half Seychellois and he says this is a good assumption. He has never seen
    a dwelling without piped water in Seychelles.
    """
    if (
        row["Country"] == "Seychelles"
        and pd.isna(row["URBANPiped"])
        and pd.isna(row["RURALPiped"])
    ):
        row["URBANPiped"] = 100.0
        row["RURALPiped"] = 100.0
    return row


def process_water_data(df):
    """
    Processes water data by calculating TOTALPiped, filling missing values, and adding necessary columns.
    Ensures that missing values are filled from previous years' data.
    """

    # Replace '>99' with 100, '<1' with 0, and '-' with NaN
    df.replace({"<1": 0, ">99": 100, "-": pd.NA}, inplace=True)

    # Convert relevant columns to numeric for proper calculations
    columns_to_convert = [
        "TOTALPiped",
        "RURALPiped",
        "URBANPiped",
        "Population \r\n(thousands)",
        "% urban",
    ]
    df[columns_to_convert] = df[columns_to_convert].replace({" ": ""}, regex=True)
    df[columns_to_convert] = df[columns_to_convert].apply(
        pd.to_numeric, errors="coerce"
    )

    # Handle special cases:
    # If % urban is 0, set URBANPiped equal to RURALPiped
    df.loc[df["% urban"] == 0, "URBANPiped"] = df["RURALPiped"]

    # If % urban is 100 and RURALPiped is NaN, set RURALPiped equal to URBANPiped
    df.loc[(df["% urban"] == 100) & (df["RURALPiped"].isna()), "RURALPiped"] = df[
        "URBANPiped"
    ]

    # Handle Japan's special case
    df = df.apply(handle_japan_case, axis=1)
    df = df.apply(handle_seychelles_case, axis=1)
    df = handle_taiwan_case(df)

    # Fill missing rural and urban piped data where total piped is 100
    df.loc[(df["TOTALPiped"] == 100) & (df["RURALPiped"].isna()), "RURALPiped"] = 100
    df.loc[(df["TOTALPiped"] == 100) & (df["URBANPiped"].isna()), "URBANPiped"] = 100

    # Sort by country and year
    df = df.sort_values(["Country", "Year"])

    # Iterate through each country to ensure URBANPiped, RURALPiped, and TOTALPiped have values
    def fill_from_older_data(group):
        group["RURALPiped"] = (
            group["RURALPiped"].ffill().bfill()
        )  # Fill forward first, then backward
        group["URBANPiped"] = group["URBANPiped"].ffill().bfill()
        group["TOTALPiped"] = group["TOTALPiped"].ffill().bfill()
        return group

    # Apply the fill_from_older_data function to each country group
    df = (
        df.groupby("Country", group_keys=False)
        .apply(fill_from_older_data)
        .reset_index(drop=True)
    )

    # Recalculate TOTALPiped for all years
    df["TOTALPiped_Recalculated"] = (df["% urban"] / 100) * df["URBANPiped"] + (
        (100 - df["% urban"]) / 100
    ) * df["RURALPiped"]

    # Filter to keep only the most recent non-empty data entry per country
    df = df.dropna(subset=["URBANPiped", "RURALPiped", "TOTALPiped"], how="all")
    df = df.sort_values(["Country", "Year"], ascending=[True, False])
    df = df.drop_duplicates("Country", keep="first").reset_index(drop=True)

    return df


def add_back_percentage_urban_data(df, df_water):
    """
    Adds percentage urban data for countries where it is was dropped in process_water_data
    """
    # Find missing % urban data, then fill using the original data from df_water
    original_data = df_water[["Country", "% urban"]].drop_duplicates(
        "Country", keep="last"
    )
    df = df.merge(
        original_data,
        left_on="Entity",
        right_on="Country",
        how="left",
        suffixes=("", "_temp"),
    )
    df["% urban"] = df["% urban"].fillna(df["% urban_temp"])
    df = df.drop(columns=["% urban_temp", "Country"])

    return df


def merge_gdp_data(df_cleaned, df_gdp, alpha2_col="alpha2", alpha3_col="Country Code"):
    """
    Merges df_cleaned with df_gdp using the Alpha-2 code from df_cleaned and Alpha-3 code from df_gdp.
    """
    # Ensure the Alpha-3 code exists in df_cleaned for the merge
    if "alpha3" not in df_cleaned.columns:
        raise ValueError(
            "df_cleaned must contain 'alpha3' column for merging with GDP data."
        )

    # Merge on the Alpha-3 code
    df_merged = df_cleaned.merge(
        df_gdp[["Country Code", "GDP per Capita (current US$)"]],
        left_on="alpha3",
        right_on=alpha3_col,
        how="left",
    )

    return df_merged


def impute_using_gdp(
    df, list_of_vars, gdp_col="GDP per Capita (current US$)", alpha3_col="alpha3"
):
    """
    Imputes missing values in the list of variables using GDP per capita via linear regression.
    Maintains alpha3 codes in the tracking DataFrame.
    Returns only the columns in list_of_vars and alpha3.
    """
    df_output = df.copy()

    # Initialize df_interp_track with an additional column for alpha3 codes
    df_interp_track = pd.DataFrame(index=df.index, columns=[alpha3_col] + list_of_vars)
    df_interp_track[alpha3_col] = df[
        alpha3_col
    ]  # Copy alpha3 codes into df_interp_track

    for variable in list_of_vars:
        # Drop rows where either the target variable or GDP is missing
        df_valid = df.dropna(subset=[variable, gdp_col])

        if df_valid.empty or df_valid[variable].nunique() <= 1:
            print(f"Not enough data to impute {variable}. Skipping.")
            continue

        # Prepare the model
        X = df_valid[[gdp_col]]  # GDP as independent variable, keep as DataFrame
        y = df_valid[variable]  # The target variable (e.g., RURALPiped)

        # Fit the regression model
        model = LinearRegression()
        model.fit(X, y)

        # Predict missing values
        missing_idx = df[df[variable].isna() & df[gdp_col].notna()].index
        if not missing_idx.empty:
            df_output.loc[missing_idx, variable] = model.predict(
                df_output.loc[missing_idx, [gdp_col]]
            )
            df_interp_track.loc[missing_idx, variable] = "gdp regression"

    # Return only the list_of_vars and the alpha3 column
    # BUG there are multiple entities for a single alpha3 code (e.g., sint maarten, curacao and netherlands for NLD)
    df_output = df_output[[alpha3_col] + list_of_vars]
    return df_output, df_interp_track


def merge_and_impute_with_gdp(
    df_cleaned, df_gdp, list_of_vars, alpha2_col="alpha2", alpha3_col="Country Code"
):
    """
    Merges the cleaned DataFrame with GDP data and then imputes the specified variables using GDP via regression.
    """
    # Merge the DataFrames
    df_merged = merge_gdp_data(
        df_cleaned, df_gdp, alpha2_col=alpha2_col, alpha3_col=alpha3_col
    )

    # Perform imputation using GDP per capita via regression
    df_output, df_interp_track = impute_using_gdp(df_merged, list_of_vars)

    return df_output, df_interp_track


def preprocess_country_info_file(input_path, output_path):
    """
    Preprocess the country info file to remove the '#' from the header line and skip comment lines.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(input_path, "r") as infile, open(output_path, "w") as outfile:
        for line in infile:
            if line.startswith("#ISO"):
                line = line.replace("#ISO", "ISO")
            if not line.startswith("#"):
                outfile.write(line)


def import_country_regions(file_path):
    """
    https://github.com/lukes/ISO-3166-Countries-with-Regional-Codes/blob/master/all/all.csv
    Imports the country regions data from the specified file
    and returns a DataFrame with the relevant columns.
    """
    # Define the new column names
    new_column_names = {
        "name": "Entity",
        "alpha-2": "alpha2",
        "alpha-3": "alpha3",
        "country-code": "ISOcode",
        "iso_3166-2": "iso_3166-2",
        "region": "region",
        "sub-region": "subregion",
        "intermediate-region": "intermediate-region",
    }

    # Read the CSV file
    df = pd.read_csv(file_path)

    # Rename the columns
    df.rename(columns=new_column_names, inplace=True)

    # Drop the specified columns
    df.drop(columns=["alpha2"], inplace=True)

    return df


def load_country_info(preprocessed_file_path):
    """
    Loads the preprocessed country information from a text file, selecting only relevant columns.
    Note that the preprocessed file should have the '#' removed from the header and comments skipped.
    Also, the 'na_values' parameter is set to handle the '_' values in the file, as North America is "NA".
    """
    country_info_df = pd.read_csv(
        preprocessed_file_path, sep="\t", keep_default_na=False, na_values=["_"]
    )
    country_info_df = country_info_df[["ISO3", "Continent", "neighbours"]]
    return country_info_df


def append_country_info(df, country_info_df):
    """
    Appends continent and neighboring countries to the dataframe using the pre-loaded country info DataFrame.
    """
    df = df.merge(country_info_df, how="left", left_on="alpha3", right_on="alpha3")
    # df.drop(columns=['alpha3'], inplace=True)
    return df


def calculate_average_weight_per_country(bmi_women_file, bmi_men_file, height_file):
    """
    This script calculates the average weight per country using BMI and height data for men and women.

    Parameters:
    - bmi_women_file: str, path to the file containing BMI data for women
    - bmi_men_file: str, path to the file containing BMI data for men
    - height_file: str, path to the file containing height data

    Data Sources:
    1. Mean Body Mass Index (BMI) in Adult Women (2016):
       - Source: https://ourworldindata.org/grapher/mean-body-mass-index-bmi-in-adult-women
    2. Mean Body Mass Index (BMI) in Adult Men (2016):
       - Source: https://ourworldindata.org/grapher/mean-body-mass-index-bmi-in-adult-men
    3. Human Height by Birth Year:
       - Source: https://ourworldindata.org/human-height

    Data is sourced from:
    - NCD RisC, Human Height (2017) – processed by Our World in Data
    - NCD RisC (2017) – processed by Our World in Data

    The script performs the following steps:
    1. Loads the datasets.
    2. Filters for the most recent year of data available for each country.
    3. Aligns BMI data to this most recent year.
    4. Calculates the average weight for men and women using the BMI and height data.
    5. Calculates the overall average weight assuming a 50/50 distribution between men and women.
    6. Outputs the final DataFrame with overall average weights per country.
    """

    # Load the datasets
    bmi_women_df = pd.read_csv(bmi_women_file)
    bmi_men_df = pd.read_csv(bmi_men_file)
    height_df = pd.read_csv(height_file)

    # Step 1: Filter for the most recent year data in the height dataset
    most_recent_height_df = height_df.loc[height_df.groupby("Entity")["Year"].idxmax()]

    # Step 2: Align BMI data to the most recent year from height data
    # Keep only the data for the most recent year in the height dataset for each country
    merged_df = pd.merge(
        most_recent_height_df, bmi_women_df, on=["Entity", "Year"], how="left"
    )
    merged_df = pd.merge(merged_df, bmi_men_df, on=["Entity", "Year"], how="left")

    # Step 3: Remove non-country entities by checking for the presence of an ISO3 country code
    merged_df = merged_df[merged_df["Code"].notna()]
    # drop OWID_WRL
    merged_df = merged_df[merged_df["Code"] != "OWID_WRL"]

    # Step 4: Calculate the weight using the formula: Weight (kg) = BMI * (Height in meters)^2
    merged_df["Average Weight (male, kg)"] = (
        merged_df["Mean BMI (male)"] * (merged_df["Mean male height (cm)"] / 100) ** 2
    )
    merged_df["Average Weight (female, kg)"] = (
        merged_df["Mean BMI (female)"]
        * (merged_df["Mean female height (cm)"] / 100) ** 2
    )

    # Step 5: Calculate the overall average weight assuming a 50/50 distribution
    merged_df["Overall Average Weight (kg)"] = (
        merged_df["Average Weight (male, kg)"]
        + merged_df["Average Weight (female, kg)"]
    ) / 2

    # Step 6: Select relevant columns for the final output and drop the 'Year' column
    average_weight_df = merged_df[["Entity", "Code", "Overall Average Weight (kg)"]]

    # Drop duplicates to ensure only one entry per country
    average_weight_df = average_weight_df.drop_duplicates(subset=["Entity", "Code"])

    # Rename columns from Code to alpha3 and Entity to Country, Average Weight to Weight
    average_weight_df.rename(
        columns={
            "Code": "alpha3",
            "Entity": "Country",
            "Overall Average Weight (kg)": "Average Weight",
        },
        inplace=True,
    )

    return average_weight_df


def load_latest_population_data(population_file_path):
    """
    Loads population data, selects the latest available population for each country,
    and removes non-country entries and the 'OWID_WRL' entry.

    Parameters:
    - population_file_path: str, path to the population CSV file.

    Returns:
    - pd.DataFrame: A DataFrame containing the latest population values for each country.
    """
    # Load the data
    df_population = pd.read_csv(population_file_path)

    # Drop non-country entries and 'OWID_WRL'
    df_population = df_population.dropna(subset=["Code"])
    df_population = df_population[df_population["Code"] != "OWID_WRL"]

    # Sort the DataFrame by 'Code' and 'Year' in descending order
    df_population.sort_values(
        by=["Code", "Year"], ascending=[True, False], inplace=True
    )

    # Drop duplicates to keep the latest population value for each country
    latest_population_df = df_population.drop_duplicates(subset=["Code"], keep="first")

    # Select only relevant columns: 'Code' (alpha-3), 'Entity' (Country), and 'Population (historical estimates)'
    latest_population_df = latest_population_df[
        ["Code", "Entity", "Population (historical estimates)"]
    ]

    # Rename columns to make them more descriptive
    latest_population_df.rename(
        columns={
            "Code": "alpha3",
            "Entity": "Country",
            "Population (historical estimates)": "Population",
        },
        inplace=True,
    )

    return latest_population_df


def main(
    water_JMP_file_path,
    bicycle_file_path,
    gdp_per_capita_file_path,
    bmi_women_file,
    bmi_men_file,
    height_file,
    population_file_path,
    country_info_output_file_path,
    household_size_file_path,
    country_regions_file_path,
):
    """
    Main function to process the input data file, interpolate missing data, and save the results.
    """

    gdp_per_capita_df = load_gdp_per_capita(gdp_per_capita_file_path)

    # Read input data
    df_water = pd.read_csv(water_JMP_file_path)

    # Process water data
    df = process_water_data(df_water)

    # Add alpha codes from ISO
    df = add_alpha_codes(df, "Country")

    # check duplicates for alpha3
    duplicates = df[df.duplicated(subset=["alpha3"], keep=False)]

    # Manual mapping for problematic entries
    manual_alpha3_mapping = {
        "Channel Islands": "CHI",
        "United States Virgin Islands": "VIR",
        "Wallis and Futuna Islands": "WLF",
        "Democratic Republic of the Congo": "COD",
        "China, Hong Kong SAR": "HKG",
        "China, Macao SAR": "MAC",
        "Republic of Korea": "KOR",
        "Curaçao": "CUW",
        "Niger": "NER",
        "Sint Maarten (Dutch part)": "SXM",
        "Guadeloupe": "GLP",
        "Martinique": "MTQ",
        "Mayotte": "MYT",
    }

    # Apply manual mapping
    df["alpha3"] = df.apply(
        lambda row: manual_alpha3_mapping.get(row["Country"], row["alpha3"]), axis=1
    )

    # check duplicates again for alpha3
    duplicates_2 = df[df.duplicated(subset=["alpha3"], keep=False)]

    assert duplicates_2.empty, "There are still duplicates in the alpha3 column."

    df = add_alpha2_from_alpha3(df, "alpha3")

    #############################
    ## add household size data ##

    # Import the latest household size data using the defined function
    household_size_df = import_and_select_latest_household_size(
        household_size_file_path
    )
    # Assuming df_imported is your DataFrame with ISO numeric codes in the 'ISO' column
    household_size_df = add_alpha3_from_ISO_numeric(household_size_df, "ISO")
    # Merge the household size data with the existing data
    df = df.merge(household_size_df, how="outer", on="alpha3")

    ######################
    ## add bicycle data ##
    df_bike = import_and_select_latest_pbo(bicycle_file_path)
    # rename ISO to alpha3
    df_bike.rename(columns={"ISO": "alpha3"}, inplace=True)
    # Update 'ROM' to 'ROU' in the 'ISO' column of the bike_df DataFrame
    df_bike.loc[df_bike["alpha3"] == "ROM", "alpha3"] = "ROU"
    

    # merge data
    df = df.merge(df_bike, how="outer", on="alpha3")

    # Load the country info from the preprocessed file
    country_info_df = load_country_info(country_info_output_file_path)
    country_info_df.rename(columns={"ISO3": "alpha3"}, inplace=True)

    df_merged_info = append_country_info(df, country_info_df)

    df_cleaned_merge = drop_countries_without_continent(df_merged_info)
    df_cleaned_merge = drop_small_island_nations(df_cleaned_merge)

    # Load further country information about subregions
    country_regions_df = import_country_regions(country_regions_file_path)

    # merge data with country regions
    df_cleaned_merge = df_cleaned_merge.merge(
        country_regions_df, on="alpha3", how="left"
    )

    # import weight data
    df_weight = calculate_average_weight_per_country(
        bmi_women_file_path, bmi_men_file_path, height_file_path
    )

    # merge weight df with df_cleaned_merge
    df_cleaned_merge = df_cleaned_merge.merge(df_weight, on="alpha3", how="left")

    # List of variables to impute
    list_of_vars = [
        "RURALPiped",
        "URBANPiped",
        "PBO",
        "Average Weight",
        "Household_Size",
    ]

    # Merge and impute using GDP regression
    df_gdp_imputation, df_gdp_imputation_track = merge_and_impute_with_gdp(
        df_cleaned_merge, gdp_per_capita_df, list_of_vars
    )

    # SPATIAL Imputation
    # rtename spatiual vars, eg: RURALPiped_spatial, URBANPiped_spatial, PBO_spatial, Weight_spatial
    # Imputation variables
    df_spatial_imputation, df_spatial_imputation_track = spatial_imputation(
        df_cleaned_merge, list_of_vars, "alpha2"
    )

    # merge the dataframes
    df_imputed = df_spatial_imputation.merge(
        df_gdp_imputation, on="alpha3", suffixes=("_spatial", "_gdp")
    )
    # Here's the problem
    df_output = df_imputed.merge(df_cleaned_merge, on="alpha3")
    df_imputed_track = df_spatial_imputation_track.merge(
        df_gdp_imputation_track, on="alpha3", suffixes=("_spatial", "_gdp")
    )

    # POPULATION
    df_population = load_latest_population_data(population_file_path)
    df_output = df_output.merge(df_population, on="alpha3")

    extra_vars = [
        "alpha2",
        "Continent",
        "% urban",
        "Population",
        "Entity",
        "region",
        "subregion",
        "intermediate-region",
    ]

    # remove columns that are are not in vars
    # Spatial imputation columns
    spatial_list_of_vars = [f"{var}_spatial" for var in list_of_vars]
    # GDP imputation columns
    gdp_list_of_vars = [f"{var}_gdp" for var in list_of_vars]
    output = "spatial"

    if output == "spatial":
        df_output = df_output[["alpha3"] + spatial_list_of_vars + extra_vars]
        # rename columns
        df_output.columns = [var.replace("_spatial", "") for var in df_output.columns]
    elif output == "gdp":
        df_output = df_output[["alpha3"] + gdp_list_of_vars + extra_vars]
        # rename columns
        df_output.columns = [var.replace("_gdp", "") for var in df_output.columns]
    else:
        df_output = df_output[["alpha3"] + spatial_list_of_vars + gdp_list_of_vars]

    # Add back % urban data
    df_output = add_back_percentage_urban_data(df_output, df_water)

    # Save dataframes as CSV
    try:
        df_output.to_csv("../../data/processed/merged_data.csv", index=False)
        df_imputed_track.to_csv(
            "../../data/processed/semi-processed/merged_data_track.csv", index=False
        )
    except:
        df_output.to_csv("./data/processed/merged_data.csv", index=False)
        df_imputed_track.to_csv(
            "./data/processed/semi-processed/merged_data_track.csv", index=False
        )


preprocess_country_info_file(country_info_csv_path, country_info_output_file_path)
main(
    water_JMP_file_path,
    bicycle_file_path,
    gdp_per_capita_file_path,
    bmi_women_file_path,
    bmi_men_file_path,
    height_file_path,
    population_file_path,
    country_info_output_file_path,
    household_size_file_path,
    country_regions_file_path,
)


# # Assuming you have an existing DataFrame `df_existing`
# df_existing = pd.read_csv('/mnt/data/your_existing_data.csv')

# # Merge the household size data with the existing data
# df_with_household_size = pd.merge(df_existing, latest_household_size_df, on='ISO', how='left')
