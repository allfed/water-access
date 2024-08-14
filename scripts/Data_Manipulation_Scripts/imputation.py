# Standard library imports
import os

# Third-party library imports
import pandas as pd
import numpy as np
import pycountry
from sklearn.linear_model import LinearRegression


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
    df = df.set_index(['Country Name', 'Country Code'])
    
    # Drop unnecessary columns (Indicator Name and Indicator Code)
    df = df.drop(columns=['Indicator Name', 'Indicator Code'])
    
    # Convert all year columns to numeric, which will make non-numeric entries NaN
    df = df.apply(pd.to_numeric, errors='coerce')
    
    # Get the most recent year that has data for each country
    most_recent_year = df.apply(lambda row: row.dropna().index[-1] if row.dropna().any() else None, axis=1)
    most_recent_gdp = df.apply(lambda row: row.dropna().iloc[-1] if row.dropna().any() else None, axis=1)
    
    # Combine the most recent year and GDP into a DataFrame
    df_recent_gdp = pd.DataFrame({
        'Most Recent Year': most_recent_year,
        'GDP per Capita (current US$)': most_recent_gdp
    })
    
    # Reset index to have a cleaner DataFrame
    df_recent_gdp = df_recent_gdp.reset_index()
    
    return df_recent_gdp

def spatial_imputation(df, list_of_vars, alpha2_col, alpha3_col='alpha3'):
    """
    Interpolates missing values in the dataframe using bordering countries and continent.
    Prioritizes original data and avoids cascading imputations.
    Maintains alpha3 codes in the tracking DataFrame.
    Returns only the columns in list_of_vars and alpha3.
    """
    df_output = df.copy()
    
    # Initialize df_interp_track with an additional column for alpha3 codes
    df_interp_track = pd.DataFrame(index=df.index, columns=[alpha3_col] + list_of_vars)
    df_interp_track[alpha3_col] = df[alpha3_col]  # Copy alpha3 codes into df_interp_track
    
    for variable in list_of_vars:
        df[variable] = df[variable].astype(float)
        
        # First pass: Impute using neighboring countries' original data
        for idx in df[df[variable].isna()].index:
            country = df.loc[idx, alpha2_col]
            neighbors = df.loc[idx, 'neighbours']
            
            if isinstance(neighbors, str):
                # Ensure both neighbor codes and alpha2_col are consistently uppercase
                neighbor_list = [n.strip().upper() for n in neighbors.split(',')]
                alpha2_values = df[alpha2_col].str.upper().values
                
                # Filter valid neighbors by checking against the existing country codes
                valid_neighbors = [n for n in neighbor_list if n in alpha2_values]
                
                if valid_neighbors:  # Check if there are any valid neighbors
                    neighbor_values = df.loc[df[alpha2_col].str.upper().isin(valid_neighbors), variable].dropna()
                    if not neighbor_values.empty:
                        df_output.at[idx, variable] = neighbor_values.mean()
                        df_interp_track.at[idx, variable] = "neighbor interpolated"
        
        # Second pass: Impute using continent's original data
        for idx in df_output[df_output[variable].isna()].index:
            country = df_output.loc[idx, alpha2_col]
            continent = df_output.loc[idx, 'Continent']
            continent_values = df_output[df_output['Continent'] == continent][variable].dropna()
            if not continent_values.empty:
                df_output.at[idx, variable] = continent_values.mean()
                df_interp_track.at[idx, variable] = "continent interpolated"

    # Return only the list_of_vars and the alpha3 column
    df_output = df_output[[alpha3_col] + list_of_vars]
    return df_output, df_interp_track



def drop_small_island_nations(df, alpha3_col='alpha3'):
    """
    Drops specified small island nations from the DataFrame based on their alpha3 codes.
    """
    # List of alpha3 codes to drop
    small_island_nations = ['MAF', 'TCA', 'VGB', 'VIR', 'CHI', 'BLM', 'AND']
    
    # Drop rows where the alpha3 code is in the list
    df_cleaned = df[~df[alpha3_col].isin(small_island_nations)].copy()
    
    return df_cleaned



def drop_countries_without_continent(df, continent_col='Continent'):
    """
    Drops countries from the dataframe that do not have continent data.
    """
    # Drop rows where the continent column is NaN or empty
    df_cleaned = df.dropna(subset=[continent_col]).copy()
    df_cleaned = df_cleaned[df_cleaned[continent_col].str.strip() != '']
    
    return df_cleaned

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
    df.sort_values(by=['ISO', 'Year'], ascending=[True, False], inplace=True)

    # Drop duplicates to keep the latest 'PBO' value for each 'ISO'
    latest_pbo_df = df.drop_duplicates(subset=['ISO'], keep='first')

    # drop othe columns (keep PBO and alpha3)
    latest_pbo_df = latest_pbo_df[['ISO', 'PBO']]

    # Return the resulting DataFrame
    return latest_pbo_df

def process_water_data(df):
    """
    Processes water data by calculating TOTALPiped, filling missing values, and adding necessary columns.
    Ensures that missing values are filled from previous years' data.
    """
    # Replace '>99' with 100, '<1' with 0, and '-' with NaN
    df.replace({'<1': 0, '>99': 100, '-': pd.NA}, inplace=True)

    # Convert relevant columns to numeric for proper calculations
    columns_to_convert = ['TOTALPiped', 'RURALPiped', 'URBANPiped', 'Population \r\n(thousands)', '% urban']
    df[columns_to_convert] = df[columns_to_convert].replace({' ': ''}, regex=True)
    df[columns_to_convert] = df[columns_to_convert].apply(pd.to_numeric, errors='coerce')

    # Fill missing rural and urban piped data where total piped is 100
    df.loc[(df['TOTALPiped'] == 100) & (df['RURALPiped'].isna()), 'RURALPiped'] = 100
    df.loc[(df['TOTALPiped'] == 100) & (df['URBANPiped'].isna()), 'URBANPiped'] = 100

    # Sort by country and year
    df = df.sort_values(['Country', 'Year'])

    # Iterate through each country to ensure URBANPiped and RURALPiped have values
    def fill_from_older_data(group):
        group['RURALPiped'] = group['RURALPiped'].ffill().bfill()  # Fill forward first, then backward
        group['URBANPiped'] = group['URBANPiped'].ffill().bfill()
        return group

    # Explicitly exclude the grouping column after applying the function
    df = df.groupby('Country', group_keys=False).apply(fill_from_older_data).reset_index(drop=True)

    # Ensure that a copy is being worked on to avoid SettingWithCopyWarning
    most_recent_df = df.drop_duplicates('Country', keep='last').copy()

    # Recalculate TOTALPiped
    most_recent_df['TOTALPiped_Recalculated'] = (
        (most_recent_df['% urban'] / 100) * most_recent_df['URBANPiped'] +
        ((100 - most_recent_df['% urban']) / 100) * most_recent_df['RURALPiped']
    )

    return most_recent_df


def merge_gdp_data(df_cleaned, df_gdp, alpha2_col='alpha2', alpha3_col='Country Code'):
    """
    Merges df_cleaned with df_gdp using the Alpha-2 code from df_cleaned and Alpha-3 code from df_gdp.
    """
    # Ensure the Alpha-3 code exists in df_cleaned for the merge
    if 'alpha3' not in df_cleaned.columns:
        raise ValueError("df_cleaned must contain 'alpha3' column for merging with GDP data.")
    
    # Merge on the Alpha-3 code
    df_merged = df_cleaned.merge(df_gdp[['Country Code', 'GDP per Capita (current US$)']],
                                 left_on='alpha3',
                                 right_on=alpha3_col,
                                 how='left')
    
    return df_merged



def impute_using_gdp(df, list_of_vars, gdp_col='GDP per Capita (current US$)', alpha3_col='alpha3'):
    """
    Imputes missing values in the list of variables using GDP per capita via linear regression.
    Maintains alpha3 codes in the tracking DataFrame.
    Returns only the columns in list_of_vars and alpha3.
    """
    df_output = df.copy()
    
    # Initialize df_interp_track with an additional column for alpha3 codes
    df_interp_track = pd.DataFrame(index=df.index, columns=[alpha3_col] + list_of_vars)
    df_interp_track[alpha3_col] = df[alpha3_col]  # Copy alpha3 codes into df_interp_track
    
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
            df_output.loc[missing_idx, variable] = model.predict(df_output.loc[missing_idx, [gdp_col]])
            df_interp_track.loc[missing_idx, variable] = "gdp regression"
    
    # Return only the list_of_vars and the alpha3 column
    df_output = df_output[[alpha3_col] + list_of_vars]
    return df_output, df_interp_track



def merge_and_impute_with_gdp(df_cleaned, df_gdp, list_of_vars, alpha2_col='alpha2', alpha3_col='Country Code'):
    """
    Merges the cleaned DataFrame with GDP data and then imputes the specified variables using GDP via regression.
    """
    # Merge the DataFrames
    df_merged = merge_gdp_data(df_cleaned, df_gdp, alpha2_col=alpha2_col, alpha3_col=alpha3_col)
    
    # Perform imputation using GDP per capita via regression
    df_output, df_interp_track = impute_using_gdp(df_merged, list_of_vars)
    
    return df_output, df_interp_track






def preprocess_country_info_file(input_path, output_path):
    """
    Preprocess the country info file to remove the '#' from the header line and skip comment lines.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(input_path, 'r') as infile, open(output_path, 'w') as outfile:
        for line in infile:
            if line.startswith('#ISO'):
                line = line.replace('#ISO', 'ISO')
            if not line.startswith('#'):
                outfile.write(line)

def load_country_info(preprocessed_file_path):
    """
    Loads the preprocessed country information from a text file, selecting only relevant columns.
    Note that the preprocessed file should have the '#' removed from the header and comments skipped.
    Also, the 'na_values' parameter is set to handle the '_' values in the file, as North America is "NA".
    """
    country_info_df = pd.read_csv(preprocessed_file_path, sep='\t', keep_default_na=False, na_values=['_'])
    country_info_df = country_info_df[['ISO3', 'Continent', 'neighbours']]
    return country_info_df

def append_country_info(df, country_info_df):
    """
    Appends continent and neighboring countries to the dataframe using the pre-loaded country info DataFrame.
    """
    df = df.merge(country_info_df, how='left', left_on='alpha3', right_on='alpha3')
    # df.drop(columns=['alpha3'], inplace=True)
    return df




def main(water_JMP_file_path, bicycle_file_path, gdp_per_capita_file_path):
    """
    Main function to process the input data file, interpolate missing data, and save the results.
    """

    gdp_per_capita_df = load_gdp_per_capita(gdp_per_capita_file_path)

    # Read input data
    df_water = pd.read_csv(water_JMP_file_path)

    # Process water data
    df = process_water_data(df_water)

    # Add alpha2 codes from ISO
    df = add_alpha_codes(df,'Country' )

    # Manual mapping for problematic entries
    manual_alpha3_mapping = {
        "Channel Islands": "CHI",
        "United States Virgin Islands": "VIR",
        "Wallis and Futuna Islands": "WLF",
        "Democratic Republic of the Congo": "COD",
        "China, Hong Kong SAR": "HKG",
        "China, Macao SAR": "MAC",
        "Republic of Korea": "KOR",
    }

    # Apply manual mapping
    df['alpha3'] = df.apply(lambda row: manual_alpha3_mapping.get(row['Country'], row['alpha3']), axis=1)

    df = add_alpha2_from_alpha3(df, "alpha3")

    # add bicycle data
    df_bike = import_and_select_latest_pbo(bicycle_file_path)
    # rename ISO to alpha3
    df_bike.rename(columns={'ISO': 'alpha3'}, inplace=True)
    # Update 'ROM' to 'ROU' in the 'ISO' column of the bike_df DataFrame
    df_bike.loc[df_bike['alpha3'] == 'ROM', 'alpha3'] = 'ROU'
    
    # merge data
    df = df.merge(df_bike, how='outer', on='alpha3')

    ### HERE NEEDS MANUAOL ATTENTION ####

    ## http://download.geonames.org/export/dump/countryInfo.txt
    country_info_csv_path =  "/Users/kevin/Documents/ProgrammingIsFun/ALLFED/Water/water-access-gis/water-access/data/lookup tables/countryInfo.txt"
    output_file_path = '/Users/kevin/Documents/ProgrammingIsFun/ALLFED/Water/water-access-gis/water-access/data/processed/semi-processed/preprocessed_countryInfo.txt'

    # Preprocess the file to remove the '#' from the header and skip comments
    preprocess_country_info_file(country_info_csv_path,output_file_path)

    # Load the country info from the preprocessed file
    country_info_df = load_country_info(output_file_path)
    country_info_df.rename(columns={'ISO3': 'alpha3'}, inplace=True)

    df_merged_info = append_country_info(df, country_info_df)

    df_cleaned_merge = drop_countries_without_continent(df_merged_info)
    df_cleaned_merge = drop_small_island_nations(df_cleaned_merge)

    # Example usage
    list_of_vars = ['RURALPiped', 'URBANPiped', 'PBO']

    # Merge and impute using GDP regression
    df_gdp_imputation, df_gdp_imputation_track = merge_and_impute_with_gdp(df_cleaned_merge, gdp_per_capita_df, list_of_vars)

    # SPATIAL Interpolation
    list_of_vars = ['RURALPiped', 'URBANPiped','PBO']
    # Interpolate variables
    df_spatial_imputation, df_spatial_imputation_track = spatial_imputation(df_cleaned_merge, list_of_vars, "alpha2")

    # merge the dataframes
    df_imputed = df_spatial_imputation.merge(df_gdp_imputation, on='alpha3', suffixes=('_spatial', '_gdp'))
    df_output = df_imputed.merge(df, on='alpha3')
    df_imputed_track = df_spatial_imputation_track.merge(df_gdp_imputation_track, on='alpha3', suffixes=('_spatial', '_gdp'))
    # # Save dataframes as CSV
    df_output.to_csv('../../data/processed/semi-processed/merged_data.csv', index=False)
    df_imputed_track.to_csv('../../data/processed/semi-processed/merged_data_track.csv', index=False)

water_JMP_file_path = "/Users/kevin/Documents/ProgrammingIsFun/ALLFED/Water/water-access-gis/water-access/data/original_data/WHO Household Water Data - 2023 Data.csv"
bicycle_file_path = "/Users/kevin/Documents/ProgrammingIsFun/ALLFED/Water/water-access-gis/water-access/data/original_data/global-bike-ownership.csv"
gdp_per_capita_file_path = '/Users/kevin/Documents/ProgrammingIsFun/ALLFED/Water/water-access-gis/water-access/data/original_data/API_NY.GDP.PCAP.CD_DS2_en_csv_v2_3189570.csv'

main(water_JMP_file_path, bicycle_file_path,gdp_per_capita_file_path)

