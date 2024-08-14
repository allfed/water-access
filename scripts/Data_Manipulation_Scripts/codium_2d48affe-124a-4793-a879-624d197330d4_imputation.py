

import pandas as pd
import numpy as np
from countryinfo import CountryInfo
import pycountry

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
        try:
            country = pycountry.countries.search_fuzzy(input_country)
            alpha3 = country[0].alpha_3
        except:
            alpha3 = "unk_" + input_country
        countries.append(alpha3)
    df["alpha3"] = countries
    return df

def append_country_info(df, alpha2_col):
    """
    Appends region, subregion, and bordering countries to the dataframe
    """
    input_countries = df[alpha2_col]
    bordering_countries = []
    region = []
    subregion = []
    for input_country in input_countries:
        try:
            country = CountryInfo(input_country)
        except:
            print("Couldn't find country")

        try:
            country_bordering_countries = ",".join(str(x) for x in country.borders())
        except:
            country_bordering_countries = "unk_" + str(input_country)

        try:
            country_region = country.region()
        except:
            country_region = "unk_" + str(input_country)

        try:
            country_subregion = country.subregion()
        except:
            country_subregion = "unk_" + str(input_country)

        bordering_countries.append(country_bordering_countries)
        region.append(country_region)
        subregion.append(country_subregion)
    df["borders"] = bordering_countries
    df["region"] = region
    df["subregion"] = subregion
    return df

def interpolate_variables(df, list_of_vars, alpha3_col):
    """
    Interpolates missing values in the dataframe using bordering countries, subregion, and region
    """
    df_output = df.copy()
    df_interp_track = pd.DataFrame(df.index)
    df_interp_track = df_interp_track.set_index(alpha3_col)
    df_interp_track[list_of_vars] = "none"

    for variable in list_of_vars:
        df[variable] = df[variable].astype(float)
        nan_locations = df[variable].isna()
        countries_requiring_interp = df[nan_locations].index

        for country in countries_requiring_interp:
            borders = df.loc[country, "borders"]
            borders = str(borders)
            border_list = borders.split(",")
            try:
                avg_var_border = df.loc[border_list, variable].mean()
            except:
                avg_var_border = np.nan

            subregion_list = df[df["subregion"] == df.loc[country, "subregion"]].index
            avg_var_subregion = df.loc[subregion_list, variable].mean()

            region_list = df[df["region"] == df.loc[country, "region"]].index
            avg_var_region = df.loc[region_list, variable].mean()

            if ~np.isnan(avg_var_border):
                df_output.loc[country, variable] = avg_var_border
                df_interp_track.loc[country, variable] = "border interpolated"
            elif ~np.isnan(avg_var_subregion):
                df_output.loc[country, variable] = avg_var_subregion
                df_interp_track.loc[country, variable] = "subregion interpolated"
            else:
                df_output.loc[country, variable] = avg_var_region
                df_interp_track.loc[country, variable] = "region interpolated"

    return df_output, df_interp_track

def process_water_data(df):
    """
    Processes water data by calculating TOTALPiped, filling missing values, and adding necessary columns.
    """
    # Replace '>99' with 100 and '<1' with 0
    df.replace({'<1': 0, '>99': 100}, inplace=True)

    # Convert relevant columns to numeric for proper calculations
    columns_to_convert = ['TOTALPiped', 'RURALPiped', 'URBANPiped', 'Population \r\n(thousands)', '% urban']
    df[columns_to_convert] = df[columns_to_convert].apply(pd.to_numeric, errors='coerce')

    # Fill missing rural and urban piped data where total piped is 100
    df.loc[(df['TOTALPiped'] == 100) & (df['RURALPiped'].isna()), 'RURALPiped'] = 100
    df.loc[(df['TOTALPiped'] == 100) & (df['URBANPiped'].isna()), 'URBANPiped'] = 100

    # Keep only the most recent year for each country
    most_recent_df = df.sort_values('Year').drop_duplicates('Country', keep='last')

    # Recalculate TOTALPiped
    most_recent_df['TOTALPiped_Recalculated'] = (
        (most_recent_df['% urban'] / 100) * most_recent_df['URBANPiped'] +
        ((100 - most_recent_df['% urban']) / 100) * most_recent_df['RURALPiped']
    )


    # Extract the needed columns, including Population and % urban
    final_df = most_recent_df[['Country', 'Year', 'Population \r\n(thousands)', '% urban', 
                               'TOTALPiped', 'RURALPiped', 'URBANPiped', 'TOTALPiped_Recalculated']]
    
    return final_df

def main(input_file_path, list_of_vars):
    """
    Main function to process the input data file, interpolate missing data, and save the results.
    """
    # Read input data
    df_input = pd.read_csv(input_file_path)

    # Process water data
    df = process_water_data(df_input)

    # Add alpha2 codes from ISO
    df = add_alpha_codes(df,'Country' )
    df = add_alpha2_from_alpha3(df, "alpha3")

    # Append country info
    df = append_country_info(df, "alpha2")


    # display(df)

    # print header names
    print(df.columns)

    # Interpolate variables
    # df_output, df_interp_track = interpolate_variables(df, list_of_vars, "alpha3")

    # Save dataframes as CSV
    # df_interp_track.to_csv("WHOinterpTrack3.csv")
    # df_output.to_csv("WHOinterp3.csv")

# Example usage
input_file_path = "/Users/kevin/Documents/ProgrammingIsFun/ALLFED/Water/water-access-gis/water-access/data/original_data/WHO Household Water Data - 2023 Data.csv"
list_of_vars = ['RURALPiped', 'URBANPiped']

main(input_file_path, list_of_vars)





# Generated by CodiumAI

# Dependencies:
# pip install pytest-mock
import pytest

class TestAppendCountryInfo:

    # Appends region, subregion, and bordering countries to the dataframe correctly
    def test_appends_country_info_correctly(self, mocker):
        import pandas as pd
        from countryinfo import CountryInfo
        from scripts.Data_Manipulation_Scripts.imputation import append_country_info

        # Mocking CountryInfo methods
        mocker.patch.object(CountryInfo, 'borders', return_value=['CountryA', 'CountryB'])
        mocker.patch.object(CountryInfo, 'region', return_value='RegionX')
        mocker.patch.object(CountryInfo, 'subregion', return_value='SubregionY')

        # Creating a sample dataframe
        df = pd.DataFrame({'alpha2': ['US', 'CA']})

        # Calling the function
        result_df = append_country_info(df, 'alpha2')

        # Asserting the results
        assert result_df['borders'].tolist() == ['CountryA,CountryB', 'CountryA,CountryB']
        assert result_df['region'].tolist() == ['RegionX', 'RegionX']
        assert result_df['subregion'].tolist() == ['SubregionY', 'SubregionY']