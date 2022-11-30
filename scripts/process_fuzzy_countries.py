import pandas as pd
import pycountry


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


def read_road_values_fuzzy(path, col):
    df = pd.read_csv(path)
    df = add_alpha_codes(df, col)
    df = df.set_index("alpha3")
    return df


col = "Country"
path = "../data/WHO Household Water Data - National Piped.csv"

df = read_road_values_fuzzy(path, col)


filename =  "WHO_household_estimates"
df.to_csv(filename + ".csv")

