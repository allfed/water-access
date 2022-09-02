import pandas as pd
import matplotlib.pyplot as plt
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


col = "Country / Economy"
path_tri = "../data/terrain-ruggedness-index.csv"  # terrain ruggedness index
path_rq = "../data/global-road-quality.csv"  # road quality

df_rq = read_road_values(path_rq, col)
