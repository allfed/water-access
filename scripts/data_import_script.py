import pandas as pd


def read_road_values(path):
    df = pd.read_csv(path)
    df.rename(columns={"Score": "RoadQuality"}, inplace=True)
    df = df.drop(["Country / Economy"], axis=1)
    df = df.set_index("alpha3")
    return df


def read_simple_csv_values(path):
    df = pd.read_csv(path)
    df = df.set_index("alpha3")
    return df


def read_pbo_values(path):
    df = pd.read_csv(path)
    df.rename(columns={"ISO": "alpha3", "Year": "YearPBO"}, inplace=True)
    df = df.set_index("alpha3")
    df = df.sort_values(by="YearPBO")  # sort ascending
    df = df[
        ~df.index.duplicated(keep="last")
    ]  # delete duplicate entries from countries, keep the most recent data
    df = df.drop(["Group", "Households"], axis=1)
    return df


def read_owid_values(path):
    """
    reads csv files downloaded from our world in data, isolates the ISO alpga 3 country code, and keeps only the most recent data per country
    """
    df = pd.read_csv(path)
    df.rename(columns={"Code": "alpha3"}, inplace=True)
    df = df.set_index("alpha3")
    df = df.sort_values(by="Year")  # sort ascending
    df = df[
        ~df.index.duplicated(keep="last")
    ]  # delete duplicate entries from countries, keep the most recent data
    df = df.drop(["Entity", "Year"], axis=1)
    return df


def read_pop_values(path):
    """
    reads csv files downloaded from our world in data, isolates the ISO alpga 3 country code, and keeps only the most recent data per country
    """
    df = pd.read_csv(path)
    df.rename(
        columns={
            "Code": "alpha3",
            "Population (historical estimates)": "Population",
        },
        inplace=True,
    )
    df = df.set_index("alpha3")
    df = df.sort_values(by="Year")  # sort ascending
    df = df[
        ~df.index.duplicated(keep="last")
    ]  # delete duplicate entries from countries, keep the most recent data
    df = df.drop(["Year"], axis=1)
    return df


def correlation_checker(df, corr_col_1, corr_col_2):
    """
    prints the correaltion between two columns denoted by their numerical index (column number)
    """
    corr_value = df.iloc[:, corr_col_1].corr(df.iloc[:, corr_col_2])
    print(
        "Correlation between %s and %s is %0.4f"
        % (df.columns[corr_col_1], df.columns[corr_col_2], corr_value)
    )


def input_data_creator():
    path_tri = "../data/terrain-ruggedness-index.csv"  # terrain ruggedness index
    path_rq = "../data/global_road_quality_processed.csv"  # road quality
    path_pbo = "../data/global-bike-ownership.csv"  # 'percent bike ownership'
    path_purb = "../data/share-of-population-urban.csv"  # share of urban population
    path_urbagg = (
        "../data/urban-agglomerations-1-million-percent.csv"  # urban agglomerations
    )
    path_pop = "../data/population.csv"  # population
    path_kummu = "../data/KummuAlpha3.csv"
    path_WHO = "../data/WHO SUMMARY WATER ISO.csv"
    path_HP = "../data/hand_pump_data.csv"

    df_rq = read_road_values(path_rq)
    df_pbo = read_pbo_values(path_pbo)
    df_tri = read_owid_values(path_tri)
    df_purb = read_owid_values(path_purb)
    df_urbagg = read_owid_values(path_urbagg)
    df_pop = read_pop_values(path_pop)
    df_kummu = read_simple_csv_values(path_kummu)
    df_WHO = read_simple_csv_values(path_WHO)
    df_HP = read_simple_csv_values(path_HP)

    dfs = [df_pop, df_pbo, df_tri, df_purb, df_urbagg, df_rq, df_kummu, df_WHO, df_HP]
    df_master = pd.concat(dfs, join="outer", axis=1)
    df_master["Risk"] = 0
    df_master.rename(
        columns={
            "Terrain Ruggedness Index 100m (Nunn and Puga 2012)": "Terrain Ruggedness"
        },
        inplace=True,
    )
    df_master.rename(
        columns={
            "Population in urban agglomerations of more than 1 million (% of total population)": "Urban Agg %"
        },
        inplace=True,
    )
    df_master.rename(
        columns={"Urban population (% of total population)": "Urban %"}, inplace=True
    )
    return df_master


"""
Start main function
"""


# df = input_data_creator()
# df.to_csv("country_data_master.csv")
