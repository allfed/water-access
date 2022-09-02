from operator import concat
from functools import reduce
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import numpy as np


def read_road_values(path):
    df = pd.read_csv(path)
    df.rename(columns={"Score": "RoadQuality"}, inplace=True)
    df = df.drop(["Country / Economy"], axis=1)
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
        % (df_master.columns[corr_col_1], df_master.columns[corr_col_2], corr_value)
    )


def bubble_plot(df):
    fig = px.scatter(
        df,
        x="PBO",
        y="Terrain Ruggedness Index 100m (Nunn and Puga 2012)",
        size="Population",
        color="Entity",
        hover_name="Entity",
        log_x=True,
        log_y=True,
        size_max=60,
    )
    fig.show()


def world_map_plot(df):
    fig = px.choropleth(
        df,
        locations=df.index,
        color="RoadQuality",
        hover_name="Entity",
        title="Risk per country",
        hover_data=[
            "Urban population (% of total population)",
            "Terrain Ruggedness Index 100m (Nunn and Puga 2012)",
            "PBO",
        ],
        color_continuous_scale=px.colors.sequential.PuRd,
    )

    fig["layout"].pop("updatemenus")
    fig.show()


def risk_creator(df3):
    bike_risk = 1 - df3.loc[:, "PBO"] / 100
    urb_risk = (
        df3.loc[
            :,
            "Population in urban agglomerations of more than 1 million (% of total population)",
        ]
        / 100
    )
    road_risk = 1 - df3.loc[:, "RoadQuality"] / 7
    TRI_quantiles = pd.qcut(
        df3.loc[:, "Terrain Ruggedness Index 100m (Nunn and Puga 2012)"],
        np.linspace(0, 1, 11),
        labels=np.linspace(0.1, 1, 10),
    )
    risk_master = TRI_quantiles.astype(float) * bike_risk * urb_risk * road_risk
    return risk_master


"""
Start main function
"""

col = "Country / Economy"
path_tri = "../data/terrain-ruggedness-index.csv"  # terrain ruggedness index
path_rq = "../data/global_road_quality_processed.csv"  # road quality
path_pbo = "../data/global-bike-ownership.csv"  # 'percent bike ownership'
path_purb = "../data/share-of-population-urban.csv"  # share of urban population
path_urbagg = (
    "../data/urban-agglomerations-1-million-percent.csv"  # urban agglomerations
)
path_pop = "../data/population.csv"  # population

df_rq = read_road_values(path_rq)
df_pbo = read_pbo_values(path_pbo)
df_tri = read_owid_values(path_tri)
df_purb = read_owid_values(path_purb)
df_urbagg = read_owid_values(path_urbagg)
df_pop = read_pop_values(path_pop)

dfs = [df_pop, df_pbo, df_tri, df_purb, df_urbagg, df_rq]
df_master = pd.concat(dfs, join="outer", axis=1)
df_master_na = df_master.dropna()

# df3 = df_master_na
# df3['quantiles'] = pd.qcut(df3["Terrain Ruggedness Index 100m (Nunn and Puga 2012)"], np.linspace(0,1,11), labels=np.linspace(0.1,1,10))

risk_master = risk_creator(df_master_na)


# 'Entity', 'Population', 'Year', 'PBO',
#        'Terrain Ruggedness Index 100m (Nunn and Puga 2012)',
#        'Urban population (% of total population)',
#        'Population in urban agglomerations of more than 1 million (% of total population)',
#        'RoadQuality'],

# world_map_plot(df_master)
