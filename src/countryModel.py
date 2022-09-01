from operator import concat
from functools import reduce
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px


def read_road_values(path):
    df = pd.read_csv(path)
    df.rename(columns={"Score": "RoadQuality"}, inplace=True)
    df = df.drop(["Country / Economy"], axis=1)
    df = df.set_index("alpha3")
    return df

def read_pbo_values(path):
    df = pd.read_csv(path)
    df.rename(columns={"ISO": "alpha3","Year": "YearPBO"}, inplace=True)
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
    df.rename(columns={"Code": "alpha3", "Population (historical estimates)": "Population", }, inplace=True)
    df = df.set_index("alpha3")
    df = df.sort_values(by="Year")  # sort ascending
    df = df[
        ~df.index.duplicated(keep="last")
    ]  # delete duplicate entries from countries, keep the most recent data
    df = df.drop(["Year"], axis=1)
    return df

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

corr_col_1 = 5
corr_col_2 = 5
corr_value = df_master.iloc[:, corr_col_1].corr(df_master.iloc[:, corr_col_2])
print("Correlation between %s and %s is %0.4f" % (df_master.columns[corr_col_1],df_master.columns[corr_col_2],corr_value))


# 'Entity', 'Population', 'Year', 'PBO',
#        'Terrain Ruggedness Index 100m (Nunn and Puga 2012)',
#        'Urban population (% of total population)',
#        'Population in urban agglomerations of more than 1 million (% of total population)',
#        'RoadQuality'],
  


# fig = px.scatter(df_master_na, x="PBO", y="Terrain Ruggedness Index 100m (Nunn and Puga 2012)",
# 	         size="Population", color="Entity",
#                  hover_name="Entity", log_x=True, log_y=True, size_max=60)
# fig.show()


# fig = px.choropleth(df_master, locations=df_master.index,
#                     color="RoadQuality",
#                     hover_name="Entity",
#                     title = "Risk per country",  
#                     hover_data=["Urban population (% of total population)",
#                      "Terrain Ruggedness Index 100m (Nunn and Puga 2012)",
#                      "PBO"],
#                     color_continuous_scale=px.colors.sequential.PuRd)

# fig["layout"].pop("updatemenus")
# fig.show()

