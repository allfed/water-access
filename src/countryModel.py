from operator import concat
from functools import reduce
import pandas as pd
import matplotlib.pyplot as plt
import pycountry

def read_road_values(path):
    df = pd.read_csv(path)
    df.rename(columns={'Score': 'RoadQuality'}, inplace=True)
    df = df.drop(['Country / Economy'], axis=1)    
    df = df.set_index('alpha3')
    return df

def read_pbo_values(path):
    df = pd.read_csv(path)
    df.rename(columns={'ISO': 'alpha3'}, inplace=True)
    df = df.set_index('alpha3')
    df = df.sort_values(by='Year') # sort ascending
    df = df[~df.index.duplicated(keep='last')] #delete duplicate entries from countries, keep the most recent data
    df = df.drop(['Group', 'Households'], axis=1)    
    return df

def read_owid_values(path):
    df = pd.read_csv(path)
    df.rename(columns={'Code': 'alpha3'}, inplace=True)
    df = df.set_index('alpha3')
    df = df.sort_values(by='Year') # sort ascending
    df = df[~df.index.duplicated(keep='last')] #delete duplicate entries from countries, keep the most recent data
    df = df.drop(['Entity', 'Year'], axis=1)    
    return df

"""
Start main function
"""

col = "Country / Economy"
path_tri = "../data/terrain-ruggedness-index.csv"   # terrain ruggedness index
path_rq = "../data/global_road_quality_processed.csv"         # road quality
path_pbo = "../data/global-bike-ownership.csv"      # 'percent bike ownership'
path_purb = "../data/share-of-population-urban.csv"    # share of urban population
path_urbagg = "../data/urban-agglomerations-1-million-percent.csv" #urban agglomerations
path_pop = "../data/population.csv"                 # population

df_rq = read_road_values(path_rq)
df_pbo = read_pbo_values(path_pbo)
df_tri = read_owid_values(path_tri)
df_purb = read_owid_values(path_purb)
df_urbagg = read_owid_values(path_urbagg)
df_pop = read_owid_values(path_pop)

dfs = [df_pop, df_pbo, df_tri, df_purb, df_urbagg, df_rq]
result_1 = pd.concat(dfs, join='outer', axis=1)


