
import pathlib
import pandas as pd
import numpy as np


def load_data(data_file: str) -> pd.DataFrame:
    '''
    Load data from /data directory
    '''
    PATH = pathlib.Path(__file__).parent
    DATA_PATH = PATH.joinpath("../data").resolve()
    return pd.read_csv(DATA_PATH.joinpath(data_file))



df1 = load_data("rough_pass_kummu_short.csv")
# df_input = load_data("country_data_master_interpolated.csv")


# assume projection is fine....

# # convert population density to percent of national popluation
# df1["pop_density"] = df1["pop_density"] / df1["pop_density"].sum()

# convert population density to percent of national population on a per country basis, grouped by ISO_CC
df1["pop_density_perc"] = df1.groupby("ISO_CC")["pop_density1"].apply(lambda x: x / x.sum())
# df1["population in region"] = df1["pop_density_perc"] * df1["population"]
