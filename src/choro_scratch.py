import pandas as pd
import plotly.express as px


# choro_df = pd.read_csv("../data/WHO_Drinking_Estimates_ISO.csv",  index_col="alpha3")
choro_df = pd.read_csv("../data/WHO_Household_Water_Data_ISO.csv",  index_col="alpha3")
choro_df_recent = choro_df.groupby(choro_df.index).last() # works because csv is already in ascending date order

#argentina annoyling doesn't have data in 2020, so use the 2015 data
choro_df_recent.loc["ARG",:] = choro_df.loc["ARG"].groupby(choro_df.loc["ARG"].index).first().values[0] #surely a better way?

cols=[i for i in choro_df_recent.columns if i not in ["Country"]]
for col in cols:
    choro_df_recent[col]=pd.to_numeric(choro_df_recent[col])

choro_df_recent.isnull().sum()

colour_col = 5

# https://plotly.com/python/choropleth-maps/
fig1 = px.choropleth(
    data_frame=choro_df_recent,
    locations=choro_df_recent.index,
    height=600,
    color=cols[colour_col],
    hover_name="Country",
    hover_data=cols,
)

fig1.show()
