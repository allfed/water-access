#!/usr/bin/env python
# coding: utf-8

# In[1]:

import pathlib
import plotly.express as px
import pandas as pd
import numpy as np
from dash import Dash, dcc, html, Output, Input, dash_table  # pip install dash
import dash_bootstrap_components as dbc  # pip install dash-bootstrap-components
from jupyter_dash import JupyterDash
import plotly.figure_factory as ff



# Declare server for Heroku deployment. Needed for Procfile.



# In[2]:

def load_data(data_file: str) -> pd.DataFrame:
    '''
    Load data from /data directory
    '''
    PATH = pathlib.Path(__file__).parent
    DATA_PATH = PATH.joinpath("../data").resolve()
    return pd.read_csv(DATA_PATH.joinpath(data_file))


def risk_creator(input_df):
    """
    Risk creator takes all
    Uses copy method for handling dataframe assignment
    https://stackoverflow.com/questions/65922241/best-practice-for-passing-pandas-dataframe-to-functions
    """
    df = input_df.copy()
    # risk from bikes
    df["PBO_risk"] = 1 - df.loc[:, "PBO"] / 100

    # find quartiles for TRI (rather than using the adjusted score as a percent)
    # CHECK IF THIS MAKES SENSE?
    df["TRI_risk"] = pd.qcut(
        df.loc[:, "Terrain Ruggedness"],
        np.linspace(0, 1, 11),
        labels=np.linspace(0.1, 1, 10),
    ).astype(float)

    df["urb_risk"] = (
        df.loc[
            :,
            "Urban %",
        ]
        / 100
    )

    df["urbagg_risk"] = (
        df.loc[
            :,
            "Urban Agg %",
        ]
        / 100
    )

    df["road_risk"] = 1 - df.loc[:, "RoadQuality"] / 7

    df["distance_risk"] = df.loc[:, "Km"]

    return df


def weight_parameter(series, slider_weight):
    """
    weights each panda series (column) of the dataframe one at a time, multiples the risk by the weighting, or if zero, let's the whole column = 0. Tis is useful because it removes NaNs (and means that more countries can be displayed)
    """
    if slider_weight == 0:
        series = 0
    else:
        series = series * slider_weight

    return series


def correlation_checker(df, corr_col_1, corr_col_2):
    """
    prints the correaltion between two columns denoted by their numerical index (column number)
    """
    corr_value = df.iloc[:, corr_col_1].corr(df.iloc[:, corr_col_2])
    print(
        "Correlation between %s and %s is %0.4f"
        % (df.columns[corr_col_1], df.columns[corr_col_2], corr_value)
    )


# In[3]:


"""
Start main function
"""
# find dataframe

df_input = load_data("country_data_master_interpolated.csv")
df_input = df_input.set_index("alpha3")

# do stuff with dataframe
df = risk_creator(df_input)

# create extra columns for drop down
df["Population Piped"] = (df["Nat Piped"]/100) *df["Population"]
df["Nat Improved"]=df["Nat Piped"]+df["Nat NonPiped"]
df["Nat Unimproved and Surface"]=100-df["Nat Improved"]
df["Population Piped Has to Relocate"] = 0 



df_table = df[
    [
        "Entity",
        "Population",
        "PBO",
        "Terrain Ruggedness",
        "Urban %",
        "Urban Agg %",
        "RoadQuality",
        "Km",
        "Risk Score",
    ]
]
ignore_columns_index = 3


# In[4]:


PBO_default = 200
Km_default = 400
scaling_default = 50
RoadQuality_default = 200
Terrain_Ruggedness_default = 50
Urban_default = 100
Urban_Agg_default = 0



# In[5]:


# Build your components
app = JupyterDash(__name__, external_stylesheets=[dbc.themes.LUX])
# app = Dash(__name__, external_stylesheets=[dbc.themes.LUX])
server = app.server  # for heroku deployment

mytitle = dcc.Markdown(children="")
mysubtitle = dcc.Markdown(children="")
choro = dcc.Graph(figure={})
choro2 = dcc.Graph(figure={})
choro3 = dcc.Graph(figure={})
bar = dcc.Graph(figure={})
bubble = dcc.Graph(figure={})
dropdown = dcc.Dropdown(
    options=df.columns.values[2:],
    value="Risk Score",  # initial value displayed when page first loads
    clearable=False,
)

table = dash_table.DataTable(
    columns=[{"name": i, "id": i} for i in df_table.columns],
    data=df_table.to_dict("records"),
    filter_action="native",
    page_size=20,
    style_data={
        "width": "150px",
        "minWidth": "150px",
        "maxWidth": "150px",
        "overflow": "hidden",
        "textOverflow": "ellipsis",
    },
)

### Create slider components on a card
controls = dbc.Card(
    [
        dbc.Label(
            "Adjust the weighting of each parameter on the final risk score with the sliders"
        ),
        html.Div(
            [
                dbc.Label("Bike Ownership %"),
                dcc.Slider(
                    0,
                    400,
                    value=PBO_default,
                    id="myslider1",
                    updatemode="drag",
                    tooltip={"placement": "bottom", "always_visible": True},
                ),
            ]
        ),
        html.Div(
            [
                dbc.Label("Terrain Ruggedness",),
                dcc.Slider(
                    0,
                    400,
                    value=Terrain_Ruggedness_default,
                    id="myslider2",
                    updatemode="drag",
                    tooltip={"placement": "bottom", "always_visible": True},
                ),
            ]
        ),
        html.Div(
            [
                dbc.Label("Urban %"),
                dcc.Slider(
                    0,
                    400,
                    value=Urban_default,
                    id="myslider3",
                    updatemode="drag",
                    tooltip={"placement": "bottom", "always_visible": True},
                ),
            ]
        ),
        html.Div(
            [
                dbc.Label("Urban Agg %"),
                dcc.Slider(
                    0,
                    400,
                    value=Urban_Agg_default,
                    id="myslider4",
                    updatemode="drag",
                    tooltip={"placement": "bottom", "always_visible": True},
                ),
            ]
        ),
        html.Div(
            [
                dbc.Label("RoadQuality"),
                dcc.Slider(
                    0,
                    400,
                    value=RoadQuality_default,
                    id="myslider5",
                    updatemode="drag",
                    tooltip={"placement": "bottom", "always_visible": True},
                ),
            ]
        ),
        html.Div(
            [
                dbc.Label("Median Distance To Water"),
                dcc.Slider(
                    0,
                    400,
                    value=Km_default,
                    id="myslider6",
                    updatemode="drag",
                    tooltip={"placement": "bottom", "always_visible": True},
                ),
            ]
        ),        
        html.Div(
            [
                dbc.Label("Risk Scale"),
                dcc.Slider(
                    0,
                    100,
                    value=scaling_default,
                    id="myslider7",
                    updatemode="drag",
                    tooltip={"placement": "bottom", "always_visible": True},
                ),
            ]
        ),
    ],
    body=True,
)



# In[6]:


# Customize your own Layout
app.layout = dbc.Container(
    [
        dbc.Row([dbc.Col([mytitle], width=6)], justify="center"),
        dbc.Row([dbc.Col([mysubtitle], width=6)], justify="center"),
        dbc.Row(
            [
                dbc.Col(controls, md=3),
                dbc.Col([choro], width=9),
            ],
            align="center",
        ),
        dbc.Row([dbc.Col([dropdown], width=6)], justify="center"),
        html.Hr(),
        dbc.Row([dbc.Col([bubble], width=12)], justify="center"),
        dbc.Row([dbc.Col([choro2], width=12)], justify="center"),
        dbc.Row([dbc.Col([choro3], width=12)], justify="center"),
        dbc.Row([dbc.Col([bar], width=12)], justify="center"),
        dbc.Row([dbc.Col([table], width=12)], justify="center"),
    ],
    fluid=True,
)


# In[7]:


# Callback allows components to interact
@app.callback(
    Output(choro, "figure"),
    Output(choro2, "figure"),
    Output(choro3, "figure"),
    Output(bar, "figure"),
    Output(bubble, "figure"),
    Output(table, "data"),
    Output(mytitle, "children"),
    Output(mysubtitle, "children"),
    Input(dropdown, "value"),
    Input("myslider1", "value"),
    Input("myslider2", "value"),
    Input("myslider3", "value"),
    Input("myslider4", "value"),
    Input("myslider5", "value"),
    Input("myslider6", "value"),
    Input("myslider7", "value"),
)
def update_graph(
    column_name, scaling1, scaling2, scaling3, scaling4, scaling5, scaling6,scaling7
):  # function arguments come from the component property of the Input
    # ignore_columns_num = 9

    dff = (
        df.copy()
    )  # create copy of dataframe to apply weightings to before creating the risk matrix
    slider_weights = [scaling1, scaling2, scaling3, scaling4, scaling5, scaling6]
    slider_cols = [
        "PBO",
        "Terrain Ruggedness",
        "Urban %",
        "Urban Agg %",
        "RoadQuality",
        "Km",
    ]
    risk_cols = [
        "PBO_risk",
        "TRI_risk",
        "urb_risk",
        "urbagg_risk",
        "road_risk",
        "distance_risk",
    ]

    # check hardcoding here, this is the number of factors that are summed in to the risk matrix
    for id, col in enumerate(slider_cols):
        series = weight_parameter(df[risk_cols[id]], slider_weights[id])
        dff[risk_cols[id]] = series

    dff = dff[dff[column_name].notnull()]
    ### Sum all of the weighted risk values in to the 'Risk' column of the dataframe
    dff["Risk Score"] = dff[risk_cols].sum(axis="columns", skipna=False)

    ### Normalise for a percentage risk,


    risk_scale = scaling7/100
    dff["Risk Score"] = dff["Risk Score"] / dff.at["NLD","Risk Score"]
    dff["Population Piped"] = (dff["Nat Piped"]/100) *dff["Population"]
    dff["Nat Improved"]=dff["Nat Piped"]+dff["Nat NonPiped"]
    dff["Nat Unimproved and Surface"]=100-dff["Nat Improved"]
    dff["Theoretical Population Piped Has to Relocate"] = (dff["Risk Score"]-1)*dff["Population Piped"]*risk_scale #this bases off netherlands no one having to move. 

    # if pandas value greater than , equals another column value
    locations_over = (dff["Theoretical Population Piped Has to Relocate"]>dff["Population Piped"])
    dff["Population Has to Relocate"] = dff["Theoretical Population Piped Has to Relocate"]
    dff["Population Has to Relocate"][locations_over] = dff["Population Piped"][locations_over]
    dff["Percent of Population Has to Relocate"] = (dff["Population Has to Relocate"]/dff["Population"])*100




    nancount = dff["Risk Score"].isnull().sum()
    mysubtitle = f"Displaying {len(df.index)-nancount} countries from a total of {len(df.index)} based on data availability"



    hover_data_list =[
            "PBO",
            "Terrain Ruggedness",
            "Urban %",
            "RoadQuality",
            "Km",
            "Nat Piped",
    ]


    # https://plotly.com/python/choropleth-maps/
    choro1 = px.choropleth(
        title=column_name, 
        data_frame=dff,
        locations=dff.index,
        height=600,
        color=column_name,
        hover_name="Entity",
        hover_data=hover_data_list
    )
    choro1.layout.coloraxis.colorbar.title = ''


    choro2 = px.choropleth(
        title="Percent of Population with Piped Water",
        data_frame=dff,
        locations=dff.index,
        height=600,
        color="Nat Piped",
        hover_name="Entity",
        hover_data=hover_data_list
    )
    choro2.layout.coloraxis.colorbar.title = ''


    choro3 = px.choropleth(
        title="Percent of Population Has to Relocate",
        data_frame=dff,
        locations=dff.index,
        height=600,
        color="Percent of Population Has to Relocate",
        hover_name="Entity",
        hover_data=hover_data_list
    )
    choro3.layout.coloraxis.colorbar.title = ''



    graph_filter = dff[column_name].notnull()
    fig2 = px.bar(
        dff[graph_filter].sort_values(by=column_name, ascending=False),
        y=column_name,
        x=dff.sort_values(by=column_name, ascending=False)[graph_filter].index,
        hover_name="Entity",
    )
    fig2.update_layout(uniformtext_minsize=14, uniformtext_mode="show")

    fig3 = px.scatter(
        dff[graph_filter].sort_values(by=column_name, ascending=False),
        x="Entity",
        y=column_name,
        size="Population",
        color="subregion",
        hover_name="Entity",
        log_y=True,
        size_max=60,
    )

    # gg = correlation_checker(dff,"National At Least Basic","Nat Piped")
    # print(gg)

    return (
        choro1,
        choro2,
        choro3,
        fig2,
        fig3,
        dff.round(decimals=1)
        .sort_values(by=column_name, ascending=False)
        .to_dict("records"),
        "# " + column_name,
        mysubtitle,
    )  # returned objects are assigned to the component property of the Output



# In[8]:







# In[9]:


# Run app
if __name__ == "__main__":
    app.run_server(mode='external')
    


# if __name__ == "__main__":
#     app.run_server(debug=True)



# In[10]:





group_labels = ['Kms to Water']
colors = ['#333F44', '#37AA9C', '#94F3E4']

# Create distplot with curve_type set to 'normal'
fig = ff.create_distplot([df["Km"].dropna()], group_labels, show_hist=False, colors=colors)

# Add title
fig.update_layout(title_text='Distribution of Distance to Water Source')
fig.update_layout(xaxis_type = "log")
fig.show()


