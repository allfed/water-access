import pathlib
import plotly.express as px
import pandas as pd
from difflib import diff_bytes
import numpy as np
from dash import Dash, dcc, html, Output, Input, dash_table  # pip install dash
import dash_bootstrap_components as dbc  # pip install dash-bootstrap-components

# Declare server for Heroku deployment. Needed for Procfile.

"""
Next steps:
Generate single csv for inpout?
Upload to heroku
"""


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


"""
Start main function
"""
# find dataframe
PATH = pathlib.Path(__file__).parent
DATA_PATH = PATH.joinpath("../data").resolve()
df_input = pd.read_csv(
    DATA_PATH.joinpath("country_data_master.csv"), index_col="alpha3"
)
# do stuff with dataframe
df = risk_creator(df_input)
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
        "Risk",
    ]
]
ignore_columns_index = 3

# Build your components
app = Dash(__name__, external_stylesheets=[dbc.themes.LUX])
server = app.server  # for heroku deployment

mytitle = dcc.Markdown(children="")
mysubtitle = dcc.Markdown(children="")
choro = dcc.Graph(figure={})
bar = dcc.Graph(figure={})
bubble = dcc.Graph(figure={})
dropdown = dcc.Dropdown(
    options=df.columns.values[2:],
    value="Risk",  # initial value displayed when page first loads
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
                    value=100,
                    id="myslider1",
                    updatemode="drag",
                    tooltip={"placement": "bottom", "always_visible": True},
                ),
            ]
        ),
        html.Div(
            [
                dbc.Label(df.columns[ignore_columns_index + 1]),
                dcc.Slider(
                    0,
                    400,
                    value=100,
                    id="myslider2",
                    updatemode="drag",
                    tooltip={"placement": "bottom", "always_visible": True},
                ),
            ]
        ),
        html.Div(
            [
                dbc.Label(df.columns[ignore_columns_index + 2]),
                dcc.Slider(
                    0,
                    400,
                    value=100,
                    id="myslider3",
                    updatemode="drag",
                    tooltip={"placement": "bottom", "always_visible": True},
                ),
            ]
        ),
        html.Div(
            [
                dbc.Label(df.columns[ignore_columns_index + 3]),
                dcc.Slider(
                    0,
                    400,
                    value=100,
                    id="myslider4",
                    updatemode="drag",
                    tooltip={"placement": "bottom", "always_visible": True},
                ),
            ]
        ),
        html.Div(
            [
                dbc.Label(df.columns[ignore_columns_index + 4]),
                dcc.Slider(
                    0,
                    400,
                    value=100,
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
                    value=100,
                    id="myslider6",
                    updatemode="drag",
                    tooltip={"placement": "bottom", "always_visible": True},
                ),
            ]
        ),
    ],
    body=True,
)


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
        dbc.Row([dbc.Col([bar], width=12)], justify="center"),
        dbc.Row([dbc.Col([table], width=12)], justify="center"),
    ],
    fluid=True,
)

# Callback allows components to interact
@app.callback(
    Output(choro, "figure"),
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
)
def update_graph(
    column_name, scaling1, scaling2, scaling3, scaling4, scaling5, scaling6
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
    ###  !!!! Why is there a +1 required here? I can't work it out.
    dff["Risk"] = dff[risk_cols].sum(axis="columns", skipna=False)

    ### Normalise for a percentage risk,

    dff["Risk"] = dff["Risk"] / max(dff["Risk"].dropna()) * 100

    nancount = dff["Risk"].isnull().sum()
    mysubtitle = f"Displaying {len(df.index)-nancount} countries from a total of {len(df.index)} based on data availability"

    # https://plotly.com/python/choropleth-maps/
    fig1 = px.choropleth(
        data_frame=dff,
        locations=dff.index,
        height=600,
        color=column_name,
        hover_name="Entity",
        hover_data=[
            "PBO",
            "Terrain Ruggedness",
            "Urban %",
            "Urban Agg %",
            "RoadQuality",
        ],
    )

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
        hover_name="Entity",
        log_y=True,
        size_max=60,
    )

    return (
        fig1,
        fig2,
        fig3,
        dff.round(decimals=1)
        .sort_values(by=column_name, ascending=False)
        .to_dict("records"),
        "# " + column_name,
        mysubtitle,
    )  # returned objects are assigned to the component property of the Output


# Run app
if __name__ == "__main__":
    app.run_server(debug=False, port=8054)
