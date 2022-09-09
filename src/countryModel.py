from operator import concat
from functools import reduce
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import numpy as np
from dash import Dash, dcc, html, Output, Input  # pip install dash
import dash_bootstrap_components as dbc    # pip install dash-bootstrap-components


"""
Next steps:
Generate single csv for inpout?
Upload to heroku
"""

def risk_creator(input_df):
    """
    Risk creator takes all
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

    return df

def weight_parameter(series, slider_weight):
    if slider_weight == 0:
        series = 0
    else:
        series = series*slider_weight
    
    return series
    #### NEXT STPE IF SLIDER WEIGHT = 0, amke stuff = 0


"""
Start main function
"""
path = "../data/country_data_master.csv"
df = pd.read_csv(path, index_col = 'alpha3')
df = risk_creator(df)
ignore_columns_index = 3

# Build your components
app = Dash(__name__, external_stylesheets=[dbc.themes.LUX])



mytitle = dcc.Markdown(children='')
mysubtitle = dcc.Markdown(children='')
mygraph = dcc.Graph(figure={})
dropdown = dcc.Dropdown(options=df.columns.values[2:],
                        value='Risk',  # initial value displayed when page first loads
                        clearable=False)
### Create slider components on a card
controls = dbc.Card(
    [
        dbc.Label("Adjust the weighting of each parameter on the final risk score with the sliders"),

        html.Div(
            [
                dbc.Label("Bike Ownership %"),
                dcc.Slider(0, 400, value=100, id="myslider1" , updatemode='drag',
                tooltip={"placement": "bottom", "always_visible": True}
                ),
            ]
        ),
        html.Div(
            [
                dbc.Label(df.columns[ignore_columns_index+1]),
                dcc.Slider(0, 400, value=100, id="myslider2" , updatemode='drag',
                tooltip={"placement": "bottom", "always_visible": True}
                ),
            ]
        ),
        html.Div(
            [
                dbc.Label(df.columns[ignore_columns_index+2]),
                dcc.Slider(0, 400, value=100, id="myslider3" , updatemode='drag',
                tooltip={"placement": "bottom", "always_visible": True}
                ),
            ]
        ),
        html.Div(
            [
                dbc.Label(df.columns[ignore_columns_index+3]),
                dcc.Slider(0, 400, value=100, id="myslider4" , updatemode='drag',
                tooltip={"placement": "bottom", "always_visible": True}
                ),
            ]
        ),
        html.Div(
            [
                dbc.Label(df.columns[ignore_columns_index+4]),
                dcc.Slider(0, 400, value=100, id="myslider5" , updatemode='drag',
                tooltip={"placement": "bottom", "always_visible": True}
                ),
            ]
        ),
    ],
    body=True,
)


# Customize your own Layout
app.layout = dbc.Container([

    dbc.Row([
        dbc.Col([mytitle], width=6)
    ], justify='center'),
    
    
    dbc.Row([
        dbc.Col([mysubtitle], width=6)
    ], justify='center'),
    
    
    dbc.Row(
        [
            dbc.Col(controls, md=3),
            dbc.Col([mygraph], width=9),
        ],
        align="center",
    ),

    dbc.Row([
        dbc.Col([dropdown], width=6)
    ], justify='center'),
            html.Hr(),



], fluid=True)

# Callback allows components to interact
@app.callback(
    Output(mygraph, 'figure'),
    Output(mytitle, 'children'),
    Output(mysubtitle, 'children'),
    Input(dropdown, 'value'),
    Input("myslider1", 'value'),
    Input("myslider2", 'value'),
    Input("myslider3", 'value'),
    Input("myslider4", 'value'),
    Input("myslider5", 'value')
)
def update_graph(column_name,scaling1,scaling2,scaling3,scaling4,scaling5):  # function arguments come from the component property of the Input
    ignore_columns_num = 9
    dff = df.copy() # create copy of dataframe to apply weightings to before creating the risk matrix
    slider_weights = [scaling1,scaling2,scaling3,scaling4,scaling5]

    for param_index in range(0,5):
        series = weight_parameter(df.iloc[:,param_index+ignore_columns_num],slider_weights[param_index])
        dff.iloc[:,param_index+ignore_columns_num] = series


    ### Sum all of the weighted risk values in to the 'Risk' column of the dataframe
    dff['Risk'] = dff.iloc[:,ignore_columns_num:ignore_columns_num+param_index+1].sum(axis='columns',skipna=False)
    ### Normalise for a percentage risk,
    dff['Risk'] = dff['Risk'] / max(dff['Risk'].dropna()) *100

    nancount = dff["Risk"].isnull().sum() 
    mysubtitle = f"there are {len(df.index)-nancount} countries from a total of {len(df.index)}"

    # dff['RiskAbsolute'] = risk_series
    # dff['Risk'] = risk_series / max(risk_series.dropna()) *100

    # https://plotly.com/python/choropleth-maps/
    fig = px.choropleth(data_frame=dff,
                        locations=dff.index,
                        height=600,
                        color=column_name,
                        hover_name="Entity",
                        hover_data=[
                            "PBO",
                            "Terrain Ruggedness",
                            "Urban %",
                            "Urban Agg %",
                            "RoadQuality"
                        ],
                        )


    return fig, "# " +column_name, mysubtitle  # returned objects are assigned to the component property of the Output


# Run app
if __name__=='__main__':
    app.run_server(debug=True, port=8054)