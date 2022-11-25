import pandas as pd
import mobility_module as mm
import numpy as np
import plotly
import plotly.express as px
import plotting_tools_water_access


######################
#### Import Data #####
with open("../data/mobility-model-parameters.csv") as csv_file:
    # read the csv file
    allHPV_param_df = pd.read_csv("../data/mobility-model-parameters.csv")


# PULL FROM CSV
# get linear space for variable range
with open("../data/mobility-model-parameters.csv") as csv_file:
    # read the csv file
    sens_df = pd.read_csv("../data/Sensitivity Analysis Variables.csv")

graph_colours = ["#3D87CB", "#F0B323", "#DC582A", "#674230", "#3A913F", "#75787B"]


full_result_dict = {}
plot_dict = {}
df_large = pd.DataFrame()


### FIlter Data
col = "Name"
filter_value = "Bicycle"
param_df = allHPV_param_df.loc[(allHPV_param_df[col] == filter_value)]

for i in range(0, len(sens_df)):

    # # initialise variables, options, MET, hpv and results classes, populated with data via 'init' functions
    mo = mm.model_options()
    mv = mm.model_variables()
    met = mm.MET_values(mv)
    hpv = mm.HPV_variables(param_df, mv)
    mr = mm.model_results(hpv, mo)

    # select model (bike...)
    mo.model_selection = 2

    ###### Start big if statement area
    ##### SENSITIVITY

    plot_min = sens_df.iloc[i]["Plotting Min"]
    plot_max = sens_df.iloc[i]["Plotting Max"]
    minval = sens_df.iloc[i]["Expected Min"]
    maxval = sens_df.iloc[i]["Expected Max"]
    def_val = sens_df.iloc[i]["Default Value"]
    res = 30
    var_string = sens_df.iloc[i]["Short Name"]  # change
    var_units = sens_df.iloc[i]["Units"]  # change

    phase_space = np.linspace(
        start=plot_min,  # define linear space of weights
        stop=plot_max,  # define maximum value for linear space
        num=res,  # how many data points?
        endpoint=True,
        retstep=False,
        dtype=None,
    )
    # include the default value
    # phase_space = np.append(phase_space,hpv.load_limit.flatten()[0])
    phase_space = np.append(phase_space, def_val)
    phase_space = np.append(phase_space, minval)
    phase_space = np.append(phase_space, maxval)

    velocitykgs = []  # create empty list to place variables in to in loop
    water_ration_metres = []  # create empty list to place variables in to in loop

    for var_test in phase_space:
        ## THIS WILL NEED A FAT IF STATEMENT AS WE NEED TO CHANGE TARGET of the assignment AND the value AND there are differening data storgae types (i.e numpy 3 dims, scalar etc)
        if var_string == "Coefficient of Rolling Resistance":
            hpv.Crr = np.array([[[var_test]]])  # change
        elif var_string == "Load Limit":
            hpv.load_limit = np.array([[[var_test]]])  # change
        elif var_string == "Reference Area":
            mv.A = var_test
        elif var_string == "Drag Coefficient":
            mv.C_d = var_test
        elif var_string == "Efficiency":
            mv.eta = var_test
        elif var_string == "Air Density":
            mv.ro = var_test
        elif var_string == "MET budget":
            met.MET_of_sustainable_excercise = var_test
        elif var_string == "T_hours":
            mv.t_hours = var_test
        elif var_string == "HPV Weight":
            hpv.m_HPV_only = np.array([[[var_test]]])  # change
        elif var_string == "Water Ration":
            mv.waterration = var_test
        elif var_string == "Human Weight":
            mv.m1 = var_test
        elif var_string == "Human Power Output":
            mv.P_t = var_test

        # # hpv.load_limit = np.array([[[var_test]]]) # change
        # mv.C_d = var_test # change

        ######## Numerical MODEL ########
        (
            mr.v_load_matrix3d,
            mr.load_matrix3d,
        ) = mm.mobility_models.numerical_mobility_model(mr, mv, mo, met, hpv)
        ####### Organise Results #######
        mr.load_results(hpv, mv, mo)

        maximum_vel_kg_per_slope = np.amax(mr.velocitykgs, axis=2)
        mean_vel_kg_per_slope = np.mean(maximum_vel_kg_per_slope)

        velocitykgs.append(mean_vel_kg_per_slope)
        water_ration_metres.append(mean_vel_kg_per_slope / mv.waterration * mr.t_secs)

    small_marker_size = 10
    med_marker_size = 20
    large_marker_size = 50
    data = {"Results": velocitykgs, "Water Ration Metres": water_ration_metres}
    df_results = pd.DataFrame(data)
    # df_results = pd.DataFrame((velocitykgs,water_ration_metres), columns=["Results", "Water Ration Metres"])
    # df_results = pd.DataFrame(water_ration_metres, columns=[)
    df_results["Adjusted Result"] = (
        df_results["Results"] - df_results.at[res, "Results"]
    )
    df_results["Adjusted Water Ration Metres"] = (
        df_results["Water Ration Metres"] - df_results.at[res, "Water Ration Metres"]
    )
    df_results["Variable"] = phase_space
    df_results["DataType"] = "Sensitivity"
    df_results["MarkerSize"] = small_marker_size
    df_results["Name"] = var_string
    df_results.at[res, "DataType"] = "Default"
    df_results.at[res + 1, "DataType"] = "Minimum Expected"
    df_results.at[res + 2, "DataType"] = "Maximum Expected"
    df_results.at[res + 1, "MarkerSize"] = med_marker_size
    df_results.at[res + 2, "MarkerSize"] = med_marker_size
    df_results.at[res, "MarkerSize"] = large_marker_size

    full_result_dict[var_string] = df_results
    df_large = pd.concat([df_large, df_results], sort=False)

    # create figures

    # for i in range(0, len(sens_df)):

    fig1 = px.scatter(
        full_result_dict[sens_df.iloc[i]["Short Name"]],
        x="Variable",
        y="Results",
        color="DataType",
        color_discrete_sequence=graph_colours,
        # markers=True,
        size="MarkerSize",
        title=f'Effect of {sens_df.iloc[i]["Short Name"]} on model results',
    ).update_layout(
        yaxis_title=r"$\text{Velocity} \times \text{Payload  } [\frac{m}{s} Kg]$",
        xaxis_title=(
            f'{sens_df.iloc[i]["Short Name"]}  <i>{sens_df.iloc[i]["Units"]}</i>'
        ),
    )
    fig1.update_yaxes(range=[80, 400])

    fig1 = plotting_tools_water_access.format_plotly_graphs(fig1)

    fig1.show()
    plot_dict[var_string] = fig1
    # filename_string = var_string + ".html"
    # plotly.offline.plot(fig1, filename=filename_string)

summary_df = df_large[
    (df_large.DataType == "Minimum Expected")
    | (df_large.DataType == "Maximum Expected")
    | (df_large.DataType == "Default")
]


fig = px.bar(
    summary_df,
    y="Name",
    x="Adjusted Water Ration Metres",
    color="DataType",
    title="Sensitivty Of Variables",  # size='MarkerSize',
)
fig.show()
