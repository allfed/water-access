import pandas as pd
import mobility_module as mm
import numpy as np
import plotly.express as px
import plotting_tools_water_access




######################
#### Import Data #####
with open("../data/mobility-model-parameters.csv") as csv_file:
    # read the csv file
    allHPV_param_df = pd.read_csv("../data/mobility-model-parameters.csv")


col = "Name"
filter_value = "Bicycle"
param_df = allHPV_param_df.loc[(allHPV_param_df[col] == filter_value)]


# # initialise variables, options, MET, hpv and results classes, populated with data via 'init' functions
mo = mm.model_options()
mv = mm.model_variables()
met = mm.MET_values(mv)
hpv = mm.HPV_variables(param_df, mv)
mr = mm.model_results(hpv, mo)

# select model (bike...)
mo.model_selection = 2




#get linear space for variable range
minval = 0.001
maxval = 0.05
res = 30
var_string = "Crr"
var_units = ""



phase_space = np.linspace(
                start=minval,  # define linear space of weights
                stop=maxval,  # define maximum value for linear space
                num=res,  # how many data points?
                endpoint=True,
                retstep=False,
                dtype=None,
            )


result = []  # create empty list to place variables in to in loop

for var_test in phase_space:
    hpv.Crr = np.array([[[var_test]]])

    ######## Numerical MODEL ########
    mr.v_load_matrix3d, mr.load_matrix3d = mm.mobility_models.numerical_mobility_model(
        mr, mv, mo, met, hpv
    )
    ####### Organise Results #######
    mr.load_results(hpv, mv, mo)

    maximum_vel_kg_per_slope = np.amax(mr.velocitykgs, axis=2)
    mean_vel_kg_per_slope = np.mean(maximum_vel_kg_per_slope)

    result.append(mean_vel_kg_per_slope)


df_results = pd.DataFrame(result, columns=["Results"])
df_results["Variable"] = phase_space


graph_colours = ["#3D87CB", "#F0B323", "#DC582A", "#674230", "#3A913F", "#75787B"]
# create figures
fig1 = px.line(
    df_results,
    x="Variable",
    y="Results",
    color_discrete_sequence=graph_colours,
    title=f"Effect of {var_string} on model results",
).update_layout(yaxis_title=r'$\text{Velocity} \times \text{Payload  } [\frac{m}{s} kg]$',
xaxis_title = (var_string + var_units))


fig1 =  plotting_tools_water_access.format_plotly_graphs(fig1)

fig1.show()


