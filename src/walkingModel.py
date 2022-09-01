import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
from matplotlib.widgets import Cursor
from scipy.optimize import fsolve
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import chart_studio
import chart_studio.plotly as py
from dotenv import load_dotenv, find_dotenv
import os


def linspace_creator(max_value_array, min_value, res):
    # creates a linsapce numpy array from the given inputs
    # max_value needs to be a numpy array (even if it is a 1x1)
    # min value is to be an int or float
    # resolution to be an int
    # returns
    # res = resoltuion, also used as a flag to set minimum/maxmum sinle load scearios

    if res == 1:  # if res =1 , calculate for max load of HPV
        load_matrix = np.zeros((len(max_value_array), res))  # initilaise numpy matrix
        load_matrix = max_value_array
    elif (
        res == 0
    ):  # if res =0 , calculate for min viable load of HPV (trick: use this to set custom load)
        load_matrix = (
            np.zeros((len(max_value_array), 1)) + min_value
        )  # initilaise numpy matrix
        load_matrix = load_matrix
    elif res > 1:
        load_matrix = np.zeros((len(max_value_array), res))  # initilaise numpy matrix

        #### Create linear space of weights
        # creates a vector for each of the HPVs, an equal number of elements spaced
        # evenly between the minimum viable load and the maximum load for that HPV
        i = 0  # initliase index
        for maxval in np.nditer(max_value_array):  # iterate through numpy array
            minval = min_value
            load_vector = np.linspace(
                start=minval,  # define linear space of weights
                stop=maxval,  # define maximum value for linear space
                num=res,  # how many data points?
                endpoint=True,
                retstep=False,
                dtype=None,
            )
            load_matrix[i:] = load_vector  # place the vector in to a matrix
            i += 1  # increment index
    else:
        print("Error: unexpected loading resolution, setting default")
        load_matrix = max_value_array

    return load_matrix


def max_safe_load(m_HPV_only, LoadCapacity, F_max, s, g):
    max_load_HPV = LoadCapacity

    #### Weight limits
    # Calculate weight limits for hills. There are some heavy loads which humans will not be able to push up certain hills
    # Note that this is for average slopes etc. This will be innacurate (i.e one VERY hilly section could render the whole thing impossible, this isn't accounted for here)
    if s != 0:  # requires check to avoid divide by zero
        max_pushable_weight = F_max / (np.sin(s) * g)
        i = 0
        if m_HPV_only.size > 1:  # can't iterate if there is only a float (not an array)
            for HPV_weight in m_HPV_only:
                if max_load_HPV[i] + HPV_weight > max_pushable_weight:
                    max_load_HPV[i] = max_pushable_weight - HPV_weight
                i += 1
        else:
            max_load_HPV = max_pushable_weight - m_HPV_only

    return max_load_HPV


def walkmodel(param_df, s, m1, P_t, F_max, L, minimumViableLoad, m_HPV_only, res):

    #### constants
    g = 9.81
    pi = 3.1416

    n_hpv = (
        param_df.Pilot.size
    )  # number of HPVs (by taking the pilot value as it is alwways a simple int [more complex data types get cnfusing wwhen the dataframe nly has 1 row])
    n = np.array(param_df.Efficiency).reshape((n_hpv, 1))
    Crr = np.array(param_df.Crr).reshape((n_hpv, 1))
    v_no_load = np.array(param_df.AverageSpeedWithoutLoad).reshape((n_hpv, 1))
    load_limit = np.array(param_df.LoadLimit).reshape((n_hpv, 1))
    load_capacity = load_limit - m1 * np.array(param_df.Pilot).reshape((n_hpv, 1))

    max_safe_load_HPV = max_safe_load(
        m_HPV_only, load_capacity, F_max, s, g
    )  # function to calaculate the max saffe loads due to hill
    max_load_HPV = np.minimum(load_capacity, max_safe_load_HPV.reshape((n_hpv, 1)))
    load_matrix = linspace_creator(max_load_HPV, minimumViableLoad, res)

    #### Derivations of further mass variables
    m_HPV_pilot = np.array(
        m1 * param_df.Pilot + m_HPV_only.reshape((param_df.Pilot.shape))
    ).reshape(
        (n_hpv, 1)
    )  # create vector with extra weights to add
    m_HPV_load_pilot = (
        load_matrix + m_HPV_pilot
    )  # weight of the HPV plus the rider (if any) plus the load
    m_walk_carry = (
        m1
        + m_HPV_load_pilot
        * (np.array(param_df.GroundContact).reshape((n_hpv, 1)) - 1)
        * -1
    )  # negative 1 is to make the wheeled = 0 and the walking = 1
    m_HPV_load = load_matrix + np.array(m_HPV_only).reshape((n_hpv, 1))
    # weight of the mass being 'walked', i.e the wieght of the human plus anything they are carrying (not pushing or riding)

    #### Constants from polynomial equation analysis
    C = ((m_walk_carry) * g / pi) * (3 * g * L / 2) ** (1 / 2)  # component of walking
    D = pi**2 / (6 * g * L)  # leg component?
    B1 = (
        m_HPV_load_pilot * g * np.cos(np.arctan(s)) * Crr
    )  # rolling resistance component
    B2 = m_HPV_load_pilot * np.sin(np.arctan(s))  # slope component
    B = B1 + B2

    ##### velocities
    v_load = (-B + np.sqrt(B**2 + (2 * C * D * P_t) / n)) / (C * D / n)
    # loaded velocity

    # if loaded speed is greater than unloaded avg speed, make equal to avg unloaded speed
    i = 0
    for maxval in v_no_load:
        indeces = v_load[i] > maxval
        v_load[i, indeces] = maxval
        i += 1

    return v_load, load_matrix


def bike_power_solution(p, *data):
    ro, C_d, A, m_t, Crr, eta, P_t, g, s = data
    v_solve = p[0]
    return (
        1 / 2 * ro * v_solve**3 * C_d * A
        + v_solve * m_t * g * Crr
        + v_solve * m_t * g * s
    ) / eta - P_t


def LCDA_solution(p, *data):
    m_load, metabolic_budget_watts, s = data
    v_solve = p[0]
    G = (s * 360 / (2 * np.pi)) / 45 * 100
    return (
        1.44
        + 1.94 * v_solve**0.43
        + 0.24 * v_solve**4
        + 0.34 * (1 - 1.05 ** (1 - 1.1 ** (G + 32)))
    ) * m_load - metabolic_budget_watts


def Lankford_solution(p, *data):
    metabolic_budget, s = data
    v_solve = p[0]
    G = (s * 360 / (2 * np.pi)) / 45
    print(G)
    print(s)
    return (
        5.43483
        + 6.47383 * v_solve
        + (-0.05372 * G)
        + 0.652298 * v_solve * G
        + 0.023761 * v_solve * G**2
        + 0.00320 * v_solve * G**3
        - metabolic_budget
    )


## Plotly creds
load_dotenv(find_dotenv())
chart_studio.tools.set_credentials_file(
    username=os.environ.get("USERNAME"), api_key=os.environ.get("API_KEY")
)

#### Start Script
with open("../data/mobility-model-parameters.csv") as csv_file:
    # read the csv file
    allHPV_param_df = pd.read_csv("../data/mobility-model-parameters.csv")

# selectHPVs you're interested in
cols = ["GroundContact"]
cols = ["Drive"]
param_df = allHPV_param_df.loc[
    (allHPV_param_df[cols] == 2).all(1)
]  # comment out if you want all
# param_df = allHPV_param_df.loc[(allHPV_param_df['Drive']==0).all(1)] # comment out if you want all

# param_df = allHPV_param_df.iloc[:]
surf_plot_index = 1

#### variables (changeable)
s_deg = 0  # slope in degrees (only used for loading scenario, is overriden in cariable slope scenario)
m1 = 83  # mass of rider/person
P_t = 75  # power output of person (steady state average)
F_max = 300  # maximum force exertion for pushing up a hill for a short amount of time
L = 1  # leg length
minimumViableLoad = 15  # in kg, the minimum useful load for such a trip
t_hours = 8  # number of hours to gather water
L = 1  # leg length
A = 1  # cross sectional area
C_d = 1  # constant for wind
ro = 1
eta = 0.8

# def run_model(s_deg=0, m1=83):

# model options
model = 3  # 1 is walking, 2 is cycling

## plot options
load_plot = 0
slope_plot = 0
surf_plot = 0
surf_plotly = 0
surf_plotly_multi = 0
load_plot_plotly = 1
slope_plot_plotly = 0
time_sensitivity = 0
time_sensitivity_plotly_grouped = 0
bar_plot_loading = 0
bar_plot_loading_distance = 0

## Options
load_res = 30  #  0 = min load, 1 = max load, >1 = linear space between min and max
slope_res = (
    30  #  0 = min slope only, 1 = max slope only, >1 = linear space between min and max
)
slope_start = 0  # slope min
slope_end = 10  # slope max

#### constants
g = 9.81
pi = 3.1416
n_hpv = param_df.Pilot.size

# data accounting

# create load vector
load_limit = np.array(param_df.LoadLimit).reshape((n_hpv, 1))
m_HPV_only = np.array(param_df.Weight).reshape(
    (n_hpv, 1)
)  # assume 5% of load is the wight of HPV

# name list
HPV_names = param_df["Name"].tolist()

# create slope vector
slope_vector_deg = linspace_creator(np.array([slope_end]), slope_start, slope_res)
slope_vector_deg = slope_vector_deg.reshape(1, slope_vector_deg.size)
# initilise and createoutput 3d matrices (dims: 0 = HPV, 1 = Slope, 2 = load)
if load_res <= 1:
    n_load_scenes = 1
else:
    n_load_scenes = load_res
v_load_matrix3d = np.zeros((n_hpv, slope_vector_deg.size, n_load_scenes))
load_matrix3d = np.zeros((n_hpv, slope_vector_deg.size, n_load_scenes))
slope_matrix3d_deg = np.repeat(slope_vector_deg, n_hpv, axis=0)
slope_matrix3d_deg = np.repeat(
    slope_matrix3d_deg[:, :, np.newaxis], n_load_scenes, axis=2
)
slope_matrix3drads = (slope_matrix3d_deg / 360) * (2 * pi)
####### MAIN LOOP SIMPLE MODEL ########
if model == 1:
    i = 0
    for slope in slope_vector_deg.reshape(slope_vector_deg.size, 1):
        s = (slope / 360) * (2 * pi)
        v_load, load_matrix = walkmodel(
            param_df, s, m1, P_t, F_max, L, minimumViableLoad, m_HPV_only, load_res
        )
        v_load_matrix3d[:, i, :] = v_load.reshape(n_hpv, load_res)
        load_matrix3d[:, i, :] = load_matrix.reshape(n_hpv, load_res)
        i += 1
####### END MAIN LOOP SIMPLE MODEL  ########

####### MAIN LOOP CYCLING MODEL ########

elif model == 2:
    v_load = np.zeros((1, load_res))
    i = 0
    for hpv_name in HPV_names:
        j = 0
        Crr = np.array(param_df.Crr).reshape((n_hpv, 1))[i]
        # print(hpv_name)
        # print(i)
        for slope in slope_vector_deg.reshape(slope_vector_deg.size, 1):
            s = (slope / 360) * (2 * pi)  # determine slope in radians
            max_load_HPV = max_safe_load(
                m_HPV_only[i], load_limit[i], F_max, s, g
            )  # find maximum pushing load
            if max_load_HPV > load_limit[i]:
                max_load_HPV = load_limit[
                    i
                ]  # see if load of HPV or load of pushing is the limitng factor.
            load_vector = linspace_creator(
                max_load_HPV, minimumViableLoad, load_res
            ).reshape((load_res, 1))
            m_t = np.array(load_vector + m1 + m_HPV_only[i])
            k = 0
            for total_load in m_t:
                data = (ro, C_d, A, m_t[k], Crr, eta, P_t, g, s)
                V_guess = 1
                V_r = fsolve(bike_power_solution, V_guess, args=data)
                v_load_matrix3d[i, j, k] = V_r[0]
                load_matrix3d[i, j, k] = load_vector[k]
                k += 1
            j += 1
        i += 1

####### END MAIN LOOP CYLCLING MODEL  ########

####### MAIN LOOP LANKFORD MODEL ########

elif model == 3:

    MET_of_sustainable_excercise = (
        6  # # https://en.wikipedia.org/wiki/Metabolic_equivalent_of_task
    )
    MET_VO2_conversion = 3.5  # milliliters per minute per kilogram body mass
    MET_watt_conversion = 1.162  # watts per kg body mass
    # so 75 Watts output (from Marks textbook) is probably equivalent to about 6 METs
    MET_budget_VO2 = (
        MET_VO2_conversion * MET_of_sustainable_excercise * m1
    )  # vo2 budget for a person
    MET_budget_watts = (
        MET_watt_conversion * MET_of_sustainable_excercise * m1
    )  # vo2 budget for a person

    v_load = np.zeros((1, load_res))
    i = 0
    for hpv_name in HPV_names:
        j = 0
        Crr = np.array(param_df.Crr).reshape((n_hpv, 1))[i]
        # print(hpv_name)
        # print(i)
        for slope in slope_vector_deg.reshape(slope_vector_deg.size, 1):
            s = (slope / 360) * (2 * pi)  # determine slope in radians
            max_load_HPV = max_safe_load(
                m_HPV_only[i], load_limit[i], F_max, s, g
            )  # find maximum pushing load
            if max_load_HPV > load_limit[i]:
                max_load_HPV = load_limit[
                    i
                ]  # see if load of HPV or load of pushing is the limitng factor.
            load_vector = linspace_creator(
                max_load_HPV, minimumViableLoad, load_res
            ).reshape((load_res, 1))
            m_t = np.array(load_vector + m1 + m_HPV_only[i])
            k = 0
            for total_load in m_t:
                data = (m_t[k], MET_budget_watts, s)
                V_guess = 1
                V_r = fsolve(LCDA_solution, V_guess, args=data)
                v_load_matrix3d[i, j, k] = V_r[0]
                load_matrix3d[i, j, k] = load_vector[k]
                k += 1
            j += 1
        i += 1

####### END MAIN LOOP LANKFORD MODEL  ########

#### Velocities
v_no_load = np.array(param_df.AverageSpeedWithoutLoad).reshape((n_hpv, 1))[
    :, np.newaxis, :
]  # unloaded speed from csv file
# calcualte average speed, half of the trip is loaded, and half is unloaded
v_avg_matrix3d = (v_load_matrix3d + v_no_load) / 2
# average velocity for a round trip

# calculation of velocity kgs
velocitykgs = v_avg_matrix3d * load_matrix3d

#### Distance
t_secs = t_hours * 60 * 60
distance_achievable = (v_avg_matrix3d * t_secs) / 1000  # kms
distance_achievable_one_hr = (v_avg_matrix3d * 60 * 60) / 1000  # kms

## Power
total_load = load_matrix3d + m1  # estimate for now withot HV weight
Ps = v_load_matrix3d * total_load * g * np.sin(np.arctan(slope_matrix3drads))
Ps = Ps / np.array(param_df.Efficiency).reshape((n_hpv, 1))[:, np.newaxis, np.newaxis]
### NEED to FIX POWER below.
# Pr = v_avg_matrix3d*total_load*g*cos(atan(slope_matrix3d)).*np.array(param_df.Crr)[:,np.newaxis,np.newaxis]/np.array(param_df.Efficiency)[:,np.newaxis,np.newaxis]
# Pw = 1/2.*C*D.*v_load.^2./n


if slope_plot == 1:
    i = 0

    #   # Slope Graph Sensitivity
    fig, ax = plt.subplots(figsize=(20, 10))
    for HPVname in param_df.Name:
        y = (
            v_avg_matrix3d[i, :, 0] * load_matrix3d[i, :, 0]
        )  # SEE ZEROS <-- this is for the minimum weight
        x = slope_vector_deg.reshape((y.shape))  # reshape to same size
        (line,) = ax.plot(x, y, label=HPVname)  # Plot some data on the axes.
        i += 1
    plt.xlabel("Slope [deg ˚]")
    plt.ylabel("Velocity Kgs [m/s]")
    plt.title(
        "Velocity Kgs as a function of Slope with Load = {}kg".format(
            load_matrix3d[0, 0, 0]
        )
    )
    plt.legend()

    plt.plot()
    plt.show()

elif load_plot == 1:
    i = 0

    #   # Slope Graph Sensitivity
    fig, ax = plt.subplots(figsize=(20, 10))
    for HPVname in HPV_names:
        y = (
            v_avg_matrix3d[i, 0, :] * load_matrix3d[i, 0, :]
        )  # SEE ZEROS <-- this is for the minimum weight
        x = load_matrix3d[i, 0, :]
        (line,) = ax.plot(x, y, label=HPVname)  # Plot some data on the axes.
        i += 1
    plt.xlabel("Load [kg]")
    plt.ylabel("Velocity Kilograms [kg][m/s]")
    plt.title(
        "Velocity Kgs as a function of Load with {}˚ Slope".format(
            slope_vector_deg[0, 0]
        )
    )
    plt.legend()
    plt.plot()
    plt.show()

elif surf_plot == 1:
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    # Make data.
    Z = distance_achievable[surf_plot_index, :, :]
    X = load_matrix3d[surf_plot_index, :, :]
    Y = slope_matrix3d_deg[surf_plot_index, :, :]

    # Plot the surface.
    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False)

    # Customize the z axis.
    ax.zaxis.set_major_locator(LinearLocator(10))
    # A StrMethodFormatter is used automatically
    ax.zaxis.set_major_formatter("{x:.02f}")

    ax.set_xlabel("Load [kg]")
    ax.set_ylabel("Slope [deg ˚]")
    ax.set_zlabel("Velocity [m/s]")
    plt.title("Slope, Load & Distance for: " + HPV_names[surf_plot_index])

    plt.show()

elif surf_plotly == 1:

    # # Make data.
    Z = distance_achievable[surf_plot_index, :, :]
    X = load_matrix3d[surf_plot_index, :, :]
    Y = slope_matrix3d_deg[surf_plot_index, :, :]

    # using the graph options from plotly, create 3d plot.
    fig = go.Figure(data=[go.Surface(z=Z, x=X, y=Y)])
    fig.update_layout(
        title="Slope, Load & Distance for: " + HPV_names[surf_plot_index],
        autosize=True,
        width=500,
        height=500,
        margin=dict(l=65, r=50, b=65, t=90),
    )
    fig.show()

elif surf_plotly_multi == 1:

    plot_height = 700
    plot_width = 900
    xaxis_title = "Load [kg]"
    yaxis_title = "Slope [˚]"
    zaxis_title = "Distannce [km]"
    # create list for plot specs based on number of hpvs
    specs_list = []
    spec_single = [{"type": "surface"}]
    for HPVname in HPV_names:
        specs_list.append(spec_single)

    # create figure and subplots
    fig = make_subplots(
        rows=n_hpv, cols=1, specs=specs_list, subplot_titles=HPV_names
    )  # name the subplots the HPV names (create a new list of strings if you'd like to rename)

    # in a for loop, create the subplots
    surf_plot_index = 0
    for HPVname in HPV_names:
        Z = distance_achievable[surf_plot_index, :, :]
        X = load_matrix3d[surf_plot_index, :, :]
        Y = slope_matrix3d_deg[surf_plot_index, :, :]

        fig.add_trace(
            go.Surface(z=Z, x=X, y=Y, colorscale="Viridis", showscale=False),
            row=1 + surf_plot_index,
            col=1,
        )

        fig.update_scenes(
            xaxis_title_text=xaxis_title,
            yaxis_title_text=yaxis_title,
            zaxis_title_text=zaxis_title,
        )

        surf_plot_index += 1

    # update the layout and create a title for whole thing
    fig.update_layout(
        title_text="3D subplots of HPVs", height=plot_height * n_hpv, width=plot_width
    )

    fig.show()

    # show the figure
    py.iplot(fig, filename="3D subplots of HPVs New")

elif load_plot_plotly == 1:

    xaxis_title = "Load [kg]"
    yaxis_title = "Kms"
    slope_scene = 0

    slope_name = slope_vector_deg.flat[slope_scene]
    chart_title = "Km/hour with different loads, Constant %0.1f slope, model %0.1f" % (
        slope_name,
        model,
    )

    i = 0

    fig = go.Figure()
    for HPVname in HPV_names:
        y = distance_achievable_one_hr[
            i, slope_scene, :
        ]  # SEE ZEROS <-- this is for the minimum weight
        x = load_matrix3d[i, slope_scene, :]
        i += 1
        fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name=HPVname))

    # Update the title
    fig.update_layout(title=dict(text=chart_title))
    # Update te axis label (valid for 2d graphs using graph object)
    fig.update_xaxes(title_text=xaxis_title)
    fig.update_yaxes(title_text=yaxis_title)
    # fig.update_yaxes(range = [2,5.5])

    fig.show()
    # py.iplot(fig, filename=chart_title)


elif slope_plot_plotly == 1:

    load_scene = -1

    xaxis_title = "Slope [˚]"
    yaxis_title = "Kms"
    load_name = load_matrix3d.flat[load_scene]
    chart_title = (
        "Km/hour with different slope, Constant %0.1f kg load, model %0.1f"
        % (load_name, model)
    )
    i = 0

    fig = go.Figure()
    for HPVname in HPV_names:
        y = distance_achievable_one_hr[
            i, :, load_scene
        ]  # SEE ZEROS <-- this is for the minimum weight
        x = slope_matrix3d_deg[i, :, load_scene]
        i += 1
        fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name=HPVname))

    # Update the title
    fig.update_layout(title=dict(text=chart_title))
    # Update te axis label (valid for 2d graphs using graph object)
    fig.update_xaxes(title_text=xaxis_title)
    fig.update_yaxes(title_text=yaxis_title)
    fig.update_yaxes(range=[0, 15])

    fig.show()
    py.iplot(fig, filename=chart_title)


elif time_sensitivity == 1:
    t_hours_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    HPV_selector = 1  # 1 = bike
    #   # Distance Sensitivity
    X = []
    Y = []
    # fig = go.Figure()
    for t_hours in t_hours_list:
        t_secs = t_hours * 60 * 60
        distance_achievable = (v_avg_matrix3d * t_secs) / 1000  # kms
        y = distance_achievable[
            HPV_selector, 0, 0
        ]  # SEE ZEROS <-- this is for the minimum weight
        x = t_hours
        print(x)
        print(X)
        i += 1
        X.append(x)
        Y.append(y)
    # fig.add_trace(go.Bar(x = X , y = Y, mode ='lines'))
    fig = px.bar(x=X, y=Y, title="Wide-Form Input")
    # fig.update_layout(title="Distance achievable, given time spent")
    fig.show()

elif time_sensitivity_plotly_grouped == 1:
    t_hours_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    load_scene = -1

    # plotly setup
    fig = go.Figure()
    i = 0
    # add trace for eat
    for HPVname in HPV_names:
        # print(col)
        X = t_hours_list
        Y = load_matrix3d[i, 0, load_scene] * (
            distance_achievable_one_hr[i, 0, load_scene] * np.array(t_hours_list)
        )
        fig.add_trace(go.Bar(x=X, y=Y, name=HPVname))
        i += 1

    fig.update_layout(
        title=dict(
            text="Distance Achievable Per HPV given different time to collect water"
        )
    )
    fig.update_layout(barmode="group")
    fig.show()

elif bar_plot_loading == 1:
    t_hours_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    load_scene = -1
    slope_scene = -1
    slope_name = slope_vector_deg.flat[slope_scene]
    chart_title = "Efficiency at %0.2f degrees" % slope_name

    df = pd.DataFrame(
        {
            "Name": HPV_names,
            "Load": load_matrix3d[:, slope_scene, load_scene],
            "Average Trip Velocity": v_avg_matrix3d[:, slope_scene, load_scene],
            "Load Velocity [kg * m/s]": v_avg_matrix3d[:, slope_scene, load_scene]
            * load_matrix3d[:, slope_scene, load_scene],
            "Loaded Velocity": v_load_matrix3d[:, slope_scene, load_scene],
            "Unloaded Velocity": param_df["AverageSpeedWithoutLoad"],
        }
    )

    fig = px.bar(
        df,
        x="Load",
        y="Load Velocity [kg * m/s]",
        hover_data=[
            "Name",
            "Average Trip Velocity",
            "Loaded Velocity",
            "Unloaded Velocity",
            "Load",
        ],
        color="Name",
        labels={"Name": "Name"},
        title=chart_title,
    )
    fig.show()
    py.iplot(fig, filename=chart_title)


elif bar_plot_loading_distance == 1:
    load_scene = -1
    slope_scene = 0
    slope_name = slope_vector_deg.flat[slope_scene]
    chart_title = "Efficiency at %0.2f degrees, with model %d" % (slope_name, model)

    df = pd.DataFrame(
        {
            "Name": HPV_names,
            "Load": load_matrix3d[:, slope_scene, load_scene],
            "Average Trip Velocity": v_avg_matrix3d[:, slope_scene, load_scene],
            "Litres * Km": distance_achievable_one_hr[:, slope_scene, load_scene]
            * load_matrix3d[:, slope_scene, load_scene],
            "Water ration * Km": distance_achievable_one_hr[:, slope_scene, load_scene]
            * load_matrix3d[:, slope_scene, load_scene]
            / minimumViableLoad,
            "Distance to Water Achievable": distance_achievable_one_hr[
                :, slope_scene, load_scene
            ]
            / 2,
            "Total Round trip Distance Achievable": distance_achievable_one_hr[
                :, slope_scene, load_scene
            ],
            "Load Velocity [kg * m/s]": v_avg_matrix3d[:, slope_scene, load_scene]
            * load_matrix3d[:, slope_scene, load_scene],
            "Loaded Velocity": v_load_matrix3d[:, slope_scene, load_scene],
            "Unloaded Velocity": param_df["AverageSpeedWithoutLoad"],
        }
    )

    fig = px.bar(
        df,
        x="Load",
        y="Litres * Km",
        hover_data=[
            "Name",
            "Average Trip Velocity",
            "Loaded Velocity",
            "Unloaded Velocity",
            "Distance to Water Achievable",
            "Total Round trip Distance Achievable",
            "Load",
        ],
        color="Name",
        labels={"Name": "Name"},
        title=chart_title,
    )
    fig.show()
    py.iplot(fig, filename=chart_title)
