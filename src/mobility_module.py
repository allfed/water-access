import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
from matplotlib.widgets import Cursor
from scipy.optimize import fsolve
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# import chart_studio
import chart_studio.plotly as py

# from dotenv import load_dotenv, find_dotenv
# import os


def linspace_creator(max_value_array, min_value, res):

    """
    creates a linsapce numpy array from the given inputs
    max_value needs to be a numpy array (even if it is a 1x1)
    min value is to be an int or float
    resolution to be an int
    returns
    res = resoltuion, also used as a flag to set minimum/maxmum single load scearios
    Output is an N1 * N2 matrix where N1 = the number of hpvs and N2 = the resolution.
    """
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
        load_matrix = np.zeros(
            (len(max_value_array), res)
        )  # initilaise numpy matrix, numbers of rows = hpvs, cols = resolution

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
    """
    #### Weight limits
    Takes in the mass of the HPV (array), the load capacity of the HPV (array), max force that a person can push up a hill, the slope, and gravity
    # Calculate weight limits for hills. There are some heavy loads which humans will not be able to push up certain hills
    # Note that this is for average slopes etc. This will be innacurate (i.e one VERY hilly section could render the whole thing impossible, this isn't accounted for here)
    """
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


class mobility_models:
    def sprott_model(hpv, mv, mo, mr):
        """
        takes the inputs from the hpv data, model variables, and model options, and returns the results in the form of a matrix :[HPV:Slope:Load] which gives the velocity
        """

        # define extra vars
        pi = np.pi

        ## loop over all of the different slopes. Dimensions for results: 0 = HPV, 1 = slope, 2 = load
        i = 0
        for slope in mr.slope_vector_deg.reshape(mr.slope_vector_deg.size, 1):
            s = (slope / 360) * (2 * pi)
            v_load, load_matrix = mobility_models.sprott_solution(hpv, s, mv, mo)
            mr.v_load_matrix3d[:, i, :] = v_load.reshape(hpv.n_hpv, mo.load_res)
            mr.load_matrix3d[:, i, :] = load_matrix.reshape(hpv.n_hpv, mo.load_res)
            i += 1
        return mr.v_load_matrix3d, mr.load_matrix3d

    def bike_power_solution(p, *data):
        ro, C_d, A, m_t, Crr, eta, P_t, g, s = data
        v_solve = p[0]
        return (
            1 / 2 * ro * v_solve**3 * C_d * A
            + v_solve * m_t * g * Crr
            + v_solve * m_t * g * s
        ) / eta - P_t

    def sprott_solution(hpv, s, mv, mo):
        """
        takes in the HPV dataframe, the slope, the model variables, and model options
        returns the velocity of walking based on the energetics of walking model outlined by Sprott
        https://sprott.physics.wisc.edu/technote/Walkrun.htm

        """

        max_safe_load_HPV = max_safe_load(
            hpv.m_HPV_only, hpv.load_capacity, mv.F_max, s, mv.g
        )
        # function to calaculate the max saffe loads due to hill
        max_load_HPV = np.minimum(
            hpv.load_capacity, max_safe_load_HPV.reshape(hpv.load_capacity.shape)
        )

        load_matrix = linspace_creator(max_load_HPV, mv.minimumViableLoad, mo.load_res)

        #### Derivations of further mass variables
        m_HPV_pilot = np.array(
            mv.m1 * hpv.Pilot + hpv.m_HPV_only.reshape((hpv.Pilot.shape))
        ).reshape(
            (hpv.n_hpv, 1)
        )  # create vector with extra weights to add
        m_HPV_load_pilot = (
            load_matrix + m_HPV_pilot
        )  # weight of the HPV plus the rider (if any) plus the load
        m_walk_carry = (
            mv.m1
            + m_HPV_load_pilot
            * (np.array(hpv.GroundContact).reshape((hpv.n_hpv, 1)) - 1)
            * -1
        )  # negative 1 is to make the wheeled = 0 and the walking = 1
        # m_HPV_load = load_matrix + np.array(m_HPV_only).reshape((n_hpv, 1))
        # weight of the mass being 'walked', i.e the wieght of the human plus anything they are carrying (not pushing or riding)

        #### Constants from polynomial equation analysis
        C = ((m_walk_carry) * mv.g / np.pi) * (3 * mv.g * mv.L / 2) ** (
            1 / 2
        )  # component of walking
        D = np.pi**2 / (6 * mv.g * mv.L)  # leg component?

        B1 = (
            m_HPV_load_pilot * mv.g * np.cos(np.arctan(s)) * hpv.Crr[:, 0, :]
        )  # rolling resistance component
        B2 = m_HPV_load_pilot * np.sin(np.arctan(s))  # slope component
        B = B1 + B2

        ##### velocities
        v_load = (-B + np.sqrt(B**2 + (2 * C * D * mv.P_t) / hpv.n[:, 0, :])) / (
            C * D / hpv.n[:, 0, :]
        )
        # loaded velocity

        # if loaded speed is greater than unloaded avg speed, make equal to avg unloaded speed
        i = 0
        for maxval in hpv.v_no_load[:, 0, :]:
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

    def numerical_mobility_model(mr, mv, mo, met, hpv):
        """
        1. Takes input of all model variables, mr, mv, mo, met, and hpv
        2. Selects the model to use based on the model options
        3. Creates 3 for loops, looping over 1) HPVs, 2) Slope, 3) load
        4. In the slope loop, determines the slope in radians, then determines the maximum safe load for that slope
        Then, creates a load vector for that partular slope (and that particular HPV), from minimum load to the max safe load
        After that calculates the total load in kg (HPV + pilot + load)
        5. Next, the function calculates the unloaded velocity for the given slope. Note the model options determine the polarity of the slope
        6. Then enters the load loop, which cycles through the recently created load vector
        Now for the ith HPV, the jth slope, and the kth load it will determine the velocity of the HPV given the metabolic/energy input
        Uses scipy.optimize's fsolve to solve the systems of equations int he walking models
        full_output=True means that a flag is returned to see if the solve was succesful.
        Many solves are not successful if the agent 'rus out of energy' and physicaly can't carry the load up the incline with the energy budget provided, these are handled in the if statement using
        """

        if mo.model_selection == 2:
            model = mobility_models.bike_power_solution
        elif mo.model_selection == 3:
            model = mobility_models.Lankford_solution
        elif mo.model_selection == 4:
            model = mobility_models.LCDA_solution
        else:
            print("Unrecognised Model Selection Number")
            exit()

        # start loop iterating over HPVs
        for i, name in enumerate(hpv.name):
            # start loop iterating over slopes
            for j, slope in enumerate(
                mr.slope_vector_deg.reshape(mr.slope_vector_deg.size, 1)
            ):
                s = (slope / 360) * (2 * np.pi)  # determine slope in radians

                # determine safe loading for hilly scenarios
                max_load_HPV = max_safe_load(
                    hpv.m_HPV_only[i], hpv.load_capacity[i], mv.F_max, s, mv.g
                )  # find maximum pushing load
                if max_load_HPV > hpv.load_capacity[i]:
                    max_load_HPV = hpv.load_capacity[
                        i
                    ]  # see if load of HPV or load of pushing is the limitng factor.

                load_vector = linspace_creator(
                    max_load_HPV, mv.minimumViableLoad, mo.load_res
                ).reshape((mo.load_res, 1))
                m_t = np.array(load_vector + mv.m1 + hpv.m_HPV_only[i])

                ## Determine unloaded velocity of this given slope
                if mo.model_selection == 2:
                    data = (
                        mv.ro,
                        mv.C_d,
                        mv.A,
                        mv.m1 + hpv.m_HPV_only.flatten()[i],  #
                        hpv.Crr.flatten()[i],  # CRR related to the HPV
                        mv.eta,
                        mv.P_t,
                        mv.g,
                        s[0] * mo.ulhillpo,  # hill polarity, see model options
                    )
                    V_guess = 12
                else:
                    data = (
                        mv.m1 + hpv.m_HPV_only.flatten()[i],
                        met,
                        s[0] * mo.ulhillpo,  # hill polarity, see model options
                    )
                    V_guess = 1

                V_un = fsolve(model, V_guess, args=data, full_output=True)
                if V_un[2] == 1:
                    mr.v_unload_matrix3d[i, j, :] = V_un[0][0]
                else:
                    mr.v_unload_matrix3d[i, j, :] = np.nan

                # start loop iterating over loads
                for k, total_load in enumerate(m_t.flatten()):

                    if mo.model_selection == 2:
                        data = (
                            mv.ro,
                            mv.C_d,
                            mv.A,
                            total_load,  #
                            hpv.Crr.flatten()[i],  # CRR related to the HPV
                            mv.eta,
                            mv.P_t,
                            mv.g,
                            s[0] * mo.lhillpo,
                        )
                        # V_guess = 12
                    else:
                        data = (total_load, met, s[0])
                        # V_guess = 1

                    V_r = fsolve(model, V_guess, args=data, full_output=True)
                    if V_r[2] == 1:
                        mr.v_load_matrix3d[i, j, k] = V_r[0][0]
                        mr.load_matrix3d[i, j, k] = load_vector[
                            k
                        ]  # amount of water carried
                    else:
                        mr.v_load_matrix3d[i, j, k] = np.nan
                        mr.load_matrix3d[i, j, k] = load_vector[k]

        return mr.v_load_matrix3d, mr.load_matrix3d

    def LCDA_solution(p, *data):
        m_load, met, s = data
        v_solve = p[0]
        G = (s * 360 / (2 * np.pi)) / 45 * 100
        return (
            1.44
            + 1.94 * v_solve**0.43
            + 0.24 * v_solve**4
            + 0.34 * (1 - 1.05 ** (1 - 1.1 ** (G + 32)))
        ) * m_load - met.budget_watts

    def Lankford_solution(p, *data):
        m_load, met, s = data
        v_solve = p[0]
        G = (s * 360 / (2 * np.pi)) / 45
        return (
            5.43483
            + (6.47383 * v_solve)
            + (-0.05372 * G)
            + (0.652298 * v_solve * G)
            + (0.023761 * v_solve * G**2)
            + (0.00320 * v_solve * G**3)
            - (met.budget_VO2 / m_load)
        )


class HPV_variables:
    """
    reshapes the incoming dataframe in to the appropriate dimensions for doing numpy maths
    """

    def __init__(self, hpv_param_df, mv):
        self.n_hpv = hpv_param_df.Pilot.size  # number of HPVs
        self.name = np.array(hpv_param_df.Name).reshape((self.n_hpv, 1))[
            :, np.newaxis, :
        ]
        self.m_HPV_only = np.array(hpv_param_df.Weight).reshape((self.n_hpv, 1))[
            :, np.newaxis, :
        ]
        self.n = np.array(hpv_param_df.Efficiency).reshape((self.n_hpv, 1))[
            :, np.newaxis, :
        ]
        self.Crr = np.array(hpv_param_df.Crr).reshape((self.n_hpv, 1))[:, np.newaxis, :]
        self.v_no_load = np.array(hpv_param_df.AverageSpeedWithoutLoad).reshape(
            (self.n_hpv, 1)
        )[:, np.newaxis, :]
        self.load_limit = np.array(hpv_param_df.LoadLimit).reshape((self.n_hpv, 1))[
            :, np.newaxis, :
        ]

        self.Pilot = np.array(hpv_param_df.Pilot).reshape((self.n_hpv, 1))[
            :, np.newaxis, :
        ]

        self.PilotLoad = mv.m1 * self.Pilot

        self.v_no_load = np.array(hpv_param_df.AverageSpeedWithoutLoad).reshape(
            (self.n_hpv, 1)
        )[:, np.newaxis, :]

        self.v_no_load_calculated = 0

        self.GroundContact = np.array(hpv_param_df.GroundContact).reshape(
            (self.n_hpv, 1)
        )[:, np.newaxis, :]

    @property
    def load_capacity(self):
        return self.load_limit - self.PilotLoad


class model_variables:
    def __init__(self):
        #### variables (changeable)
        self.s_deg = 0  # slope in degrees (only used for loading scenario, is overriden in variable slope scenario)
        self.m1 = 62  # mass of rider/person
        self.P_t = 75  # power output of person (steady state average)
        self.F_max = 300  # maximum force exertion for pushing up a hill for a short amount of time
        self.L = 1  # leg length
        self.minimumViableLoad = 0  # in kg, the minimum useful load for such a trip
        self.t_hours = 8  # number of hours to gather water
        self.L = 1  # leg length
        self.A = 0.51  # cross sectional area
        self.C_d = 0.9  # constant for wind
        self.ro = 1.225
        self.eta = 0.92
        self.g = 9.81
        self.waterration = 15


class model_options:
    def __init__(self):

        # model options
        self.model_selection = 2  # 1 is sprott, 2 is cycling 3 is lankford, 4 is LCDA

        #  0 = min load, 1 = max load, >1 = linear space between min and max
        self.load_res = 15
        #  0 = min slope only, 1 = max slope only, >1 = linear space between min and max
        self.slope_res = 15

        # slope start and end
        self.slope_start = 0  # slope min degrees
        self.slope_end = 10  # slope max degrees

        # is it uphill or downhill? -1 is downhill, +1 is uphill. Any value between 0 to 1 will lesson the impact of the "effective hill", useful for exploring/approximating braking on bikes for donwhill (otherwise we get up to 60km/h on an unloaded bike pleting down a 10 deg hill!)
        self.lhillpo = 1  # loaded hill polarity
        self.ulhillpo = 0  # unloaded hill polarity

        # for plotting of single scenarios, likle on surf plots
        self.slope_scene = (
            0  # 0 is flat ground, -1 will be the steepest hill (slope_end)
        )
        self.load_scene = (
            -1  # 0 is probably 15kg, -1 will be max that the HPV can manage
        )
        self.surf_plot_index = 0  # this one counts the HPVs (as you can only plot one per surf usually, so 0 is the first HPV in the list, -1 will be the last in the list)

        # # name the model
        # if self.model_selection == 1:
        #     self.model_name = "Sprott"
        # elif self.model_selection == 2:
        #     self.model_name = "Cycling"
        # elif self.model_selection == 3:
        #     self.model_name = "Lankford"
        # elif self.model_selection == 4:
        #     self.model_name = "LCDA"
        # else:
        #     self.model_name = "Unknown"

        if self.load_res <= 1:
            self.n_load_scenes = 1
        else:
            self.n_load_scenes = self.load_res  # number of load scenarios


    @property
    def model_name(self):
        if self.model_selection == 1:
            return "Sprott"
        elif self.model_selection == 2:
            return "Cycling"
        elif self.model_selection == 3:
            return "Lankford"
        elif self.model_selection == 4:
            return "LCDA"
        else:
            return "Unknown"


class MET_values:
    def __init__(self, mv):
        # Metabolic Equivalent of Task
        self.MET_of_sustainable_excercise = (
            4  # # https://en.wikipedia.org/wiki/Metabolic_equivalent_of_task
        )
        self.MET_VO2_conversion = 3.5  # milliliters per minute per kilogram body mass
        self.MET_watt_conversion = 1.162  # watts per kg body mass
        # so 75 Watts output (from Marks textbook) is probably equivalent to about 6 METs
        self.budget_VO2 = (
            self.MET_VO2_conversion * self.MET_of_sustainable_excercise * mv.m1
        )  # vo2 budget for a person
        self.budget_watts = (
            self.MET_watt_conversion * self.MET_of_sustainable_excercise * mv.m1
        )  # vo2 budget for a person


class model_results:
    def __init__(self, hpv, mo):
        self.hpv_name = (hpv.name,)

        # create slope vector
        self.slope_vector_deg = linspace_creator(
            np.array([mo.slope_end]), mo.slope_start, mo.slope_res
        )
        self.slope_vector_deg = self.slope_vector_deg.reshape(
            1, self.slope_vector_deg.size
        )

        # initilise and createoutput 3d matrices (dims: 0 = HPV, 1 = Slope, 2 = load)
        # uses load resolution as a flag, so converts it to "load scenarios", as load_res = 0

        ## initialise matrices
        self.v_load_matrix3d = np.zeros(
            (hpv.n_hpv, self.slope_vector_deg.size, mo.n_load_scenes)
        )
        self.v_unload_matrix3d = np.zeros(
            (hpv.n_hpv, self.slope_vector_deg.size, mo.n_load_scenes)
        )
        self.load_matrix3d = np.zeros(
            (hpv.n_hpv, self.slope_vector_deg.size, mo.n_load_scenes)
        )
        self.slope_matrix3d_deg = np.repeat(self.slope_vector_deg, hpv.n_hpv, axis=0)
        self.slope_matrix3d_deg = np.repeat(
            self.slope_matrix3d_deg[:, :, np.newaxis], mo.n_load_scenes, axis=2
        )
        self.slope_matrix3drads = (self.slope_matrix3d_deg / 360) * (2 * 3.1416)

    def create_dataframe_single_scenario(self, hpv, mv, load_scene, slope_scene):
        df = pd.DataFrame(
            {
                "Name": self.hpv_name[0].transpose()[0][0],
                "Load": self.load_matrix3d[:, slope_scene, load_scene],
                "Slope" : self.slope_matrix3d_deg[:, slope_scene, load_scene],
                "Average Trip Velocity": self.v_avg_matrix3d[
                    :, slope_scene, load_scene
                ],
                "Litres * Km": self.distance_achievable_one_hr[
                    :, slope_scene, load_scene
                ]
                * self.load_matrix3d[:, slope_scene, load_scene],
                "Water ration * Km": self.distance_achievable_one_hr[
                    :, slope_scene, load_scene
                ]
                * self.load_matrix3d[:, slope_scene, load_scene]
                / mv.waterration,
                "Distance to Water Achievable": self.distance_achievable_one_hr[
                    :, slope_scene, load_scene
                ] * mv.t_hours
                / 2,
                "Total Round trip Distance Achievable": self.distance_achievable_one_hr[
                    :, slope_scene, load_scene
                ] * mv.t_hours,
                "Load Velocity [kg * m/s]": self.v_avg_matrix3d[
                    :, slope_scene, load_scene
                ]
                * self.load_matrix3d[:, slope_scene, load_scene],
                "Loaded Velocity": self.v_load_matrix3d[:, slope_scene, load_scene],
                "Unloaded Velocity": hpv.v_no_load.transpose()[0][0],
                "Hours Collecting Water Max": mv.t_hours,
                "Hours Spent Collecting Single Person Water" : mv.waterration / (
                self.distance_achievable_one_hr[
                    :, slope_scene, load_scene
                ]
                * self.load_matrix3d[:, slope_scene, load_scene])
            }
        )
        return df

    def load_results(self, hpv, mv, mo):

        self.model_name = mo.model_name

        ## Calculate average speed
        # ### CAUTION! this currently uses a hardcoded 'average speed' which is from lit serach, not form the model
        # # Thismight be appropriate, as the model currentl assumes downhill to the water, uphill away, so it is conservative.
        # # average velocity for a round trip
        self.v_avg_matrix3d = (self.v_load_matrix3d + hpv.v_no_load) / 2
        self.v_avg_matrix3d = (self.v_load_matrix3d + self.v_unload_matrix3d) / 2

        self.velocitykgs = self.v_avg_matrix3d * self.load_matrix3d

        # #### Distance
        self.t_secs = mv.t_hours * 60 * 60
        self.distance_achievable = (self.v_avg_matrix3d * self.t_secs) / 1000  # kms
        self.distance_achievable_one_hr = (self.v_avg_matrix3d * 60 * 60) / 1000  # kms

        # power
        self.total_load = self.load_matrix3d + (
            mv.m1 * hpv.Pilot + hpv.m_HPV_only
        )  # total weight when loaded
        # self.Ps = self.v_load_matrix3d * self.total_load * mv.g * np.sin(np.arctan(self.slope_matrix3drads))
        # self.Ps = self.Ps / hpv.n[:, np.newaxis, np.newaxis]


"""
Plotting Class
"""


class plotting_hpv:

    def surf_plot(mr, mo, hpv):

        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        # Make data.
        Z = mr.distance_achievable[mo.surf_plot_index, :, :]
        X = mr.load_matrix3d[mo.surf_plot_index, :, :]
        Y = mr.slope_matrix3d_deg[mo.surf_plot_index, :, :]

        # Plot the surface.
        surf = ax.plot_surface(
            X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False
        )

        # Customize the z axis.
        ax.zaxis.set_major_locator(LinearLocator(10))
        # A StrMethodFormatter is used automatically
        ax.zaxis.set_major_formatter("{x:.02f}")

        ax.set_xlabel("Load [kg]")
        ax.set_ylabel("Slope [deg ˚]")
        ax.set_zlabel("Velocity [m/s]")
        plt.title(
            "Slope, Load & Distance for: " + mr.hpv_name[0][mo.surf_plot_index][0][0]
        )

        plt.show()

    def surf_plotly(mr, mo, hpv):

        # # Make data.
        Z = mr.distance_achievable[mo.surf_plot_index, :, :]
        X = mr.load_matrix3d[mo.surf_plot_index, :, :]
        Y = mr.slope_matrix3d_deg[mo.surf_plot_index, :, :]

        # using the graph options from plotly, create 3d plot.
        fig = go.Figure(data=[go.Surface(z=Z, x=X, y=Y)])
        fig.update_layout(
            title="Slope, Load & Distance for: "
            + mr.hpv_name[0][mo.surf_plot_index][0][0],
            autosize=True,
            width=500,
            height=500,
            margin=dict(l=65, r=50, b=65, t=90),
        )
        fig.show()

    def surf_plotly_multi(mr, mo, hpv):

        plot_height = 700
        plot_width = 900
        xaxis_title = "Load [kg]"
        yaxis_title = "Slope [˚]"
        zaxis_title = "Distannce [km]"
        # create list for plot specs based on number of hpvs
        specs_list = []
        spec_single = [{"type": "surface"}]
        for HPVname in mr.hpv_name:
            specs_list.append(spec_single)

        # create figure and subplots
        fig = make_subplots(
            rows=mr.n_hpv, cols=1, specs=specs_list, subplot_titles=mr.hpv_name
        )  # name the subplots the HPV names (create a new list of strings if you'd like to rename)

        # in a for loop, create the subplots
        mo.surf_plot_index = 0
        for HPVname in mr.hpv_name:
            Z = mr.distance_achievable[mo.surf_plot_index, :, :]
            X = mr.load_matrix3d[mo.surf_plot_index, :, :]
            Y = mr.slope_matrix3d_deg[mo.surf_plot_index, :, :]

            fig.add_trace(
                go.Surface(z=Z, x=X, y=Y, colorscale="Viridis", showscale=False),
                row=1 + mo.surf_plot_index,
                col=1,
            )

            fig.update_scenes(
                xaxis_title_text=xaxis_title,
                yaxis_title_text=yaxis_title,
                zaxis_title_text=zaxis_title,
            )

            mo.surf_plot_index += 1

        # update the layout and create a title for whole thing
        fig.update_layout(
            title_text="3D subplots of HPVs",
            height=plot_height * mr.n_hpv,
            width=plot_width,
        )

        fig.show()

        # show the figure
        # py.iplot(fig, filename="3D subplots of HPVs New")

    def load_plot_plotly(mr, mo, hpv):

        xaxis_title = "Load [kg]"
        yaxis_title = "Speed [m/s]"

        slope_name = mr.slope_vector_deg.flat[mo.slope_scene]
        chart_title = "m/s with different loads, Constant %0.1f slope, model %s" % (
            slope_name,
            mr.model_name,
        )

        i = 0

        fig = go.Figure()
        for HPVname in mr.hpv_name[0].transpose()[0][0]:
            y = mr.v_load_matrix3d[i, mo.slope_scene, :]
            x = mr.load_matrix3d[i, mo.slope_scene, :]
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

    def slope_plot_plotly(mr, mo, hpv):

        xaxis_title = "Slope [˚]"
        yaxis_title = "m/s"
        if mo.load_scene == 0:
            load_name = mr.load_matrix3d.flat[mo.load_scene]
        elif mo.load_scene == -1:
            load_name = "maximum"
        else:
            load_name = "variable"

        chart_title = (
            f"m/s with different slope, {load_name} kg load, model {mr.model_name}"
        )

        i = 0
        fig = go.Figure()
        for HPVname in mr.hpv_name[0].transpose()[0][0]:
            # y = mr.velocitykgs[
            y = mr.v_load_matrix3d[
                i, :, mo.load_scene
            ]  # SEE ZEROS <-- this is for the minimum weight
            x = mr.slope_matrix3d_deg[i, :, mo.load_scene]
            i += 1
            fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name=HPVname))

        # Update the title
        fig.update_layout(title=dict(text=chart_title))
        # Update te axis label (valid for 2d graphs using graph object)
        fig.update_xaxes(title_text=xaxis_title)
        fig.update_yaxes(title_text=yaxis_title)
        # fig.update_yaxes(range=[0, 15])

        fig.show()
        # py.iplot(fig, filename=chart_title)

    def slope_velcoity_kgs(mr, mo, hpv):

        xaxis_title = "Slope [˚]"
        yaxis_title = "Velocity Kgs"

        if mo.load_scene == 0:
            load_name = mr.load_matrix3d.flat[mo.load_scene]
        elif mo.load_scene == -1:
            load_name = "maximum"
        else:
            load_name = "variable"

        chart_title = f"Velocity Kgs with different slope, {load_name} kg load, model {mr.model_name}"

        i = 0

        fig = go.Figure()
        for HPVname in mr.hpv_name[0].transpose()[0][0]:
            # y = mr.velocitykgs[
            y = mr.velocitykgs[
                i, :, mo.load_scene
            ]  # SEE ZEROS <-- this is for the minimum weight
            x = mr.slope_matrix3d_deg[i, :, mo.load_scene]
            i += 1
            fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name=HPVname))

        # Update the title
        fig.update_layout(title=dict(text=chart_title))
        # Update te axis label (valid for 2d graphs using graph object)
        fig.update_xaxes(title_text=xaxis_title)
        fig.update_yaxes(title_text=yaxis_title)
        # fig.update_yaxes(range=[0, 15])

        fig.show()
        # py.iplot(fig, filename=chart_title)

    def slope_velocities(mr, mo, hpv):

        HPV = 0

        xaxis_title = "Slope [˚]"
        yaxis_title = "Velocity [m/s]"
        load_name = mr.load_matrix3d[HPV][0][mo.load_scene]

        chart_title = f"m/s with different slope, {load_name} kg load,\n for HPV: {hpv.name.flatten()[0]}, model {mr.model_name}"

        fig = go.Figure()
        x = mr.slope_matrix3d_deg[HPV, :, mo.load_scene]
        y = mr.v_avg_matrix3d[HPV, :, mo.load_scene]
        fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name="Average"))
        y = mr.v_load_matrix3d[HPV, :, mo.load_scene]
        fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name="Loaded"))
        y = mr.v_unload_matrix3d[HPV, :, mo.load_scene]
        fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name="Unloaded"))

        # Update the title
        fig.update_layout(title=dict(text=chart_title))
        # Update te axis label (valid for 2d graphs using graph object)
        fig.update_xaxes(title_text=xaxis_title)
        fig.update_yaxes(title_text=yaxis_title)
        # fig.update_yaxes(range=[0, 15])

        fig.show()
        # py.iplot(fig, filename=chart_title)

    def time_sensitivity_plotly_grouped(mr, mo, hpv):
        t_hours_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        load_scene = -1

        # plotly setup
        fig = go.Figure()
        i = 0
        # add trace for eat
        for HPVname in mr.hpv_name[0].transpose()[0][0]:
            X = t_hours_list
            Y = mr.load_matrix3d[i, 0, load_scene] * (
                mr.distance_achievable_one_hr[i, 0, load_scene] * np.array(t_hours_list)
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

    def bar_plot_loading(mr, mo, hpv, mv):
        t_hours_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

        slope_name = mr.slope_vector_deg.flat[mo.slope_scene]
        chart_title = "Efficiency at %0.2f degrees" % slope_name

        df = mr.create_dataframe_single_scenario(hpv, mv, mo.load_scene, mo.slope_scene)

        fig = px.bar(
            df,
            x="Load",
            y="Load Velocity [kg * m/s]",
            hover_data=[
                "Name",
                "Average Trip Velocity",
                "Loaded Velocity",
                "Unloaded Velocity",
                "Distance to Water Achievable",
                "Total Round trip Distance Achievable",
                "Load",
                "Slope",
                "Hours Collecting Water Max",
                "Hours Spent Collecting Single Person Water" 
            ],
            color="Name",
            labels={"Name": "Name"},
            title=chart_title,
        )
        fig.show()
        # py.iplot(fig, filename=chart_title)

    def bar_plot_loading_distance(mr, mo, hpv, mv):

        slope_name = mr.slope_vector_deg.flat[mo.slope_scene]
        chart_title = "Efficiency at %0.2f degrees, with model %s" % (
            slope_name,
            mr.model_name,
        )

        df = mr.create_dataframe_single_scenario(hpv, mv, mo.load_scene, mo.slope_scene)

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
                "Slope",
                "Hours Collecting Water Max",
                "Hours Spent Collecting Single Person Water" 
            ],
            color="Name",
            labels={"Name": "Name"},
            title=chart_title,
        )
        fig.show()
        # py.iplot(fig, filename=chart_title)
