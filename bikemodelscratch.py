import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
from matplotlib.widgets import Cursor
from scipy.optimize import fsolve


def linspace_creator(max_value_array,min_value,res):
    # creates a linsapce numpy array from the given inputs
    # max_value needs to be a numpy array (even if it is a 1x1)
    # min value is to be an int or float
    # resolution to be an int
    # returns
    # res = resoltuion, also used as a flag to set minimum/maxmum sinle load scearios

    if res == 1:  # if res =1 , calculate for max load of HPV
        load_matrix = np.zeros((len(max_value_array),res))    # initilaise numpy matrix
        load_matrix = max_value_array
    elif res == 0: # if res =0 , calculate for min viable load of HPV (trick: use this to set custom load)
        load_matrix = np.zeros((len(max_value_array),1)) + min_value  # initilaise numpy matrix
        load_matrix = load_matrix
    elif res > 1:
        load_matrix = np.zeros((len(max_value_array),res))    # initilaise numpy matrix

        #### Create linear space of weights
        # creates a vector for each of the HPVs, an equal number of elements spaced 
        # evenly between the minimum viable load and the maximum load for that HPV
        i=0                                                         # initliase index
        for maxval in np.nditer(max_value_array): # iterate through numpy array
            minval = min_value
            load_vector =   np.linspace(start = minval,             # define linear space of weights
                            stop =maxval,                           # define maximum value for linear space
                            num = res,                              # how many data points?
                            endpoint = True,
                            retstep = False,
                            dtype = None)
            load_matrix[i:] = load_vector                           # place the vector in to a matrix
            i += 1                                                  # increment index
    else:
        print('Error: unexpected loading resolution, setting default')
        load_matrix = max_value_array

    return load_matrix

def max_safe_load(m_HPV_only,LoadCapacity,F_max,s,g):
    max_load_HPV = LoadCapacity  

    #### Weight limits
    # Calculate weight limits for hills. There are some heavy loads which humans will not be able to push up certain hills
    # Note that this is for average slopes etc. This will be innacurate (i.e one VERY hilly section could render the whole thing impossible, this isn't accounted for here)
    if s != 0:  #requires check to avoid divide by zero
        max_pushable_weight = F_max/(np.sin(s)*g)
        i=0
        if    m_HPV_only.size > 1: # can't iterate if there is only a float (not an array)
            for  HPV_weight in m_HPV_only:
                if max_load_HPV[i]+HPV_weight>max_pushable_weight:
                    max_load_HPV[i] = max_pushable_weight-HPV_weight
                i+=1
        else:
            max_load_HPV = max_pushable_weight-m_HPV_only
            

    
    return max_load_HPV

#### Start Script
with open("data/ModelParams.csv") as csv_file:
    # read the csv file
    allHPV_param_df = pd.read_csv("data/ModelParams.csv") 

print('bike solver')

# selectHPVs you're interested in
param_df = allHPV_param_df.iloc[:]

#### variables (changeable)
s_deg = 0       # slope in degrees (only used for loading scenario, is overriden in cariable slope scenario)
m1=83           # mass of rider/person
P_t=75          # power output of person (steady state average)
F_max = 300     # maximum force exertion for pushing up a hill for a short amount of time
L=1             # leg length
minimumViableLoad = 15 # in kg, the minimum useful load for such a trip
t_hours=8       # number of hours to gather water
L=1             # leg length
A = 1           # cross sectional area
C_d = 1         # constant for wind




## plot options
load_plot = 0
slope_plot = 0
surf_plot = 1

## Options
load_res = 3        #  0 = min load, 1 = max load, >1 = linear space between min and max
slope_res = 4      #  0 = min slope only, 1 = max slope only, >1 = linear space between min and max
slope_start = 0     # slope min
slope_end = 20      # slope max
res = load_res

#### constants
g=9.81
pi = 3.1416
s =  (s_deg/360)*(2*pi)

# create slope vector
slope_vector = linspace_creator(np.array([slope_end]),slope_start,slope_res)
slope_vector = slope_vector.reshape(1,slope_vector.size)
# initilise and createoutput 3d matrices (dims: 0 = HPV, 1 = Slope, 2 = load)
n_hpv = param_df.Pilot.size
if load_res <= 1:
    n_load_scenes = 1
else:
    n_load_scenes = load_res
v_load_matrix3d     = np.zeros((n_hpv,slope_vector.size,n_load_scenes))
load_matrix3d       = np.zeros((n_hpv,slope_vector.size,n_load_scenes))
slope_matrix3d      = np.repeat(slope_vector, n_hpv, axis=0)
slope_matrix3d      = np.repeat(slope_matrix3d[:,:,np.newaxis], n_load_scenes, axis=2)
slope_matrix3drads  = (slope_matrix3d/360)*(2*pi)


#### constants
g=9.81
pi = 3.1416
ro=1
eta = 0.8

n_hpv = param_df.Pilot.size # number of HPVs (by taking the pilot value as it is alwways a simple int [more complex data types get cnfusing wwhen the dataframe nly has 1 row])
n = np.array(param_df.Efficiency).reshape((n_hpv,1))
Crr = np.array(param_df.Crr).reshape((n_hpv,1))
v_no_load = np.array(param_df.AverageSpeedWithoutLoad).reshape((n_hpv,1))
LoadCapacity = np.array(param_df.LoadCapacity).reshape((n_hpv,1))
m_HPV_only = param_df.LoadCapacity*0.05;                    # assume 5# of load is the wight of HPV

max_load_HPV = max_safe_load(m_HPV_only,LoadCapacity,F_max,s,g) # function to calaculate the max saffe loads due to hill
load_matrix = linspace_creator(max_load_HPV,minimumViableLoad,res)

#### Derivations of further mass variables
m_HPV_pilot = np.array(m1*param_df.Pilot + m_HPV_only).reshape((n_hpv,1)) # create vector with extra weights to add
m_HPV_load_pilot = load_matrix + m_HPV_pilot # weight of the HPV plus the rider (if any) plus the load
m_walk_carry = m1 + m_HPV_load_pilot * (np.array(param_df.GroundContact).reshape((n_hpv,1))-1)*-1 # negative 1 is to make the wheeled = 0 and the walking = 1
m_HPV_load = load_matrix + np.array(m_HPV_only).reshape((n_hpv,1)) 
# weight of the mass being 'walked', i.e the wieght of the human plus anything they are carrying (not pushing or riding)

m_t = m_HPV_load_pilot

func = lambda v_solve :  ( (1/2*ro*v_solve**3 * C_d*A 
                            + v_solve*m_t*g*Crr 
                            + v_solve*m_t*g*s) /eta - P_t )

# Use the numerical solver to find the roots

v_solve_initial_guess = 2
v_solve_solution = fsolve(func, v_solve_initial_guess)

# if loaded speed is greater than unloaded avg speed, make equal to avg unloaded speed
i=0
for maxval in v_no_load:
    indeces = v_load[i]>maxval; v_load[i,indeces]=maxval ; i+=1

return v_load , load_matrix