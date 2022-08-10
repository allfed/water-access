import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

with open("ModelParams.csv") as csv_file:
    # read the csv file
    param_df = pd.read_csv("ModelParams.csv") 

#### variables (changeable)
s_deg=0         # slope in degrees
m1=83           # mass of rider/person
P_t=75          #power output of person (steady state average)
F_max = 300     # maximum force exertion for pushing up a hill for a short amount of time
L=1             #leg length
minimumViableLoad = 15 # in kg, the minimum useful load for such a trip
res = 3        # resolution (how many datapoints for linear space)

#### constants
g=9.81
pi = 3.1416

#### Data Accounting
s = (s_deg/360)*(2*pi) #converts s in to radians
n_hpv = len(param_df.Name)
n = np.array(param_df.Efficiency).reshape((n_hpv,1))
Crr = np.array(param_df.Crr).reshape((n_hpv,1))
v_no_load = np.array(param_df.AverageSpeedWithoutLoad).reshape((n_hpv,1))
max_load_HPV = np.array(param_df.LoadCapacity).reshape((n_hpv,1))
if s != 0:  #requires check to avoid divide by zero
    max_pushable_weight = F_max/(np.sin(s)*g)
else:
    max_pushable_weight = 10000 #some large number, weight will be determined by HPV loading


#### Weight limits
# Calculate weight limits for hills. There are some heavy loads which humans will not be able to push up certain hills
# Note that this is for average slopes etc. This will be innacurate (i.e one VERY hilly section could render the whole tri impossible, this isn't accounted for here)

max_load_HPV[max_load_HPV>max_pushable_weight]=max_pushable_weight



#### Create linear space of weights
# creates a vector for each of the HPVs, an equal number of elements spaced 
# evenly between the minimum viable load and the maximum load for that HPV
load_matrix = np.zeros((len(param_df.LoadCapacity),res))    # initilaise numpy matrix
i=0                                                         # initliase index
for maxval in np.nditer(max_load_HPV):
    minval = minimumViableLoad
    load_vector =   np.linspace(start = minval,             # define linear space of weights
                    stop =maxval,                           # define maximum value for linear space
                    num = res,                              # how many data points?
                    endpoint = True,
                    retstep = False,
                    dtype = None)
    load_matrix[i:] = load_vector                           # place the vector in to a matrix
    i += 1                                                  # increment index

#### Derivations of further mass variables
m_HPV_only = param_df.LoadCapacity*0.05;                    # assume 5# of load is the wight of HPV
vecnp = np.array(m1*param_df.Pilot + m_HPV_only).reshape((n_hpv,1)) # crate vector with extra weights to add
m_HPV_load_pilot = load_matrix + vecnp # weight of the HPV plus the rider (if any) plus the load
m_walk_carry = m1 + m_HPV_load_pilot * (np.array(param_df.GroundContact).reshape((n_hpv,1))-1)*-1 # negative 1 is to make the wheeled = 0 and the walking = 1
m_HPV_load = load_matrix + np.array(m_HPV_only).reshape((n_hpv,1)) 
# weight of the mass being 'walked', i.e the wieght of the human plus anything they are carrying (not pushing or riding)

#### Constants from polynomial equation analysis
C = ((m_walk_carry)*g/pi)*(3*g*L/2)**(1/2)
D = pi**2/(6*g*L)
B1 = m_HPV_load_pilot*g*np.cos(np.arctan(s))*Crr     # rolling resistance component
B2 = m_HPV_load_pilot * np.sin(np.arctan(s))         # slope component
B = B1 + B2

##### velocities
v_load = (-B + np.sqrt(B**2+(2*C*D*P_t)/n))/(C*D/n);    # loaded velocity

# if loaded speed is greater than unloaded avg speed, make equal to avg unloaded speed
i=0
for maxval in v_no_load:
    indeces = v_load[i]>maxval
    v_load[i,indeces]=maxval
    i+=1

# calcualte average speed, half of the trip is loaded, and half is unbload
v_avg = (v_load+v_no_load)/2; #average velocity for a round trip

#### Forces and Power
F_push_hill = np.sin(s)*m_HPV_load*g # force required to push uphill (approx)


#### Distance
t_hours=8
t_secs=8*60*60
distance_achievable = (v_avg*t_secs)/1000 #kms

#### Plotting
i=0
fig, ax = plt.subplots(figsize=(20, 10))

for HPVname in param_df.Name:
    ax.plot(load_matrix[i], v_avg[i], label=HPVname)  # Plot some data on the axes.
    i += 1
plt.xlabel('Load [kg]')
plt.ylabel('Velocity [m/s]')
plt.title("Simple Plot")
plt.legend();

filename_output = 'velocity_slope' + str(s_deg)
fig.savefig(filename_output, transparent=False, dpi=80, bbox_inches="tight")