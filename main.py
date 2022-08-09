# import pandas module
import pandas as pd
import numpy as np

 
with open("ModelParams.csv") as csv_file:
    # read the csv file
    param_df = pd.read_csv("ModelParams.csv") 


# variables (changeable)
s_deg=0         # slope in degrees
m1=83           # mass of rider/person
P_t=75          #power output of person (steady state average)
L=1             #leg length
minimumViableLoad = 15


# constants
g=9.81
pi = 3.1416
## Data Accounting
s = (s_deg/360)*(2*pi) #converts s in to radians


dfs = []        # initialise empty list of dataframes
res = 50        # how many datapoints for linear space




for maxval in param_df.LoadCapacity:
    minval = minimumViableLoad
    load_vector =   np.linspace(start = minval, #define linear space of weights
                    stop =maxval,           # define maximum value for linear space
                    num = res,              # how many data points?
                    endpoint = True,
                    retstep = False,
                    dtype = None)
    df = pd.DataFrame(load_vector)          # create dataframe with load linspace data
    dfs.append(df)                          # create list of dataframes

load_matrix_df = pd.concat(dfs, axis=1)
load_matrix_df.columns= param_df.Name


## Derivations of further mass variables
m_HPV = param_df.LoadCapacity*0.05; # assume 5# of load is the wight of HPV
m_t = m1*param_df.Pilot #+load_matrix_df+m_HPV; #total weight for calculation of


print(m_t)




## Derivations of further mass variables
# m_HPV = param_df.LoadCapacity*0.05; # assume 5# of load is the wight of HPV
# m_t = m1*param_df.Pilot+load_matrix+m_HPV; #total weight for calculation of

# print(m_t)

#create linear space vector for load scenarios
# multiLoad=1
# if multiLoad ==1
# linres = 1000
# load_matrix=zeros(length(m2),linres)
# for i=1:length(m2)
#     load_matrix(i,:) = linspace(minimumViableLoad,m2(i),linres)
# end
# else
    
# static_load_matrix = [0,20,50,75,100,150,200,250,300]    
# load_matrix=zeros(length(m2),length(static_load_matrix))+static_load_matrix
# end




