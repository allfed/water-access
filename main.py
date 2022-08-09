
# import pandas module
import pandas as pd
 
with open("ModelParams.csv") as csv_file:
    # read the csv file
    df = pd.read_csv("ModelParams.csv") 


# variables (changeable)
s_deg=0; # slope in degrees
m1=83; # mass of rider/person
P_t=75;#power output of person (steady state average)
L=1; #leg length
minimumViableLoad = 15;


# constants
g=9.81



print(df)


