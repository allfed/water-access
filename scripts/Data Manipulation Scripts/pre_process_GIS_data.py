import pandas as pd
""" 



NAN VALUES IN COLUMNS:

dtw            6474098
pop_density    7070981
GHS_SMOD         95090
dtype: int64

Correlation between NaN values in columns:
                  dtw  pop_density  GHS_SMOD
dtw          1.000000     0.825685  0.067406
pop_density  0.825685     1.000000  0.057366
GHS_SMOD     0.067406     0.057366  1.000000

"""

input_file = '../../data/GIS/updated_GIS_output.csv'


# read the data
df = pd.read_csv(input_file)

# drop the columns 'shapeType'
df = df.drop(['shapeType'], axis=1)

# drop rows that don't have a country code
df = df.dropna(subset=['shapeGroup'])

# Create a DataFrame with True (or 1) where NaN is present, and False (or 0) where it is not
nan_indicator = df[['dtw', 'pop_density', 'GHS_SMOD', 'slope1']].isna()

# Convert the boolean values to integers (1 for NaN, 0 for not NaN)
nan_indicator = nan_indicator.astype(int)

# Calculate correlation matrix for NaN presence
nan_correlation = nan_indicator.corr()

# Display the correlation matrix
print("Correlation between NaN values in columns:")
print(nan_correlation)

# Check for rows where GHS_SMOD is the only NaN value
ghs_smod_only_nan = df[df['GHS_SMOD'].isna() & df['dtw'].notna() & df['pop_density'].notna()]
# check the others too
dtw_only_nan = df[df['dtw'].isna() & df['GHS_SMOD'].notna() & df['pop_density'].notna() & df['slope1'].notna()]
pop_density_only_nan = df[df['pop_density'].isna() & df['GHS_SMOD'].notna() & df['dtw'].notna() & df['slope1'].notna()]
slope1_only_nan = df[df['slope1'].isna() & df['GHS_SMOD'].notna() & df['dtw'].notna() & df['pop_density'].notna()]

# new line
print("\n")

# print the number of NaNs in each column
print(f"Number of NaNs in dtw: {df['dtw'].isna().sum()}")
print(f"Number of NaNs in pop_density: {df['pop_density'].isna().sum()}")
print(f"Number of NaNs in GHS_SMOD: {df['GHS_SMOD'].isna().sum()}")
print(f"Number of NaNs in slope1: {df['slope1'].isna().sum()}")

# new line
print("\n")


# Print the number of such rows
print(f"Number of rows where GHS_SMOD is the only NaN value: {len(ghs_smod_only_nan)}")
print(f"Number of rows where dtw is the only NaN value: {len(dtw_only_nan)}")
print(f"Number of rows where pop_density is the only NaN value: {len(pop_density_only_nan)}")
print(f"Number of rows where slope1 is the only NaN value: {len(slope1_only_nan)}")



# since the correlation between the NaNs is so high, we can just drop the rows where there are any NaNs in the columns
# drop rows where there are any NaNs in the columns
print('\nDropping rows where there are any NaNs in the columns...\n')
pre_len = len(df)
df = df.dropna(subset=['dtw', 'pop_density', 'GHS_SMOD', 'slope1'])

# print the number of rows dropped, compared to the original number of rows
print(f"Number of rows dropped: {pre_len - len(df)}")

# Remap the names# ... existing code ...

column_mapping = {
    # 'GHS_SMOD': 'URBAN_1',
    'dtw': 'dtw_1',
    # 'pop_density': 'pop_count_15_1',
    'slope1': 'slope_1',
    'shapeGroup': 'ISOCODE',
    # Add mappings for grip columns if they exist in your current DataFrame
    'tp51': 'grip_1_1',
    'tp41': 'grip_2_1',
    'tp31': 'grip_3_1',
    'tp21': 'grip_4_1',
    'tp11': 'grip_5_1',
    'id': 'fid',
    # 'id' already exists
    # 'shapeName' already exists
}

# Rename the columns
df = df.rename(columns=column_mapping)

print('Saving the data...\n')

# save the data
df.to_csv('../../data/GIS/updated_GIS_output_cleaned.csv', index=False)

print('Data saved!\n')
