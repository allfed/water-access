import pandas as pd

# Define the input file path
input_file = "../../data/GIS/NaN_count_with_centroids.csv"

# Read the CSV file
df = pd.read_csv(input_file)

# Slice the dataframe to exclude Antarctica (latitude > -60 degrees)
df_no_antarctica = df[df["latitude_centroid"] > -60]

# Define the columns to check for NaN

#  dtw_nan
# pop_density_nan
# GHS_SMOD_nan
# slope1_nan
# any_nan

columns_to_check = [
    "dtw_nan",
    "pop_density_nan",
    "GHS_SMOD_nan",
    "slope1_nan",
    "any_nan",
]


# Print header
print(
    "\n{:<20} {:<15} {:<15} {:<15}".format(
        "Column", "Total NaNs", "Non-Antarctica", "Difference"
    )
)
print("-" * 65)  # Add a separator line


for col in columns_to_check:
    total_nans = df[col].sum()
    non_antarctica_nans = df_no_antarctica[col].sum()
    difference = total_nans - non_antarctica_nans

    print(
        "{:<20} {:<15} {:<15} {:<15}".format(
            col, total_nans, non_antarctica_nans, difference
        )
    )

# Calculate and print the total number of rows
total_rows = len(df)
non_antarctica_rows = len(df_no_antarctica)
rows_difference = total_rows - non_antarctica_rows

print("\nTotal Rows:")
print(f"All data: {total_rows}")
print(f"Excluding Antarctica: {non_antarctica_rows}")
print(f"Rows in Antarctica: {rows_difference}")

# create an antarctica df
antarctica_df = df[df["latitude_centroid"] < -60]

# CONFIRMED VISUALLY IN QGIS THAT ANTARCTICA IS ACCURATE


# analyiss of the population density in NOT ANTARCTICA
# less than 0.1
# NaN
# Total


pop_less_than_01 = len(df_no_antarctica[df_no_antarctica["pop_density"] < 0.1])
print(f"Number of population less than 0.1 in World: {pop_less_than_01}")

# NaN
pop_nan_count = len(df_no_antarctica[df_no_antarctica["pop_density"].isna()])
print(f"Number of NaN in World: {pop_nan_count}")

# Total
print(f"Total number of World: {len(df_no_antarctica)}")

# percentage of the total
print(
    f"Percentage of population less than 0.1 in World: "
    f"{pop_less_than_01 / len(df_no_antarctica) * 100:.2f}%"
)
print(
    f"Percentage of NaN in World: "
    f"{pop_nan_count / len(df_no_antarctica) * 100:.2f}%"
)
