import pandas as pd

# Define the input and output file paths

input_file = "../../data/GIS/updated_GIS_output_with_nan_indicators.csv"
output_file_name = input_file.split(".")[0] + "_with_centroids.csv"
output_file_path = "../../data/GIS/" + output_file_name


# Read the CSV file
df = pd.read_csv(input_file)

# Calculate the longitude and latitude centroids
df["longitude_centroid"] = (df["left"] + df["right"]) / 2
df["latitude_centroid"] = (df["top"] + df["bottom"]) / 2

# Save the updated DataFrame to a new CSV file
df.to_csv(output_file_path, index=False)

print(f"Data with centroids saved to {output_file_path}")


# invetsiagte pop_density, I want to know the cut off which is "close to zero"

# 1. Calculate the mean and standard deviation of the pop_density column
