from pathlib import Path
import pandas as pd
import numpy as np

"""
This script processes and merges simulation results with geographical data, adding centroid coordinates.

The script performs the following steps:
1. Reads simulation results from a Parquet file.
2. Reads original geographical data from a CSV file.
3. Calculates longitude and latitude centroids for each geographical area.
4. Merges the simulation results with the centroid data.
5. Exports the merged data to a new CSV file.

The resulting file contains the original simulation results enhanced with centroid coordinates,
allowing for easier geographical visualization and analysis of the simulation outcomes.

Input files:
- Parquet file: 'zone_simulation_result_0.parquet' in the 'results/parquet_files' directory
- CSV file: 'updated_GIS_output.csv' in the 'data/GIS' directory

Output file:
- CSV file: 'GIS_merged_output_processed_with_centroids.csv' in the 'results' directory

Note: This script assumes a specific directory structure and file naming convention.
Adjust paths and filenames if necessary.
"""


# Define the root of the repository by going up one level from the script directory
data_script_dir = Path(__file__).resolve().parent
script_dir = data_script_dir.parent
repo_root = script_dir.parent

# Define the data directories relative to the repo root
parquet_data_dir = repo_root / "results" / "median_parquet"
results_dir = repo_root / "results"
csv_data_dir = repo_root / "data" / "GIS"

# Define the paths to the Parquet file and the original CSV file
parquet_file_path = parquet_data_dir / "zone_simulation_result_median.parquet"
original_csv_file_path = csv_data_dir / "GIS_country_points_fid.csv"
output_file_path = results_dir / "GIS_merged_output_processed_with_centroids_right.csv"

# Step 1: Read the Parquet file into a DataFrame
processed_df = pd.read_parquet(parquet_file_path)

# Step 2: Read the original CSV file
original_df = pd.read_csv(original_csv_file_path)

# Step 3: Calculate the longitude and latitude centroids
original_df["longitude_centroid"] = (original_df["left"] + original_df["right"]) / 2
original_df["latitude_centroid"] = (original_df["top"] + original_df["bottom"]) / 2

# Step 4: Keep only the fid and centroid columns
original_df = original_df[["fid", "longitude_centroid", "latitude_centroid"]]


## OPTION LEFT OR RIGHT
# left treats non analsyies areas and ocean as the same (missing data)
# end result is minimum dataframe with just the analysis areas and centroid data
# right treats non analsyses areas and ocean as different
# end result is a dataframe with all land mass areas and centroid data
# difference is due to dropping of non populated areas within the python model.
# Step 5: Merge the processed DataFrame with the centroid data on the 'fid' column
merged_df = pd.merge(
    processed_df, original_df, left_on="fid", right_on="fid", how="right"
)

# Step 6: Save the merged DataFrame to a new CSV file
merged_df.to_csv(output_file_path, index=False)

# count the number of nan values in the merged_df
nan_count = merged_df.isna().sum()
print(f"Number of NaN values in the merged DataFrame:\n{nan_count}")


# alteanrive csv created with NaNs as zeros
merged_df_no_nans = merged_df.fillna(0)

# fix any negative values in the merged_df_no_nans in zone_pop_with_water and zone_pop_without_water
merged_df_no_nans["zone_pop_with_water"] = merged_df_no_nans[
    "zone_pop_with_water"
].apply(lambda x: 0 if x < 0 else x)
merged_df_no_nans["zone_pop_without_water"] = merged_df_no_nans[
    "zone_pop_without_water"
].apply(lambda x: 0 if x < 0 else x)


output_file_path_no_nans = (
    results_dir / "GIS_merged_output_processed_with_centroids_no_nans.csv"
)
# save the merged_df_no_nans to a new CSV file
merged_df_no_nans.to_csv(output_file_path_no_nans, index=False)
