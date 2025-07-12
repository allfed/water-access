from pathlib import Path
import pandas as pd

# Define the root of the repository by going up one level from the script
# directory
data_script_dir = Path(__file__).resolve().parent
script_dir = data_script_dir.parent
repo_root = script_dir.parent

# Define the data directories relative to the repo root
parquet_data_dir = repo_root / "results" / "parquet_files"
results_dir = repo_root / "results"
csv_data_dir = repo_root / "data" / "GIS"

# Define the paths to the Parquet file and the original CSV file
parquet_file_path = parquet_data_dir / "zone_simulation_result_0.parquet"
original_csv_file_path = csv_data_dir / "gis_data_adm1.csv"
output_file_path = (
    results_dir / "GIS_merged_output_processed.csv"
)  # Adjust the path as needed

# Step 1: Read the Parquet file into a DataFrame
processed_df = pd.read_parquet(parquet_file_path)


columns_to_keep = [
    "fid",  # Ensure 'fid' is included for the merge
    "left",
    "top",
    "right",
    "bottom",
]


original_df = pd.read_csv(original_csv_file_path, usecols=columns_to_keep)

# Step 3: Merge the processed DataFrame with the original data on the
# 'fid' column
merged_df = pd.merge(processed_df, original_df, on="fid", how="left")

# Step 4: Save the merged DataFrame to a new CSV file
merged_df.to_csv(output_file_path, index=False)

print(f"Merged data saved to {output_file_path}")
