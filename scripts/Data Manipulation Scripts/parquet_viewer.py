from pathlib import Path
import pandas as pd

# Define the root of the repository by going up one level from the script directory
data_script_dir = Path(__file__).resolve().parent
script_dir = data_script_dir.parent
repo_root = script_dir.parent

# Define the data directory relative to the repo root
parquet_data_dir = repo_root / "results" / "parquet_files"

# Define the path to the specific Parquet file
parquet_file_path = parquet_data_dir / "zone_simulation_result_0.parquet"

# Read the Parquet file into a DataFrame
df = pd.read_parquet(parquet_file_path)

# Display the first few rows of the DataFrame
print(df.head())
