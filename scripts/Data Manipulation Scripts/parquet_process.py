from pathlib import Path
import pandas as pd


"""
Output for the Septemeber 1000 run:

Run closest to median 'zone_pop_without_water':
Name: 85,

Run closest to 95th percentile of 'zone_pop_without_water':
Name: 54,


Run closest to 95th percentile of 'zone_pop_without_water':
Name: 211,


"""

# Define the root of the repository by going up one level from the script directory
data_script_dir = Path(__file__).resolve().parent
script_dir = data_script_dir.parent
repo_root = script_dir.parent

# Define the data directory relative to the repo root
parquet_data_dir = repo_root / "results" / "parquet_files"

# Initialize an empty list to store the sums
sums_list = []

# Number of files to process
num_files = 1000

# Loop over the files
for i in range(num_files):
    # Construct the filename
    filename = f"zone_simulation_result_{i}.parquet"
    parquet_file_path = parquet_data_dir / filename

    # Check if file exists
    if not parquet_file_path.exists():
        print(f"File {parquet_file_path} does not exist. Skipping.")
        continue

    # Read the Parquet file into a DataFrame
    df = pd.read_parquet(parquet_file_path)

    # Compute the sums of the two columns
    sum_with_water = df['zone_pop_with_water'].sum()
    sum_without_water = df['zone_pop_without_water'].sum()

    # Store the sums into the list
    sums_list.append({
        'run_id': i,
        'sum_zone_pop_with_water': sum_with_water,
        'sum_zone_pop_without_water': sum_without_water
    })

    # Remove df from memory
    del df

# Create a DataFrame from the list
sums_df = pd.DataFrame(sums_list)

# Save the sums DataFrame to a CSV file (optional)
sums_df.to_csv(parquet_data_dir / "sums_of_zone_populations.csv", index=False)

# Compute the median sums
median_with_water = sums_df['sum_zone_pop_with_water'].median()
median_without_water = sums_df['sum_zone_pop_without_water'].median()

print(f"Median sum of 'zone_pop_with_water' across runs: {median_with_water}")
print(f"Median sum of 'zone_pop_without_water' across runs: {median_without_water}")

# Identify the run closest to the median sum
sums_df['diff_with_water'] = (sums_df['sum_zone_pop_with_water'] - median_with_water).abs()
sums_df['diff_without_water'] = (sums_df['sum_zone_pop_without_water'] - median_without_water).abs()

# Find the run with minimal difference to the median
median_run_with_water = sums_df.loc[sums_df['diff_with_water'].idxmin()]
median_run_without_water = sums_df.loc[sums_df['diff_without_water'].idxmin()]

print("\nRun closest to median 'zone_pop_with_water':")
print(median_run_with_water)

print("\nRun closest to median 'zone_pop_without_water':")
print(median_run_without_water)


# Compute the 5th and 95th percentiles for 'zone_pop_with_water' and 'zone_pop_without_water'
percentile_5_with_water = sums_df['sum_zone_pop_with_water'].quantile(0.05)
percentile_95_with_water = sums_df['sum_zone_pop_with_water'].quantile(0.95)

percentile_5_without_water = sums_df['sum_zone_pop_without_water'].quantile(0.05)
percentile_95_without_water = sums_df['sum_zone_pop_without_water'].quantile(0.95)

# Find the run closest to the 5th and 95th percentiles for 'zone_pop_with_water'
sums_df['diff_to_5_with_water'] = (sums_df['sum_zone_pop_with_water'] - percentile_5_with_water).abs()
sums_df['diff_to_95_with_water'] = (sums_df['sum_zone_pop_with_water'] - percentile_95_with_water).abs()

run_closest_to_5_with_water = sums_df.loc[sums_df['diff_to_5_with_water'].idxmin()]
run_closest_to_95_with_water = sums_df.loc[sums_df['diff_to_95_with_water'].idxmin()]

# Find the run closest to the 5th and 95th percentiles for 'zone_pop_without_water'
sums_df['diff_to_5_without_water'] = (sums_df['sum_zone_pop_without_water'] - percentile_5_without_water).abs()
sums_df['diff_to_95_without_water'] = (sums_df['sum_zone_pop_without_water'] - percentile_95_without_water).abs()

run_closest_to_5_without_water = sums_df.loc[sums_df['diff_to_5_without_water'].idxmin()]
run_closest_to_95_without_water = sums_df.loc[sums_df['diff_to_95_without_water'].idxmin()]

# Print the results
print(f"Run closest to 5th percentile of 'zone_pop_with_water':")
print(run_closest_to_5_with_water)

print(f"\nRun closest to 95th percentile of 'zone_pop_with_water':")
print(run_closest_to_95_with_water)

print(f"\nRun closest to 5th percentile of 'zone_pop_without_water':")
print(run_closest_to_5_without_water)

print(f"\nRun closest to 95th percentile of 'zone_pop_without_water':")
print(run_closest_to_95_without_water)

# Optional: Save the runs closest to the 5th and 95th percentiles
percentile_runs = {
    'run_5th_with_water': run_closest_to_5_with_water['run_id'],
    'run_95th_with_water': run_closest_to_95_with_water['run_id'],
    'run_5th_without_water': run_closest_to_5_without_water['run_id'],
    'run_95th_without_water': run_closest_to_95_without_water['run_id']
}

# Convert to DataFrame and save
percentile_runs_df = pd.DataFrame([percentile_runs])
percentile_runs_df.to_csv(parquet_data_dir / "percentile_runs_of_zone_populations.csv", index=False)

print("\nPercentile run numbers have been saved to 'percentile_runs_of_zone_populations.csv'")
