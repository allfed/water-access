import pandas as pd
import os

# Print the current working directory (for debugging purposes)
print(os.getcwd())

# Define the input and output file paths
input_file = '../../data/GIS/GIS_stripped_back.csv'
output_file = '../../data/GIS/GIS_data_zones_sample_stripped.csv'

# Step 1: Load the dataset
gis_data = pd.read_csv(input_file)

# Step 2: Define the sample size per country
sample_size_per_country = 20  # Adjust sample size as needed

# Step 3: Sample the data
gis_sample = gis_data.groupby('ISOCODE', group_keys=False).apply(lambda x: x.sample(min(len(x), sample_size_per_country)))

# Step 4: Save the sampled data
gis_sample.to_csv(output_file, index=False)

print(f"Sampled data saved to {output_file}")
