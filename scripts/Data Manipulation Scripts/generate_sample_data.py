import pandas as pd

# print directory
import os
print(os.getcwd())

# Step 1: Load the datasets
gis_data = pd.read_csv("./data/GIS/GIS_data_zones.csv")
country_data = pd.read_csv("./data/processed/country_data_master_interpolated.csv")

# Step 2: Identify common countries
common_countries = set(gis_data['ISOCODE']).intersection(set(country_data['alpha3']))

# Step 3: Sample rows for each country
sample_size_per_country = 10  # Adjust sample size as needed

gis_sample = pd.concat([gis_data[gis_data['ISOCODE'] == country].sample(n=min(len(gis_data[gis_data['ISOCODE'] == country]), sample_size_per_country), replace=True) for country in common_countries])
country_sample = pd.concat([country_data[country_data['alpha3'] == country].sample(n=min(len(country_data[country_data['alpha3'] == country]), sample_size_per_country), replace=True) for country in common_countries])

# Step 4: Save the sampled data
gis_sample.to_csv("./data/GIS/GIS_data_zones_sample.csv", index=False)
country_sample.to_csv("./data/processed/country_data_master_interpolated_sample.csv", index=False)