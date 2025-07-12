import pandas as pd

"""


"""

input_file = "../../data/GIS/updated_GIS_output.csv"
country_input_file = "../../data/processed/merged_data.csv"  # contains population data
output_file_path = "../../results/GIS_data_country_slice_for_results_pop.csv"


# read the data
df = pd.read_csv(input_file)

# Read the country data
df_country = pd.read_csv(country_input_file)

# Merge the dataframes
df = df.merge(
    df_country[["alpha3", "Population"]],
    left_on="shapeGroup",
    right_on="alpha3",
    how="inner",
)

# Calculate population density percentage
df["pop_density_perc"] = df.groupby("shapeGroup")["pop_density"].transform(
    lambda x: x / x.sum()
)

# Calculate population for each zone
df["pop_zone"] = df["pop_density_perc"] * df["Population"]

# Calculate total population for each country
df["country_pop_raw"] = df.groupby("shapeGroup")["pop_zone"].transform("sum")

# Filter out zones with very low population
df["any_pop"] = df["pop_zone"].apply(lambda x: 1 if x > 10 else 0)
df = df[df["any_pop"] == 1]


print(f"Processed {len(df)} zones with population data")

# Calculate the longitude and latitude centroids
df["longitude_centroid"] = (df["left"] + df["right"]) / 2
df["latitude_centroid"] = (df["top"] + df["bottom"]) / 2

# drop all columns except id, centroid lat and long, and country pop

# Save the updated DataFrame to a new CSV file
df.to_csv(output_file_path, index=False)

print(f"Data with centroids saved to {output_file_path}")
