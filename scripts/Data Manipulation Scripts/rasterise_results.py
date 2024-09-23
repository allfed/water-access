import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import rasterio
from rasterio.features import rasterize
from rasterio.transform import from_origin
import numpy as np
import time

# Start timing the script
start_time = time.time()

# Load your data into a DataFrame
filename = "../../results/GIS_merged_output_processed_with_centroids_right.csv"
df = pd.read_csv(filename)
output_filename = "output_raster_5_arcmin_partial_percentage.tif"

# Create a GeoDataFrame with the coordinates and specify the initial CRS (WGS84)
gdf = gpd.GeoDataFrame(
    df,
    geometry=gpd.points_from_xy(df.longitude_centroid, df.latitude_centroid),
    crs="EPSG:4326"
)

# Define the resolution: 5 arc-minutes is 5/60 degrees, which equals 0.08333 degrees
resolution = 5 / 60  # 0.08333 degrees

# Calculate the raster bounds based on the extent of the data (minx, miny, maxx, maxy)
bounds = gdf.total_bounds  # (min_longitude, min_latitude, max_longitude, max_latitude)

# Ensure the bounds are aligned to the 5 arc-minute grid
x_min = bounds[0] - (bounds[0] % resolution)
x_max = bounds[2] + (resolution - bounds[2] % resolution)
y_min = bounds[1] - (bounds[1] % resolution)
y_max = bounds[3] + (resolution - bounds[3] % resolution)

# Calculate the number of rows and columns for the raster grid
width = int((x_max - x_min) / resolution)
height = int((y_max - y_min) / resolution)

# Define the transform for the raster (top-left corner and resolution)
transform = from_origin(x_min, y_max, resolution, resolution)

# Create percentage of people with water access
gdf['percentage_with_water'] = (
    gdf['zone_pop_with_water'] /
    (gdf['zone_pop_without_water'] + gdf['zone_pop_with_water'])
)

# Define the variables to rasterize
variables_to_rasterize = [
    'zone_pop_with_water',
    'zone_pop_without_water',
    'percentage_with_water'
]
num_bands = len(variables_to_rasterize)

# Create an empty array to hold raster data
raster_data = np.empty((num_bands, height, width), dtype=np.float32)

for i, variable in enumerate(variables_to_rasterize):
    print(f"Rasterizing variable: {variable}")

    # Create a generator that yields the (geometry, value) pairs for rasterization
    def geometry_value_pairs():
        for geom, value in zip(gdf.geometry, gdf[variable]):
            yield geom, value

    # Rasterize the GeoDataFrame, filling cells that don't have data with -9999.0
    raster = rasterize(
        geometry_value_pairs(),
        out_shape=(height, width),
        transform=transform,
        fill=-9999.0,
        dtype='float32'
    )

    # Store the raster in the raster_data array
    raster_data[i, :, :] = raster

# Save the raster to a GeoTIFF file
with rasterio.open(
    output_filename,
    'w',
    driver='GTiff',
    height=height,
    width=width,
    count=num_bands,
    dtype='float32',
    crs=gdf.crs.to_string(),
    transform=transform,
    compress='deflate',
    nodata=-9999.0  # Set nodata value
) as dst:
    for i in range(num_bands):
        dst.write(raster_data[i, :, :], i + 1)  # Bands are 1-indexed
    # Set band descriptions
    dst.descriptions = tuple(variables_to_rasterize)

print(f"Raster file '{output_filename}' created successfully.")

# End timing the script
end_time = time.time()
print(f"Time taken: {end_time - start_time} seconds")
