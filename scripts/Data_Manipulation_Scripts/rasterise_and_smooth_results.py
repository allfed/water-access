import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import rasterio
from rasterio.features import rasterize
from rasterio.transform import from_origin
import numpy as np
import time
from scipy.ndimage import uniform_filter

# Start timing the script
start_time = time.time()

# Load your data into a DataFrame
# filename = "../../results/GIS_merged_output_processed_with_centroids_right.csv"
filename = "./results/GIS_merged_output_processed_with_centroids_right.csv"
df = pd.read_csv(filename)
output_filename = "./results/TIFs/output_raster_5_arcmin_smoothed.tif"

# Create a GeoDataFrame with the coordinates and specify the initial CRS (WGS84)
gdf = gpd.GeoDataFrame(
    df,
    geometry=gpd.points_from_xy(df.longitude_centroid, df.latitude_centroid),
    crs="EPSG:4326",
)

# Define the resolution: 5 arc-minutes is 5/60 degrees, which equals approximately 0.08333 degrees
resolution = 5 / 60  # 0.08333 degrees

# Calculate the raster bounds based on the extent of the data (minx, miny, maxx, maxy)
bounds = gdf.total_bounds  # (min_longitude, min_latitude, max_longitude, max_latitude)

# Ensure the bounds are aligned to the 5 arc-minute grid
x_min = bounds[0] - (bounds[0] % resolution)
x_max = bounds[2] + (resolution - bounds[2] % resolution)
y_min = bounds[1] - (bounds[1] % resolution)
y_max = bounds[3] + (resolution - bounds[3] % resolution)

# Calculate the number of rows and columns for the raster grid
width = int(np.ceil((x_max - x_min) / resolution))
height = int(np.ceil((y_max - y_min) / resolution))

# Define the transform for the raster (top-left corner and resolution)
transform = from_origin(x_min, y_max, resolution, resolution)

# Create percentage of people with water access
gdf["percentage_with_water"] = gdf["zone_pop_with_water"] / (
    gdf["zone_pop_without_water"] + gdf["zone_pop_with_water"]
)

# Define the variables to rasterize
variables_to_rasterize = [
    "zone_pop_with_water",
    "zone_pop_without_water",
    "percentage_with_water",
]
num_bands = len(variables_to_rasterize)

# Create an empty array to hold raster data
raster_data = np.empty((num_bands, height, width), dtype=np.float32)

# Rasterize each variable
for i, variable in enumerate(variables_to_rasterize):
    print(f"Rasterizing variable: {variable}")

    # Create a generator that yields the (geometry, value) pairs for rasterization
    def geometry_value_pairs():
        for geom, value in zip(gdf.geometry, gdf[variable]):
            yield geom, value

    # Rasterize the GeoDataFrame, filling cells that don't have data with np.nan
    raster = rasterize(
        geometry_value_pairs(),
        out_shape=(height, width),
        transform=transform,
        fill=np.nan,
        dtype="float32",
    )

    # Store the raster in the raster_data array
    raster_data[i, :, :] = raster

# Smoothing parameters
apply_second_smoothing = False  # Set to True to apply the second smoothing pass
first_window_size = 3  # Window size for the first smoothing pass
second_window_size = 5  # Window size for the second smoothing pass

# Apply the smoothing filter to each band
print("Applying first smoothing filter...")
for i in range(num_bands):
    print(f"First smoothing band {i+1}: {variables_to_rasterize[i]}")
    band_data = raster_data[i, :, :]

    # Create a mask of valid data (True where data is not NaN)
    valid_mask = ~np.isnan(band_data)

    # Replace NaNs with zeros for computation
    data_filled = np.nan_to_num(band_data, nan=0.0)

    # Apply uniform filter to the data and the mask
    smoothed_data = uniform_filter(
        data_filled, size=first_window_size, mode="constant", cval=0.0
    )
    smoothed_mask = uniform_filter(
        valid_mask.astype(float), size=first_window_size, mode="constant", cval=0.0
    )

    # Avoid division by zero
    with np.errstate(invalid="ignore", divide="ignore"):
        smoothed_band = smoothed_data / smoothed_mask
        smoothed_band[smoothed_mask == 0] = np.nan

    # Restore original NaN values
    smoothed_band[~valid_mask] = np.nan

    # Store the smoothed band back
    raster_data[i, :, :] = smoothed_band

# Apply second smoothing pass if enabled
if apply_second_smoothing:
    print("Applying second smoothing filter...")
    for i in range(num_bands):
        print(f"Second smoothing band {i+1}: {variables_to_rasterize[i]}")
        band_data = raster_data[i, :, :]

        # Create a mask of valid data (True where data is not NaN)
        valid_mask = ~np.isnan(band_data)

        # Replace NaNs with zeros for computation
        data_filled = np.nan_to_num(band_data, nan=0.0)

        # Apply uniform filter to the data and the mask
        smoothed_data = uniform_filter(
            data_filled, size=second_window_size, mode="constant", cval=0.0
        )
        smoothed_mask = uniform_filter(
            valid_mask.astype(float), size=second_window_size, mode="constant", cval=0.0
        )

        # Avoid division by zero
        with np.errstate(invalid="ignore", divide="ignore"):
            smoothed_band = smoothed_data / smoothed_mask
            smoothed_band[smoothed_mask == 0] = np.nan

        # Restore original NaN values
        smoothed_band[~valid_mask] = np.nan

        # Store the smoothed band back
        raster_data[i, :, :] = smoothed_band

# Save the raster to a GeoTIFF file
with rasterio.open(
    output_filename,
    "w",
    driver="GTiff",
    height=height,
    width=width,
    count=num_bands,
    dtype="float32",
    crs=gdf.crs.to_string(),
    transform=transform,
    compress="deflate",
    nodata=np.nan,  # Set nodata value to NaN
) as dst:
    for i in range(num_bands):
        dst.write(raster_data[i, :, :], i + 1)  # Bands are 1-indexed
    # Set band descriptions
    dst.descriptions = tuple(variables_to_rasterize)

print(f"Raster file '{output_filename}' created successfully.")

# End timing the script
end_time = time.time()
print(f"Time taken: {end_time - start_time} seconds")
