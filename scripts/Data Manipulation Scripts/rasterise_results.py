import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import rasterio
from rasterio.features import rasterize
from rasterio.transform import from_origin
import numpy as np

# Load your data into a DataFrame
filename = "../../results/GIS_merged_output_processed_with_centroids_no_nans.csv"
df = pd.read_csv(filename)

# Create a GeoDataFrame with the coordinates and specify the initial CRS (WGS84)
gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.longitude_centroid, df.latitude_centroid), crs="EPSG:4326")

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

# Create a generator that yields the (geometry, value) pairs for rasterization
def geometry_value_pairs():
    for geom, value in zip(gdf.geometry, gdf['zone_pop_without_water']):
        yield geom, value

# Rasterize the GeoDataFrame, filling cells that don't have data with NaN
raster = rasterize(geometry_value_pairs(), out_shape=(height, width), transform=transform, fill=np.nan)

# Save the raster to a GeoTIFF file
with rasterio.open(
    'output_raster_5_arcmin2.tif',
    'w',
    driver='GTiff',
    height=height,
    width=width,
    count=1,
    dtype=raster.dtype,
    crs=gdf.crs.to_string(),
    transform=transform,
    compress='deflate'  # Optional compression to save space
) as dst:
    dst.write(raster, 1)

print("Raster file 'output_raster_5_arcmin2.tif' created successfully.")
