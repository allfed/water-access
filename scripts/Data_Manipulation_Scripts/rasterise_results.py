"""Rasterise zone population results to a GeoTIFF.

Default grid is 30 arc-seconds. Reads the Parquet written by
``export_results_with_centroids.py`` (CSV is no longer used at 30as).
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize
from rasterio.transform import from_origin

repo_root = Path(__file__).resolve().parent.parent.parent
results_dir = repo_root / "results"
tif_dir = results_dir / "TIFs"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--label",
        default="median",
        choices=("median", "5th_percentile", "95th_percentile"),
        help="which exported centroids Parquet to rasterise",
    )
    p.add_argument(
        "--resolution",
        choices=("30as", "5min"),
        default="30as",
        help="output grid spacing (default 30as)",
    )
    p.add_argument(
        "--filled",
        action="store_true",
        help="use the *_no_nans.parquet export (missing cells as 0)",
    )
    return p.parse_args()


def resolution_degrees(name: str) -> float:
    if name == "30as":
        return 30 / 3600  # 0.008333...°
    if name == "5min":
        return 5 / 60
    raise ValueError(name)


def main() -> None:
    args = parse_args()
    start_time = time.time()

    suffix = "_no_nans" if args.filled else ""
    input_path = (
        results_dir
        / f"GIS_merged_output_processed_with_centroids_{args.label}{suffix}.parquet"
    )
    if not input_path.exists():
        raise SystemExit(
            f"Missing {input_path}. Run export_results_with_centroids.py "
            f"--label {args.label} first."
        )

    print(f"Reading {input_path} ...")
    df = pd.read_parquet(input_path)
    resolution = resolution_degrees(args.resolution)
    tif_dir.mkdir(parents=True, exist_ok=True)
    output_filename = tif_dir / (
        f"output_raster_{args.resolution}_{args.label}_partial_percentage.tif"
    )

    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df.longitude_centroid, df.latitude_centroid),
        crs="EPSG:4326",
    )

    bounds = gdf.total_bounds
    x_min = bounds[0] - (bounds[0] % resolution)
    x_max = bounds[2] + (resolution - bounds[2] % resolution)
    y_min = bounds[1] - (bounds[1] % resolution)
    y_max = bounds[3] + (resolution - bounds[3] % resolution)

    width = int(np.ceil((x_max - x_min) / resolution))
    height = int(np.ceil((y_max - y_min) / resolution))
    transform = from_origin(x_min, y_max, resolution, resolution)
    print(f"Grid {width} x {height} at {args.resolution} ({resolution}°)")

    denom = gdf["zone_pop_without_water"] + gdf["zone_pop_with_water"]
    gdf["percentage_with_water"] = np.where(
        denom > 0, gdf["zone_pop_with_water"] / denom, np.nan
    )

    variables_to_rasterize = [
        "zone_pop_with_water",
        "zone_pop_without_water",
        "percentage_with_water",
    ]
    raster_data = np.empty((len(variables_to_rasterize), height, width), dtype=np.float32)

    for i, variable in enumerate(variables_to_rasterize):
        print(f"Rasterizing {variable} ...")
        values = gdf[variable].to_numpy()
        geoms = gdf.geometry
        pairs = (
            (geom, float(val) if np.isfinite(val) else -9999.0)
            for geom, val in zip(geoms, values)
        )
        raster_data[i] = rasterize(
            pairs,
            out_shape=(height, width),
            transform=transform,
            fill=-9999.0,
            dtype="float32",
        )

    with rasterio.open(
        output_filename,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=len(variables_to_rasterize),
        dtype="float32",
        crs=gdf.crs.to_string(),
        transform=transform,
        compress="deflate",
        nodata=-9999.0,
    ) as dst:
        for i in range(len(variables_to_rasterize)):
            dst.write(raster_data[i], i + 1)
        dst.descriptions = tuple(variables_to_rasterize)

    print(f"Wrote {output_filename}")
    print(f"Time taken: {time.time() - start_time:.1f}s")


if __name__ == "__main__":
    main()
