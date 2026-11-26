# Reproducing the full analysis

Prerequisites: complete the [installation steps](../README.md) (conda environment, `pip install -e .`, Git LFS).

## 1. Build the 30 arc-second grid

The model runs on a 30 arc-second (~1 km) grid built from public source rasters (distance to water, population density, urban/rural classification, slope, roads, and ADM1 boundaries). Download the rasters and build `data/GIS/extracted/master_30as.parquet` following [`scripts/extraction/README.md`](../scripts/extraction/README.md):

```bash
python scripts/extraction/extract_grid.py --out data/GIS/extracted/master_30as.parquet
```

## 2. Run Monte Carlo simulations

From the repository root, either:

(a) Run a single model run with central parameter values:

```bash
python src/gis_global_module.py
```

(b) Run Monte Carlo simulations locally (this is very RAM intensive; one draw at a time needs about 16 GB):

```bash
python scripts/run_monte_carlo.py
```

(c) Run Monte Carlo simulations on Google Cloud. This will involve setting up a server instance; see [`gcp/gcp-setup.md`](../gcp/gcp-setup.md) for details or reach out to us.

```bash
./gcp/deploy-spot.sh deploy
./gcp/monitor.sh
```

The uncertainty ranges used in the publication are defined in [`src/monte_carlo_parameters.py`](../src/monte_carlo_parameters.py); adjust as needed. Outputs are written to `results/parquet_files/` (one country and one district parquet per draw), and summary CSVs are written to `results/` when aggregation completes. The analysis notebooks read the per-draw country parquets from `results/parquet_files/`.

## 3. Generate key results

Run [`scripts/key_results.ipynb`](../scripts/key_results.ipynb) and [`scripts/distribution_plots.ipynb`](../scripts/distribution_plots.ipynb) for country- and district-level tables and plots.

## 4. Post-process for QGIS maps

Generate GeoTIFFs for map visualization:

1. Run [`scripts/Data_Manipulation_Scripts/parquet_process.py`](../scripts/Data_Manipulation_Scripts/parquet_process.py) to select the median / 5th / 95th percentile draws and re-run them locally with per-cell output. It reads the draw parameters from `results/parquet_files/checkpoint.json`, which the cloud driver (`scripts/run_monte_carlo_gcp.py`) writes.

```bash
python scripts/Data_Manipulation_Scripts/parquet_process.py
# selection only: add --no-rerun-zones-locally
```

That writes `zone_simulation_result_{median,5th_percentile,95th_percentile}.parquet` (add `--force` to overwrite existing zone files).

2. Export centroids (from `master_30as.parquet`):

```bash
python scripts/Data_Manipulation_Scripts/export_results_with_centroids.py --label median
python scripts/Data_Manipulation_Scripts/export_results_with_centroids.py --label 5th_percentile
python scripts/Data_Manipulation_Scripts/export_results_with_centroids.py --label 95th_percentile
```

3. Rasterise:

```bash
python scripts/Data_Manipulation_Scripts/rasterise_results.py --label median
python scripts/Data_Manipulation_Scripts/rasterise_results.py --label 5th_percentile
python scripts/Data_Manipulation_Scripts/rasterise_results.py --label 95th_percentile
```

Outputs land in `results/TIFs/output_raster_30as_<label>_partial_percentage.tif`.

## 5. Visualize in QGIS (external)

1. Load the GeoTIFF from `results/TIFs/` into QGIS.
2. Reproject to your map CRS (Winkel Tripel was used for publication maps; use **ESRI:54042**, World Winkel Tripel NGS).
3. Style as singleband pseudocolor (publication maps used Magma reversed).
4. For the district-level maps, join `results/districts_median_results.csv` to a global ADM1 shapefile on `shapeID` and colour with Magma reversed.
