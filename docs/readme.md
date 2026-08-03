# Reproducing the full analysis

Prerequisites: complete the [installation steps](../README.md) (conda environment, `pip install -e .`, Git LFS).

## 1. Pre-process GIS data in QGIS (external)

Pre-process spatial inputs in QGIS to produce `data/GIS/updated_GIS_output_cleaned.csv`, following the methods outlined in the paper (link to be added after publication). Please feel free to reach out to the ALLFED team for more information or assistance here.

## 2. Run Monte Carlo simulations

From the repository root, either:

(a) Run a single model run:

```bash
python src/gis_global_module.py
```

(b) Run Monte Carlo simulations locally (this is very RAM intensive):
```bash
python scripts/run_monte_carlo.py
```

(c) Run Monte Carlo simulations on Google Cloud. This will involve setting up a server instance; see gcp/ for details or reach out to us.

```bash
./gcp/deploy-spot.sh deploy
./gcp/monitor.sh
```

Parameters at the top of the Monte Carlo script define the uncertainty ranges used in the publication; adjust as needed. Outputs are written to `results/parquet_files/` (one parquet per iteration for zones, districts, and countries). Summary CSVs and pickles are written to `results/` when aggregation completes.

## 3. Generate key results

Run [`scripts/key_results.ipynb`](../scripts/key_results.ipynb) and [`scripts/distribution_plots.ipynb`](scripts/distribution_plots.ipynb) for country- and district-level tables and plots.

## 4. Post-process for QGIS maps

Generate GeoTIFFs for map visualization:

1. Run [`scripts/Data_Manipulation_Scripts/parquet_process.py`](../scripts/Data_Manipulation_Scripts/parquet_process.py) to identify runs closest to the median, 5th, and 95th percentiles.
2. Rename the identified zone parquet files to:
   - `zone_simulation_result_median.parquet`
   - `zone_simulation_result_5th_percentile.parquet`
   - `zone_simulation_result_95th_percentile.parquet`
3. Run [`scripts/Data_Manipulation_Scripts/export_results_with_centroids.py`](../scripts/Data_Manipulation_Scripts/export_results_with_centroids.py)
4. Run one or both raster scripts:
   - [`scripts/Data_Manipulation_Scripts/rasterise_results.py`](../scripts/Data_Manipulation_Scripts/rasterise_results.py) — unsmoothed 5-arcminute grid (`results/TIFs/output_raster_5_arcmin_partial_percentage.tif`)
   - [`scripts/Data_Manipulation_Scripts/rasterise_and_smooth_results.py`](../scripts/Data_Manipulation_Scripts/rasterise_and_smooth_results.py) — smoothed variant for global-scale maps

## 5. Visualize in QGIS (external)

1. Load the GeoTIFF from `results/TIFs/` into QGIS.
2. Reproject to your map CRS (Winkel Tripel was used for publication maps; use **ESRI:54042**, World Winkel Tripel NGS).
3. Style as singleband pseudocolor (publication maps used Magma reversed).
4. For the district-level maps, join `results/districts_median_results.csv` to a global ADM1 shapefile on `shapeID` and colour with Magma reversed.