# Overview of how to re-run the full analysis.

### 1. Pre-process data in QGIS to generate updated_GIS_output_cleaned file
This is covered in the related publication (preprint link to be added here shortly). Please feel free to reach out to the ALLFED team for more information or assistance here.  <br />

### 2. Run `scripts/run_monte_carlo.ipynb`
The parameters currently selected are the ones used in the publication, but feel free to change these to fit your assumptions or context.  <br />

### 3. Run `scripts/key_results.ipynb`
This will output the most important findings in terms of who can access water at what resolutions and by walking/cycling/using non-piped sources.  <br />

### 4. Run postprocessing scripts to format data for QGIS visualisation
You can generate TIF files for making result maps by doing the following:
* Run`/scripts/Data_Manipulation_Scripts/parquet_process.py` to find the median/5th/95th runs
* Rename the identified run files to “zone_simulation_result_median.parquet”, “zone_simulation_result_5th_percentile.parquet”, and “zone_simulation_result_95th_percentile.parquet”
* Run `/scripts/Data_Manipulation_Scripts/export_results_with_centroids`
* Run `/scripts/Data_Manipulation_Scripts/rasterize_results`: this will provide unsmoothed results for mapping which are better for high-resolution analysis of results
* Run `/scripts/Data_Manipulation_Scripts/rasterize_and_smooth_results`: this will provide smoothed results which can look better for a global-scale map
* Put output into QGIS
* Reproject TIF files into chosen map project (Winkel Tripel was used for our publication)
* Update symbology as desired (we used singleband pseudocolor with Magma reversed as the colormap)
