# `scripts/extraction` — building the model input grid

These scripts build the model's 30 arc-second input table,
`data/GIS/extracted/master_30as.parquet`, from public source rasters.

---

## 1. Environment

Python 3.11+, with `rasterio`, `geopandas`, `pyarrow`, `pandas`, `numpy`.

```bash
python3.11 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/pip install -e .          # makes `src` importable
```

Verified on: Python 3.11.15, rasterio 1.4.4, geopandas 1.1.4, pyarrow 25.0.0,
pandas 2.3.3, numpy 2.4.6, macOS 15 (Apple M4, 32 GB).

Nothing in this package imports `src/`, so it also runs standalone.

## 2. Source data

Everything lands in `data/GIS/source_rasters/` (gitignored, ~3.9 GB).

| Layer | File | Source URL | Version | Licence |
|---|---|---|---|---|
| Distance to water (**master grid**) | `distance2water_30arcsec.tif` | Dryad `doi:10.5061/dryad.71c6r` / Zenodo `4956396` | Kummu et al. 2011 | CC0 |
| Population density | `gpw_v4_density_2020_30sec.tif` | `https://data.ghg.center/sedac-popdensity-yeargrid5yr-v4.11/gpw_v4_population_density_rev11_2020_30_sec_2020.tif` | GPW v4.11, 2020, 30 arc-sec, unadjusted | CC BY 4.0 (CIESIN/SEDAC) |
| Slope | `slope_1KMmd_GMTEDmd.tif` | `https://data.earthenv.org/topography/slope_1KMmd_GMTEDmd.tif` | EarthEnv 1 km, median aggregation, GMTED median DEM | cite Amatulli et al. 2018 `doi:10.1038/sdata.2018.40` |
| Degree of urbanisation | `GHS_SMOD_E2020_GLOBE_R2023A_54009_1000_V2_0.tif` | `https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/GHSL/GHS_SMOD_GLOBE_R2023A/GHS_SMOD_E2020_GLOBE_R2023A_54009_1000/V2-0/GHS_SMOD_E2020_GLOBE_R2023A_54009_1000_V2_0.zip` | R2023A V2-0, 1 km, Mollweide | EC reuse policy (CC BY 4.0 equivalent) |
| Road density | `grip4/grip4_tp{1..5}_dens_m_km2.asc` | `https://dataportaal.pbl.nl/downloads/GRIP4/GRIP4_density_tp{1..5}.zip` | GRIP4, 5 arcmin, m road / km² | CC0; cite Meijer et al. 2018 |
| Admin boundaries | `geoBoundariesCGAZ_ADM1.gpkg` | `https://github.com/wmgeolab/geoBoundaries/raw/main/releaseData/CGAZ/geoBoundariesCGAZ_ADM1.gpkg` | CGAZ ADM1 (3,224 polygons) | CC BY 4.0 |

```bash
cd data/GIS/source_rasters
# population, slope
curl -LO https://data.ghg.center/sedac-popdensity-yeargrid5yr-v4.11/gpw_v4_population_density_rev11_2020_30_sec_2020.tif
mv gpw_v4_population_density_rev11_2020_30_sec_2020.tif gpw_v4_density_2020_30sec.tif
curl -LO https://data.earthenv.org/topography/slope_1KMmd_GMTEDmd.tif
# degree of urbanisation
curl -LO https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/GHSL/GHS_SMOD_GLOBE_R2023A/GHS_SMOD_E2020_GLOBE_R2023A_54009_1000/V2-0/GHS_SMOD_E2020_GLOBE_R2023A_54009_1000_V2_0.zip
unzip -o -j GHS_SMOD_E2020_GLOBE_R2023A_54009_1000_V2_0.zip '*.tif' -d .
# roads
mkdir -p grip4 && cd grip4
for t in 1 2 3 4 5; do curl -LO "https://dataportaal.pbl.nl/downloads/GRIP4/GRIP4_density_tp${t}.zip" && unzip -o -j "GRIP4_density_tp${t}.zip"; done
cd ..
# admin boundaries
curl -Lo geoBoundariesCGAZ_ADM1.gpkg https://github.com/wmgeolab/geoBoundaries/raw/main/releaseData/CGAZ/geoBoundariesCGAZ_ADM1.gpkg
# distance to water (Kummu et al. 2011) — manual step:
#   download from https://doi.org/10.5061/dryad.71c6r (or Zenodo record 4956396)
#   and place as distance2water_30arcsec.tif
```

The only manual step is the Kummu distance-to-water raster — Dryad serves it
behind an interactive download page. Everything else is a plain `curl`.

Population uses the unadjusted GPW product. The model only uses
`pop_density` to share each country's population between its cells
(`merge_and_adjust_population` in `src/gis_global_module.py`), and national
totals come from the country table, so a per-country scaling of the density
layer has no effect on results.

## 3. Build the table

Run from the repository root.

```bash
.venv/bin/python scripts/extraction/extract_grid.py \
    --out data/GIS/extracted/master_30as.parquet
```

`data/GIS/extracted/` is gitignored — the output is data, never committed.
Budget ~3.5 GB of free disk for it.

## 4. Master grid

The master grid is the **Kummu distance-to-water lattice**, read verbatim off
the GeoTIFF rather than reconstructed from constants:

| | |
|---|---|
| size | 43200 × 21600 |
| cell | 0.00833333333° |
| origin (x, y) | −179.999999007, +90.002416350 |
| CRS | EPSG:4326 |

The origin is **not** (−180, +90). Every other raster is therefore slightly
off-lattice and is read through `rasterio.warp.reproject` onto the block affine
(nearest-neighbour), never by raw array indexing. `dtw_1` alone is read
natively and is bit-exact.

**`fid` = `col * grid_height + row + 1`** — column-major, 1-based, row varying
fastest.

`left/top/right/bottom` are the cell bounds in degrees. The model uses
`top/bottom` for the cos-latitude population weighting, and infers the cell
size for the population filter from `right − left`. The cell size is also
written into the Parquet footer as `cell_size_degrees`, along with the full
source-file provenance:

```python
import pyarrow.parquet as pq
pq.ParquetFile(path).schema_arrow.metadata  # resolution, cell size, sources, date
```

### Row selection

1. drop cells whose centre falls in no CGAZ ADM1 polygon;
2. drop cells with nodata in **any** of `dtw_1`, `pop_density`, `GHS_SMOD`,
   `slope_1`. GRIP nodata is *not* a drop reason.

Because step 2 runs before writing, the `*_nan` indicator columns are all-zero
in the output; they are kept because they are part of the model's input
columns.

### Layer sources and resampling

| Column | Source | Onto the master grid |
|---|---|---|
| `dtw_1` | Kummu, the grid-defining raster | native windowed read, bit-exact |
| `pop_density` | GPW v4.11 density 30as | nearest |
| `slope_1` | EarthEnv GMTED 1 km, median aggregation | nearest |
| `GHS_SMOD` | GHS-SMOD R2023A 1 km Mollweide | true reprojection, **nearest only** (categorical) |
| `grip_1_1` … `grip_5_1` | GRIP4 tp5 … tp1, 5 arcmin | each 5 arcmin cell is broadcast to its 10 × 10 block of 30as cells |
| `shapeName`, `shapeID`, `ISOCODE` | CGAZ ADM1 (`shapeGroup` → ISOCODE) | rasterised to int32 per block, cell centres |

The road columns run from local roads to highways: `grip_1_1` is tp**5**
(local roads) and `grip_5_1` is tp**1** (highways).

CRS traps handled in code: GRIP `.asc` ships `crs=None` (EPSG:4326 asserted per
its ReadMe); CGAZ declares an "Undefined geographic SRS"
(`set_crs(4326, allow_override=True)`); GHS-SMOD is the only Mollweide layer.

## 5. Performance

Measured on a MacBook Air M4 / 32 GB, `/usr/bin/time -l`, warm page cache.
Peak memory scales with `--block-rows`, not with grid size — the full
43200 × 21600 stack is never materialised.

| | |
|---|---|
| block rows | 480 (default) → 45 blocks |
| wall clock | 6 min 40 s |
| peak RSS | 2.3 GB (plan for up to 5.5 GB) |
| rows written | 218,986,194 |
| output size | 3.36 GB Parquet (~87 bytes/row in pandas) |

Row accounting: 933,120,000 cells scanned, 627,610,021 dropped for falling in
no ADM1 polygon, 86,523,785 dropped for nodata in any of `dtw_1` /
`pop_density` / `GHS_SMOD` / `slope_1`, leaving 218,986,194 land cells. The
population filter is applied later by the model, not here.

**Reading the table without running out of memory:** a full `pd.read_parquet`
is ~19 GB. The file is written with bounded row groups (`--row-group-rows`,
default 2,000,000), so read a column subset or one row group at a time:

```python
import pyarrow.parquet as pq
pf = pq.ParquetFile("data/GIS/extracted/master_30as.parquet")
for g in range(pf.num_row_groups):
    chunk = pf.read_row_group(g, columns=["fid", "ISOCODE", "dtw_1", "slope_1"]).to_pandas()
```

The model itself never loads the whole table: `src/chunked_analysis.py` runs a
draw one group of whole countries at a time, so its peak memory follows the
chunk budget rather than the grid size.

### Monte Carlo runs on this table

Measured on the same machine with `--workers 1`:

| | |
|---|---|
| modelled cells (after the population filter) | 137,930,520 |
| chunk plan | 12 chunks, 12,000,000-cell budget, Russia split by district |
| chunk planning | ~19 min (once per run) |
| simulating | ~5–8 min per draw |
| peak RSS | 16.4 GB |

Peak memory is roughly 1.0 GB + 1.2 GB per million cells in the largest chunk.
Lower `--chunk-rows` if a machine has less headroom. A 1000-draw run is
therefore several days single-threaded, which is what the cloud driver
(`gcp/`) is for.

## 6. Files

| File | Role |
|---|---|
| `grid.py` | master-grid definition, `fid` encoding, cell bounds, row blocking |
| `layers.py` | one reader per source layer; all resampling and CRS handling |
| `extract_grid.py` | command-line script — builds the Parquet table |
