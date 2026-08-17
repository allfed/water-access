# `scripts/extraction` — scripted raster extraction

This package replaces the undocumented manual QGIS workflow that produced
`data/GIS/updated_GIS_output_cleaned.csv`. It rebuilds that table — same
columns, same `fid` convention, same row-selection rules — from public source
rasters, at either **5 arc-minute** (validation) or **30 arc-second**
(production).

A cold reader should be able to go from a fresh checkout to
`data/GIS/extracted/master_30as.parquet` using only §1–§4 below.

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
`data/GIS/source_rasters/MANIFEST.md` carries the full verification lines;
this table is the download recipe.

| Layer | File | Source URL | Version | Licence |
|---|---|---|---|---|
| dtw (30as, **master grid**) | `distance2water_30arcsec.tif` | Dryad `doi:10.5061/dryad.71c6r` / Zenodo `4956396` | Kummu et al. 2011 | CC0 |
| dtw (5min) | `distance2water_5arcmin.tif` | same DOI | Kummu et al. 2011 | CC0 |
| Population density | `gpw_v4_density_2020_30sec.tif` | `https://data.ghg.center/sedac-popdensity-yeargrid5yr-v4.11/gpw_v4_population_density_rev11_2020_30_sec_2020.tif` | GPW v4.11, 2020, 30 arc-sec, **unadjusted** | CC BY 4.0 (CIESIN/SEDAC) |
| Slope (30as) | `slope_1KMmd_GMTEDmd.tif` | `https://data.earthenv.org/topography/slope_1KMmd_GMTEDmd.tif` | EarthEnv 1 km, **median** aggregation, GMTED median DEM | cite Amatulli et al. 2018 `doi:10.1038/sdata.2018.40` |
| Slope (5min) | `slope_10KMmd_GMTEDmd.tif` | `https://data.earthenv.org/topography/slope_10KMmd_GMTEDmd.tif` | EarthEnv 10 km (= 5 arcmin), median aggregation | as above |
| Degree of urbanisation | `GHS_SMOD_E2020_GLOBE_R2023A_54009_1000_V2_0.tif` | `https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/GHSL/GHS_SMOD_GLOBE_R2023A/GHS_SMOD_E2020_GLOBE_R2023A_54009_1000/V2-0/GHS_SMOD_E2020_GLOBE_R2023A_54009_1000_V2_0.zip` | R2023A V2-0, 1 km, Mollweide | EC reuse policy (CC BY 4.0 equivalent) |
| Road density | `grip4/grip4_tp{1..5}_dens_m_km2.asc` | `https://dataportaal.pbl.nl/downloads/GRIP4/GRIP4_density_tp{1..5}.zip` | GRIP4, 5 arcmin, m road / km² | CC0; cite Meijer et al. 2018 |
| Admin boundaries | `geoBoundariesCGAZ_ADM1.gpkg` | `https://github.com/wmgeolab/geoBoundaries/raw/main/releaseData/CGAZ/geoBoundariesCGAZ_ADM1.gpkg` | CGAZ ADM1 (3,224 polygons) | CC BY 4.0 |

Optional / cross-check only (not read by the pipeline):
`ghs_pop_2020_30ss.tif` (GHS-POP, alternative population layer),
`geoBoundariesCGAZ_ADM0.gpkg`, `GHS_SMOD_..._V1_0.tif`,
`slope_1KMmn_GMTEDmd.tif` (mean-aggregated slope; see §6), and
`gpw_v4_density_adjusted_2020_30sec.tif` (the **UN WPP-adjusted** GPW sibling —
a documented alternate selectable with `--pop-raster`; see §6 for why it is not
the default).

**GPW UN WPP-adjusted variant (alternate).** SEDAC's own host is dead; the
product now lives on NASA Earthdata and **requires a (free) Earthdata login**,
so it cannot be `curl`ed unauthenticated. Collection `C3540929692-ESDIS`,
`doi:10.7927/H4F47M65`, CC BY 4.0 (CIESIN/SEDAC). Download the `_asc` bundle
(8 ESRI-ASCII tiles, 182 MB zipped / 6.1 GB unpacked) from

```
https://data.earthdata.nasa.gov/nasa-earth/human-dimensions/sedac-root/downloads/data/gpw-v4/gpw-v4-population-density-adjusted-to-2015-unwpp-country-totals-rev11/gpw-v4-population-density-adjusted-to-2015-unwpp-country-totals-rev11_2020_30_sec_asc.zip
```

(the sibling `..._tif.zip` also exists) into `data/GIS/`, then mosaic the tiles
into one master-lattice GeoTIFF:

```bash
.venv/bin/python scripts/extraction/convert_gpw_asc.py \
    --zip data/GIS/gpw-v4-population-density-adjusted-to-2015-unwpp-country-totals-rev11_2020_30_sec_asc.zip \
    --out data/GIS/source_rasters/gpw_v4_density_adjusted_2020_30sec.tif
```

`--zip` unpacks to a scratch directory and deletes each `.asc` as soon as it has
been written, so it needs ~6.1 GB of transient free space. The script verifies
every tile header against the reference lattice rather than assuming the 4×2
layout, refuses to stamp the reference transform if any tile is more than
`--max-origin-drift` (1e-6°) off it, re-reads every written block to prove the
DEFLATE round-trip is bit-exact, and finishes with a country-ratio sanity check
against the published UN WPP adjustment factors. Runtime ~45 s after
extraction; output 146 MB, tiled 432×432, DEFLATE, float32, nodata −9999,
43200×21600 on exactly the reference GPW transform.

```bash
cd data/GIS/source_rasters
# population, slope (both resolutions)
curl -LO https://data.ghg.center/sedac-popdensity-yeargrid5yr-v4.11/gpw_v4_population_density_rev11_2020_30_sec_2020.tif
mv gpw_v4_population_density_rev11_2020_30_sec_2020.tif gpw_v4_density_2020_30sec.tif
curl -LO https://data.earthenv.org/topography/slope_1KMmd_GMTEDmd.tif
curl -LO https://data.earthenv.org/topography/slope_10KMmd_GMTEDmd.tif
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
#   and place as distance2water_30arcsec.tif and distance2water_5arcmin.tif
```

The only manual step is the Kummu distance-to-water rasters — Dryad serves them
behind an interactive download page. Everything else is a plain `curl`.

## 3. Regenerate everything (one sequence)

Run from the repository root.

```bash
# 5 arc-minute table + legacy CSV, then Gate 1 against the historical artefact
.venv/bin/python scripts/extraction/extract_grid.py \
    --resolution 5min --format csv \
    --out data/GIS/extracted/master_5min.parquet

.venv/bin/python scripts/extraction/validate_extraction.py \
    --new data/GIS/extracted/master_5min.parquet \
    --old data/GIS/updated_GIS_output_cleaned.csv \
    --report results/validation/gate1_report.md

# 30 arc-second production table, then structural self-checks
.venv/bin/python scripts/extraction/extract_grid.py \
    --resolution 30as \
    --out data/GIS/extracted/master_30as.parquet

.venv/bin/python scripts/extraction/check_consistency.py \
    --table data/GIS/extracted/master_30as.parquet --resolution 30as
```

All four commands exit 0 on success. `data/GIS/extracted/` is gitignored —
these outputs are data, never committed.

## 4. Master grid 

The master grid is the **Kummu distance-to-water lattice**, read verbatim off
the GeoTIFF rather than reconstructed from constants:

| | 30 arc-second | 5 arc-minute |
|---|---|---|
| size | 43200 × 21600 | 4320 × 2160 |
| cell | 0.00833333333° | 0.08333333330° |
| origin (x, y) | −179.999999007, +90.002416350 | identical |
| CRS | EPSG:4326 | EPSG:4326 |

The origin is **not** (−180, +90) — it is offset by ~0.03 of a 5-arcmin cell in
latitude. Every other raster is therefore slightly off-lattice and is read
through `rasterio.warp.reproject` onto the block affine (nearest-neighbour for
categorical and for layers already at or below the target resolution), never by
raw array indexing. `dtw_1` alone is read natively and is bit-exact.

**`fid` = `col * grid_height + row + 1`** — column-major, 1-based, row varying
fastest. This is the legacy QGIS convention; it reproduces all 2,190,446 legacy
`fid` values exactly, whereas the "obvious" row-major `row * ncols + col`
matches none of them.

`left/top/right/bottom` are the cell bounds in degrees and are load-bearing:
the resolution-aware population filter  infers the cell size from
`right − left`. The cell size is also written into the Parquet footer as
`cell_size_degrees`, along with the full source-file provenance, so a consumer
can skip the four float64 coordinate columns (~37% of the in-memory footprint
at 30as) and still know the resolution:

```python
import pyarrow.parquet as pq
pq.ParquetFile(path).schema_arrow.metadata  # resolution, cell size, sources, date
```

### Row selection

1. drop cells whose centre falls in no CGAZ ADM1 polygon (legacy
   `dropna(subset=["shapeGroup"])`);
2. drop cells with nodata in **any** of `dtw_1`, `pop_density`, `GHS_SMOD`,
   `slope_1` (legacy `drop_nan_rows`). GRIP nodata is *not* a drop reason — the
   legacy CSV keeps 7,675 such rows and so does this pipeline.

Because step 2 runs before writing, the `*_nan` indicator columns are all-zero
in the output — exactly as in the legacy CSV. `--keep-nan-rows` disables step 2
and makes them informative.

### Layer sources and resampling 

| Column | Source | Onto the master grid |
|---|---|---|
| `dtw_1` | Kummu, the grid-defining raster | native windowed read, bit-exact |
| `pop_density` | GPW v4.11 density 30as | 30as: grids coincide, nearest. 5min: `average` (conserves totals; `--pop-agg` also offers `nearest`/`bilinear`) |
| `slope_1` | EarthEnv GMTED, median aggregation | native at both resolutions (1 km file at 30as, 10 km file at 5min), nearest |
| `GHS_SMOD` | GHS-SMOD R2023A 1 km Mollweide | true reprojection, **nearest only** (categorical) |
| `grip_1_1` … `grip_5_1` | GRIP4 tp5 … tp1, 5 arcmin | exact `//10` broadcast at 30as; information floor stays 5 arcmin |
| `shapeName`, `shapeID`, `ISOCODE` | CGAZ ADM1 (`shapeGroup` → ISOCODE) | rasterised to int32 per block, cell centres |

Note the **deliberate GRIP inversion**: `grip_1_1` is tp**5** (local roads) and
`grip_5_1` is tp**1** (highways), matching
`scripts/Data_Manipulation_Scripts/pre_process_GIS_data.py:63-73`. Verified
against the legacy CSV.

CRS traps handled in code: GRIP `.asc` ships `crs=None` (EPSG:4326 asserted per
its ReadMe); CGAZ declares an "Undefined geographic SRS"
(`set_crs(4326, allow_override=True)`); GHS-SMOD is the only Mollweide layer.

## 5. Performance actuals

Measured on a MacBook Air M4 / 32 GB, `/usr/bin/time -l`, warm page cache.
Peak RSS scales with `--block-rows`, not with grid size — the full
43200 × 21600 stack is never materialised.

### 5 arc-minute (measured, full run)

| | |
|---|---|
| command | `extract_grid.py --resolution 5min --out …/master_5min.parquet` |
| block rows | 540 (default) → 4 blocks |
| wall clock | **13.5 s** (28.5 s with `--format csv`) |
| peak RSS | **2.92 GB** (3.24 GB with `--format csv`) |
| rows written | 2,191,074 |
| output size | 35.8 MB Parquet (411 MB legacy CSV) |
| Gate 1 | `validate_extraction.py` **5.5 s**, **4.88 GB** peak (loads both tables) |
| self-checks | `check_consistency.py` ~25 s, <2 GB |

### 30 arc-second (measured, full run — Phase D, 2026-08-08)

| | |
|---|---|
| command | `extract_grid.py --resolution 30as --out data/GIS/extracted/master_30as.parquet` |
| block rows | 480 (default) → 45 blocks |
| wall clock | **399.7 s** (6 min 40 s) |
| peak RSS | **2.31 GB** (`maximum resident set size` = 2,485,321,728 B; `peak memory footprint` 6.24 GB) |
| rows written | **218,986,194** |
| output size | **3.36 GB** Parquet (3,521,298,754 B — 16.1 bytes/row on disk, ~87 bytes/row in pandas) |
| free disk | 17 GB before → 13 GB after |
| self-checks | `check_consistency.py` **11.4 s**, 2.34 GB peak — **8/8 passed** |

Row accounting for the full run: 933,120,000 cells scanned, 627,610,021 dropped
for falling in no ADM1 polygon, 86,523,785 dropped for nodata in any of
`dtw_1` / `pop_density` / `GHS_SMOD` / `slope_1` (per-layer nodata counts:
dtw_1 25,460,475 · pop_density 86,121,286 · GHS_SMOD 4,778,564 · slope_1
85,760,167 — they overlap heavily), leaving 218,986,194 written. These are raw
land cells; the resolution-aware population filter is applied downstream by the model, not
here.

Both projections made from the earlier 18-of-45-block smoke were conservative:
it extrapolated ~3 min / ~5.5 GB / 250–300M rows / 4–5 GB Parquet, against
actuals of 6 min 40 s / **2.31 GB** / 219M rows / 3.36 GB. The row count came in
under because the smoke covered latitudes +90° → +18°, the land-densest band, so
per-block row counts fell off in the remaining 27 blocks. The peak-RSS gap
(5.49 GB smoke vs 2.31 GB full run, same accounting, same `--block-rows`) is
**not explained** — it is favourable and was not chased further; treat 5.5 GB as
the safe planning number rather than 2.3 GB.

**Disk:** budget ~3.5 GB free for the 30as Parquet, and check `df -h` first.
The volume this was measured on had 17 GB free and finished with 13 GB.

**Reading the 30as table without blowing the memory budget:** the file is
written with bounded row groups (`--row-group-rows`, default 2,000,000), so it
streams:

```python
import pyarrow.parquet as pq
pf = pq.ParquetFile("data/GIS/extracted/master_30as.parquet")
for g in range(pf.num_row_groups):
    chunk = pf.read_row_group(g, columns=["fid", "ISOCODE", "dtw_1", "slope_1"]).to_pandas()
```

At ~87 bytes/row a full `pd.read_parquet` of the 30as table's 218,986,194 rows
is ~19 GB — use a column subset or row groups. The model itself never does this:
`src/chunked_analysis.py` runs a draw one group of whole countries at a time, so
its peak follows the chunk budget rather than the grid size .

### Monte Carlo runs off these tables (measured, Phase D)

Ten-iteration smoke runs, same machine, same `--seed 20260808`, `--workers 1` on
both sides. These are the chunked-vs-monolithic benchmark numbers; the full comparison is
`results/validation/smoke_comparison.md`.

| | 30 arc-second | 5 arc-minute |
|---|---|---|
| command | `run_monte_carlo.py --resolution 30as --iterations 10 --input …/master_30as.parquet` | same with `--resolution 5min --input …/master_5min.parquet` |
| modelled cells | 137,930,520 | 1,419,773 |
| chunk plan | 12 chunks, 12,000,000-cell budget, RUS split by district | not chunked |
| chunk planning | 19.4 min (one-off per run, streams 3 columns of the whole table) | n/a |
| simulating | **80.7 min** for 10 draws (~5-8 min/draw, 12 chunks at ~20-43 s each) | 3.0 min |
| total wall clock | **100.1 min** | 3.0 min |
| peak RSS | **16.43 GB** (`maximum resident set size` 17,639,522,304 B) | 3.19 GB |

Peak RSS was independently corroborated by sampling the summed RSS of the whole
Python process tree every 15 s during the run (16.19 GB peak,
`results/validation/smoke30_rss_samples.csv`) — worth doing because
`/usr/bin/time -l` reports `getrusage`'s max-over-children, not parent+worker.
Note that the same `/usr/bin/time -l` output reports a `peak memory footprint` of
42.3 GB; that counter accumulates across the run rather than reporting a
simultaneous high-water mark, and it is contradicted by both the RSS figure and
the sampler on a 32 GB machine that never swapped. **`maximum resident set size`
is the figure the chunking validation asks for and the one to trust.**

Against the chunked memory model (peak GB ≈ 1.0 + 1.2 × millions of cells in the
largest chunk), a 12M-cell budget projected ~15 GB and measured 16.4 GB — the
model is good to ~10% and the 24 GB budget held with room to spare. Lower
`--chunk-rows` if a machine has less headroom; raising it buys little, since the
solver LUT is cached across chunks.

A 10-iteration run at 30 arc-second is therefore a ~1.7 h job, not the 4-7 h
originally projected, so the 1000-iteration production run extrapolates to
roughly 5-6 days single-threaded — which is what the GCP driver is for.

## 6. Deviations from the layer-extraction spec, and why

Two source choices differ from the letter of the layer spec. Both are evidence-driven
and both are recorded in the Gate 1 report.

**Slope — `md` (median) not `mn` (mean) aggregation.** The layer spec named
`slope_1KMmn_GMTEDmd.tif`. Testing against the legacy CSV showed its `slope_1`
is **bit-exact** `slope_10KMmd_GMTEDmd.tif` (EarthEnv's natively 5-arcmin,
*median*-aggregated product): 100.00% exact match on 5,000 random cells, and
100.0000% agreement over all 2,190,444 joined cells in the full Gate 1 run. The
`mn` variant is ~15% higher globally (mean ratio 1.155, corr 0.990). The 30
arc-second analogue of the legacy layer is therefore `slope_1KMmd_GMTEDmd.tif`,
which is what the pipeline uses. `slope_1KMmn_GMTEDmd.tif` remains on disk and
is still selectable (`--slope-raster`, and it is the automatic fallback if the
`md` file is absent), so the mean-slope configuration still runs.

**Population — GPW adjusted vs unadjusted (hypothesis tested and refuted,
2026-08-09).** The pipeline uses the **unadjusted** GPW v4.11 30as raster named
in the layer spec, and Gate 1 carries a wildcard `pop_total_pct` exception for it.

Phase A hypothesised that the residual existed because the legacy manual-QGIS
artefact had used the **UN WPP-adjusted** GPW variant while the pipeline used
the unadjusted one. The adjusted product has since been obtained (see §2) and
the hypothesis is **refuted** — the legacy layer is the *unadjusted* product:

| evidence (from `validate_extraction.py`, 5 arcmin, 160 gated countries) | unadjusted (default) | adjusted |
|---|---|---|
| cell-level `pop_density` agreement, `--pop-agg average` | **61.12%** | 27.22% |
| cell-level agreement, `--pop-agg nearest` | **73.24%** | 28.47% |
| countries within Gate 1's 1% on `pop_total_pct` | **45 / 160** | 15 / 160 |
| median \|`pop_total_pct`\| | **1.94%** | 6.22% |
| worst \|`pop_total_pct`\| | **26.29% (LBN)** | 56.74% (LBN) |
| uncovered Gate 1 failures | **0** | 14 |

Three quarters of the legacy cells are *bit-equal* to the unadjusted raster and
therefore cannot be the adjusted one; swapping triples the median per-country
error and breaks the gate. The three ratios previously cited as "published
adjustment factors" (NPL 0.966, GBR 1.014, NGA 0.995) were simply the observed
legacy/unadjusted ratios — the true GPW adjustment factors for those countries
are 0.739, 0.987 and 0.954, which do not match.

What the residual actually is: the undocumented QGIS resampling of a highly
skewed density field onto the coarse 5-arcmin lattice, where each output cell
covers 100 30-arcsec source cells. Measured median / worst \|`pop_total_pct`\|
by resampling: `average` 1.94% / 26.29%, `bilinear` 2.02% / 29.03%, `nearest`
3.15% / 94.73%. `average` is the closest reconstruction and is the default.

Either way the choice is **downstream-neutral**: the two products differ by a
per-country scalar only (verified — the ratio is constant to 4 dp across a
country's cells, e.g. 189,032 of Nepal's ~190k cells share one value), and
`merge_and_adjust_population` (`src/gis_global_module.py:323-327`) renormalises
`pop_density` to a within-country share before multiplying by an external
country population, cancelling any such scalar exactly. Global
population-weighted difference is −0.26%.

The adjusted raster is kept on disk as a documented alternate. To use it:

```bash
.venv/bin/python scripts/extraction/extract_grid.py --resolution 5min \
    --pop-raster data/GIS/source_rasters/gpw_v4_density_adjusted_2020_30sec.tif \
    --out data/GIS/extracted/master_5min.parquet
```

or flip `layers.GPW_DENSITY` from `GPW_DENSITY_UNADJUSTED` to
`GPW_DENSITY_ADJUSTED`. **Doing so currently fails Gate 1** (14 uncovered
failures), so it needs a deliberate decision and a new exception list, not a
silent switch.

## 7. Files

| File | Role |
|---|---|
| `grid.py` | master-grid definition, `fid` encoding, cell bounds, row blocking |
| `layers.py` | one reader per source layer; all resampling and CRS handling |
| `extract_grid.py` | CLI — builds the Parquet (and optional legacy CSV) |
| `validate_extraction.py` | CLI — Gate 1 (Gate 1), writes `results/validation/gate1_report.md` |
| `check_consistency.py` | CLI — structural self-checks that need no legacy artefact; the only validation available at 30as |
| `gate1_exceptions.json` | the approved, magnitude-capped Gate 1 exception list  |
