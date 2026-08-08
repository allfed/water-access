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
`slope_1KMmn_GMTEDmd.tif` (mean-aggregated slope; see §6).

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

## 4. Master grid (ADR-1)

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
the resolution-aware population filter (ADR-8) infers the cell size from
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

### Layer sources and resampling (ADR-3)

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
land cells; the ADR-8 population filter is applied downstream by the model, not
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
its peak follows the chunk budget rather than the grid size (ADR-5).

## 6. Deviations from SPEC ADR-3, and why

Two source choices differ from the letter of ADR-3. Both are evidence-driven
and both are recorded in the Gate 1 report.

**Slope — `md` (median) not `mn` (mean) aggregation.** ADR-3 named
`slope_1KMmn_GMTEDmd.tif`. Testing against the legacy CSV showed its `slope_1`
is **bit-exact** `slope_10KMmd_GMTEDmd.tif` (EarthEnv's natively 5-arcmin,
*median*-aggregated product): 100.00% exact match on 5,000 random cells, and
100.0000% agreement over all 2,190,444 joined cells in the full Gate 1 run. The
`mn` variant is ~15% higher globally (mean ratio 1.155, corr 0.990). The 30
arc-second analogue of the legacy layer is therefore `slope_1KMmd_GMTEDmd.tif`,
which is what the pipeline uses. `slope_1KMmn_GMTEDmd.tif` remains on disk and
is still selectable (`--slope-raster`, and it is the automatic fallback if the
`md` file is absent), so the ADR-3 configuration still runs.

**Population — unadjusted GPW, not UN WPP-adjusted.** The legacy `pop_density`
is GPW v4.11 **UN WPP-adjusted** density: per-country ratios reproduce the
published adjustment factors (NPL 0.966, GBR 1.014, NGA 0.995) and the extremes
are the classic high-migration cases (LBN −26%, QAT +24%). The adjusted product
sits behind an Earthdata login and was not obtained, so the pipeline uses the
unadjusted 30as raster named in ADR-3. This is immaterial downstream:
`merge_and_adjust_population` (`src/gis_global_module.py:323-327`) renormalises
`pop_density` to a within-country share and multiplies by an external country
population, which cancels any uniform per-country scale factor exactly. Global
population-weighted difference is −0.26%. If the adjusted product is ever
obtained, point `layers.GPW_DENSITY` at it — nothing else needs to change.

## 7. Files

| File | Role |
|---|---|
| `grid.py` | master-grid definition, `fid` encoding, cell bounds, row blocking |
| `layers.py` | one reader per source layer; all resampling and CRS handling |
| `extract_grid.py` | CLI — builds the Parquet (and optional legacy CSV) |
| `validate_extraction.py` | CLI — Gate 1 (AC-3), writes `results/validation/gate1_report.md` |
| `check_consistency.py` | CLI — structural self-checks that need no legacy artefact; the only validation available at 30as |
| `gate1_exceptions.json` | the approved, magnitude-capped Gate 1 exception list (ADR-9) |
