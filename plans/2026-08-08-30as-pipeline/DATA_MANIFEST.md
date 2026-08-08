# Source raster / vector manifest

Directory: `data/GIS/source_rasters/` (gitignored — nothing here is committed).

All downloads dated **2026-08-08**. Verification lines below were produced with
`.venv-phase0/bin/python` (rasterio for rasters, geopandas/pyogrio for GeoPackages).

---

## 1. Distance to surface water — 30 arc-second (pre-existing)

| | |
|---|---|
| **Filename** | `distance2water_30arcsec.tif` |
| **Source** | Kummu et al. 2011, Dryad `doi:10.5061/dryad.71c6r` / Zenodo `4956396` |
| **Downloaded** | pre-existing (2026-08-08, earlier session) |
| **Size on disk** | 260 MB |
| **Licence** | CC0 (public domain dedication) |
| **Verification** | `width=43200 height=21600 count=1 dtype=float32 crs=EPSG:4326 nodata=-9999.0 res=(0.00833333, 0.00833333)` |

This is the **master grid** for the 30as pipeline (SPEC ADR-1). Kummu computed at
1 km natively and aggregated to 5 arcmin "for computational reasons" — so the
5 arcmin file below is the derivative, not the source.

## 2. Distance to surface water — 5 arc-minute (pre-existing)

| | |
|---|---|
| **Filename** | `distance2water_5arcmin.tif` |
| **Source** | Kummu et al. 2011, Dryad `doi:10.5061/dryad.71c6r` / Zenodo `4956396` (same DOI as above) |
| **Downloaded** | pre-existing |
| **Size on disk** | 36 MB |
| **Licence** | CC0 |
| **Verification** | `width=4320 height=2160 count=1 dtype=float32 crs=EPSG:4326 nodata=-9999.0 res=(0.08333333, 0.08333333)` |

Exactly the 30as lattice ÷ 10. Used as the validation/regression grid.

## 3. GHS-POP R2023A 2020, 30 arc-second WGS84 (pre-existing)

| | |
|---|---|
| **Filename** | `GHS_POP_E2020_GLOBE_R2023A_4326_30ss_V1_0.tif` (symlink `ghs_pop_2020_30ss.tif`) |
| **Source** | JRC GHSL — `https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/GHSL/GHS_POP_GLOBE_R2023A/GHS_POP_E2020_GLOBE_R2023A_4326_30ss/V1-0/GHS_POP_E2020_GLOBE_R2023A_4326_30ss_V1_0.zip` |
| **Downloaded** | pre-existing (zip `ghs_pop_2020_30ss.zip`, 460 MB, retained) |
| **Size on disk** | 365 MB (+ 111 MB `.ovr`) |
| **Licence** | European Commission reuse policy — open and free, reuse authorised with source acknowledgement (CC BY 4.0 equivalent) |
| **Verification** | `width=43202 height=21384 count=1 dtype=float64 crs=EPSG:4326 nodata=None res=(0.00833333, 0.00833333)` |

> **Grid gotcha:** 43202 × 21384, **not** 43200 × 21600 — GHS-POP is 2 columns wider
> and 216 rows shorter than the Kummu master grid, and declares **no nodata value**.
> It must be windowed/aligned onto the Kummu lattice rather than assumed co-registered.

## 4. geoBoundaries CGAZ ADM0 (pre-existing)

| | |
|---|---|
| **Filename** | `geoBoundariesCGAZ_ADM0.gpkg` |
| **Source** | `https://github.com/wmgeolab/geoBoundaries/raw/main/releaseData/CGAZ/geoBoundariesCGAZ_ADM0.gpkg` |
| **Downloaded** | pre-existing |
| **Size on disk** | 155 MB |
| **Licence** | CC BY 4.0 (geoBoundaries open licence) |
| **Verification** | `layer=globalADM0 rows=218 columns=['id','shapeGroup','shapeType','shapeName','geometry']` |

## 5. GPW v4.11 population density 2020, 30 arc-second — **NEW**

| | |
|---|---|
| **Filename** | `gpw_v4_density_2020_30sec.tif` |
| **Source** | `https://data.ghg.center/sedac-popdensity-yeargrid5yr-v4.11/gpw_v4_population_density_rev11_2020_30_sec_2020.tif` (CIESIN/SEDAC GPW v4.11, mirrored by the US GHG Center) |
| **Downloaded** | 2026-08-08 |
| **Size on disk** | 407 MB |
| **Licence** | CC BY 4.0 (CIESIN, Columbia University / NASA SEDAC) |
| **Verification** | `width=43200 height=21600 count=1 dtype=float32 crs=EPSG:4326 nodata=-9999.0 res=(0.00833333, 0.00833333)`, bounds `(-180, -90, 180, 90)` |

Units: persons per km². **Grid-identical to the Kummu 30as master grid** — no
resampling needed. This is the ADR-3 alternative population layer.

## 6. EarthEnv 1 km slope (GMTED) — **NEW**

| | |
|---|---|
| **Filename** | `slope_1KMmn_GMTEDmd.tif` |
| **Source** | `https://data.earthenv.org/topography/slope_1KMmn_GMTEDmd.tif` (EarthEnv, https://www.earthenv.org/topography) |
| **Downloaded** | 2026-08-08 |
| **Size on disk** | 768 MB |
| **Licence** | No explicit licence on the EarthEnv page; cite Amatulli et al. 2018, *Scientific Data* 5:180040 (`doi:10.1038/sdata.2018.40`). Treat as CC BY-equivalent with attribution. |
| **Verification** | `width=43200 height=16800 count=1 dtype=float32 crs=EPSG:4326 nodata=-32768.0 res=(0.00833333, 0.00833333)`, bounds `(-180, -56, 180, 84)`; sample window min/max/mean = 0.0 / 19.87 / 1.03 |

**Variant taken:** `slope_1KMmn_GMTEDmd` = slope aggregated to the 1 km grain by
**mean** (`mn`), derived from the 250 m GMTED **median**-elevation source layer
(`GMTEDmd`) — i.e. the requested "aggregation = mean, DEM = median" product.
`slope_1KMmn_GMTEDmn.tif` **does not exist** (404); EarthEnv only publishes
derived variables off the median DEM, so `GMTEDmd` is the only mean-aggregated
option.

> **Superseded as the pipeline's slope layer.** Phase B established that the
> legacy CSV's `slope_1` is bit-exact EarthEnv's **median**-aggregated (`md`)
> product, not this mean-aggregated (`mn`) one — see items 10 and 11 below and
> `scripts/extraction/README.md` §6. This file stays on disk as the ADR-3
> configuration and as the automatic fallback if the `md` file is absent, but
> nothing reads it by default.

> **Extent gotcha:** latitude coverage is **−56° to +84°** (16800 rows), not full
> globe. Columns align with the master grid but rows do **not** — needs a windowed
> read with an offset, and cells outside the GMTED envelope have no slope value.

## 7. GHS-SMOD R2023A 2020, 1 km Mollweide — **NEW**

| | |
|---|---|
| **Filename** | `GHS_SMOD_E2020_GLOBE_R2023A_54009_1000_V2_0.tif` (+ `.ovr`, `.aux.xml`; rest of the archive in `ghs_smod/`) |
| **Source** | `https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/GHSL/GHS_SMOD_GLOBE_R2023A/GHS_SMOD_E2020_GLOBE_R2023A_54009_1000/V2-0/GHS_SMOD_E2020_GLOBE_R2023A_54009_1000_V2_0.zip` (**V2-0**, 34 MB) |
| **Downloaded** | 2026-08-08 |
| **Size on disk** | 17 MB (tif) · 75 MB (`ghs_smod/` incl. zip + DUC/UC shapefiles) |
| **Licence** | European Commission reuse policy — open and free, reuse authorised with source acknowledgement |
| **Verification** | `width=36082 height=18000 count=1 dtype=int16 crs=ESRI:54009 nodata=-200.0 res=(1000.0, 1000.0)`, bounds `(-18041000, -9000000, 18041000, 9000000)`; unique classes `[-200, 10, 11, 12, 13, 21, 22, 23, 30]` |

Degree-of-Urbanisation classes: 10/11/12/13 = water & rural grades, 21/22/23 =
urban cluster grades, 30 = urban centre. `-200` = nodata.
V2-0 was available, so **no fallback to V1-0 was needed** — though V1-0 was
later fetched anyway (item 12) to check whether the version the legacy code
names changes anything. It does not.
The zip also contained the DUC and UC settlement-boundary shapefiles, kept in
`ghs_smod/`.

> **Projection gotcha:** this is the only layer in **Mollweide (ESRI:54009)**, not
> EPSG:4326 — it requires a genuine reprojection onto the master grid, not a
> windowed read.

## 8. GRIP4 road-density rasters, 5 arc-minute — **NEW**

Source page: `https://www.globio.info/download-grip-dataset`
Files: `https://dataportaal.pbl.nl/downloads/GRIP4/GRIP4_density_{total,tp1..tp5}.zip`
Downloaded 2026-08-08. **Licence: CC0 — free to use.** Cite Meijer et al. 2018,
*Environ. Res. Lett.* (`doi:10.1088/1748-9326/aabd42`).
Units: **metres of road per km² per cell**.

All grids share the identical verification line:
`width=4320 height=2160 dtype=int32 crs=None nodata=-9999.0 res=(0.08333333, 0.08333333)` bounds `(-180, -90, 180, 90)`

| Filename (in `grip4/`) | Road type | Size |
|---|---|---|
| `grip4_total_dens_m_km2.asc` | all roads, equally weighted | 46 MB |
| `grip4_tp1_dens_m_km2.asc` | type 1 — highways | 45 MB |
| `grip4_tp2_dens_m_km2.asc` | type 2 — primary roads | 45 MB |
| `grip4_tp3_dens_m_km2.asc` | type 3 — secondary roads | 46 MB |
| `grip4_tp4_dens_m_km2.asc` | type 4 — tertiary roads | 46 MB |
| `grip4_tp5_dens_m_km2.asc` | type 5 — local roads | 46 MB |
| `grip4_area_land_km2.asc` | km² land area per cell (bonus, `dtype=float32`) | 33 MB |

These are the five grids behind the legacy QGIS columns `tp11, tp21, tp31, tp41,
tp51`. The 6 source zips and `ReadMe.txt` / `GRIP4_density_m_km2.lyr` are retained
in `grip4/`.

> **CRS gotcha:** the `.asc` files ship **without a `.prj`**, so rasterio reports
> `crs=None`. The ReadMe states WGS84 lat/lon degrees — **EPSG:4326 must be
> assigned explicitly** on read. The grid is otherwise an exact match for
> `distance2water_5arcmin.tif`.

## 9. geoBoundaries CGAZ ADM1 — **NEW**

| | |
|---|---|
| **Filename** | `geoBoundariesCGAZ_ADM1.gpkg` |
| **Source** | `https://github.com/wmgeolab/geoBoundaries/raw/main/releaseData/CGAZ/geoBoundariesCGAZ_ADM1.gpkg` |
| **Downloaded** | 2026-08-08 |
| **Size on disk** | 138 MB |
| **Licence** | CC BY 4.0 (geoBoundaries open licence) |
| **Verification** | `layer=globalADM1 rows=3224 columns=['id','shapeName','shapeID','shapeGroup','shapeType','geometry']` |

`shapeGroup` carries the ISO3 country code (join key to ADM0); `id` is all-null.

> **CRS gotcha:** the layer declares an **"Undefined geographic SRS"** (unknown
> datum, WGS84 ellipsoid params) rather than EPSG:4326. Coordinates *are* WGS84
> lon/lat, but `gdf.crs` is not `EPSG:4326`, so a `set_crs(4326, allow_override=True)`
> is needed before any CRS-checked spatial join.

## 10. EarthEnv 1 km slope, **median** aggregation — **NEW (Phase B)**

| | |
|---|---|
| **Filename** | `slope_1KMmd_GMTEDmd.tif` |
| **Source** | `https://data.earthenv.org/topography/slope_1KMmd_GMTEDmd.tif` (EarthEnv, https://www.earthenv.org/topography) |
| **Downloaded** | 2026-08-08 (Phase B) |
| **Size on disk** | 693 MB |
| **Licence** | No explicit licence on the EarthEnv page; cite Amatulli et al. 2018, *Scientific Data* 5:180040 (`doi:10.1038/sdata.2018.40`). Treat as CC BY-equivalent with attribution. |
| **Verification** | `width=43200 height=16800 count=1 dtype=float32 crs=EPSG:4326 nodata=-32768.0 res=(0.00833333, 0.00833333)`, bounds `(-180, -56, 180, 84)` |

**This is the slope layer the 30 arc-second pipeline actually uses** (`slope_1`).
Variant: aggregated to the 1 km grain by **median** (`md`), off the GMTED
median-elevation source (`GMTEDmd`).

Why `md` and not the `mn` of item 6: the legacy QGIS CSV's `slope_1` was
tested against both and is **bit-exact** the `md` product — 100.00% exact match
on 5,000 random cells and 100.0000% agreement across all 2,190,444 joined cells
in the full Gate 1 run, whereas `mn` runs ~15% higher globally (mean ratio
1.155, correlation 0.990). So the median-aggregation product, not the mean one,
is the true 30as analogue of the layer the published model was built on. This
is a deliberate documented deviation from SPEC ADR-3; see
`scripts/extraction/README.md` §6.

> **Extent gotcha:** same as item 6 — latitude coverage is **−56° to +84°**
> (16800 rows), not full globe, so rows are offset from the master grid and
> cells outside the GMTED envelope have no slope value.

## 11. EarthEnv 10 km slope, median aggregation (5 arc-minute) — **NEW (Phase B)**

| | |
|---|---|
| **Filename** | `slope_10KMmd_GMTEDmd.tif` |
| **Source** | `https://data.earthenv.org/topography/slope_10KMmd_GMTEDmd.tif` (EarthEnv) |
| **Downloaded** | 2026-08-08 (Phase B) |
| **Size on disk** | 6.6 MB |
| **Licence** | As item 10 — cite Amatulli et al. 2018 (`doi:10.1038/sdata.2018.40`) |
| **Verification** | `width=4320 height=1680 count=1 dtype=float32 crs=EPSG:4326 nodata=-32768.0 res=(0.08333333, 0.08333333)`, bounds `(-180, -56, 180, 84)` |

EarthEnv's natively 10 km (= 5 arc-minute) median-aggregated product, and the
`slope_1` source at 5 arc-minute. This is the file the bit-exactness result in
item 10 was established against: it reproduces the legacy CSV's `slope_1`
exactly, which is what identified the whole `md` family as the right one.

## 12. GHS-SMOD R2023A 2020, 1 km Mollweide — **V1-0** — **NEW (Phase B)**

| | |
|---|---|
| **Filename** | `GHS_SMOD_E2020_GLOBE_R2023A_54009_1000_V1_0.tif` (archive `ghs_smod_v1.zip`, 29 MB, retained) |
| **Source** | `https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/GHSL/GHS_SMOD_GLOBE_R2023A/GHS_SMOD_E2020_GLOBE_R2023A_54009_1000/V1-0/GHS_SMOD_E2020_GLOBE_R2023A_54009_1000_V1_0.zip` |
| **Downloaded** | 2026-08-08 (Phase B) |
| **Size on disk** | 18 MB |
| **Licence** | European Commission reuse policy — open and free, reuse authorised with source acknowledgement |
| **Verification** | `width=36082 height=18000 count=1 dtype=int16 crs=ESRI:54009 nodata=-200.0 res=(1000.0, 1000.0)`, bounds `(-18041000, -9000000, 18041000, 9000000)` |

The **older** release of item 7, downloaded because `src/gis_global_module.py:171`
names V1-0 — i.e. it is the version the legacy pipeline referenced. Fetched to
settle whether the version difference mattered: cell-level class agreement with
the legacy CSV is **91.9% (V1-0) vs 91.7% (V2-0)**, i.e. not materially
different (`scripts/extraction/gate1_exceptions.json`). V2-0 therefore remains
the default; V1-0 stays on disk and is selectable with `--smod-raster`.

Grid, classes and the Mollweide projection gotcha are identical to item 7.

---

## Disk usage

| | |
|---|---|
| **Total `data/GIS/source_rasters/`** | **3.7 GB** (`du -sh`, measured 2026-08-08 after Phase B) |
| Items 5–9 (first download session) | ~1.7 GB (GPW 407 MB · slope `mn` 768 MB · SMOD V2-0 75 MB · GRIP4 321 MB · ADM1 138 MB) |
| Items 10–12 (added in Phase B) | ~747 MB (slope `md` 1 km 693 MB · slope `md` 10 km 6.6 MB · SMOD V1-0 18 MB + 29 MB zip) |
| Pre-existing (items 1–4) | ~1.4 GB |
| Free space remaining on volume | 20 GB |

Derived output, not source, and gitignored:
`data/GIS/extracted/master_30as.parquet` is **3.4 GB** and
`master_5min.parquet` **36 MB**. Budget ~3.5 GB of free space to rebuild them.

## Summary of cross-layer alignment

| Layer | Grid | Aligns with Kummu 30as master? |
|---|---|---|
| Kummu dtw 30as | 43200 × 21600, EPSG:4326 | master |
| GPW v4.11 density | 43200 × 21600, EPSG:4326 | **yes, exact** |
| EarthEnv slope 1 km (`md`, **in use**, item 10) | 43200 × 16800, EPSG:4326 | columns yes, rows offset (lat −56…84) |
| EarthEnv slope 1 km (`mn`, item 6) | 43200 × 16800, EPSG:4326 | as above; not read by default |
| EarthEnv slope 10 km (`md`, item 11) | 4320 × 1680, EPSG:4326 | matches the 5 arcmin grid in x, rows offset |
| GHS-POP R2023A | 43202 × 21384, EPSG:4326 | **no** — different width and height |
| GHS-SMOD R2023A | 36082 × 18000, ESRI:54009 | **no** — Mollweide, needs reprojection |
| GRIP4 densities | 4320 × 2160, CRS unset | matches the 5 arcmin grid exactly |
