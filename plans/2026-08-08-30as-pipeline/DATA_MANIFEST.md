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
option. (`slope_1KMmd_GMTEDmd.tif`, the median-aggregated variant, exists at
693 MB but was not taken.)

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
V2-0 was available, so **no fallback to V1-0 was needed**.
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

---

## Disk usage

| | |
|---|---|
| **Total `data/GIS/source_rasters/`** | **3.1 GB** |
| New this session (items 5–9) | ~1.7 GB (GPW 407 MB · slope 768 MB · SMOD 75 MB · GRIP4 321 MB · ADM1 138 MB) |
| Pre-existing (items 1–4) | ~1.4 GB |
| Free space remaining on volume | 16 GB |

## Summary of cross-layer alignment

| Layer | Grid | Aligns with Kummu 30as master? |
|---|---|---|
| Kummu dtw 30as | 43200 × 21600, EPSG:4326 | master |
| GPW v4.11 density | 43200 × 21600, EPSG:4326 | **yes, exact** |
| EarthEnv slope | 43200 × 16800, EPSG:4326 | columns yes, rows offset (lat −56…84) |
| GHS-POP R2023A | 43202 × 21384, EPSG:4326 | **no** — different width and height |
| GHS-SMOD R2023A | 36082 × 18000, ESRI:54009 | **no** — Mollweide, needs reprojection |
| GRIP4 densities | 4320 × 2160, CRS unset | matches the 5 arcmin grid exactly |
