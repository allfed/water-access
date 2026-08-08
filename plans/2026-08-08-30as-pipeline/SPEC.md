# SPEC: 30 arc-second pipeline upgrade (scripted extraction + solver/memory refactor)

**Date:** 2026-08-08 · **Repo:** water-access, branch `kr/30as-feasibility` (worktree at `.claude/worktrees/30as-feasibility`)
**Companion doc:** `FEASIBILITY_30_ARCSEC.md` (repo root) — read §1–§6 first; it is the evidence base for every decision here.
**Signed-off decisions from Kevin (2026-08-08):** scope = Phases 1+2 + 30as smoke run (no production run); canonical TIME_GATHERING = 2–4 h; Gate 1 = statistical match bar; compute target = local M4 32 GB first, GCP fallback.

---

## 1. PRD

### What we're building

1. **A scripted raster-extraction pipeline** (pure Python: rasterio + geopandas) that replaces the undocumented manual QGIS workflow which produced `data/GIS/updated_GIS_output_cleaned.csv`. It must run at both 5 arc-minute (for validation against the old output) and 30 arc-second (the new target), from freshly downloaded public rasters.
2. **A model refactor** making the Monte Carlo tractable at 30 arc-second (~140M populated cells, 100× today): replace the per-cell `fsolve` loops with a per-draw lookup table + interpolation, cut memory with float32/categoricals/chunking, and persist per-draw district/country aggregates instead of full zone frames.
3. **A 30 arc-second end-to-end smoke run** (10 iterations) on a MacBook Air M4 / 32 GB proving the whole path works within memory budget.

### Why

- Reviewer major comment #5 (cell size ~10 km vs ~5 km walk threshold). Phase 0 measured the global effect as small (+0.46 pp at 6.5 km) but found **144.5M coastal people silently dropped** by 5-arcmin raster overlap and country-level biases of 4–6 pp (see `FEASIBILITY_30_ARCSEC.md` §6). The upgrade is an improvement + reproducibility win, not a rescue.
- The manual QGIS step is a documented reviewer risk ("follow the methods in the paper") and a bus-factor problem. Scripting it removes both permanently.
- The refactor makes even 5-arcmin re-runs take minutes instead of days — it pays for itself during the remaining review cycles.

### Scope

- New `scripts/extraction/` pipeline + validation harness (Gate 1).
- Refactor of the solver path and memory profile in `src/gis_global_module.py` / `src/mobility_module.py` / `src/gis_monte_carlo.py`, preserving public function contracts (Gate 2).
- Environment modernisation (py3.11+, pandas 2.x) and fixing the 8 currently-failing tests (see ADR-6).
- Unifying the two Monte Carlo scripts' sampling parameters (ADR-7).
- 10-iteration 30as smoke run + a short results report.

### Non-goals (do NOT build)

- The full 1000-iteration 30as production run (separate go/no-go after gates pass).
- GRIP re-rasterisation from vectors (ADR-3: roads stay at native 5 arcmin, broadcast).
- WorldPop population sensitivity, the unpiped/borehole breakdown (lives on `kr/unpiped-breakdown`), manuscript edits, map styling.
- Any change to model *logic* (velocity equations, threshold rules, JMP handling) beyond what the refactor mechanically requires. If old and new outputs differ beyond tolerance, the refactor is wrong — never "improve" the model to make a gate pass.
- Resampling anything below 30 arc-second (spurious precision — see FEASIBILITY §1a).

---

## 2. ADRs

### ADR-1: Master grid = the Kummu 30as lattice
Grid: 43200 × 21600, EPSG:4326, cell 0.00833333°, origin (−179.999999, 90.0024163) — i.e. exactly the grid of `distance2water_30arcsec.tif`. The 5-arcmin validation grid is the same lattice ÷10 (4320 × 2160), which is exactly the grid of the existing CSV.
**Rationale:** dtw is the decision-driving layer (FEASIBILITY §2); sampling it on its own lattice is lossless. All other rasters get windowed/reprojected reads onto this grid (nearest-neighbour for categorical/coarse layers, as QGIS evidently did — the CSV's dtw values are exact Kummu lattice values).
**Rejected:** a "clean" −180/+90-origin grid (would put every Kummu read off-lattice); 3as (no information below 30as in the source vectors).

### ADR-2: Extraction outputs Parquet with the legacy column contract
Output: partitioned Parquet (float32 for rasters, category for ISOCODE/shapeID/shapeName), columns exactly:
`fid, left, top, right, bottom, shapeName, shapeID, ISOCODE, GHS_SMOD, pop_density, dtw_1, grip_1_1..grip_5_1, slope_1, dtw_1_nan, pop_density_nan, GHS_SMOD_nan, slope_1_nan, any_nan`
(the verbatim schema of `updated_GIS_output_cleaned.csv`; QGIS→model renames documented at `scripts/Data_Manipulation_Scripts/pre_process_GIS_data.py:63-73`, e.g. tp51→grip_1_1 … tp11→grip_5_1 — note the deliberate order inversion). `fid` = `row * 43200 + col` on the master grid. A `--format csv` flag emits the legacy CSV for the 5-arcmin validation diff. The model's loader (`load_data`, `src/gis_global_module.py:126`) gains Parquet support; CSV path stays working.
**Rationale:** identical contract lets Gate 1 be a mechanical diff and lets the refactored model consume either resolution unchanged. 30as CSV would be ~40 GB; Parquet float32 is ~8–15 GB.
**Rejected:** new "clean" schema (breaks every downstream script for zero benefit now).

### ADR-3: Data sources at 30as
| Layer | Source (30as build) | Notes |
|---|---|---|
| dtw_1 | Kummu `distance2water_30arcsec.tif` (already at `data/GIS/source_rasters/`) | native |
| pop_density | **GPW v4.11 population density 2020, 30as — CONFIRMED available, no login**: `https://data.ghg.center/sedac-popdensity-yeargrid5yr-v4.11/gpw_v4_population_density_rev11_2020_30_sec_2020.tif` (426 MB; verified HTTP 200 on 2026-08-08; downloaded to `data/GIS/source_rasters/gpw_v4_density_2020_30sec.tif`). GHS-POP E2020 30ss remains on disk as a cross-check layer only | Gate 1's pop criteria (AC-3, 1% totals) apply in full — same product as the manuscript. See `data/GIS/source_rasters/MANIFEST.md` for all source provenance (feeds AC-9) |
| GHS_SMOD | GHS-SMOD R2023A 1 km (Mollweide) via `rasterio.vrt.WarpedVRT` nearest → master grid | categorical; nearest only |
| slope_1 | EarthEnv 1 km slope, **GMTED-based variant** (mn aggregation) | SRTM variant holes >60°N (FEASIBILITY §1a) |
| grip_*_1 | GRIP4 5-arcmin density rasters (tp11–tp51 ASCII grids) broadcast 10×10 onto the 30as grid | information floor is 5 arcmin; document |
| shapeID/shapeName/ISOCODE | geoBoundaries CGAZ **ADM1** + ADM0 gpkg rasterised to int32 on the master grid (Phase 0 did ADM0 in 13 s) | cells with no polygon → dropped, count reported |

### ADR-4: Solver lookup table, built per draw
Per Monte-Carlo draw, solve the Lankford + bike models on a grid of (slope × Crr × body-mass): slope = ~500 points spanning observed range (finer near 0), Crr = the 6 road classes, mass = ~21 points spanning country weights; then bilinear-interpolate (slope, mass) per Crr onto cells. ~60–130k fsolve calls ≈ 10–25 s per draw, independent of cell count. Verified claim: velocity depends only on (slope, Crr, mass) + draw scalars (`src/mobility_module.py:203-335`).
**Acceptance-tested against direct fsolve** (AC-4). Existing `single_bike_run`/`single_lankford_run` stay untouched and remain the ground truth; the LUT is a new layer used by `run_bicycle_model`/`run_walking_model`.
**Rejected:** one global pre-computed LUT over draw parameters too (5-D+, needless); vectorising fsolve itself (fragile).

### ADR-5: Memory model — chunked draws, aggregate-only persistence, ≤24 GB peak
Static per-cell frame built once (float32/categorical, memory-mapped Parquet). Per draw: LUT + vectorised column ops + district/country groupby; persist ONLY the per-draw district and country aggregate frames (~3k rows). Per-cell across-draw quantiles (map products): second pass recomputing draws per spatial chunk. Hard budget: peak RSS ≤ 24 GB at 30as on the M4 (AC-6); the existing GCP scripts remain the >32 GB fallback.
**Rejected:** persisting per-cell per-draw parquets (2 TB at 30as); Polars/duckdb rewrite (new dependency, bigger diff — revisit only if numpy/pandas chunking misses the budget).

### ADR-6: Target environment is Python 3.11+ / pandas ≥2.1; fix the 8 drift failures
The pinned conda env (py3.9, pandas 1.4.3 — `environment.yml`) doesn't exist on the dev machine; on ambient py3.11/pandas 2.1.4 the suite is **101 passed / 8 failed**, all traceable to pandas-2.x `groupby(...).apply` reindex behaviour in `merge_and_adjust_population` (`src/gis_global_module.py:245`) cascading into 8 tests. The refactor touches this code anyway.
**Decision:** modernise — make the suite green on py3.11+/pandas 2.x (minimal-diff fixes FIRST, before any refactor, so a trustworthy baseline exists); update `environment.yml` and add `requirements.txt` (uv-compatible); keep `pip install -e .` as the import mechanism and document it in the extraction README. Python 3.9 is EOL.
**Rejected:** rebuilding the legacy conda env (fights the ecosystem; blocks rasterio/geopandas versions needed for extraction).

### ADR-7: Canonical Monte-Carlo sampling = the GCP script's values
`run_monte_carlo.py` and `run_monte_carlo_gcp.py` disagree: TIME_GATHERING normal(4,7) vs lognormal(2,4); met (3,6) vs (3,5); watts (20,80) vs (20,60). Kevin ruled **2–4 h canonical** (matches the manuscript's ~6.5 km median walk limit). Unify BOTH scripts on the GCP values (lognormal 2–4 h, met 3–5, watts 20–60), hoisted into one shared constants module so they can never diverge again. Record in the changelog that `run_monte_carlo.py`'s previous values were non-canonical.
(Gate 2 is unaffected: it compares old-vs-new code at *fixed* parameters.)

### ADR-8: `pop_zone > 10` filter becomes density-equivalent
`src/gis_global_module.py:257` keeps cells with >10 people. At 5 arcmin that's ~0.1 person/km²; verbatim at 30as it would be ~10 people/km² and delete the rural tail. **Decision:** threshold = `10 * (cell_area / five_arcmin_cell_area)` — i.e. >10 at 5 arcmin (unchanged), >0.1 at 30as. Implemented as a resolution-aware constant, documented inline.

### ADR-9: Gate 1 pass bar is statistical, not cell-exact (Kevin sign-off)
The QGIS provenance is undocumented (resampling method, tie-breaking, an extra unexplained transformation between `pre_process_GIS_data.py` outputs and the cleaned CSV). Chasing cell-exactness would burn build loops on artefacts of the old manual process. Bar defined in AC-3. Residual per-country exceptions get a documented list with a one-line cause hypothesis each.

---

## 3. Acceptance criteria

Run everything from repo root with the project venv active (`pip install -e .` done). Every criterion is a command + observable.

**AC-1 (env + drift fixes).** `python -m pytest tests/ -q` → **109 passed, 0 failed** on py3.11+/pandas ≥2.1. The fixes to reach green must precede the refactor in history (separate commit(s)) and not change any numeric output (spot-check: `TestRunGlobalAnalysis::test_run_global_analysis_null_run` passes with its existing MAPE < 20% assertion untouched).

**AC-2 (baseline capture).** With the suite green and BEFORE the refactor: `python scripts/capture_baseline.py` (new, ~30 lines) runs `run_global_analysis(crr_adjustment=0, time_gathering_water=2.84, practical_limit_bicycle=40, practical_limit_buckets=20, met=4, watts=40, hill_polarity="downhill_uphill", urban_adjustment=1.3, rural_adjustment=1.4, calculate_distance=True, plot=False)` on the full 5-arcmin CSV and writes `results/regression_baseline/{countries,districts}.csv`. Committed.

**AC-3 (Gate 1 — extraction fidelity at 5 arcmin).** `python scripts/extraction/extract_grid.py --resolution 5min --out data/GIS/extracted/master_5min.parquet` then `python scripts/extraction/validate_extraction.py --new data/GIS/extracted/master_5min.parquet --old data/GIS/updated_GIS_output_cleaned.csv --report results/validation/gate1_report.md` exits 0, where exit 0 requires, per country (pop-weighted where applicable), over countries with ≥1M people:
  - modelled-cell count within **2%**;
  - `dtw_1`: cell-level exact match (same value or both-nodata) on **≥99%** of spatially-aligned cells; pop-weighted mean within **2%**;
  - `slope_1`, `GHS_SMOD` (mode), dominant road class: pop-weighted/statistical match within **2%** (SMOD urban-share within 2 pp);
  - `pop_density`: totals within **1%** *if* the same population product (GPW) was obtained; otherwise distribution-level report only (ADR-3) with the source swap stated in the report header;
  - a **documented exception list** (country, metric, magnitude, one-line cause hypothesis) for any residual failures, each individually approved as a known-QGIS-artefact — the report is the gate artefact.

**AC-4 (solver LUT fidelity).** `python scripts/validate_lut.py --samples 10000 --report results/validation/lut_report.md` draws 10k random (slope from the real slope distribution, Crr class, country mass, draw-parameter tuples sampled from the ADR-7 canonical ranges), compares LUT-interpolated loaded/unloaded walk+bike velocities against direct `single_lankford_run`/`single_bike_run` calls: **max |Δv| ≤ 0.5% of v (or ≤0.005 m/s absolute, whichever is larger); NaN/no-solve cases agree exactly**. Exits 0.

**AC-5 (Gate 2 — refactor regression at 5 arcmin).** After the refactor: `python scripts/capture_baseline.py --out results/regression_new` (same fixed parameters, new code) then `python scripts/validate_regression.py --report results/validation/gate2_report.md` exits 0 requiring, against AC-2's committed baseline: for every country, `percent_with_water` within **0.1 pp**; every population-count column within **0.1%**; district-level `percent_with_water` within **0.25 pp** for districts >100k people; row sets identical (same countries, same shapeIDs).

**AC-6 (memory + 30as smoke).** `python scripts/run_monte_carlo.py --resolution 30as --iterations 10 --input data/GIS/extracted/master_30as.parquet` completes on the M4/32 GB with **peak RSS ≤ 24 GB** (measured via `/usr/bin/time -l`, `maximum resident set size` reported in the run log) and writes country+district mean/median/5th/95th CSVs in the standard schema. Wall-clock reported; target ≤ 12 h for the 10-iteration smoke (soft — report actuals).

**AC-7 (speed at 5 arcmin).** Single fixed-parameter run (AC-5's command) completes in **≤ 90 s** on the M4 excluding initial Parquet load (vs ~4 min solver time today) — proves the LUT path is active. Reported in gate2_report.md.

**AC-8 (30as sanity vs Phase 0).** From the smoke run: global `percent_with_water` (median across the 10 iterations) differs from the 5-arcmin equivalent (same 10 seeds/params re-run at 5 arcmin) by **≤ 2 pp**, and the direction of the walking-access shift matches Phase 0 (30as ≥ 5arcmin at the ~6.5 km margin, i.e. slightly MORE access). Both runs' outputs + a 15-line comparison table land in `results/validation/smoke_comparison.md`.

**AC-9 (reproducibility docs).** `scripts/extraction/README.md` documents: every source dataset (URL, version, licence, download command or manual step), the master-grid definition, runtime + peak RAM actuals for both resolutions, and the one-command sequence to regenerate everything from a clean checkout. A cold reader must be able to re-derive `master_30as.parquet` from it.

**AC-10 (no contract breakage).** `grep`-verifiable: `run_simulation` and `run_global_analysis` signatures unchanged (new optional kwargs allowed at the end); `python -m pytest tests/ -q` still 109-green after the refactor; the legacy CSV input path still works (`use_sample_data=True` test exercises it).

---

## 4. Phase plan (draft)

Phases B and C are genuinely independent (disjoint files) and MAY run as parallel worktree implementers after A lands. D needs B+C.

- **Phase A — env + baseline (sequential, first).** ADR-6 fixes → AC-1 green → AC-2 baseline committed. Also hoist ADR-7 canonical constants and the ADR-8 resolution-aware filter constant (mechanical, small).
  *Deliverable: green suite + committed baseline CSVs.*
- **Phase B — scripted extraction (independent).** `scripts/extraction/` package: source download manifest (Kummu/GHS-POP already at `data/GIS/source_rasters/`; add EarthEnv slope, GHS-SMOD, GRIP tp11–tp51, geoBoundaries ADM0+ADM1; GPW attempt per ADR-3), windowed extraction to Parquet at `--resolution {5min,30as}`, ADM rasterisation, nan-indicator columns, `--format csv` legacy export, validation harness → **AC-3, AC-9**. Phase 0's `scripts/feasibility/phase0_resolution_bias.py` is the working skeleton for windowed grid-aligned reads.
- **Phase C — solver LUT + memory refactor (independent).** LUT module + integration behind the existing function contracts, float32/categorical dtypes, chunked per-draw execution, aggregate-only persistence, quantile second pass, Parquet loader support → **AC-4, AC-5, AC-7, AC-10**.
- **Phase D — 30as smoke (needs B+C).** Extract `master_30as.parquet` (AC-9 runtime actuals), 10-iteration smoke + 5-arcmin twin run → **AC-6, AC-8**; write `results/validation/smoke_comparison.md`.

**Verifier notes:** every AC has a report artefact under `results/validation/`; the verifier re-runs AC-1, AC-3, AC-4, AC-5, AC-7 commands cold and spot-checks AC-6's logged RSS. Data downloads (~3–6 GB) are the only network dependency; all sources are public except the optional GPW/EarthData step.

**All source data is already on disk and verified** (2026-08-08): see `plans/2026-08-08-30as-pipeline/DATA_MANIFEST.md` (tracked copy; original sits gitignored beside the rasters). Phase B needs **no downloads** — skip straight to extraction. Slope file is `slope_1KMmn_GMTEDmd.tif` (mean aggregation of the GMTED median-DEM layer; the `GMTEDmn` variant does not exist).

**Measured alignment facts (from the download verification — do not rediscover these):**
- **GPW v4.11 density is grid-identical to the Kummu 30as master grid** (43200×21600, EPSG:4326, nodata −9999) — read directly, zero resampling.
- **GHS-POP is NOT co-registered**: 43202×21384, fractional-cell offset (~0.95 col / ~0.3 row), no declared nodata. Cross-check use only; window via `from_bounds`, never by index.
- **EarthEnv slope covers lat −56°…+84° only** (16800 rows): columns align with the master grid, rows need a fixed offset; cells outside coverage → nodata handling.
- **CRS traps**: GRIP4 `.asc` grids ship with `crs=None` (ReadMe says WGS84 — set EPSG:4326 explicitly); geoBoundaries CGAZ ADM1 declares an undefined geographic SRS — `set_crs(4326, allow_override=True)` before any join; GHS-SMOD is the sole Mollweide layer and needs true reprojection (WarpedVRT nearest, per ADR-3).
- GRIP4 units: metres of road per km², tp1=highways … tp5=local roads; grid-exact against `distance2water_5arcmin.tif`.

**Known risks for the builder:**
- Gate 1 discrepancies of unknown QGIS origin are EXPECTED — investigate briefly, then document in the exception list rather than chase (ADR-9). The unexplained extra transformation between `pre_process_GIS_data.py` output and the cleaned CSV is on record.
- GHS-POP raster is float64 with no declared nodata and a fractionally-offset grid (~0.3 cell in lat) — Phase 0 handled both; reuse that code.
- `run_monte_carlo_gcp.py` computes `project_root` CWD-relatively (fragile) — fix while unifying constants (ADR-7), don't inherit it.
- The pickle outputs (`*_simulation_results.pkl`) are consumed only by two notebooks; keep writing them at 5 arcmin, skip at 30as (they'd be huge) with a log line.
