# Feasibility analysis: re-running the model at 30 arc-seconds

**Date:** 2026-08-08 · **Author:** Kevin + Claude (two Opus research agents: repo audit + public-data survey; synthesis and all judgement calls by the orchestrating session)
**Question:** Reviewer major comment #5 — grid cells (~10 km) are the same order as the walk/no-walk threshold (~5 km). Is moving to 30 arc-second (~1 km) resolution (a) supported by real data, (b) an improvement rather than noise, and (c) computationally feasible on cloud infra?

**Bottom line: Yes to all three — with a staged plan.** The critical layer (Kummu distance-to-water) exists *natively* at 30 arc-second; the 5 arc-minute file we use is the lossy downsample of it. Population and settlement layers are also natively ≤1 km. The compute blocker that stopped the original run is removable with one refactor (the per-cell `fsolve` is cacheable), after which the full 1000-iteration run costs **tens of dollars, not weeks**. The real cost is ~1–1.5 weeks of engineering. A half-day gating experiment (§5, Phase 0) should be run first to quantify the bias before committing.

The original author already knew this. `src/gis_global_module.py:236`:
> *"To reduce this error, use the 30 arc second data. Too much heavy lifting for my computer to do this at the moment."*

That was a hardware problem in ~2022. It is a software problem now, and a solved one.

---

## 1. State of play: every GIS parameter, current vs available

### 1a. The master table

| Layer | Column(s) | Source | Used at | **Native / best available** | 30as upgrade path | Trade-offs / caveats |
|---|---|---|---|---|---|---|
| **Distance to surface water** | `dtw_1` | Kummu et al. 2011 (Dryad `doi:10.5061/dryad.71c6r`) | 5 arcmin | **30 arcsec, same DOI** — `distance2water_30arcsec.tif`, 272 MB, CC0. Kummu computed at 1 km and aggregated "for computational reasons"; the 5 arcmin file is the derivative | **Free — download the other file** | This is the layer that drives the result (see §2). Its real weakness is *content* not resolution: perennial water only, no small streams/ephemeral water (overstates distance in a crisis — conservative direction). Don't resample below 30as: source vectors (VMAP0, ~1:1M) carry no finer information |
| **Population** | `pop_density` | GPW v4 (per manuscript) | 5 arcmin | **30 arcsec native** (GPW v4.11). GHS-POP R2023A also ships 30as; WorldPop 3as | Free download | Láng-Ritter et al. 2025 (Nat Commun 16:2170) report all gridded products underestimate *rural* population (GHS-POP worst, WorldPop best; magnitudes disputed by JRC, direction well-supported). A WorldPop sensitivity run is arguably higher-value than the resolution change — cheap to add once the pipeline is scripted |
| **Settlement (urban/rural)** | `GHS_SMOD` | GHS-SMOD R2023A | 5 arcmin | **1 km native** (Mollweide) | Free — currently *purely* lost information | Drives the piped/unpiped urban split and detour index; degrading it to 10 km was pure loss |
| **Slope** | `slope_1` | EarthEnv topography (Amatulli 2018) | 10 km product (per manuscript) | **1 km product exists** — slope computed at 250 m (GMTED) then aggregated, so magnitude is preserved, not flattened | Free download | Use the GMTED variant (SRTM variant holes out >60°N). Walking barely cares (velocity spread ±3.5% globally); cycling cares moderately. Mean-of-fine-slope is the physically correct input for energetics |
| **Roads / rolling resistance** | `grip_1_1`–`grip_5_1` | GRIP4 raster (Meijer 2018) | 5 arcmin | **5 arcmin is the only published raster.** Global vector (>2 GB) exists | **Keep at 5 arcmin, nearest-neighbour onto the 30as grid, and say so** | Re-rasterising the vector to 1 km would manufacture false precision: GRIP source density varies by country, so 30as detail would encode data availability, not road surface. The one layer where finer genuinely is noise |
| **Admin boundaries** | `shapeID`, `ISOCODE` | geoBoundaries CGAZ ADM1 | vector join | Resolution-independent | Point-in-polygon over ~140M points — needs a spatial index (STRtree/exactextract), solved problem | — |
| **JMP piped/unpiped, bike ownership (PBO), household size, body weight, national population** | country scalars | WHO/JMP, Oke 2015, UNDESA, NCD-RisC, OWID | country → painted on grid | No spatial resolution exists | None needed | Finer grid adds nothing here directly — but JMP piped % *is* modulated by the gridded urban flag, so SMOD resolution does propagate into the piped split |

### 1b. Repo state (what we actually have on disk)

- **No source rasters are in the repo** — `data/GIS/` (3.3 GB, LFS) holds only the post-QGIS CSV exports. Every upstream raster must be re-downloaded (all free; ~2–8 GB total).
- The master input is `data/GIS/updated_GIS_output_cleaned.csv`: **2,190,446 land cells** at exactly 5 arcmin, EPSG:4326, of which **1,403,388 populated cells** are modelled per Monte Carlo iteration (after the `pop_zone > 10` filter).
- The GIS extraction (QGIS 3.36: reproject → resample to 5 arcmin → grid centroids → join-by-location → sample-raster-values → CSV) is **manual and undocumented beyond the manuscript** — `docs/readme.md:5-7` says "follow the methods in the paper / contact ALLFED". This is exactly the step James remembers having to "relearn". Everything downstream of the CSV is scripted and ports unchanged.
- ~1.05 GB of `data/GIS/` (`gis_data_adm1.csv`, `GIS_stripped_back.csv`) is dead — no code references. Cleanup candidate regardless of this decision.
- **Smoking gun on the dtw layer:** its unique values are exactly `1000·√(a²+b²)` for integer (a, b) — 55,787 distinct values like 1000, 1414.214, 2236.068… That is the fingerprint of a Euclidean distance transform on a 1 km lattice, nearest-neighbour-sampled at one point per 10 km cell. **The reviewer's within-cell criticism is precisely correct, and the information to fix it already exists at the same DOI.**

---

## 2. Will 30 arc-second improve things, or is it noise?

**Signal, not noise — for the layers that matter.** The argument in four steps:

1. **The walk decision is almost purely a threshold on dtw.** Measured across the whole world, walking velocity spans only ±3.5% (5th–95th), so "can this cell's population walk to water" reduces to `dtw < ~6.5–7 km` (at the current time budget). The result is therefore dominated by the *joint distribution of population and distance-to-water* — exactly the two layers that exist natively at ≤30 arcsec and are currently degraded 10× .
2. **Coarse cells destroy the population–distance covariance, and the error is systematic, not random.** People settle preferentially at the near-water end of a 10 km cell; block-averaging assigns them the cell-centroid distance instead. Direction of bias (inference — to be confirmed by Phase 0): population-weighted mean distance is *overestimated* at 5 arcmin, while the extreme far tail is *underestimated* (variance compression hides the genuinely remote). These do **not** wash out under national aggregation because the sign is consistent — it's a textbook MAUP scale effect. Closest quantitative analogue: Punger & West 2013 found population-weighted PM₂.₅ exposure biased 30–40% at coarse grids because the field varies at short scales where people live; the bias scales with the field's autocorrelation length relative to cell size. Distance-to-water is short-scale in humid, densely-populated regions (large bias expected) and long-scale in arid ones (small bias) — a benign pattern for our headline conclusions, and worth stating in the paper either way.
3. **The model response is nonlinear (threshold), so smoothing does not cancel** (Jensen's inequality): the error in "% who can reach water" has a definite sign per region that depends on where the ~6.5 km threshold sits in the local distance distribution. This is measurable, not guessable.
4. **No literature answers this directly — which is an opportunity.** Kummu ran no resolution sensitivity; nobody has published a 5 arcmin vs 30 arcsec distance-to-water comparison. Because both rasters sit at one DOI, we can *measure* the bias in an afternoon and cite our own sensitivity analysis. That satisfies the reviewer regardless of which way it comes out.

**What does NOT improve:** roads (information floor at 5 arcmin), all country scalars (JMP, PBO, weights), and the Kummu layer's perennial-only content bias (unchanged at any resolution; remains our conservative-direction caveat). The honest framing for the paper is a **hybrid-resolution model**: decision variables (dtw × population × settlement) at 30as, road class at its native 5 arcmin, country scalars at country level.

**Expected effect on results (prior, to be tested):** population-weighted distances shrink in humid/dense regions → somewhat *fewer* people modelled as losing access there; the far tail (deep rural, arid) becomes visible instead of averaged away → district maps get more credible extremes. Net effect on the global 21.4% headline: probably a modest decrease, uncertain sign by continent — which is itself a reason to run it before a reviewer makes us.

---

## 3. Compute feasibility

### 3a. Why it was impossible before

Current architecture: flat dataframe of cells; per iteration, a Python for-loop calls `scipy.optimize.fsolve` **4× per cell** (2 walking, 2 cycling). Measured: 174 µs/cell → ~4 min/iteration at 1.4M cells on a laptop; 15–30 min/iteration on GCP (n2-standard-32, 24 workers × ~3.3 GB peak each, 2–3 days for 1000 iterations).

Naive 100× scaling (10× per axis): ~140M populated cells, **~7 h/iteration solver time, ~165 GB dataframe (330 GB peak/worker), ~200–300 days wall-clock, ~2 TB of per-iteration outputs.** Infeasible as written — this is exactly the wall James hit.

### 3b. The refactor that removes the wall

**The fsolve is a lookup table in disguise** (verified in `src/mobility_module.py:203-335`): velocity depends only on `(slope, Crr, country body-weight)` plus four per-draw global scalars. It does **not** depend on dtw, population, urban/rural, or anything else that proliferates at 30as. Observed slope cardinality: 3,260 unique values at 2 dp (finer than the slope layer's real accuracy) × 6 Crr classes × ~200 country weights.

So per draw: solve on a (slope-grid × Crr × weight) lattice — **~10⁴–10⁵ solves ≈ 10–100 s** — and interpolate onto cells. Solver cost becomes independent of grid resolution. Everything else in the pipeline is already vectorised pandas/numpy and becomes a chunking problem:

| Stage (refactored) | Cost at 30as (~140M cells) |
|---|---|
| Static per-cell frame (built once): load, country merge, pop shares, urban flag, road class | ~15–25 GB in RAM as float32/categorical Parquet; built once, memory-mapped |
| Per-draw: solver LUT + interpolation | 10–100 s |
| Per-draw: vectorised column ops + threshold + district groupby | ~30–60 s (memory-bandwidth bound) |
| Per-draw persistence | district/country aggregates only (~3k rows) — the 2 TB problem disappears |
| Per-cell across-draw quantiles (for maps) | second pass, recompute per spatial chunk (~5M cells × 1000 draws in ~20 GB) — a few hours |

**Total: ~0.5–1 day wall-clock on one large-memory machine.**

### 3c. Cloud cost estimate

| Option | Machine | Est. wall-clock | Est. cost |
|---|---|---|---|
| GCP (existing tooling: `gcp/deploy-spot.sh` already targets n2 spot VMs) | n2-highmem-64 (64 vCPU / 512 GB) | ~1 day | **~$10–35 spot / ~$30–105 on-demand** |
| AWS equivalent | r6i.16xlarge (64 vCPU / 512 GB) | ~1 day | similar |
| Storage | inputs ~2–8 GB downloads; master Parquet ~10–25 GB; output GeoTIFFs ~4 GB/band | — | negligible |

For comparison, the *current* 5-arcmin run costs 2–3 days on n2-standard-32. **After the refactor, the 30as run is cheaper than the 5-arcmin run is today.** The refactor also makes 5-arcmin re-runs take ~minutes, which pays for itself during the remaining review cycles.

### 3d. The real cost: engineering time

| Phase | Work | Estimate (Claude-assisted) |
|---|---|---|
| **0 — Gate** | Raster-level bias measurement (§5) | **0.5 day, laptop** |
| **1 — Scripted extraction** | Replace the manual QGIS steps with rasterio/xarray stack-read aligned to the Kummu 30as grid + STRtree boundary join. **Validate by regenerating the 5-arcmin CSV and diffing against `updated_GIS_output_cleaned.csv`** — this regression test is the safety net, and scripting the step permanently kills the "relearn QGIS" problem and the undocumented-provenance reviewer risk | 2–4 days |
| **2 — Solver LUT + chunked MC** | LUT/interpolation, float32 dtypes, chunk-parallel draws, aggregate-only persistence | 1–2 days |
| **3 — 30as production run** | Cloud run + rasterise (`rasterise_results.py` needs `resolution = 30/3600`) + map re-styling | 1 day |

**Total ≈ 1–1.5 weeks.** Phases 1–2 are worth doing even if we ultimately stay at 5 arcmin (reproducibility + speed); Phase 3 is the only 30as-specific spend.

---

## 4. Design decisions and traps (flagged for sign-off)

1. **The `pop_zone > 10` filter must be rescaled** (`src/gis_global_module.py:257`). It is an absolute persons-per-cell cutoff: at 10 km cells it means ~0.1 person/km²; kept verbatim at 1 km cells it means 10 persons/km² and would silently delete the sparse rural tail — the exact population this paper is about. Use ~0.1 person-per-1km-cell (or a density threshold) to preserve semantics.
2. **Resampling method for GRIP onto the 30as grid:** nearest-neighbour, documented as such (the information is genuinely 5 arcmin).
3. **Slope product switch** (10 km → 1 km EarthEnv) changes cycling results slightly, independent of the grid change. If we want to isolate the resolution effect for the paper, run Phase 3 twice: once changing only dtw+pop+SMOD, once also upgrading slope. Cheap after the refactor.
4. **⚠️ Pre-existing discrepancy found during the audit, independent of resolution:** `scripts/run_monte_carlo.py:43-44` samples `TIME_GATHERING` from **4–7 h** while `scripts/run_monte_carlo_gcp.py:54-55` uses **2–4 h**. That is a factor-of-two difference in the walk threshold (~9.5–16.7 km vs ~4.8–9.5 km) between two scripts that both claim to produce the published run. **Resolve which one produced the published numbers before any re-run** — this dwarfs the resolution effect.
5. Population product: stay on the manuscript's GPW v4 (natively 30as) for the resolution comparison so only one thing changes; add a WorldPop sensitivity as a separate, cheap run afterwards (see §1a).

---

## 5. Recommended staged plan

**Phase 0 (do first, half a day, laptop):** download `distance2water_30arcsec.tif` + `distance2water_5arcmin.tif` + GPW v4 30as. Compute population-weighted distance distributions per country on both resolutions and diff. This directly measures the bias the reviewer alleges, produces a citable sensitivity result whichever way it goes, and gates the rest:
- **Bias material** (moves population-weighted access shares by ≳1–2 pp in populous countries) → proceed to Phases 1–3; the paper gets a genuine improvement plus a methods paragraph.
- **Bias immaterial** → stop; the paper gets a quantified one-paragraph rebuttal of comment #5, which is nearly as valuable and costs nothing more.

My read of the evidence (systematic covariance loss + threshold nonlinearity + the 1-km fingerprint in our own data) is that Phase 0 will show a material, regionally-patterned bias — but that's a prior, and Phase 0 exists to replace it with a measurement.

---

## 6. Phase 0 RESULTS (measured 2026-08-08, MacBook Air M4, ~70 s runtime)

Script: `scripts/feasibility/phase0_resolution_bias.py` · outputs: `results/phase0/` (per-country CSV, summary CSV, histograms, figure). Method: pop-weighted distance statistics on Kummu 30as vs Kummu 5arcmin broadcast back onto the 30as grid (mathematically identical to what the 5arcmin pipeline computes), GHS-POP E2020 30as weights and geoBoundaries CGAZ ADM0 in both columns, paired-valid cells only (7,609M people).

### 6a. Global and continental: the errors DO largely wash out

% of population within one-way walking distance of surface water (30as − 5arcmin, percentage points):

| Region | pop (M) | pw-mean dtw 30as / 5' (km) | Δpp @2 km | Δpp @5 km | **Δpp @6.5 km** | Δpp @10 km |
|---|---|---|---|---|---|---|
| **GLOBAL** | 7,609 | 7.92 / 7.99 | +1.11 | +0.75 | **+0.46** | +0.06 |
| Africa | 1,332 | 13.80 / 13.82 | +0.06 | +0.66 | **+0.62** | +0.29 |
| Asia | 4,519 | 7.10 / 7.17 | +1.45 | +0.70 | **+0.47** | +0.04 |
| Europe | 722 | 4.52 / 4.62 | +2.62 | +1.19 | **−0.12** | +0.02 |
| Oceania | 38 | 6.81 / 6.85 | +2.14 | −0.31 | **+0.07** | +0.53 |
| S.+N. America | 416* | 6.41 / 6.48 | +0.18 | +1.17 | **+0.61** | −0.04 |

*continent labels from `merged_data.csv`; NA/SA split available in the per-country CSV.

At the model's operative walking threshold (~6.5 km) the global resolution effect is **+0.46 pp** — an order of magnitude smaller than the paper's 90% parameter-uncertainty interval on the headline result (~8–39%). Direction as predicted: the coarse grid slightly *overstates* pop-weighted mean distance (7.99 vs 7.92 km) and *understates* access near thresholds — i.e. **the published 5-arcmin numbers slightly overstate the population losing access**, and the effect shrinks toward zero by 10 km (cycling range). Kevin's instinct that the errors wash out at aggregate level was right.

### 6b. Country level: material for a meaningful minority

Distribution of |Δpp @6.5 km| across 157 countries with >1M people: median 0.96 pp, 90th pct 3.7 pp, max 6.0 pp. Biggest movers (>20M pop): Angola +5.8, North Korea +5.2, Syria +4.6, Mali +4.5, South Korea +4.5, DR Congo +4.0, Egypt +2.8; both signs occur (Spain −2.1, Morocco −1.9, Pakistan −1.6, Afghanistan −1.6). So **country-level results and the district maps do change visibly** at 30as, even though continental rankings would not.

### 6c. Bonus finding: the raster-overlap loss is 145M people

**144.5M people (~1.9% of world population) sit on cells where the 30as distance layer is valid but the 5arcmin layer is nodata** — overwhelmingly coastal fringe, exactly the imperfect-overlap data loss the original author's code comment flagged (`gis_global_module.py:236`). Only 20.1M go the other way. Moving to 30as recovers ~125M net people currently falling out of the analysis silently. This is arguably a stronger reason to upgrade than the resolution bias itself.

### 6d. Verdict

- **Reviewer comment #5 is now answerable with a measurement either way:** "We quantified the resolution sensitivity using the native 1 km version of the distance layer: at the median walking threshold the global effect is +0.5 pp and all continental effects are ≤0.7 pp — well within the reported uncertainty interval. Coarse resolution slightly overstates, not understates, the population losing access." That paragraph alone likely satisfies the reviewer.
- **The full 30as re-run (Phases 1–3) is still worth doing, but as an improvement, not a rescue:** it recovers 145M coastal people, fixes 4–6 pp country-level biases (Angola, DPRK, Syria, Mali, DRC...), and sharpens the district maps — at ~1 week engineering + tens of dollars of compute. It will not move the headline global number by more than ~half a point.
- Priority order vs other open work: below the borehole/unpiped breakdown (multi-pp effect on headline), roughly on par with a WorldPop population sensitivity (§1a), above everything cosmetic.

---

## Appendix: key evidence trail

- Kummu 30as native: Dryad `doi:10.5061/dryad.71c6r` / Zenodo 4956396 — both files, CC0; methods state 1 km computation aggregated for compute reasons.
- 1-km lattice fingerprint in `dtw_1`: unique values are exactly 1000·√(a²+b²), 55,787 distinct.
- Solver cacheability: `src/mobility_module.py:203-239` (Lankford), `:241-335` (bike) — inputs are slope, Crr, body mass, draw scalars only. Measured 173.7 µs/cell combined.
- Cell counts: 2,190,446 land / 1,403,388 modelled (measured from the master CSV).
- Memory: ~1,176 B/row final frame → 1.65 GB now, ~165 GB naive at 30as; worker peak ~3.3 GB matches `run_monte_carlo_gcp.py:31-34` comments.
- GCP history: `gcp/gcp-setup.md:25-32` (2–3 days on 32 vCPU), OOM history at `run_monte_carlo_gcp.py:197-203`.
- GRIP raster only at 5 arcmin: globio.info/download-grip-dataset.
- EarthEnv slope at 1 km (computed at 250 m, aggregated): earthenv.org/topography, Amatulli et al. 2018.
- Rural population bias: Láng-Ritter et al. 2025, Nat Commun 16:2170 (JRC dispute noted).
- Resolution-bias analogue: Punger & West 2013, Air Qual Atmos Health 6(3).
