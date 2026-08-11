# 30 arc-second pipeline upgrade — build summary & status

**Dates:** 2026-08-08 → 2026-08-09 · **Branch:** `kr/30as-feasibility` (28 commits, `b487093..59d4331`+) · **Process:** fable-orchestration build against `SPEC.md`, independently verified · **Spec:** `plans/2026-08-08-30as-pipeline/SPEC.md` · **Evidence base:** `FEASIBILITY_30_ARCSEC.md`

## What was built

1. **Environment + baseline (Phase A).** Suite green on py3.11/pandas 2.x (was 8 failures from pandas drift); Monte Carlo sampling parameters unified in `src/monte_carlo_constants.py` (GCP values canonical: lognormal 2–4 h TIME_GATHERING, met 3–5, watts 20–60); resolution-aware population filter (ADR-8); pre-refactor regression baseline frozen at `results/regression_baseline/`.
2. **Scripted extraction (Phase B)** — `scripts/extraction/`. Replaces the undocumented manual QGIS workflow. Runs at 5 arcmin and 30 arcsec from public rasters on the Kummu master grid. **Gate 1:** dtw/slope/all-GRIP **100% cell-exact** vs the legacy CSV; 2,190,444/2,190,446 legacy rows reproduced; 5-entry documented exception list. Full provenance in `scripts/extraction/README.md` + manifests.
3. **Solver LUT + memory refactor (Phases C/C2)** — `src/solver_lut.py`, `src/chunked_analysis.py`. Per-draw lookup table replaces per-cell fsolve (validated to max 6×10⁻⁵ m/s vs bar 0.005); country-group chunked execution with district-boundary splits, proven **bit-identical** to monolithic. Fixed-parameter run: ~4 min → **~25 s**. **Gate 2:** all outputs within 0.002 pp of frozen baseline.
4. **30as smoke (Phase D).** Full 30as extraction: 219M rows / 3.36 GB / 6m40s. 10-iteration Monte Carlo: **100 min, 16.4 GB peak RSS** (budget 24 GB / 12 h). AC-8: global access differs from 5 arcmin by **+0.018 pp**; walking-access shift +0.52 pp matches Phase 0's +0.46 pp prediction.

**Independent verification (air-gapped, 2026-08-09): PASS — all 10 acceptance criteria**, re-run cold; adversarial probes confirmed the validators genuinely fail when they should.

## Findings made along the way (each evidence-backed, committed)

| Finding | Consequence |
|---|---|
| Legacy slope is EarthEnv **GMTEDmd (median)**, not mn — bit-exact proof | ADR-3 amended (Kevin sign-off, `1f8128a`); md is default, mn a `--slope-raster` fallback |
| Legacy population is **unadjusted** GPW, NOT UN-WPP-adjusted (earlier hypothesis refuted by cell-level evidence) | Unadjusted stays default; adjusted converted + archived as `--pop-raster` alternate; downstream-neutral either way (renormalisation cancels per-country scalars) |
| `--input` never reached spawned MC workers (pre-existing) | Fixed; earlier smoke evidence re-run |
| Input-swap (legacy CSV → scripted extraction, fixed params): **global +0.044 pp**; country pop-weighted 0.44 pp; tails DJI +19 pp, district p90 4.7 pp | `results/validation/input_swap_comparison.md`; global story robust, district tails pipeline-sensitive |
| **cos(latitude) missing from population placement** (people = density × area; area term omitted) | Real bug, modest effect: global −0.083 pp, worst NOR +1.32 pp, only 1 country >0.5 pp, resolution-stable. Implemented behind `area_weighted_population`, **default ON since 2026-08-09** (Kevin: leaving a genuine bug off-by-default is a trap); legacy split reachable via `--no-area-weighted*`. Chunk-safe, tests 109→123, regression baseline re-frozen. `results/validation/cos_lat_diff.md` |
| Legacy boundaries were originally Garmin-via-ArcGIS per the handwritten note, switched to CGAZ undocumented | Explains residual cell-count exceptions; recorded in methodology review |

## Decisions log (Kevin)

- 2026-08-08: scope = Phases 1+2 + smoke only; TIME_GATHERING 2–4 h canonical; Gate 1 statistical bar; local M4 first.
- 2026-08-09: ADR-3 slope amended to md. Unadjusted GPW retained (evidence-driven). **cos(lat): adopt at the production run** (flip `--area-weighted-population` ON and re-freeze the regression baseline in the same commit — NOT retroactively). **Superseded the same day: adopt as the DEFAULT now** — an off-by-default correction is a trap for the next reader; baseline re-frozen against the corrected model (legacy behaviour stays reachable, and the pre-correction baseline stays in git history). **Production run: one more run, 1000 iterations at 30 arcsec, cos(lat) ON.** Population product ensemble: **deferred to the Kummu follow-up paper**; current manuscript takes a limitation note (see below).
- Kummu 2026 follow-up paper approved as a separate effort: `plans/2026-08-09-kummu-2026-followup/IDEA.md`.

## Manuscript limitation notes (draft language)

1. *Population placement:* "Gridded population (GPW v4.11) distributes census-unit counts uniformly within administrative units; in sparsely-enumerated rural areas this smooths population toward a uniform density, and global gridded products are known to under-represent rural populations (Láng-Ritter et al. 2025). Country-level results are insensitive to the choice of gridded product (totals are renormalised to national populations), but district-level figures inherit placement uncertainty; we quantify pipeline sensitivity at the district level in SI [input-swap table]."
2. *Water layer:* "Distance to water uses Kummu et al. (2011), which maps perennial water bodies from ~1:1M-scale sources and excludes seasonal water, wetlands and small streams; modelled distances are therefore upper bounds and access estimates a floor."
3. *Roads:* "Road density retains its native 5 arc-minute resolution (GRIP4 rasters); the 30 arc-second grid sharpens water distance and population, not road information."
4. *Distance metric:* "Straight-line distance with urban/rural circuity multipliers approximates route distance; least-cost-path methods over friction surfaces (Weiss et al. 2018) are a successor approach, not currently generalisable to loaded water-carrying off-network."

## Next steps

1. **Production run (pending Kevin's go):** 1000 iterations, 30as — cos(lat) is now the default and the baseline is already re-frozen, so this is just GCP dry-run smoke, then the 1000. Timing: ~8 min/draw single-worker ⇒ **~5.6 days on the M4**; options to compress: 2 workers at `--chunk-rows 6000000` (verify RSS on a 10-draw smoke first, projected ~2× ~8.2 GB), chunk-plan caching, or the GCP fallback. Also revisit whether 1000 draws are needed vs ~300 for the reported quantiles.
2. **Kummu 2026 follow-up** (separate paper): updated distance-to-water layer + assumption-matrix release (water definition × population product — see IDEA.md).
3. Optional pre-submission: exactextract district-aggregation cross-check (~1 day, certifies the boundary-assignment shortcut).

## Key artefacts

`results/validation/`: `gate1_report.md`, `gate2_report.md`, `lut_report.md`, `chunking_report*.md`, `smoke_comparison.md`, `smoke30_run.log`, `input_swap_comparison.md`, `cos_lat_diff.md`, `gis_methodology_review.md` · `results/regression_baseline/` (frozen) · `scripts/extraction/README.md` (reproducibility) · `plans/2026-08-08-30as-pipeline/{SPEC,DATA_MANIFEST,BUILD_SUMMARY}.md` · `plans/2026-08-09-kummu-2026-followup/IDEA.md`
