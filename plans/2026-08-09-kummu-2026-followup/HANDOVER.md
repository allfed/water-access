# Handover: plan the "Distance to Water 2026" follow-up (Kummu update-replication)

**For:** a fresh `/fable-plan` session · **From:** the 30as pipeline build session (2026-08-09) · **Owner:** Kevin Rassool (ALLFED water-access)

## The task to plan

A short follow-up paper + open data release: an **updated global distance-to-surface-water layer** (successor to Kummu et al. 2011, PLoS ONE, doi:10.1371/journal.pone.0020578) built from modern hydrography, published as an **update-replication** of Kummu's population-distance analysis, with an **assumption-matrix data release** (water definition × population product) so downstream users choose their assumptions. Kevin approved this 2026-08-09; effort budget ~2–4 weeks part-time.

## Read these first (in order — they carry the verified evidence; do NOT re-research what they settle)

1. `plans/2026-08-09-kummu-2026-followup/IDEA.md` — the pitch, the verified literature gap (no public successor layer exists; no published VMAP0-vs-satellite distance comparison), the 2026 rebuild recipe, the assumption-matrix extension, scope guards, key sources.
2. `plans/2026-08-08-30as-pipeline/BUILD_SUMMARY.md` — state of the water-access pipeline this feeds into (all gates green; ADR-10 cos-lat default; production run pending).
3. `scripts/extraction/README.md` — the extraction pipeline's mechanics: master grid definition, layer readers, provenance conventions, runtime actuals. The new layer must slot into this machinery.
4. `FEASIBILITY_30_ARCSEC.md` §1–2 — why dtw is the decision-driving layer; the Kummu fingerprinting work (how we proved which lattice/values the legacy used).
5. `results/validation/gis_methodology_review.md` §D — where this idea came from; the honest best-practice landscape.

## Facts already verified (2026-08-09, web-grounded — cite, don't re-derive)

- **Kummu 2011 method:** VMAP0 (~1:1M, 1990s) rivers + GLWD lakes ≥0.1 km² + coastline; **perennial water only** (seasonal/wetlands/small streams explicitly excluded, listed as limitations in the paper); Euclidean distance at 1 km native, aggregated to 5 km; delivered EPSG:4326 (planar-degree distance — a known distortion to fix). Data: Dryad doi:10.5061/dryad.71c6r, CC0.
- **No successor exists:** Aalto WDRG lists no update; JRC GSW (Pekel et al. 2016) is a 30 m water *mask*, no distance derivative published; MERIT Hydro / HydroLAKES / GRWL are inputs, not distance layers.
- **Rebuild inputs:** MERIT Hydro (3 arcsec, river width + upstream area — the minimum-river threshold is THE key methodological decision), HydroLAKES (≥0.1 km², matches GLWD cutoff) or GSW permanent class, GSW seasonality classes for the permanent-vs-seasonal fork, GSHHG/OSM coastline.
- **Compute:** 30as global grid = 43200×21600 ≈ 933M cells ≈ 3.7 GB float32 — laptop-feasible distance transform (GDAL proximity / chunked scipy EDT); 3 arcsec native = modest cloud job. The M4/32GB handled the full 30as extraction in ~7 min.
- **Known critique to cite:** even 30 m products miss small rivers (Yao/Liu et al. 2022, IJRS) — VMAP0-era products miss far more.

## Hard constraints

- **The 22-column contract is frozen** (`plans/2026-08-08-30as-pipeline/SPEC.md` ADR-2): the new layer must be producible as a drop-in `dtw_1` on the Kummu master grid (ADR-1: 43200×21600, EPSG:4326, origin −179.999999/90.0024163) so the water-access model re-runs with zero code change. Native-resolution or alternate-grid versions may ALSO be released, but the master-grid variant is mandatory.
- **Strictly separate from the current paper's review response.** The manuscript takes limitation notes only (drafted in BUILD_SUMMARY.md). Nothing here blocks or touches the pending 1000-iteration production run.
- A Kummu-comparable variant (permanent-only water) MUST be one of the released layers, or the "replication" framing fails.

## Decisions the plan must resolve WITH Kevin (interactive — that's what fable-plan is for)

1. **Minimum river size** — width threshold (GRWL-comparable) vs upstream-area threshold (MERIT), and the sensitivity story around it.
2. **The matrix's exact axes** — proposed: water ∈ {permanent-only, permanent+seasonal} × population ∈ {GPW v4.11, WorldPop constrained, GHS-POP} = 6 variants; confirm or trim. (Population plumbing costs: WorldPop 100m→30as aggregation, GHS-POP fractional-offset windowing — offset already measured in Phase B of the pipeline build.)
3. **Distance metric** — geodesic-corrected (recommended upgrade over Kummu's planar) vs planar-for-comparability, or both with the delta reported.
4. **Output resolutions** — 30as on the master grid (mandatory); is a 3as native release worth the cloud job?
5. **Replication scope** — which of Kummu 2011's actual analyses/figures get re-run (population within X km of water, by country/continent; the update-replication table)? Re-running them through the existing pipeline is cheap.
6. **Venue/format** — data descriptor (Scientific Data, ESSD) vs short research note; determines how much analysis vs documentation.
7. **Where the code lives** — this repo (e.g. `scripts/dtw2026/`, reusing the extraction package's grid/manifest conventions) vs a fresh repo with this one as a dependency. Kevin leans reproducibility; the extraction README's provenance pattern is the house style either way.
8. **Data deposit target** — Zenodo/Dryad, licence (Kummu was CC0), and whether the water-access model outputs per matrix cell are part of the release.

## Suggested phase shape (for the plan to refine, not adopt blindly)

acquire+manifest sources → build water masks (per water-definition variant) → distance transform (+geodesic correction, validation against Kummu on his own terms) → drop into `dtw_1` / re-run population analysis (replication tables) → assumption-matrix runs → deposit + paper draft.

## House conventions (from the pipeline build — keep them)

- Every source dataset: URL, version, licence, download command in a MANIFEST (see `data/GIS/source_rasters/MANIFEST.md` for format).
- Every claim gate-tested; validation artefacts committed under `results/validation/`.
- Reproducibility README such that a cold reader can re-derive everything (AC-9 pattern).
- Use `.venv/bin/python` (py3.11, rasterio/geopandas installed); black for style; short imperative commit messages.
