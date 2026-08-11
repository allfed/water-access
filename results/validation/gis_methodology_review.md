# GIS extraction methodology — landscape review (ALLFED water-access model)

*Independent review, 2026-08-09 (Claude Fable). Inputs: legacy handwritten workflow note + Roam project export, the scripted extraction pipeline (scripts/extraction/), SPEC ADRs, FEASIBILITY_30_ARCSEC.md, Gate 1 / input-swap / smoke validation artefacts. Constraint: recommendations must preserve the 22-column output contract.*

## A. What was done (legacy)

**Reconstruction.** From the handwritten note and the Roam export (`data/GIS/Project- water access...md`, "# GIS" section, lines 527–629): import Kummu dtw, GPW v4, a Garmin-derived country shapefile (geometries needed fixing); build a QGIS vector polygon grid; take centroids; join-by-location to country polygons; run "sample raster values" per raster at those centroids; export CSV. All in the QGIS 3.36 UI, undocumented beyond "follow the methods in the paper / contact ALLFED". The Roam export shows the dataset choices were deliberate: GRIP density rasters chosen over vectors as "too unwieldy" (line 538), GHS-SMOD chosen because its settlement-approach urban/rural definition matches JMP's (lines 573–576), classes 11–13 = rural mapped from first principles (lines 605–619).

**Specific weaknesses.**
- *Centroid point-sampling, not area statistics.* Each 10 km cell got the single nearest source-pixel value at its centre. For dtw this is provably what happened — the CSV's dtw values are exact Kummu-lattice values (FEASIBILITY §1b). For a 1 km categorical layer (SMOD) sampled at one point per 10 km cell, 99% of the information is discarded.
- *Boundary provenance drift.* The note names a Garmin-via-ArcGIS shapefile, but the final artefact matches geoBoundaries CGAZ ADM1 (ISOCODE 100%, shapeID 99.9995% — gate1_report). The boundary source changed mid-project with no record; the Garmin layer's licence and comparability were never resolved.
- *UI irreproducibility.* No script, no parameter record; an "unexplained extra transformation" between pre-process outputs and the cleaned CSV is on record (SPEC ADR-9); the population resampling could never be reconstructed exactly (Gate 1: 61% cell agreement).
- *Collateral defects:* mojibake in non-ASCII shapeNames; 144.5M coastal people silently dropped by 5-arcmin nodata overlap (FEASIBILITY §6c); 5 arcmin used at all only because 30as was "too much heavy lifting for my computer" (src/gis_global_module.py:236 comment).

**What it got right.** It implicitly adopted the Kummu lattice as master grid (lossless for the decision-driving layer); it sampled consistently (nearest everywhere), so the new pipeline could reproduce it; the *dataset choices* were sound and survive unchanged; and for a global threshold analysis where the decision is essentially `dtw < ~6.5 km`, centroid sampling of dtw at 5 arcmin was adequate for the headline number — Phase 0 measured the global resolution bias at +0.46 pp.

## B. What has been done (current pipeline)

**At or above contemporary standard:**
- Master grid read verbatim off the grid-defining raster including its non-obvious origin, everything else windowed/reprojected onto it (grid.py, layers.py). Blocked streaming reads: full 30as build in 6m40s / 2.3 GB RSS.
- Full provenance: URL, version, licence per source (README §2, MANIFEST.md); Parquet footer carries cell size and provenance.
- A real validation regime: Gate 1 fidelity diff with thresholds and a magnitude-capped exception list (ADR-9); structural self-checks at 30as; an end-to-end input-swap experiment (input_swap_comparison.md) — rarer than it should be in published pipelines.
- Forensic source identification: legacy slope proven bit-exact to the EarthEnv *median* product; the GPW adjusted-variant hypothesis tested and refuted with cell-level evidence. dtw, slope, all five GRIP layers: 100.0000% agreement.
- CRS traps handled explicitly (GRIP crs=None, CGAZ undefined SRS, SMOD Mollweide).

**Legacy-compatibility choices that are NOT best practice (deliberate, documented):**
- *Point-lattice semantics.* Every cell value is a nearest-neighbour sample at the cell location. At 30as the harm is small (most layers are native ~1 km), but SMOD via nearest and slope via nearest are point choices, not area choices.
- *Boundary assignment by cell centre.* Cells whose centre falls in no ADM1 polygon are dropped (627.6M of 933M cells at 30as); border cells assigned wholly to one district.
- *Plate-carrée grid with no equal-area correction.* Critically, the model computes within-country population shares as `pop_density / Σ pop_density` with **no cos(latitude) cell-area weighting** (src/gis_global_module.py:442-456, verified). On an equal-angle grid, people per cell = density × area, and area shrinks ~cos(φ); a country spanning 17°–56° latitude has its poleward cells over-weighted by up to ~70% relative to equatorial ones. **A genuine, latent, legacy-inherited bias — unmeasured.**
- *GRIP broadcast at the 5-arcmin information floor* (defensible — re-rasterising the vector would encode data availability — but a floor).
- *No uncertainty propagation from input layers:* the Monte Carlo samples behavioural parameters only; population-product error (Láng-Ritter et al. 2025 rural undercount) and dtw content error enter nowhere.
- Within-country population *placement* remains unverifiable against legacy (pop_share_l1 median 12.3%), with measured district-tail consequences (input swap: district p90 4.7 pp; DJI +19.4 pp).

## C. What we SHOULD do (compromise tier, contract-preserving, ranked by benefit ÷ effort)

1. **Run production at 30as with the pipeline as-built.** Effort: compute only. Moves: recovers ~125M net coastal people, fixes 4–6 pp country biases (AGO, PRK, SYR, MLI, COD); headline moves ≤0.5 pp. Paper: one methods paragraph (drafted in FEASIBILITY §6d). Single highest-value action, already validated.
2. **Add cos(latitude) area weighting to the population share** (§B finding). Effort: ~half a day incl. before/after diff. Moves: within-country weighting in high-latitude/elongated countries (RUS, CAN, CHL, NOR, ARG); country shifts plausibly O(0.1–1 pp) there, global ~0. One methods sentence. Do the diff first — if it moves nothing, it becomes a verified non-issue for free.
3. **Quantify the rasterised-boundary error with an exactextract cross-check** (github.com/isciences/exactextract). District population totals and pop-weighted dtw via exact fractional-coverage zonal statistics, diffed against centre-assignment. Effort: ~1 day. Validation artefact, not a contract change: if deltas ≪ input-swap tails, centre-assignment is certified; if not, you know which districts to caveat.
4. **Publish the input-swap and smoke comparisons as SI**; state district-tail pipeline sensitivity explicitly. Effort: hours. Pre-empts the reviewer.
5. **One-flag sensitivity runs:** slope mn vs md (--slope-raster; cycling-sensitive), GPW-adjusted population (--pop-raster). Hours each; one SI line each.
6. **Coastal populated-cell rescue at 30as** — assign nearest polygon within one cell for populated no-polygon cells. 1–2 days; moves small-island/coastal states. Only if #3 shows the drop still material at 30as; otherwise a named limitation.

Not worth it: mode-based SMOD at 5 arcmin (validation-only resolution once production is 30as); GRIP vector re-rasterisation (manufactures false precision — correctly rejected).

## D. What we COULD do (best-practice tier)

What a top-tier global accessibility study (Weiss et al. 2018/2020, Malaria Atlas Project) does, and what each buys THIS model:

- **Friction surface + least-cost path instead of straight-line dtw × velocity.** MAP builds a ~1 km friction surface (OSM/Google roads, rail, rivers, slope, landcover), runs cumulative-cost travel time in Earth Engine (Weiss et al., Nature 2018, doi:10.1038/nature25181). This model approximates it with Euclidean dtw × slope-velocity × fixed circuity multipliers (urban 1.3 / rural 1.4 — themselves grounded in circuity literature per the Roam export, lines 819–867). A least-cost rewrite could keep the contract by emitting "effective walking distance" into dtw_1, but changes the methods wholesale, needs a defensible loaded-carry friction surface that doesn't exist off the shelf, and the evidence says the headline is threshold-dominated and robust at ±0.5 pp — gains land in district tails you can caveat. Right call for the NEXT paper (the Roam notes already propose it).
- **Better water layer, not better geometry.** Kummu 2011's weakness is content: perennial VMAP0 water only. Best practice: recompute distance-to-water from JRC Global Surface Water (Pekel et al. 2016, 30 m Landsat, seasonality) or MERIT Hydro. Contract-preserving and genuinely decision-relevant (moves the layer the model thresholds on), but a results-changing swap needing its own validation cycle and a crisis-relevance argument (is seasonal water usable?). The strongest COULD; the current layer's conservative direction is a legitimate interim defence.
- **Exact/area-weighted extraction everywhere + equal-area grids** (GHSL Mollweide, H3/ISEA DGGS): mostly superseded by C-2 + C-3 at a fraction of the cost; full regrid breaks the fid/bounds contract — excluded.
- **Dasymetric population ensemble** (GPW + WorldPop constrained + GHS-POP as scenarios via --pop-raster): cheap after the refactor, and rural population is exactly this paper's population of interest (Láng-Ritter 2025 rural undercount). The COULD most worth promoting to a sensitivity run.
- **Uncertainty propagation from inputs** (layer error as Monte Carlo dimensions): rarely done even in top-tier work; the product-ensemble above is the pragmatic 80%.
- **Cloud-native stack** (COG/STAC/Zarr, xarray/Dask, DuckDB-spatial, GEE): buys scalability no longer needed (whole 30as build is 6m40s on a laptop). Publishing the extracted Parquet + manifest as a data deposit gets 90% of the open-science value for ~zero effort.

## E. Verdict

The current pipeline is **defensible for publication as-is** — scripted, provenance-complete, validated at three levels (cell, gate, end-to-end input swap); on reproducibility it now exceeds typical practice for this class of study. Before the production run, exactly three changes: **(1)** commit to 30as as the production resolution (fixes the 145M coastal drop and country-tail biases); **(2)** add cos-latitude area weighting to the within-country population share with a before/after diff — the one latent methodological error found in this review; **(3)** run the exactextract district-aggregation cross-check and attach it plus the input-swap table as SI. Name as limitations rather than fix: the 5-arcmin roads information floor; straight-line distance with fixed circuity factors (citing the friction-surface approach as successor method); and the Kummu layer's perennial-only water content (biases conservative).

*References: exactextract (isciences.github.io), exactextractr docs, MAP Global Friction Surface 2019 & Accessibility 2019 (GEE catalog), Weiss et al. Nature 2018 doi:10.1038/nature25181, Pekel et al. Nature 2016, Láng-Ritter et al. 2025.*
