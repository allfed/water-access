# Follow-up paper idea: "Distance to water, 2026" — an updated Kummu et al. (2011) layer

**Status:** idea approved by Kevin 2026-08-09; not started. This note captures the evidence so the project can be specced cold later (e.g. via /fable-plan).

## The pitch

Kummu et al. 2011 (PLoS ONE, "How close do we live to water?", doi:10.1371/journal.pone.0020578) is the standard global distance-to-surface-water layer and made a splash. Its water mask is VMAP0 (1:1,000,000-scale, 1990s vintage) + GLWD lakes ≥0.1 km², **perennial water only** — the paper itself lists seasonal rivers, wetlands, small streams and springs as excluded. Modern inputs (Landsat/Sentinel water masks, MERIT Hydro) see far more water.

**Verified gap (researched 2026-08-09, web-grounded):**
- No public successor exists: Kummu's group (Aalto WDRG) lists no update; JRC publishes the 30 m water mask (Pekel et al. 2016) but no distance derivative; MERIT Hydro / HydroLAKES / GRWL are inputs, not distance layers.
- No published study quantifies how distance-to-water changes when swapping VMAP0-era hydrography for satellite-derived water. Novel contribution, update-replication framing.

## Recipe sketch (2026 rebuild)

- **Rivers:** MERIT Hydro (3 arcsec; width / upstream-area threshold — THE key methodological decision, controls how much closer rural people get vs Kummu's river classes). Cross-check vs GRWL centerlines/widths.
- **Lakes:** HydroLAKES (≥0.1 km², matches GLWD cutoff) or JRC GSW permanent class as independent check.
- **Seasonal water:** JRC GSW seasonality classes. For crisis use, seasonal water arguably counts → run permanent-only AND permanent+seasonal, report the delta (also keeps a Kummu-comparable variant).
- **Coastline:** GSHHG / OSM / Natural Earth (low controversy).
- **Compute:** binary mask on the 30 arcsec master grid (~933M cells, ~3.7 GB float32) → distance transform (GDAL proximity / chunked scipy EDT; GEE fastDistanceTransform as alternative). Laptop-feasible at 30as (days); 3 arcsec native = modest cloud job. Fix Kummu's planar-degrees distance with geodesic-aware or per-tile equal-distance projection (documented upgrade).

## Why it's cheap for us specifically

The scripted extraction pipeline (scripts/extraction/, built 2026-08) has a frozen 22-column contract: a new layer drops into the `dtw_1` slot and the ENTIRE water-access model re-runs with zero code change (fixed-param run ≈ 26 s at 5 arcmin; 10-draw 30as MC ≈ 100 min). The comparison table (Kummu vs 2026 layer, population-weighted, by country) falls straight out of the existing validation harness pattern.

## Framing / scope guards

- Companion note / short paper, NOT part of the current review response (results-changing input swap mid-review = new validation cycle + second reviewer front).
- Current paper states the limitation with direction: Kummu perennial-only ⇒ modelled distances are an over-estimate ⇒ access numbers are a floor.
- Estimated effort: 2–4 weeks part-time.

## Key sources

Kummu et al. 2011 (PLoS ONE); Dryad doi:10.5061/dryad.71c6r (CC0); Pekel et al. 2016 (Nature, JRC GSW); Yamazaki et al. 2019 (WRR, MERIT Hydro); Allen & Pavelsky (GRWL); Yao/Liu et al. 2022 (IJRS — even 30 m products miss small rivers); Láng-Ritter et al. 2025 (rural population undercount — relevant to the pop-weighting side).
