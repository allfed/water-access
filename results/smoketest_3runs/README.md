# SMOKE TEST ONLY — not paper results

3-iteration end-to-end test of the borehole lose-fraction wiring
(urban/rural_borehole_lose_fraction sampled uniformly over the HEMP ranges
from ELECTRIC_BOREHOLE_SHARE_ESTIMATE.md: urban 0.63-0.80, rural 0.23-0.35).
Run 2026-08-10 on a 4-core container (MAX_WORKERS=3), pandas 1.5.3.

These outputs only demonstrate that the pipeline completes and the sampled
lose fractions vary across runs (see sampled_parameters_smoketest.csv and the
per-run countries_simulation_result_*.parquet). With 3 iterations the
statistics are noise — the manuscript needs the full NUM_ITERATIONS=1000
batch, which is still pending. Zone-level parquets are not committed (size).
