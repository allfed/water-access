# HANDOFF — branch `kr/integration-30as-unpiped`

Updated 2026-08-11. For the next agent, zero context assumed. The
what/how is documented in the repo; this file covers only what isn't.

## Where we got to

`kr/unpiped-breakdown` and `kr/30as-feasibility` are MERGED here (merge
commit `1a4c660`); this branch is the integration line. Four phases:

1. **Unified source breakdown** (53 of the 76 blank JMP rows filled, 23
   judged not fillable; model switched over, `merged_data.csv`
   regenerated). Read
   `HANDOVER_unified_source_breakdown.md`, then
   `data/original_data/researched_source_breakdown_README.md`, then README
   §"Unpiped source breakdown".
2. **Electric vs hand-pumped borehole share — DONE.** Read
   `ELECTRIC_BOREHOLE_SHARE_ESTIMATE.md` (method, verification, recommended
   Monte Carlo ranges, caveats) and `BOREHOLE_PUMP_TYPE_BY_COUNTRY.md`
   (per-country table). Headline, HEMP scenario:
   `urban_borehole_lose_fraction` 0.63–0.80 (central ~0.72),
   `rural_borehole_lose_fraction` 0.23–0.35 (central ~0.29);
   loss-of-central-infrastructure = HEMP × 0.75–0.95 urban / 0.85–1.0 rural.
   Data: `data/original_data/borehole_pump_type_estimates.csv` (+ raw research
   in `borehole_pump_type_research.json`); aggregation script:
   `scripts/Data_Manipulation_Scripts/aggregate_borehole_pump_estimates.py`.
3. **Lose fractions wired into the Monte Carlo — DONE.**
   `urban/rural_borehole_lose_fraction` are sampled uniformly over the HEMP
   ranges above and threaded through `run_simulation()` into
   `calculate_population_water_access()`. Smoke-tested end-to-end with
   3 iterations (commit `5811ddf`, `results/smoketest_3runs/` — pipeline
   completes, sampled lose fractions vary per run; that smoke predates the
   canonical mobility ranges, so treat it purely as evidence the wiring
   works, never as results).
4. **30-arcsecond pipeline merged in — DONE 2026-08-11 (`1a4c660`).** See
   §"State after the merge" below before touching anything.

## State after the merge

- **Parameters live in exactly one place**: `src/monte_carlo_constants.py`
  (ADR-7). It owns the mobility distributions *and* the two borehole
  lose-fraction ranges, the unallocated bounds, and the scenario switch
  `UNALLOCATED_VULNERABLE_FRACTION_FIXED` (`None` = sample g ~ U(0,1);
  `"prorata"` = central; `0.0`/`1.0` = brackets). `PARAMETER_NAMES` is
  ordered to match `run_simulation`'s positional signature.
- **Three drivers, one sampler.** `scripts/run_monte_carlo_gcp.py` is the
  production/GCP driver (checkpointing, per-worker disk writes — this is
  what `gcp/deploy-spot.sh` launches). `scripts/run_monte_carlo.py` is the
  local/CLI harness that came back with the 30as work (`--resolution`,
  `--chunk-rows`, `--seed`, `--input`). `scripts/run_monte_carlo_test.py`
  is a 2-iteration smoke. All three import `sample_parameters`; a test
  asserts the GCP driver holds no private copy of the constants.
- **Chunked ≡ monolithic, re-proved with the new parameters.**
  `scripts/validate_chunking.py` now runs at urban 0.72 / rural 0.29 /
  `"prorata"` — deliberately off the code defaults. Country and district
  rows bit-identical, GLOBAL and split-country rows within 1.8e-16
  relative (`results/validation/chunking_report.md`). The chunk column
  contract had to gain the twelve source-share columns: the model *defaults*
  four of the six shares to zero when they are missing, so a chunked run
  would otherwise have modelled packaged/delivered/borehole users as piped,
  silently.
- **The GLOBAL row is now exactly the sum of the country rows it is
  published beside** (the outlier cut runs before it is built), in both
  execution modes and at both resolutions.
- **`results/regression_baseline/` was RE-FROZEN in the merge commit** — a
  deliberate model change (new `merged_data.csv`, Unallocated accounting,
  trim ordering), not a regression. Precedent: `be80d80`. Anything that
  diffs against the old baseline will be comparing across a model change.
- **Output columns renamed** `population_piped_*` → `population_affected_*`
  (and `percent_piped_*` → `percent_affected_*`). Run directories written
  before 2026-08-11 carry the old names and cannot be read by
  `compare_smoke_runs.py`; rerun rather than renaming, since their numbers
  predate the model change too.
- Environment: python ≥ 3.11 / pandas 2.x (`requirements.txt`,
  `environment.yml`). Full suite: **275 passed**.

## Single next action

Run the full NUM_ITERATIONS=1000 production Monte Carlo via
`gcp/deploy-spot.sh` and re-derive the manuscript numbers from it.
**Before launching: wipe the GCS bucket state**
(`gs://<bucket>/checkpoints` and `gs://<bucket>/results/parquet_files`) —
a leftover checkpoint silently restores the OLD parameter arrays and
leftover parquets count as completed iterations (hazard notes are in the
script and deploy-spot.sh). Scenario logic lives in the
`urban/rural_borehole_lose_fraction` AND `unallocated_vulnerable_fraction`
parameters of `gis_global_module.calculate_population_water_access()`
(HEMP vs CIL as different parameter ranges), NOT baked into the dataset —
per Kevin's decision; the ranges themselves are now edited in
`src/monte_carlo_constants.py`. Don't quote anything run on the old
code-default fractions. Note a checkpoint that predates
`unallocated_vulnerable_fractions` is rejected loudly — wipe and restart
rather than resuming across it.

Open decision that gates the *resolution* of that run: whether to go to
30 arc-seconds. The pipeline is merged and works, but the extraction inputs
(`data/GIS/source_rasters/`, `data/GIS/extracted/`) are gitignored and are
NOT in this checkout — they live in the `.claude/worktrees/30as-feasibility`
worktree. Everything above was validated at 5 arc-minutes from
repo-committed data. See `FEASIBILITY_30_ARCSEC.md`.

## In-flight / out of scope

- **All Monte Carlo results in `results/` are stale** (except the
  explicitly-labelled `results/smoketest_3runs/`, a wiring test, and
  `results/regression_baseline/` + `results/regression_new/`, which are
  fixed-parameter regression references re-frozen in the merge commit, not
  Monte Carlo output): they predate the new `merged_data.csv`, the new
  lose-fraction ranges and the merge. Rerun before quoting any numbers in
  the manuscript. `results/validation/smoke30*` and `twin5min*` also predate
  the column rename.
- Per-country lose fractions (rather than the two global scalars) are
  supported by the CSV but not implemented; a single global scalar
  misallocates *where* displaced people are (matters for spatial results) —
  see the estimate doc's caveats. Optional upgrade, not required.
- Venezuela source breakdown still deliberately unfilled (ENCOVI microdata is
  the fix); retrievable-but-not-retrieved sources listed in
  `HANDOVER_unified_source_breakdown.md` §follow-ups.
- **RESOLVED 2026-08-11 (commits 3a4c507, 53f6788):** the 348M share-sum
  gap and the USA/SLB >100 rows. Investigation in
  `UNALLOCATED_GAP_INVESTIGATION.md`; implementation: parser reference and
  extraction fixes, JMP-ladder fill into OtherUnpiped (~128M, audit CSV in
  semi-processed), explicit `URBAN/RURALUnallocated` columns (169.5M
  people), and the sampled `unallocated_vulnerable_fraction` parameter
  (Uniform(0,1) in the GCP driver; deterministic scenarios: 0 = old
  behaviour, "prorata" = central, 1 = all-vulnerable). Every stratum's six
  shares now sum to exactly 100. Air-gapped verifier: SHIP, all criteria
  reproduced.
- **Open follow-ups from that change (Kevin holds the first):**
  (a) ROU/BIH now select sparse EUSILC/HBS JMP rows and the "JMP beats
  research" rule demotes richer research fills (ROU rural borehole 37→0,
  wrong direction for review comment 2; loud WARNING in the build output).
  Candidate fix is a per-category merge — JMP wins only where it measured —
  but that refines a Kevin decision, so it awaits his call. (b) Parser
  still ignores the "Other improved sources" sub-'Other' rows (102–103);
  a validator warning when row-99 total ≫ cart+tanker was specced
  (memo §6 L1.3) but not yet implemented — matters for PHL (45–61pp).
  (c) ISR/FLK workbooks use a variant row layout and land in
  jmp_source_breakdown_errors.csv rather than parsing. (d) ITA relabelled
  research→jmp with identical values (benign, noting for provenance).
- Deferred from the 2026-08-11 three-agent review (Kevin holds):
  downhill walking velocity is uncapped and peaks at 3.74 m/s (~13.5
  km/h) around 6° downhill — an artifact of the Lankford polynomial exposed
  by the slope-units fix; capping it is a modelling decision, not plumbing.

## Decisions made (don't relitigate without new information)

- lose_fraction = 1 − handpump share **among borehole users** (JMP row 57
  denominator, never all-households). The adversarial verifiers re-derived
  this arithmetic for IND/PAK (CONFIRMED digit-for-digit) and
  BGD/IDN/UGA/NGA/UKR (PLAUSIBLE, none refuted).
- MC bounds are the population-weighted envelope of per-country low/high
  bounds — defensible because the dominant uncertainties are systematic
  across countries (same inference method); uniform or triangular sampling
  both fine.
- India 2024 values are *extrapolated* from measured 2011/2018 anchors
  (post-2018 surveys publish only the combined handpump+tubewell category) —
  documented, not a bug.
- Regional defaults fill the ~5% tail; Northern Africa's default is known-bad
  for Egypt's Nile Delta handpumps (~0.4% of global weight, flagged).
- Prior-phase decisions (JMP beats research fills, multi-response only in
  bounds_notes, piped reconciliation carve-outs, dug wells ≠ boreholes) stand
  — see the source-breakdown README.

## Dead ends and corrections from the research pass

- Nigeria: WASH-NORM functionality-by-type numbers first cited by research
  were wrong (they matched nothing in the report); the real national figures
  run the other way (handpump boreholes 78% functional vs motorized 64%),
  which is why NGA urban was corrected UP to 10–25% handpump.
- Uganda: RWSN's "67% of rural population rely on a handpump" is a
  conflation (67% is the technology mix *of improved supply*); the 88–99%
  rural bound survives via category structure instead (motorized boreholes
  feed piped schemes, whose users leave the Borehole category).
- Don't read pump-stock counts as user shares (motorized points serve more
  people); inventory-derived rows are bounds, flagged in the CSV.
- (Prior phase) spatial/GDP imputation of breakdowns, and utility raw-water
  mixes as household shares: both rejected, see source-breakdown README.

## Blocking open questions / who holds the ball

- Rerunning the MC with the new ranges + manuscript numbers: James. The
  wiring as committed samples uniform over the envelope ranges; the estimate
  doc says uniform or triangular are both defensible, so switching to
  triangular is a two-line change in `src/monte_carlo_constants.py` if
  preferred — and it changes every driver at once, which is the point.
- India post-2018 handpump/tubewell split: unresolvable until the next NSS
  drinking-water round or census; the 70–80/25–38 ranges carry it.
- 30-arcsec go/no-go for the production run: Kevin. The code is merged;
  what is outstanding is the input data (see §Single next action) and the
  compute call.
- Manuscript edits from review comments + James's own branch push: James.

## Environment

`environment.yml` and `requirements.txt` now pin python >= 3.11 and
pandas >= 2.1, < 3.0 (the 30as branch modernised them; the old 1.4.3 pin
could not supply the rasterio/geopandas versions the extraction needs). The
8 pandas-2 failures the unpiped branch carried in `test_gis_global_module`/
`test_gis_monte_carlo` were fixed on the 30as side and are gone: the whole
suite is green on system python3.11 / pandas 2.1.4.

    python3.11 -m pytest tests/ -q                      # 275 passed
    python3.11 scripts/validate_chunking.py             # chunked == monolithic
    python3.11 scripts/capture_baseline.py --out results/regression_new
    python3.11 scripts/validate_regression.py           # Gate 2
