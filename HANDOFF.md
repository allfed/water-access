# HANDOFF — branch `kr/unpiped-breakdown`

Updated 2026-08-11. For the next agent, zero context assumed. The
what/how is documented in the repo; this file covers only what isn't.

## Where we got to

Three phases done on this branch:

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
3. **Lose fractions wired into the Monte Carlo — DONE, single driver.**
   `scripts/run_monte_carlo_gcp.py` (the script `gcp/deploy-spot.sh`
   launches) samples `urban/rural_borehole_lose_fraction` uniformly over the
   HEMP ranges above and threads them through `run_simulation()` into
   `calculate_population_water_access()`. The old local driver
   `scripts/run_monte_carlo.py` was RETIRED 2026-08-11 (Kevin's call: two
   drivers had let parameter ranges drift apart — the local copy still had
   the pre-review mobility ranges, and its retained-futures memory pattern
   could not survive 1000 iterations anyway). Smoke-tested end-to-end with
   3 iterations (commit `5811ddf`, `results/smoketest_3runs/` — pipeline
   completes, sampled lose fractions vary per run; note the smoke ran on
   the since-retired local script, so its *mobility* draws used the stale
   ranges — treat it purely as evidence the wiring works, never as results).

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
per Kevin's decision. Don't quote anything run on the old code-default
fractions. Note a checkpoint that predates `unallocated_vulnerable_fractions`
is rejected loudly — wipe and restart rather than resuming across it.

Merge note for `kr/30as-feasibility`: that branch reintroduces its own
`run_monte_carlo.py` (the 30as CLI harness) plus a shared sampling module
`src/monte_carlo_constants.py`. At merge, take the 30as versions and move
the lose-fraction sampling from `run_monte_carlo_gcp.py` into
`sample_parameters()` there, so parameters live in exactly one place.

## In-flight / out of scope

- **All Monte Carlo results in `results/` are stale twice over** (except the
  explicitly-labelled `results/smoketest_3runs/`, which is a wiring test, not
  results): they predate the new `merged_data.csv` AND the new lose-fraction
  ranges. Rerun before quoting any numbers in the manuscript.
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
  triangular is a two-line change in `run_monte_carlo_gcp.py` if preferred.
- India post-2018 handpump/tubewell split: unresolvable until the next NSS
  drinking-water round or census; the 70–80/25–38 ranges carry it.
- 30-arcsec feasibility call: Kevin (branch `kr/30as-feasibility`).
- Manuscript edits from review comments + James's own branch push: James.

## Environment gotcha

`environment.yml` pins pandas 1.4.3; system python3.11 (pyenv, pandas 2.1.4)
makes 8 pre-existing tests fail in `test_gis_global_module`/
`test_gis_monte_carlo` (groupby/MultiIndex differences) — NOT from this
branch. Parser, reconciliation and mobility tests pass on 3.11.
