# HANDOFF — branch `kr/unpiped-breakdown`

Updated 2026-08-10 (evening). For the next agent, zero context assumed. The
what/how is documented in the repo; this file covers only what isn't.

## Where we got to

Two phases done on this branch:

1. **Unified source breakdown** (76 blank JMP rows filled, model switched
   over, `merged_data.csv` regenerated). Read
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

## Single next action

Wire the estimate into the Monte Carlo scenario configs and rerun. Model code
was deliberately NOT touched: per Kevin's decision, scenario logic lives in
the `urban/rural_borehole_lose_fraction` parameters of
`gis_global_module.calculate_population_water_access()` (HEMP vs CIL as
different parameter ranges), NOT baked into the dataset. The code defaults
(0.5/0.25) understate urban losses — don't quote results run on them.

## In-flight / out of scope

- **All Monte Carlo results in `results/` are stale twice over**: they
  predate the new `merged_data.csv` AND the new lose-fraction ranges. Rerun
  before quoting any numbers in the manuscript.
- Per-country lose fractions (rather than the two global scalars) are
  supported by the CSV but not implemented; a single global scalar
  misallocates *where* displaced people are (matters for spatial results) —
  see the estimate doc's caveats. Optional upgrade, not required.
- Venezuela source breakdown still deliberately unfilled (ENCOVI microdata is
  the fix); retrievable-but-not-retrieved sources listed in
  `HANDOVER_unified_source_breakdown.md` §follow-ups.
- USA rural / SLB urban sum >100 in merged_data: pre-existing parser
  mismatch, deliberately untouched, flagged to James.

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

- Rerunning the MC with the new ranges + manuscript numbers: James (Kevin to
  confirm the scenario parameterization choice — envelope vs triangular).
- India post-2018 handpump/tubewell split: unresolvable until the next NSS
  drinking-water round or census; the 70–80/25–38 ranges carry it.
- 30-arcsec feasibility call: Kevin (branch `kr/30as-feasibility`).
- Manuscript edits from review comments + James's own branch push: James.

## Environment gotcha

`environment.yml` pins pandas 1.4.3; system python3.11 (pyenv, pandas 2.1.4)
makes 8 pre-existing tests fail in `test_gis_global_module`/
`test_gis_monte_carlo` (groupby/MultiIndex differences) — NOT from this
branch. Parser, reconciliation and mobility tests pass on 3.11.
