# HANDOFF — branch `kr/unpiped-breakdown`

Updated 2026-08-10. For the next agent, zero context assumed. The what/how is
documented in the repo; this file covers only what isn't.

## Where we got to

The 76 blank JMP source-breakdown rows are filled from researched national
sources, the model reads the unified file, and `merged_data.csv` is
regenerated. Read in this order:

1. `HANDOVER_unified_source_breakdown.md` (repo root): summary for James,
   covers the dataset, verification results, and reconciliation rules.
2. `data/original_data/researched_source_breakdown_README.md`: mapping
   rubric, flags, reconciliation details.
3. README section "Unpiped source breakdown": the 4-step rebuild pipeline.

## Single next action

Estimate the **electric vs hand-pumped borehole share** (per country or
income tier, as a range for Monte Carlo). No new plumbing needed: the knobs
are `urban/rural_borehole_lose_fraction` in
`gis_global_module.calculate_population_water_access()`. Kevin's decision:
keep scenario logic in those parameters (HEMP: all electric boreholes fail;
loss-of-central-infrastructure: some survive), NOT baked into the dataset.
Seed evidence is collected in the dataset's `bounds_notes` column and README
§pump-type (Bosnia ~11:1 electric:manual, Sint Maarten census G-12, French
Guiana's 36 handpumps). Kevin holds this; a research fan-out like the one
that built the dataset is the expected approach.

## In-flight / out of scope

- **Monte Carlo results in `results/` are stale**: they predate the new
  `merged_data.csv`. Rerun before quoting any numbers in the manuscript.
- Venezuela (the case study) is deliberately NOT filled: only multi-response
  data exists. Highest-value fix: ENCOVI (UCAB) microdata main-source table.
- Retrievable-but-not-retrieved sources (see handover doc §follow-ups):
  Brunei BPP 2021 census PDF, French Polynesia ISPF 2022 (JS-only site),
  Seychelles census (geo-blocked), Guernsey Water AR, Chuuk census H03.
- USA rural and SLB urban sum >100 in merged_data: pre-existing JMP-parser
  vs process_water_data mismatch, deliberately untouched, flagged to James.

## Decisions made (don't relitigate without new information)

- JMP-parsed values always beat research fills.
- Multi-response and attitudinal survey numbers are never inserted as
  values, only recorded in `bounds_notes` (kills VEN, AUS, Gulf states).
- Piped reconciliation: James's unpiped-budget proportional scaling is the
  default; carve-out when WHO piped >= 99.5 (rounded ">99") where categories
  win and piped becomes 100 - categories; when piped is spatially IMPUTED it
  also yields to measured categories. Rationale + examples in the dataset
  README; tests in `tests/test_build_unified_source_breakdown.py`.
- Dug wells map to OtherUnpiped, not Borehole; standpipes/taphouses count as
  piped (this is why Mayotte is ~3% not 25%, and Greenland is unfilled).
- `.claude/worktrees/` is gitignored (was an accidental gitlink); the
  30-arcsec feasibility work lives on branch `kr/30as-feasibility`.

## Dead ends already explored

- Spatial/GDP imputation of the source breakdown itself: James tried it,
  produces artifacts (Netherlands inheriting neighbor breakdowns). Blanks
  that remain should stay zero-filled, not imputed.
- Don't read utility raw-water mixes ("96% groundwater-sourced") as
  household shares; several were rejected for this (Iceland, Liechtenstein).

## Environment gotcha

`environment.yml` pins pandas 1.4.3; the system python3.11 (pyenv) has
2.1.4, which makes 8 pre-existing tests fail in
`test_gis_global_module`/`test_gis_monte_carlo` (groupby/MultiIndex
differences). Those failures are NOT from this branch's changes. The parser,
reconciliation and mobility tests all pass on 3.11.

## Who holds what

- Electric-borehole share research + the 30-arcsec feasibility call: Kevin.
- Manuscript edits from the review comments + pushing his branch: James.
