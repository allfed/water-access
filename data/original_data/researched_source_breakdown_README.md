# researched_source_breakdown.csv — manually researched water-source fills

Companion dataset to `data/processed/semi-processed/jmp_source_breakdown_latest.csv`,
covering the 76 countries/territories whose JMP country files carry **no**
source-type breakdown (blank rows). Values were compiled from national
censuses, DHS/EPHS final reports, drinking-water regulator statistics and
statistical-office publications, then mapped into the JMP schema under a
strict rubric (see "Mapping rubric" below). Each row records its source,
survey year and confidence.

## How it is used

`scripts/Data_Manipulation_Scripts/build_unified_source_breakdown.py` merges
this file with the JMP-parsed breakdown into
`data/processed/semi-processed/jmp_source_breakdown_unified.csv`:

- **Best source wins.** A JMP row keeps precedence only when it is
  *informative*: a positive value in at least one of packaged / delivered /
  borehole (`jmp_row_is_informative`). A JMP row that is blank, or whose only
  content is the UnpipedUnaffected catch-all (sparse EU-SILC / HBS blocks), is
  demoted and the research row is used instead, with the demotion recorded in
  `research_notes` and printed as a WARNING. Precedence is per country row,
  not per column: an informative JMP row replaces the whole research row.
  The parser writes 0.0 for categories a survey did not ask, so an all-zero
  JMP row is treated as uninformative rather than as measured zeros.
- Provenance travels on every filled row (`data_origin`, `research_tier`,
  `research_confidence`, `research_source`, `research_url`, `research_flags`,
  `research_notes`, `research_bounds`).
- `--exclude-low-confidence` drops fills whose confidence is `low`.
- A validation report (`jmp_source_breakdown_unified_validation.csv`) lists
  rows whose categories conflict with the WHO/JMP piped share (see below).

`scripts/Data_Manipulation_Scripts/imputation_script.py` reads the unified
file (`jmp_source_breakdown_path`) and carries `data_origin` through to
`data/processed/merged_data.csv`.

## Columns

- `alpha3`, `Entity` — ISO 3166-1 alpha-3 and English name.
- `URBAN…`/`RURAL…` × `Packaged`, `Delivered`, `Borehole`, `UnpipedUnaffected` —
  percent (0–100) of the urban (resp. rural) population whose **main
  drinking-water source** is that category. Null = unknown; 0 = evidence
  supports ≈0 (an intentional, citable zero).
- `survey`, `survey_year` — short source id (`ROU_2021_CEN`) and reference year.
- `tier` — `census` | `survey` | `regulator_proxy` | `derived_assumption`.
- `confidence` — `high` | `med` | `low`.
- `source`, `url` — citation.
- `flags` — semicolon-separated caveats. The important ones:
  - `national_applied_to_both` — one national figure written into both strata.
  - `residual_allocation_uncertain` — a "% not connected" residual was
    allocated to a category on partial evidence.
  - `proxy_ceiling_not_measurement` — value is a connection-rate residual
    (ceiling), not a household-survey measurement.
  - `household_supply_not_drinking_question` — census asked about supply, not
    drinking water, so Packaged is structurally invisible (left null).
  - `deviates_from_section_suggested_mapping` / `..._report_...` — we
    deliberately mapped differently from the research report's suggestion
    (rubric wins; reasons in `mapping_notes`).
- `mapping_notes` — the category-by-category reasoning and arithmetic.
- `bounds_notes` — numbers that must NOT be inserted as values (multi-response
  surveys, attitudinal bottled-water shares, ceilings) but that bound them.
- `fill_status` — `filled` | `partial_fill` | `not_fillable`. `not_fillable`
  rows carry no values but preserve what was searched and why nothing usable
  exists.

## Mapping rubric (summary)

- Categories follow JMP conventions: Borehole = drilled tubewell/borehole
  (incl. own drilled well with electric pump/hydrophore); UnpipedUnaffected =
  dug wells + springs + rainwater/cisterns + surface + other; Packaged =
  bottled/sachet; Delivered = tanker/cart/vendor. Public taps, standpipes and
  taphouses count as **piped** and are excluded.
- Where a census asks both "main household supply" and "main drinking source",
  the drinking table wins (Anguilla, BVI).
- Multi-response and attitudinal statistics are never inserted as values —
  they live in `bounds_notes` (HumVenezuela multi-response figures for Venezuela
  remain bounds-only; see the VEN row's `partial_fill` from ENCOVI 2024 p.4).
- Connection-rate residuals are allocated only where the source names the
  residual's composition; otherwise flagged.

## Reconciliation with the WHO/JMP piped share (the "unpiped budget" rule)

Research fills can conflict with the WHO/JMP piped share in three ways:

1. **JMP rounding**: WHO reports piped = 100 (rounded from ">99") while the
   research measures 4–16% self-supply (DNK, FIN, NOR, NZL, CZE, SVK, MAF).
2. **Drinking vs supply**: Anguilla's WHO piped ≈ 100 describes household
   supply, but 60.7% drink bottled → sums to ~198.
3. **Reference-year drift**: DHS/census years differ from the WHO piped year
   (COG, ERI, GNQ, IRN, SXM, VCT — sums of ~103–114).

One rule handles all three: **the source hierarchy is JMP global (piped) →
JMP country file → researched fills → spatial interpolation, and a stratum
that sums past 100 scales its unpiped categories down with piped protected.** Implemented in
`reconcile_categories_over_100_pct()` (`jmp_source_parser.py`) and applied in
three passes:

- **Build time** (`build_unified_source_breakdown.py`, research rows): if the
  five categories sum past `100 − piped`, scale them all down proportionally.
  Unknown (null) categories are untouched. Recorded in
  `URBAN_budget_adjustment` / `RURAL_budget_adjustment` as
  `scaled_to_unpiped_budget(scale=…)`, so raw research values remain
  recoverable from `researched_source_breakdown.csv`.
- **Imputation, before spatial interpolation**
  (`imputation_script.reconcile_research_breakdown_with_piped`): the same
  scale-down rule for research and JMP rows against the piped share the
  model actually uses.
- **Imputation, after spatial interpolation**
  (`imputation_script.reconcile_research_rows_with_final_piped`): for any row
  whose piped share had to be neighbour- or continent-interpolated, the
  measured categories are kept and piped is lowered to
  `100 − sum(categories)` instead. Interpolation is the bottom of the
  hierarchy and must never scale real survey or census data down. Rows with
  a measured piped share use the scale-down rule.

A stratum that sums to *less* than 100 is never scaled up; the remainder is
written to `URBANUnallocated` / `RURALUnallocated`
(`apply_unallocated_from_source_gap`). Proportional scaling assumes the
overlap is with piped users; if respondents overlap between unpiped
categories instead, the redistribution is approximate.

Tests: `tests/test_build_unified_source_breakdown.py`,
`tests/test_unallocated_from_source_gap.py`, `tests/test_jmp_source_parser.py`.

## Pump type

Private wells in high-income countries (Nordics, Austria, Jersey, NZ) are
near-universally electric-pumped: the Borehole populations added here for
rich countries are mostly CIL-vulnerable, unlike DHS-country handpump
boreholes. The `urban/rural_borehole_lose_fraction` ranges are derived in
`borehole_pump_type_estimates_README.md`. Direct pump-type signals
captured in `bounds_notes`: Sint Maarten census G-12 (windmill/electric/no
pump), Bosnia hydrophore vs dug wells (~11:1), French Guiana's 36 documented
handpumps, Iran's gravity-fed qanats.
