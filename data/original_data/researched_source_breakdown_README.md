# researched_source_breakdown.csv — manually researched water-source fills

Companion dataset to `data/processed/semi-processed/jmp_source_breakdown_latest.csv`,
covering the 76 countries/territories whose JMP country files carry **no**
source-type breakdown (blank rows). Values were compiled from national
censuses, DHS/EPHS final reports, drinking-water regulator statistics and
statistical-office publications, then mapped into the JMP schema under a
strict rubric (see "Mapping rubric" below). Research provenance:
`~/Documents/ProgrammingIsFun/ALLFED/borehole-gap-research/REPORT.md`
(2026-08-07) and its per-region section files.

## How it is used

`scripts/Data_Manipulation_Scripts/build_unified_source_breakdown.py` merges
this file with the JMP-parsed breakdown into
`data/processed/semi-processed/jmp_source_breakdown_unified.csv`:

- **JMP-parsed values always win**; research values only fill rows where the
  JMP extract is entirely blank.
- Provenance travels on every filled row (`data_origin`, `research_tier`,
  `research_confidence`, `research_source`, `research_url`, `research_flags`,
  `research_notes`, `research_bounds`).
- `--exclude-low-confidence` drops fills whose confidence is `low`.
- A validation report (`jmp_source_breakdown_unified_validation.csv`) lists
  rows whose categories conflict with the WHO/JMP piped share (see below).

To point the model at the unified file, change `jmp_source_breakdown_path` in
`scripts/Data_Manipulation_Scripts/imputation_script.py` to
`jmp_source_breakdown_unified.csv` and rerun it.

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
  they live in `bounds_notes` (this is why Venezuela and Australia are
  `not_fillable` despite rich research).
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

`build_unified_source_breakdown.py` resolves these automatically
(`reconcile_stratum_with_piped_budget()`), applied to research-origin rows
only — JMP-origin rows were already fitted by the parser. The rule follows
the approach James proposed on the 2026-08-06 call, with one carve-out agreed
2026-08-08:

- **Default (WHO piped < 99.5)** — James's unpiped-budget rule, identical to
  `normalize_unpiped_subcategories()` in `jmp_source_parser.py`: if the four
  categories sum past `100 − piped`, scale them all down proportionally to
  fit. Piped wins, because breakdown surveys are often multi-response and the
  overlap sits mostly with piped users, who are already counted as losing
  access. This handles class 3 (e.g. Congo rural ×0.964) and genuine
  piped-wins conflicts (e.g. St Vincent urban ×0.124, where WHO says 98%
  piped but the 2012 census said 83%).
- **Carve-out (WHO piped ≥ 99.5)** — the unpiped budget is a rounding
  artifact, and scaling to it would zero out measured regulator/census values
  (Denmark's GEUS-counted 4% boreholes would vanish). Instead the categories
  are kept (proportionally capped only if they alone exceed 100) and an
  **effective piped share** `100 − sum(categories)` is written to
  `URBANPipedEffective` / `RURALPipedEffective`. Examples: Denmark → 96.0;
  NZ → 83.6 (independently matching Taumata Arowai's registered-supply
  share); Anguilla urban → 2.4 (piped-as-main-drinking-source, resolving
  class 2).

Every adjustment is recorded in `URBAN_budget_adjustment` /
`RURAL_budget_adjustment` (`scaled_to_unpiped_budget(scale=…)` or
`piped_capped`), so raw research values remain recoverable from
`researched_source_breakdown.csv`. Strata with no WHO piped reference at all
(e.g. Anguilla rural) are left unadjusted.

**Integration requirement**: when the model adopts the unified breakdown, any
row with a non-null `PipedEffective` must also override that stratum's piped
share (else piped=100 + packaged users double-counts the affected
population). Kevin's recorded caveat from the call still applies to the
default rule: proportional scaling assumes the overlap is with piped; if
respondents overlap between unpiped categories instead, the redistribution is
approximate.

Tests: `tests/test_build_unified_source_breakdown.py`.

## Caution for the manual-vs-electric pump phase

Private wells in high-income countries (Nordics, Austria, Jersey, NZ) are
near-universally electric-pumped: the Borehole populations added here for
rich countries are mostly CIL-vulnerable, unlike DHS-country handpump
boreholes. The `urban/rural_borehole_lose_fraction` parameters — or a future
per-country pump-type split — must reflect this. Direct pump-type signals
captured in `bounds_notes`: Sint Maarten census G-12 (windmill/electric/no
pump), Bosnia hydrophore vs dug wells (~11:1), French Guiana's 36 documented
handpumps, Iran's gravity-fed qanats.
