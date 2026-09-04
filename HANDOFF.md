# HANDOFF — source-hierarchy verification (2026-09-04)

Branch under review: `origin/unpiped_breakdown` (James, 25 Aug, "30s trial run").
This file lives on `kr/hierarchy-review-2026-09-04`, branched from it.

**Status (2026-09-04, later the same day): the Lebanon bug below is FIXED on
this branch** (commit after 803fd96). Diff vs James's branch:
`imputation_script.py` (guard dropped, ~6 lines, plus a comment), the locking
test replaced by three tests, `merged_data.csv` (LBN row only), and doc-only
edits to the research README and the build script docstring. Full suite 76 passed /
4 skipped. James needs to merge or cherry-pick this branch before the runs.

## Context (from the 2026-09-04 Kevin/James call)
Agreed trust order for the per-country source breakdown: JMP global → JMP
country file → researched fills → interpolation only as a last resort. If a
stratum sums to >100 the unpiped categories scale down (piped protected).
James is holding the full model runs until Kevin confirms the code matches.
Kevin owns: this verification, re-emailing the Venezuela contacts (cc James),
the manuscript diagram. James owns: intro rewrite, running the model.

## Verdict so far
Mostly compliant. Precedence and scaling are in
`scripts/Data_Manipulation_Scripts/build_unified_source_breakdown.py` (row-level
JMP-vs-research merge, `jmp_row_is_informative` demotion rule) and
`jmp_source_parser.reconcile_categories_over_100_pct` (scale = (100−piped)/total
over all five unpiped categories, piped untouched). Piped itself comes from the
WHO/JMP global CSV via `imputation_script.process_water_data`, with spatial
neighbour/continent interpolation applied last. Interpolation never touches the
breakdown columns (they are `fillna(0)` before it runs). Sum <100 is not scaled
up; the remainder goes to `URBAN/RURALUnallocated` via
`apply_unallocated_from_source_gap`, and `src/gis_global_module._check_share_sums`
asserts ≤100.5. Tests: 74 pass, 4 skipped (missing raw xlsx only).

## One real bug (FIXED on this branch — kept for the record)
`imputation_script.reconcile_research_rows_with_final_piped` (~line 660):
`piped_was_imputed` is only True for `data_origin == "research"`. For a `jmp`
row whose piped share was itself neighbour-interpolated, the code still trusts
that interpolated piped and scales the real survey categories down. Confirmed
live for **Lebanon** only (of 25 interpolated-piped countries): unified urban
categories 31.3% → 7.0% in `merged_data.csv`, rural 31.3% → 14.5%, driven by
`URBANPiped=93.0` tagged "neighbor interpolated". This is the exact
"interpolation beats real data" case the hierarchy forbids.
`tests/test_unallocated_from_source_gap.py::test_jmp_row_never_takes_the_piped_lowering_branch`
locks the wrong behaviour in and will need changing with the fix.
Fix applied: `piped_was_imputed` is now True whenever the spatial track shows
interpolation for that stratum, regardless of `data_origin`. Baseline rebuild
before the fix reproduced the committed `merged_data.csv` byte-for-byte, so the
post-fix diff is clean: **only the LBN row changes.** Urban piped 93.0 → 68.72
with the five survey categories restored to their measured 31.28 total
(packaged 16.89, unknown 8.04, unaffected 4.63, delivered 0.97, borehole 0.75);
rural piped 85.5 → 68.72 with the same categories. Unallocated stays 0 and
`unallocated_gap_report.csv` / `merged_data_track.csv` are unchanged. The other
five JMP-origin countries with interpolated piped (BHS, COK, DMA, LBY, TTO)
never had categories exceeding 100 − piped, so they are untouched.

## Lesser findings (all closed 2026-09-04, nothing outstanding for James)
Kevin's call: these are all consequences of the hierarchy already agreed, not
new decisions, so they were settled here rather than pushed back.
- Precedence is per country row, not per column: an informative JMP row
  replaces the whole research row. This is "JMP beats research" applied
  conservatively (no mixing sources within a stratum). Kept; now documented
  in the README.
- All-zero JMP rows demote to research because the parser writes 0.0 for
  "not asked". An all-zero row carries no information, so demotion is the
  only sane reading. Kept; documented in code and README.
- Stale docs FIXED in this branch: README "How it is used" and the
  reconciliation section rewritten to the current three-pass rule with the
  real function names; `imputation_script.py` path comment and the
  `build_unified_source_breakdown.py` docstring updated (the latter was also
  made stale by the Lebanon fix).
- Interpolation provenance is still not joined into `merged_data.csv`. A
  future nicety for auditability, not a hierarchy violation, and not worth a
  schema change before the runs. Left as is.

## Dead ends
- "Interpolation" here is spatial (neighbour/continent), not year
  interpolation. Don't go looking for a year-interp step.
- Libya looked like a second case but its categories are not actually scaled.
- Recent commits e771833/e52ea23/bf858ee did not touch the hierarchy files;
  `build_unified_source_breakdown.py` last changed at 33ede3b.

## Next action
Done on Kevin's side: bug fixed, test updated, data rebuilt, suite green.
Remaining: James merges `kr/hierarchy-review-2026-09-04` into
`unpiped_breakdown` and starts the runs. No open decisions. Kevin still owes the Venezuela re-email and the manuscript
diagram.

## Working notes
Use Sonnet 5 subagents for any further code tracing to preserve context. A
worktree of the branch may still exist under the Claude scratchpad; otherwise
`git worktree add <path> origin/unpiped_breakdown`.
