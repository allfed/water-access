# Handover: unified source breakdown (the 76 blank JMP rows)

Kevin, 10 Aug 2026. Branch: `kr/unpiped-breakdown`.

## TL;DR

The 76 countries with blank rows in the JMP source breakdown are now filled
from national censuses, DHS/EPHS reports and regulator statistics where real
data exists (53 of them), with provenance and confidence on every value. The
model now reads the unified file: `imputation_script.py` points at
`jmp_source_breakdown_unified.csv` and `merged_data.csv` has been
regenerated. Your unpiped-budget rule from our call is applied throughout,
with two extensions described below. 68 of the 76 are live model inputs, and
before this they were all silently getting 0% for every unpiped category
(blanks were zero-filled), i.e. treated as fully CIL-resilient.

## What's new

- `data/original_data/researched_source_breakdown.csv`: all 76 territories,
  one row each. Mapped values, tier (census / survey / regulator_proxy),
  confidence, citation + URL, the category-by-category arithmetic, and
  numbers that must NOT be used as values (multi-response and attitudinal
  surveys) kept separately in `bounds_notes`.
- `data/original_data/researched_source_breakdown_README.md`: the mapping
  rubric, all caveats, and the reconciliation rule. Read this one first.
- `scripts/Data_Manipulation_Scripts/build_unified_source_breakdown.py`:
  merges JMP-parsed + researched values (JMP always wins) into
  `data/processed/semi-processed/jmp_source_breakdown_unified.csv`.
- `tests/test_build_unified_source_breakdown.py`: 11 tests on the
  reconciliation rule.
- `imputation_script.py` changes: reads the unified file, applies the
  reconciliation against final piped values, and fixes a small pre-existing
  bug where OtherUnpiped was dropped from the output selection and re-added
  as all zeros in `merged_data.csv`.

## Where the numbers came from, and how much to trust them

Research was done per-country against primary sources (the full report with
per-country write-ups is in a separate `borehole-gap-research` folder, happy
to send it). I then ran an adversarial fact-check on the ten highest-stakes
values: 9 of 10 CONFIRMED against the primary source exactly (Congo, Eritrea
and Eq. Guinea DHS/EPHS tables, Romania, Bosnia and Slovakia censuses,
Denmark GEUS, BVI census, USVI census API). FSM rated PLAUSIBLE only, single
non-official source, though its Yap and Pohnpei numbers verify exactly
against state census tables.

Fill quality varies and is labelled: 14 territories fully filled, 39
partially (often a regulator connection-rate residual, ceiling not
measurement), 23 not fillable. The not-fillable list includes Venezuela
(HumVenezuela is multi-response so inserting it would fabricate precision;
ENCOVI microdata is the highest-value follow-up) and Australia (ABS is
multi-source only). `--exclude-low-confidence` on the build script drops the
low-confidence fills if we want a sensitivity run.

## The reconciliation rule (your call rule, plus two extensions)

Your rule from the call is the default: if the four categories sum past the
unpiped budget (100 minus piped), scale them all down proportionally. Same
logic as `normalize_unpiped_subcategories` in the parser. Two extensions,
both only for research-origin rows:

1. **Rounded-piped carve-out.** Where WHO piped is >= 99.5 it's a rounded
   ">99", and the budget is a rounding artifact. Scaling would delete
   measured values (Denmark's GEUS-counted 4% private boreholes would go to
   zero). Instead the categories win and piped becomes 100 minus the
   categories: Denmark 96, NZ 83.6 (independently matches Taumata Arowai's
   registered-supply share), Anguilla urban 2.4 (their census: 60.7% drink
   bottled, cisterns 35.9%, so piped-as-main-drinking-source really is tiny).
2. **Imputed-piped rule.** Some countries have no piped data in the WHO CSV
   at all (Austria, Slovenia, Croatia) or no urban/rural split (FSM,
   Grenada), so their piped share is spatially imputed. An imputed piped is
   a weaker datum than a measured census breakdown, so there the categories
   win too: FSM lands on piped 31, which is exactly the census's own
   utility+community share.

Your caveat from the call still stands and is recorded in the README: the
proportional scaling assumes the overlap sits with piped users. If people
overlap between unpiped categories instead, the redistribution is
approximate.

Two pre-existing quirks I did NOT touch, flagging for you: USA rural and
Solomon Islands urban sum past 100 because the parser's piped reference
disagrees with what `process_water_data` fills. JMP-origin rows, predates
this work, worth a look at some point.

## Next phase: electric vs hand-pumped boreholes

No new plumbing needed to "turn off" electric boreholes: the knobs already
exist as `urban_borehole_lose_fraction` / `rural_borehole_lose_fraction` in
`gis_global_module.py`. What's missing is the number. Pump-type signals
collected in passing are in the README and in `bounds_notes` (best ones:
Bosnia's census splits hydrophore vs dug wells at roughly 11:1
electric-to-manual, Sint Maarten's census table G-12 splits wells by
windmill / electric / no pump, French Guiana documents exactly 36
handpumps). One warning for the estimate: every high-income-country source
independently says private wells there are near-universally electric, so
rich-country boreholes should fail in a HEMP scenario, unlike the
DHS-country handpump picture.

Scenario separation thought (from Kevin, for discussion): keep the electric
share as scenario parameters rather than baked into the data, so we can run
loss-of-central-infrastructure (electric boreholes on private power might
survive) vs HEMP (all electric fails) off the same dataset.

## To reproduce

```
python scripts/Data_Manipulation_Scripts/build_unified_source_breakdown.py
python scripts/Data_Manipulation_Scripts/imputation_script.py
```

Test suite note: 8 failures in `test_gis_global_module` / `test_gis_monte_carlo`
are pre-existing pandas-version issues (they fail identically without these
changes); the parser and reconciliation tests all pass.
