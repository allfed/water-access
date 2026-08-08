# AC-5 (Gate 2): refactor regression at 5 arc-minutes

- Baseline (pre-refactor, committed): `/Users/kevin/Documents/ProgrammingIsFun/ALLFED/water-access/.claude/worktrees/30as-feasibility/results/regression_baseline`
- New (post-refactor): `/Users/kevin/Documents/ProgrammingIsFun/ALLFED/water-access/.claude/worktrees/30as-feasibility/results/regression_new`
- Countries compared: 181; districts compared: 2895 (2483 over 100,000 people)

Both runs use the AC-2 fixed parameters: `crr_adjustment=0, time_gathering_water=2.84, practical_limit_bicycle=40, practical_limit_buckets=20, met=4, watts=40, hill_polarity="downhill_uphill", urban_adjustment=1.3, rural_adjustment=1.4, calculate_distance=True, plot=False`.

## Checks

| check                                             | observed                                         | bar                | pass |
| ------------------------------------------------- | ------------------------------------------------ | ------------------ | ---- |
| countries row set identical                       | 0 missing, 0 added                               | 0 missing, 0 added | PASS |
| districts row set identical                       | 0 missing, 0 added                               | 0 missing, 0 added | PASS |
| country percent_with_water                        | max 0.00152 pp (DEU)                             | <= 0.1 pp          | PASS |
| country country_pop_raw                           | max 0.00004% (BHS)                               | <= 0.1%            | PASS |
| country country_pop_with_water                    | max 0.00185% (DEU)                               | <= 0.1%            | PASS |
| country country_pop_without_water                 | max 0.00859% (DEU)                               | <= 0.1%            | PASS |
| country population_piped_with_access              | max 0.00285% (TZA)                               | <= 0.1%            | PASS |
| country population_piped_with_cycling_access      | max 0.00575% (TZA)                               | <= 0.1%            | PASS |
| country population_piped_with_walking_access      | max 0.00004% (BHS)                               | <= 0.1%            | PASS |
| country population_piped_with_only_cycling_access | max 0.02502% (TZA)                               | <= 0.1%            | PASS |
| district percent_with_water (>100,000 people)     | max 0.04872 pp (36957248B69769580879919, n=2483) | <= 0.25 pp         | PASS |
| AC-7 run time excluding data load                 | 112.9 s                                          | <= 90 s            | FAIL |

## AC-7 timing

- Input: `/Users/kevin/Documents/ProgrammingIsFun/ALLFED/water-access/.claude/worktrees/30as-feasibility/data/GIS/updated_GIS_output_cleaned.csv`
- Total wall clock: **121.3 s**
- Initial data load: 8.4 s
- **Excluding data load: 112.9 s** (bar: <= 90 s; the pre-refactor solver alone took ~4 minutes)

## Memory (ADR-5 datum for Phase D)

- Peak RSS of the single fixed-parameter run: **2.15 GB**
- Modelled cells: 1,403,388

This is the 5-arcmin datum; AC-6's 24 GB budget is measured at 30 arc-seconds by Phase D.

> **AC-7 measurement caveat — re-measure on a quiet machine.** This run was taken while an unrelated GPU/CPU-heavy application (a game) held ~47% CPU, which on Apple silicon pushes the model onto efficiency cores and inflates *both* wall-clock and user CPU time by roughly 4x. A bare `pandas.read_csv` of the same input went 1.5 s -> 8.3 s over the same interval with no code involved. An A/B under identical load shows no regression from the chunking work: pre-C2 commit `1475b11` took **112.6 s** excluding load, the C2 tree **111.7 s**. On the quiet machine the same command measured **26-28 s** against the 90 s bar. The independent verifier should simply re-time it.

## Per-column population detail

| column                                    | max_rel_error_pct | worst_country | baseline_value | new_value   |
| ----------------------------------------- | ----------------- | ------------- | -------------- | ----------- |
| country_pop_raw                           | 3.5e-05           | BHS           | 396914         | 396914      |
| country_pop_with_water                    | 0.001846          | DEU           | 6.90604e+07    | 6.90616e+07 |
| country_pop_without_water                 | 0.008591          | DEU           | 1.48401e+07    | 1.48388e+07 |
| population_piped_with_access              | 0.002849          | TZA           | 1.57793e+07    | 1.57797e+07 |
| population_piped_with_cycling_access      | 0.005752          | TZA           | 7.80791e+06    | 7.80836e+06 |
| population_piped_with_walking_access      | 3.5e-05           | BHS           | 220074         | 220074      |
| population_piped_with_only_cycling_access | 0.02502           | TZA           | 1.7937e+06     | 1.79415e+06 |

## Verdict

**FAIL**
