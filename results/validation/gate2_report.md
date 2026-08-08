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
| AC-7 run time excluding data load                 | 24.9 s                                           | <= 90 s            | PASS |

## AC-7 timing

- Input: `/Users/kevin/Documents/ProgrammingIsFun/ALLFED/water-access/.claude/worktrees/30as-feasibility/data/GIS/updated_GIS_output_cleaned.csv`
- Total wall clock: **26.3 s**
- Initial data load: 1.4 s
- **Excluding data load: 24.9 s** (bar: <= 90 s; the pre-refactor solver alone took ~4 minutes)

## Memory (ADR-5 datum for Phase D)

- Peak RSS of the single fixed-parameter run: **2.88 GB**
- Modelled cells: 1,403,388

This is the 5-arcmin datum; AC-6's 24 GB budget is measured at 30 arc-seconds by Phase D.

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

**PASS**
