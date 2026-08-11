# AC-5 (Gate 2): refactor regression at 5 arc-minutes

- Baseline (pre-refactor, committed): `/Users/kevin/Documents/ProgrammingIsFun/ALLFED/water-access/results/regression_baseline`
- New (post-refactor): `/Users/kevin/Documents/ProgrammingIsFun/ALLFED/water-access/results/regression_new`
- Countries compared: 181; districts compared: 2895 (2482 over 100,000 people)

Both runs use the AC-2 fixed parameters: `crr_adjustment=0, time_gathering_water=2.84, practical_limit_bicycle=40, practical_limit_buckets=20, met=4, watts=40, hill_polarity="downhill_uphill", urban_adjustment=1.3, rural_adjustment=1.4, calculate_distance=True, plot=False`.

## Checks

| check                                                | observed                                        | bar                | pass |
| ---------------------------------------------------- | ----------------------------------------------- | ------------------ | ---- |
| countries row set identical                          | 0 missing, 0 added                              | 0 missing, 0 added | PASS |
| districts row set identical                          | 0 missing, 0 added                              | 0 missing, 0 added | PASS |
| country percent_with_water                           | max 0.00000 pp (AFG)                            | <= 0.1 pp          | PASS |
| country country_pop_raw                              | max 0.00000% (AFG)                              | <= 0.1%            | PASS |
| country country_pop_with_water                       | max 0.00000% (AFG)                              | <= 0.1%            | PASS |
| country country_pop_without_water                    | max 0.00000% (AFG)                              | <= 0.1%            | PASS |
| country population_affected_with_access              | max 0.00000% (AFG)                              | <= 0.1%            | PASS |
| country population_affected_with_cycling_access      | max 0.00000% (AFG)                              | <= 0.1%            | PASS |
| country population_affected_with_walking_access      | max 0.00000% (AFG)                              | <= 0.1%            | PASS |
| country population_affected_with_only_cycling_access | max 0.00000% (AFG)                              | <= 0.1%            | PASS |
| district percent_with_water (>100,000 people)        | max 0.00000 pp (10019604B1094996199402, n=2482) | <= 0.25 pp         | PASS |
| AC-7 run time excluding data load                    | 37.2 s                                          | <= 90 s            | PASS |

## AC-7 timing

- Input: `/Users/kevin/Documents/ProgrammingIsFun/ALLFED/water-access/data/GIS/updated_GIS_output_cleaned.csv`
- Total wall clock: **39.2 s**
- Initial data load: 2.0 s
- **Excluding data load: 37.2 s** (bar: <= 90 s; the pre-refactor solver alone took ~4 minutes)

## Memory (ADR-5 datum for Phase D)

- Peak RSS of the single fixed-parameter run: **3.07 GB**
- Modelled cells: 1,395,693

This is the 5-arcmin datum; AC-6's 24 GB budget is measured at 30 arc-seconds by Phase D.

## Per-column population detail

| column                                       | max_rel_error_pct | worst_country | baseline_value | new_value   |
| -------------------------------------------- | ----------------- | ------------- | -------------- | ----------- |
| country_pop_raw                              | 0.0               | AFG           | 3.98354e+07    | 3.98354e+07 |
| country_pop_with_water                       | 0.0               | AFG           | 3.1261e+07     | 3.1261e+07  |
| country_pop_without_water                    | 0.0               | AFG           | 8.57448e+06    | 8.57448e+06 |
| population_affected_with_access              | 0.0               | AFG           | 6.07543e+06    | 6.07543e+06 |
| population_affected_with_cycling_access      | 0.0               | AFG           | 2.5184e+06     | 2.5184e+06  |
| population_affected_with_walking_access      | 0.0               | AFG           | 5.18573e+06    | 5.18573e+06 |
| population_affected_with_only_cycling_access | 0.0               | AFG           | 889707         | 889707      |

## Verdict

**PASS**
