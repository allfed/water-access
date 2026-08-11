# AC-4: solver lookup-table fidelity

- Samples: **10000** (10 parameter draws x 1000)
- Seed: `20260808`
- Slope pool: `/Users/kevin/Documents/ProgrammingIsFun/ALLFED/water-access/.claude/worktrees/30as-feasibility/data/GIS/updated_GIS_output_cleaned.csv` (`slope_1`, 2190446 cells, 0.0000-50.5199 deg)
- Mass pool: `/Users/kevin/Documents/ProgrammingIsFun/ALLFED/water-access/.claude/worktrees/30as-feasibility/data/processed/merged_data.csv` (`Average Weight`, 226 countries, 47.47-99.09 kg)
- Draw parameters: canonical ranges from `src/monte_carlo_constants.py` (ADR-7)
- LUT grid: 1000 slope nodes x 21 mass nodes x 6 Crr classes (ADR-4)
- Wall clock: 153.0 s

## Pass bar

`max |dv| <= max(0.005 m/s, 0.5% of v)`; NaN / no-solve cases must agree exactly.

## Results

| quantity                  | n     | max_abs_error | max_rel_error_pct | tolerance_breaches | nan_mismatches |
| ------------------------- | ----- | ------------- | ----------------- | ------------------ | -------------- |
| bicycle loaded velocity   | 10000 | 6.065e-05     | 0.00230           | 0                  | 0              |
| bicycle unloaded velocity | 10000 | 3.606e-05     | 0.00304           | 0                  | 0              |
| bicycle max load          | 10000 | 0.000e+00     | 0.00000           | 0                  | 0              |
| walking loaded velocity   | 10000 | 5.300e-05     | 0.00227           | 0                  | 0              |
| walking unloaded velocity | 10000 | 5.046e-05     | 0.00162           | 0                  | 0              |
| walking max load          | 10000 | 0.000e+00     | 0.00000           | 0                  | 0              |

## Parameter draws sampled

| draw | hill_polarity   | watts | met   | crr_adjustment | practical_limit_bicycle | practical_limit_buckets |
| ---- | --------------- | ----- | ----- | -------------- | ----------------------- | ----------------------- |
| 0    | uphill_downhill | 50.08 | 3.442 | 0              | 40                      | 23                      |
| 1    | uphill_flat     | 44.55 | 3.736 | 1              | 39                      | 17                      |
| 2    | flat_uphill     | 36.11 | 4.553 | -1             | 37                      | 19                      |
| 3    | downhill_uphill | 48.36 | 3.652 | 1              | 44                      | 15                      |
| 4    | uphill_downhill | 60.21 | 4.52  | -1             | 33                      | 20                      |
| 5    | uphill_flat     | 33.46 | 3.992 | -1             | 40                      | 21                      |
| 6    | flat_uphill     | 35.62 | 4.342 | 1              | 38                      | 19                      |
| 7    | downhill_uphill | 43.07 | 4.074 | 0              | 30                      | 22                      |
| 8    | uphill_downhill | 46.93 | 4.298 | 1              | 37                      | 21                      |
| 9    | uphill_flat     | 39.58 | 4.142 | 0              | 35                      | 20                      |

## Verdict

**PASS**
