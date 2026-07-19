# Water access — model review & changes

Branch: `fix/model-review-2026-07` (not merged to main). Nothing here is final; parameter values are provisional pending sign-off.

---

## Key finding: current headline numbers are stale

- A unit-conversion bug in `calculate_max_distances` (velocity treated as km/h instead of m/s) underestimated all distances by 3.6×. Fixed in commit `c560ab4`, 2025-08-13.
- The stored results (`results/country_median_results.csv` etc.) are dated 2025-07-18 — before that fix — and were never regenerated.
- So the paper's 24.3% no-access / 3.6 km walk / 11.8 km cycle are pre-fix (distances too small). A re-run is required regardless of any parameter change.
- This explains the earlier single-run result (17% / 4.9 km / 21 km): that is post-fix output. Lower parameters did reduce distances; the apparent jump vs the paper is the 3.6× fix, not the parameters.

---

## Changes on the branch

### Code — bug fixes
- **Lankford slope units** (`src/mobility_module.py`): slope was computed as `degrees/45` instead of true percentage grade (`tan×100`), underweighting slope ~80×. Walking was near slope-insensitive (~4% speed drop from flat to 20°). Corrected and clipped to Lankford's validated −18%…+40% range. Now drops ~1.12→0.20 m/s over 0°→20°.
- **Lankford cubic-term sign**: `+0.000320` → `−0.000320`. Confirmed against the Lankford 2020 PDF (Table 4: coefficient −0.0001431, full CI below zero). Latent until the units fix made the term significant.
- **Walking hill polarity**: `single_lankford_run` now applies polarity to the loaded/unloaded legs (was ignoring it; cycling already did).
- **Velocity floor**: non-converged or negative solves now return 0 reachable velocity instead of NaN.

### Code — Monte Carlo params & plumbing
- **Shared config** `src/monte_carlo_config.py`: all three run scripts now import from it (previously each had its own copy of the constants; the cloud runs `run_monte_carlo_gcp.py`, so edits to only the local script would have had no effect on cloud results).
- **GCP checkpoint fix**: resumed runs previously crashed (NameError) on the periodic/final checkpoint save.

### Parameter changes (provisional)
- **METs (walking)**: 3–6 (median 4.5) → 3–5 (median 4). Normal.
- **Time gathering**: normal 4–7 (median 5.5) → lognormal 2–5 (median ~3.16).
- **Watts (cycling)**: independent normal 20–80 → derived per iteration from the sampled METs via the ACSM cycle-ergometry equation (62 kg reference, 5 W floor).

### Manuscript (already applied to `manuscript/[Manuscript v2.0] Water Access ALLFED.md`, now tracked in repo)
- Deleted the parameter-worksheet scratch notes left in Supplementary Materials.
- Added air density ρ to Eq 1 (was defined but missing from the equation).
- Fixed cross-reference "2.3 Data Processing" → "2.4 Data pre-processing".
- Fixed two garbled sentences (abstract "would from be able"; §3.3 "openingopportunity…interventionstargeted").
- Fixed citation key NMDA → NDMA (both in-text and reference).

---

## Parameters: proposed vs recommended

| Parameter | Original | Proposed | Recommended (on branch) |
|---|---|---|---|
| METs (walking) | normal 3–6 (med 4.5) | normal 2–5 (med 3.5) | normal 3–5 (med 4) |
| Time gathering (h) | normal 4–7 (med 5.5) | lognormal 2–5 (med 3.16) | lognormal 2–5 (med 3.16) |
| Watts (cycling) | normal 20–80 (med 50) | lognormal 10–60 (med 24.5) | derived from METs via ACSM |

---

## Fact-check results

- **METs**: the cited sources center water-carrying at ~4.3–5.0 METs (Mozambique head-hauling 4.3; Baka water-fetching 7.28). Four of the six Compendium values in the notes were the 2000-edition figures; the true 2011 values are slightly higher. A 2-MET floor is unsupported (empty-handed walking ≈ 3.1). → median ~4 is better supported than 3.5.
- **Watts / ACSM**: the 10–60 W ≈ 2–5 METs conversion (at 62 kg) checks out; no mechanical/metabolic confusion (ACSM bridges them). Deriving watts from the sampled METs ties walker and cyclist to one metabolic budget and removes physiologically impossible pairings (e.g. strong walker + weak cyclist) that widen the CI.
- **Lankford slope units**: confirmed = percent grade, validated −18%…+40% (PDF + authors' data repo). Fit stats as cited (adj R² 0.89, RMSE 5.92 kJ/min).
- **Cycling:walking distance ratio**: was 3.3 (validated vs Larsen 2010, median 3.43). New parameters push it to ~4.3. Re-sourceable: Larsen's own mean/85th-percentile is ~3.9× ("nearly four times"); FTA/TCRP transit-catchment standard supports 3–4×.
- **15 L/person/day ration**: standard (Sphere-derived); unchanged.

---

## Still to do

1. Agree final parameter values (esp. METs). One-line change in `src/monte_carlo_config.py`.
2. Full 1000-run Monte Carlo re-run (cloud). Before running: delete `results/parquet_files/checkpoint.json` and stale `results/*.csv` (a stale checkpoint restores old parameters on resume).
3. Manuscript Eq 5: flip the cubic sign `+0.000320` → `−0.000320` (one character).
4. Re-derive all headline numbers from the new run (abstract, continental breakdowns, Venezuela, conclusion; the "nearly a third" in the conclusion vs 24.3%).
5. Update the Methods distribution wording (time and watts descriptions change; add the ACSM-derivation note).
6. Re-source the cycling:walking ratio sentence (§3.1) to ~4.3×.
7. Correct the Compendium citations in Supplementary Table S1 if the mislabeled values appear there.

---

## Uncertainties

- The slope fix reduces access (esp. hilly regions — Venezuela is mountainous); lower METs also reduce it; the distance fix increases it. Net direction of the headline number is unknown until the re-run.
- The METs value is a judgment call, not a fixed fact. Branch uses median 4 as a compromise.
- A single median-parameter run is not representative: cycling distance swings 2–4× on the hill-polarity draw alone (downhill leg is speed-capped). Only the 1000-run medians/CIs are meaningful.
- Parameter values are marked PROVISIONAL throughout the code pending sign-off.

---

## Test status

- 108 passed, 9 failed. The 9 failures are pre-existing (pandas version incompatibility in GIS/mock paths); identical on the pre-change baseline. 8 new tests added (Lankford slope monotonicity, velocity non-negativity, sampler medians, watts derivation).
