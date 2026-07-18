# Water-Access Model & Manuscript — Review and Implementation Plan

**Prepared for:** a fresh implementing agent (and Kevin/James) picking this up cold.
**Date of review:** 2026-07-18.
**Repo:** `/Users/kevin/Documents/ProgrammingIsFun/ALLFED/water-access`
**Manuscript:** `/Users/kevin/Documents/ProgrammingIsFun/ALLFED/water access manuscript/[Manuscript v2.0] Water Access ALLFED(1).md`
(⚠️ that file is 560 KB; only lines 1–403 are text, the rest are base64 images. Read in chunks, never past line 403.)

---

## 0. Context you must understand before touching anything

The paper estimates the % of world population that could still reach freshwater by walking/cycling if piped-water infrastructure fails (EMP / geomagnetic storm / cyberattack / pandemic). Headline in the current draft: **24.3% without access; median one-way walking 3.6 km; cycling 11.8 km.** Results come from 1000 Monte Carlo runs over a GIS model (Lankford walking model + Martin cycling model).

**Constraints (read these — they set the bar for change):**
- The paper has been through many revision rounds and is near resubmission. Kevin is now a volunteer/part-time on it.
- **The bar for change is HIGH.** Only make changes that are (a) outright bugs/errors, (b) internal inconsistencies a reviewer would catch, or (c) very cheap high-value fixes. Do NOT restructure, add analyses, or rewrite for style.
- A **full Monte Carlo re-run is already mandatory** (see §1.0) — so any code fix that lands *before* that re-run is essentially free on compute. Bundle all agreed code changes into the single re-run.

**THE key discovery driving all of this:** The paper's headline numbers are **stale pre-bug-fix outputs**. A unit-conversion bug that underestimated all distances by exactly **3.6×** was fixed in commit `c560ab4` (2025-08-13), but `results/country_median_results.csv` — the file behind the paper's numbers — is dated **2025-07-18** and was never regenerated. The repo's own results contradict its own code. This is why co-author James's recent single run (17% no-access, 4.9 km walk, 21 km cycle) looks paradoxically *higher* on distance despite *lower* fitness parameters: it's the post-fix code; the paper is pre-fix. **Everything below assumes a re-run through corrected code is going to happen.**

**Two parameters of note that appear throughout:**
- Reference body mass 62 kg (country-specific weights used in the real model; 62 kg is the sensitivity/analytic reference).
- The Martin cycling model's power input `P_t` is **mechanical** watts at the drivetrain; the walking budget (`budget_VO2`, `budget_watts`) is **metabolic**. The ACSM cycle-ergometry equation is the correct bridge between them (see §1.7).

---

# SECTION 1 — CODE CHANGES

Priority key: 🔴 **must-fix** · 🟠 **should-fix (recommended)** · 🟡 **optional / author decision**

Ranked by (value ÷ effort ÷ risk). Read §1.0–§1.2 as a coupled block.

---

## 1.0 🔴 Re-run the Monte Carlo through current (post-fix) code — the umbrella task

No code change by itself, but this is the reason the rest exists. The stored results predate the `c560ab4` distance fix.

**Verify before re-running:**
- `src/gis_global_module.py` `calculate_max_distances` (~line 838–843) contains the post-fix form:
  `average_velocity_* * time_gathering_water * 3600 / 2 / 1000` (the `3600/1000` factor is the fix). Confirm it's present.
- Delete any stale `results/*.csv` / `*.pkl` and stale per-zone velocity CSVs (`results/**/bicycle_velocity_by_zone.csv`, `walk_velocity_by_zone.csv`) so nothing old is silently reused.
- Delete any `checkpoint.json` in the parquet output dir (see §1.8 — it caches sampled parameters and overrides new ones on resume).

**Acceptance:** headline numbers regenerated from current code, with all §1.1–§1.8 changes applied, in ONE run.

---

## 1.1 🔴 Lankford walking model — slope is in the wrong units (nearly slope-blind)

**File:** `src/mobility_module.py`, `mobility_models.Lankford_solution`, **line ~600**.

**Bug:** The Lankford (2020) equation expects `sp` = **percent grade** (rise/run × 100), validated over −18% to +40% (confirmed against the authors' data repo; high confidence). The code computes:
```python
G = (s * 360 / (2 * np.pi)) / 45      # s is slope in radians  →  G = degrees / 45
```
So `G` = degrees/45 (range ~0–1.1 over real terrain) instead of `tan(degrees)*100` (range ~0–120%). Slope is underweighted by **~79–100×**, making walking speed almost independent of terrain.

**Proof it's an oversight, not a choice:** the sibling `LCDA_solution` (line ~578) does the *same* conversion *with* the missing factor: `G = (s * 360 / (2 * np.pi)) / 45 * 100`. (Note even that is a crude linear approximation of true % grade; prefer `np.tan(s)*100` for the real fix.)

**Impact (62 kg + 15 kg load, 4.5 MET):** current code, walking speed drops only ~4% from flat to a 20° slope; corrected, it **halves by ~5°** and falls ~80% by the 95th-percentile slope. The model currently **overstates walking access in hilly terrain** — which directly undermines the **Venezuela case study** (mountainous; a headline result) and steep-terrain poor regions generally. Median global terrain is ~1° (fine), so flat regions barely move; the effect concentrates where it matters most for the paper's argument.

**Fix (COUPLED with §1.2 and §1.3 — do all three together or not at all):**
```python
G = np.tan(s) * 100        # s in radians → true percentage grade
```

**Do NOT ship this alone** — once G is in percent, the `sp³` term (§1.2) and slope clipping (§1.3) become load-bearing.

---

## 1.2 🔴 Lankford cubic-term SIGN error (only bites once §1.1 lands)

**File:** `src/mobility_module.py`, `Lankford_solution`, **line ~607**.

**Bug:** code has `+ (0.000320 * v_solve * G**3)`. The authors' fitted model gives this coefficient as **negative (−0.00032)** (verified in their public data repo, `results/model.md`; all five other coefficients match the manuscript to the digit). With the buggy units this term was negligible (`G³ ≈ 0–1`); once §1.1 makes `G` a real percentage, `G³` reaches ~64,000 at 40% grade and the sign determines whether the model is sane. The magnitude `0.000320` (not `0.00320`) is correct — an earlier git fix to that was right; the leftover `# this was previously 0.00320 - bug?` comment can be deleted.

**Fix:**
```python
- (0.000320 * v_solve * G**3)   # negative per Lankford 2020 fitted model
```

**⚠️ Verify first:** confirm the sign against the Lankford 2020 full text (Kevin may have the PDF; https://doi.org/10.1007/s00421-020-04428-z). Confidence the true coefficient is negative: **high**. Confidence on whether the published paper itself carries the typo: medium. If the PDF confirms negative, fix both code and manuscript Eq 5 (§2.2).

---

## 1.3 🔴 Clip slope to Lankford's validated range after the units fix

**File:** `src/mobility_module.py`, `single_lankford_run` (~line 203) or where `s` is derived.

Once §1.1 is in, real terrain (max ~50° = ~120% grade) exceeds Lankford's validated envelope (−18%…+40%). Clip the **percentage grade** to `[-18, 40]` before use (or clip the degrees to ~±21.8°), and note the extrapolation. This prevents the polynomial from doing unphysical things in the steep tail (p99 terrain ≈ 37% grade is near the edge; the max is well past it).

```python
G = np.clip(np.tan(s) * 100, -18, 40)
```

**Acceptance for §1.1–1.3:** add a regression test (see §1.9) asserting loaded walking velocity is monotonically non-increasing as slope goes 0°→5°→10°→20°, and finite at the max.

---

## 1.4 🟠 Walking model ignores hill polarity (asymmetric with cycling)

**File:** `src/mobility_module.py`, `single_lankford_run` (~line 203).

`single_lankford_run` passes the same raw `s` to both loaded and unloaded legs and **never** references `mo.ulhillpo`/`mo.lhillpo`, whereas `single_bike_run` uses `s*mo.ulhillpo` (unloaded) and `s*mo.lhillpo` (loaded). So the production walking path treats every trip as the same uphill magnitude both ways and gets no downhill relief — and silently ignores the `hill_polarity` the Monte Carlo sets up via `map_hill_polarity` (`gis_global_module.py:616`). (A second, non-production Lankford path in the batch loop ~line 457–542 *does* apply polarity — so the codebase is internally inconsistent.)

**Decision (author call):**
- **Option A (recommended, matches cycling):** apply `s*mo.ulhillpo` to the unloaded solve and `s*mo.lhillpo` to the loaded solve in `single_lankford_run`, mirroring `single_bike_run`. This makes walking consistent with the paper's "slope polarity" methods paragraph. Moderate change; must be covered by the §1.9 monotonicity test and re-validated.
- **Option B (cheaper):** leave as-is but **document explicitly** in the manuscript methods/limitations that walking uses symmetric slope magnitude. Lower risk before a deadline.

Recommend **A** if time allows, since the paper's Methods (line ~130) implies polarity applies to human-powered transport generally, and B otherwise. Flag which was chosen.

---

## 1.5 🟠 Lankford solver accepts negative velocities (garbage propagates)

**File:** `src/mobility_module.py`, `single_lankford_run`, **line ~237**.

`fsolve` result is accepted on convergence flag only (`V_un[2] == 1`), not sign — unlike `single_bike_run`, which loops guesses and rejects negatives. With lower METs (see §1.7), ~4% of runs draw below ~1.93 METs where the **loaded** Lankford velocity is negative, and ~11% below where it's near-zero. Negative/garbage loaded velocities can flow into distance/coverage.

**Fix:** clamp loaded and unloaded velocities to `>= 0` (and treat non-convergence / negative as 0 reachable distance), e.g.:
```python
loaded_velocity = max(loaded_velocity, 0.0) if V_load[2] == 1 else 0.0
```
Add an assertion/test (see §1.9). This is a correctness floor independent of the parameter debate, but becomes important the moment METs are lowered.

---

## 1.6 🟡 Cycling downhill cap makes single runs polarity-sensitive (no code change, but know it)

**File:** `src/mobility_module.py`, `single_bike_run`, lines ~301 & ~332 (velocity capped at 7 m/s).

Under `uphill_downhill` / `downhill_uphill` polarity draws, the loaded downhill leg pins at the 7 m/s cap regardless of watts, so average cycling velocity ≈ (slow-uphill + 7)/2 and becomes **watts-insensitive**; a single run's cycling distance can swing ~2–4× on the polarity draw alone. This is fine across 1000 MC runs (it averages out) but means **no single median-parameter run is representative** — relevant when sanity-checking (see the note to James in §3). No change recommended; documented so nobody "validates" against one run.

---

## 1.7 🟠 Monte Carlo parameters — METs / time / watts (the core of James's proposal)

**Files (⚠️ constants are DUPLICATED — see §1.8):** `scripts/run_monte_carlo.py` lines ~40–86; `scripts/run_monte_carlo_gcp.py` lines ~44–90; `scripts/run_monte_carlo_test.py` lines ~42–72.

Samplers in `src/gis_monte_carlo.py`: `sample_normal(low, high, n)` and `sample_lognormal(low, high, n)` treat `[low, high]` as a 90% CI. `sample_lognormal` is currently **unused in production** — switching a parameter to lognormal requires changing the sampler call too, not just the numbers.

### Evidence-based recommendation (PENDING Kevin/James sign-off — do not hard-code as final without it)

| Parameter | Current code | James proposed | Fact-check verdict | **Recommended** |
|---|---|---|---|---|
| **METs (walking)** | normal 3–6 (med 4.5) | normal 2–5 (med 3.5) | James's own sources center water-carrying at **4.3–5.0** (Mozambique 4.3, Baka 7.28); 4 of 6 Compendium values were mis-cited 2000-edition (true values higher). 2-MET floor & 5-MET cap both unsupported. | **normal 3–5 (med 4)** — a defensible compromise: floor at empty-walking cost (~3.1), cap below the Baka outlier, median between old 4.5 and James's 3.5. |
| **Time gathering (h)** | normal 4–7 (med 5.5) | lognormal 2–5 (med 3.16) | Not challenged by fact-check; manuscript sources support up to ~4 h/day. Distance is **linear** in time, so this is the single biggest lever — flag sensitivity. | **lognormal 2–5 (med 3.16)** as James proposes — reasonable; document the lognormal choice. |
| **Watts (cycling, mechanical)** | normal 20–80 (med 50) | lognormal 10–60 (med 24.5) | ACSM maps 20–80 W ↔ **exactly 3–6 METs** at 62 kg — the old range was deliberately calibrated to the old walking budget. Watts must track whatever METs range is chosen. | **Derive from METs (preferred)** or calibrated **normal 20–60 (med 40 ↔ 4 METs)** to match recommended METs 3–5. |

### Watts: two implementation options

**Option A (preferred, more rigorous & reviewer-proof): derive mechanical watts from the per-iteration sampled MET via the ACSM cycle-ergometry equation.** This ties the cyclist and walker to one shared metabolic budget, kills the "strong-walker + feeble-cyclist" impossible combinations that inflate the CI (METs & watts are currently sampled *independently*), and makes the efficiency assumption auditable.

Inversion of `VO2 = 1.8*(W*6.12)/mass + 7`, `VO2 = MET*3.5`:
```python
# per iteration, reference mass 62 kg (or country weight if made country-specific)
watts = np.maximum((mets * 3.5 - 7.0) * 62.0 / (1.8 * 6.12), 5.0)   # floor ~5 W (≈2-MET unloaded cost)
```
Check values: MET 3→~20 W, 4→~39 W, 4.5→~49 W, 5→~59 W (matches the old 20–80↔3–6 calibration and James's own converter). Floor at the natural ~2-MET unloaded cost, **not** an arbitrary 10 W.
This deletes the WATTS constants entirely and replaces the independent `watts_values` sampling with a derivation from the already-sampled `mets` array. ~3 lines per script + one methods sentence.

**Option B (minimal change): keep independent sampling but calibrate the range** to the chosen METs via ACSM and document the mapping. If METs = normal 3–5, set WATTS = normal 20–60 (median 40 ↔ 4 METs). Keeps watts normal (no lognormal switch needed). Does NOT fix the impossible-combination CI inflation.

Recommend **A** if the ~1 methods sentence is acceptable; **B** if minimizing change footprint before the deadline.

### Sampling caveats to preserve
- `sample_lognormal` asserts `low > 0` (fine for time=2, watts=10).
- Lower METs interacts with §1.5 (negative-velocity clamp) — do §1.5 regardless.
- Document the **ACSM extrapolation caveat**: the equation is validated ≥50 W; 10–60 W is below that, but its fixed +7 intercept extrapolates safely (cite low-power VO₂-linearity literature) — see §2.13.

---

## 1.8 🟠 Parameter plumbing — de-duplicate constants; the cloud runs a different script

**The bug that will silently ruin a cloud run:** the three run scripts each hold their **own copy-pasted** copy of every constant (no shared config). `gcp/deploy-spot.sh:168` runs **`scripts/run_monte_carlo_gcp.py`**, not `run_monte_carlo.py`. **Editing only the local script → the Spot VM produces old-parameter results.**

**Fixes (in priority order):**
1. **Extract a single shared constants module** (e.g. `src/monte_carlo_config.py`) imported by all three scripts. Permanently eliminates drift. Recommended.
   - If not doing the refactor: apply every §1.7 change identically to **all three** scripts and switch `mc.sample_normal → mc.sample_lognormal` at each relevant call site.
2. **`checkpoint.json`** (written by `run_monte_carlo_gcp.py` ~lines 200–264) serializes the sampled parameter arrays and restores them on resume — a stale checkpoint overrides new constants. **Delete it on any parameter change.**
3. **Ensure `calculate_distance=True`** in the production run. If `False`, `calculate_and_merge_*_distance` (`gis_global_module.py` ~698–707, ~804–813) silently reuses stale `*_velocity_by_zone.csv` instead of recomputing with new watts/METs.

---

## 1.9 🟠 Add regression tests (currently nothing guards this)

**File:** `tests/test_mobility_module.py` (no Lankford test today); `tests/test_gis_monte_carlo.py`.

The Lankford path has **zero direct tests** — the GIS tests mock `single_lankford_run` with canned values, so none of the §1.1–§1.5 bugs are visible to CI. Add:
1. **Slope monotonicity:** loaded walking velocity is non-increasing as slope 0°→5°→10°→20°, and finite/non-negative at the clipped max (guards §1.1–1.3).
2. **Velocity non-negativity:** across a MET sweep including low METs (2.0), returned velocities are `>= 0` (guards §1.5).
3. **Sampler sanity (optional):** `sample_lognormal(2,5)` median ≈ 3.16; `sample_normal(3,5)` median ≈ 4 — so parameter/sampler swaps don't silently mis-behave.
4. **Constants integrity (optional):** if a shared config module is created, one test asserting the three scripts import it (prevents drift regressions).

Existing tests do NOT pin distribution shape, so switching time/watts samplers needs no test edits — but `test_valid_input_returns_result` uses `watts=75` (above the new range; still type-valid).

---

## 1.10 🟡 Housekeeping
- Delete the resolved `# this was previously 0.00320 - bug?` comment (`mobility_module.py:607`) once §1.2 is done.
- Remove the stray `print(type(crr_adjustment))` debug line in `src/gis_monte_carlo.py` `run_simulation` (~line 135) before a production run (noisy over 1000×).

---

# SECTION 2 — MANUSCRIPT CHANGES

Line numbers refer to the markdown file (see header). All are low-effort unless noted. Priority key as above.

---

## 2.1 🔴 Delete co-author's raw parameter worksheet left in Supplementary Materials

**Lines 350–396.** Under `# Supplementary Materials`, after the reference list, sits unfinished scratch working — verbatim quotes plus lines like *"Maybe reduce 3-6 mets to 2-5 mets? Median goes 4.5 to 3.5?"*, *"Still would probably get one way of ~7km, which seems too high"*, *"Currently 3.6km, whereas 5 still seems reasonable one way?"* This is the parameter-change reasoning, accidentally left in the submitted document. **Highly embarrassing if a reviewer sees it. Delete lines 350–396 entirely; keep only line 348 (the legitimate "Link to tables") and the footnote at 398.**

---

## 2.2 🔴 Eq 5 (Lankford) cubic-term sign — `+` must become `−` (CONFIRMED via PDF)

**Lines 119–120.** Manuscript writes `+0.000320vsp3`. **Confirmed against the Lankford 2020 PDF:** Table 4 (the fitted coefficient table, p. 2102) lists this term as **−0.0001431 with the entire 95% CI below zero**, so the true sign is unambiguously **negative**. The published paper prints its mph-form equation with a `+` sign typo, which the manuscript inherited. Change to `−0.000320·v·sp³`. The code (src/mobility_module.py, §1.2) is already correct (negative) on the branch. All other Eq 5 coefficients verified correct AND are the genuine **m/s-form** coefficients (intercept 5.43483, +6.47383·v, −0.05372·sp, +0.652298·v·sp, +0.023761·v·sp²) — the "v is velocity (m.s⁻¹)" label at line ~117 is correct.

**Resolved non-issue (do not re-raise):** a concern that the code used mph coefficients while treating the solved speed as m/s was investigated and dismissed. By unit invariance the code's larger coefficients ARE the m/s form (mph coeffs would be smaller: ×0.44704); solved speeds (~4.8–5.7 km/h flat) and cached velocities are realistic only as m/s. The walking velocity is correctly in m/s throughout — no change needed.

---

## 2.3 🟠 Eq 1 (aerodynamic drag) is missing air density ρ

**Line 107.** Text (line 105) defines "⍴ is air density," but Eq 1 as written is `PAD = ½ CD A vA² vG` — ρ never appears. Standard aerodynamic-drag power is `P = ½·ρ·C_D·A·v³`. Insert ρ: `PAD = ½ ρ CD A vA² vG`. **Also check the typeset PDF** — a drag term with no density is an easy reviewer catch. (The code's `bike_power_solution` does include `ro`, so this is a manuscript-only transcription slip.)

---

## 2.4 🟠 Wrong internal cross-reference

**Line 144.** "Following the GIS processing described in **2.3 Data Processing**…" — the GIS pre-processing is in **§2.4 "Data pre-processing"** (line 89); §2.3 "Data" just lists datasets. Line 53 cites it correctly as "2.4". Change "2.3 Data Processing" → "2.4 Data pre-processing".

---

## 2.5 🟠 Garbled sentence (tracked-changes collision)

**Line 190.** "With relatively little funding dedicated to CIL preparedness [cite] **While**, this lack of preparation leaves an **openingopportunity** for alternative targeted **interventionstargeted**, high-impact actions." Suggested fix: *"With relatively little funding dedicated to CIL preparedness [cite], this lack of preparation leaves an opening for alternative, targeted, high-impact actions."*

---

## 2.6 🟠 Abstract garble

**Line 17.** "28.2% **would from be able** to reach water by walking" → "28.2% would be able to reach water by walking".

---

## 2.7 🟠 Citation key typo NMDA → NDMA (both occurrences)

**Lines 198 and 300.** The organisation is the **N**ational **D**isaster **M**anagement **A**uthority = **NDMA**; the key is transposed to "NMDA" in both the in-text cite (198) and the reference (300). Fix both so they still match.

---

## 2.8 🔴 Methods §2.5.6 — update distribution descriptions & parameter ranges

**Line 150.** This sentence currently says time gathering was "sampled as a **normal** distribution between 4 and 7 hours," power output 20–80 W, METs 3–6. After the re-run it must state the **final agreed** ranges and distribution types (time and possibly watts become **lognormal**; METs stays normal). **This is the one methods sentence whose structure changes, not just its numbers.** If watts are derived from METs (§1.7 Option A), add one sentence describing the ACSM-based derivation and the efficiency/extrapolation caveat (§2.13).

---

## 2.9 🟠 Validation of the cycling:walking distance ratio (§3.1)

**Line 174.** Currently: cycling:walking = 11.8/3.6 = "3.3×… relatively close to… three times." After the re-run the ratio rises to ~4.3×, breaking the "three times" claim. **Do not delete the validation — re-source it:**
- Keep Larsen et al. 2010 but cite its **mean / 85th-percentile** ratio (~3.9×); the authors themselves say "nearly four times" (median is 3.43).
- Add the **FTA/TCRP** transit-catchment standard (0.5-mile walk vs ~3-mile bike; FTA states cyclists cover "three or four times" the walking distance).
- Demote Solheim & Stangeby 1997 to loose corroboration (it's really ~2.5×, a weak anchor).
Reword to something like "…acceptable cycling distances are typically three to four times walking distances, consistent with our median factor of ~4.3."

---

## 2.10 🔴 Re-derive EVERY headline number after the re-run (checklist)

All of these change. Update from the new MC output:
- **Line 17 (abstract):** 24.3% (14.4–34.8); 28.2% walking; 12.0% cycling; 35.6% unpiped.
- **Line 160:** 24.3% / ~1.9 bn; Americas 39.2 (24.6–54.2); Europe 24.6 (10.7–42.3); Africa 22.7 (15.3–30.0); Asia 21.3 (12.2–31.5); Oceania 18.8 (9.0–32.2).
- **Line 166:** 12.0% / ~1 bn; Europe 19.1; Americas 18.1; Africa 2.9; Africa unpiped 60.4; Asia 38.5; Oceania 23.3.
- **Line 168:** China + India 560 million.
- **Line 174:** walking 3.6 km (1.6–6.1); cycling 11.8 km (6.7–16.6); ratio (§2.9).
- **Line 180 (Venezuela):** 48.0% (33.9–59.4); capital 85.1; La Guaira 66.5. ⚠️ Venezuela is mountainous — §1.1 (slope fix) will move these more than most regions. External figures (31% Punto Fijo, 78% piped, 98% alternatives) stay.
- **Line 222 (conclusion):** "nearly a third" — reconcile with the new median (was already inconsistent with 24.3% ≈ a quarter).
- **Cross-check** the abstract mode-% sum (currently 24.3+28.2+12.0+35.6 = 100.1%) reconciles with the "undefined access ≤2%" footnote (Fig 4, line 164) once new medians are in.

---

## 2.11 🟠 Parameter-justification framing (avoid "reverse-engineered" critique)

Wherever the new parameters are justified (Methods §2.5.4/2.5.6 and Supplementary Table S1), make the case **source-driven, not distance-target-driven.** The deleted worksheet (§2.1) chose values partly so distances "go up a bit, but not a lot" — a reviewer would call that motivated reasoning. Justify each range by its citations. **And note the fact-check finding:** the evidence centers water-carrying METs at ~4.3–5.0, so a median of ~4 (recommended) is far more defensible than 3.5; if the authors keep 3.5 they need an explicit whole-round-trip-with-rest-breaks argument, because the raw carrying values do not support it.

---

## 2.12 🟠 Correct the Compendium of Physical Activities citations (Supplementary Table S1)

If Table S1 (the linked spreadsheet) lists the per-activity MET values, **4 of 6 are 2000-edition values mislabeled as 2011:** walking 2 mph slow = **2.8** (not 2.5); pushing a wheelchair = **3.8** (not 4.0); lifting 10–20 lb limited walking = **4.5** (not 4.0); farming/hauling water = **4.3** (not 4.5). Correct = walking the dog 3.0 and carrying 25–49 lb 5.0. Source: Ainsworth et al. 2011, *MSSE* 43(8):1575. The 2024 Compendium even adds "hauling water, head hauling, flat = 4.5 METs." Correcting these strengthens a median-~4 choice.

---

## 2.13 🟡 Add a one-line caveat on the ACSM low-power cycling extrapolation

If watts are derived from/calibrated to METs via ACSM (§1.7): the ACSM cycle-ergometry equation is validated ≥50 W, and the fetching regime is below that. Note that its fixed intercept (unloaded-pedalling + rest cost) makes downward extrapolation stable, and cite the low-power VO₂-linearity literature. One sentence in Methods or Limitations. (Low priority; only if a reviewer is likely to probe the conversion.)

---

## Deliberately LEFT ALONE (considered and rejected as not worth the churn)
- Export artifacts: nested markdown section numbering, a missing bold-period on the "Slope polarity" bullet (line 130). Cosmetic.
- Eqs 2, 3 (rolling resistance, potential energy) — checked, dimensionally correct.
- Eq 4 renders division-by-Ec as a trailing multiplier in markdown — verify the typeset PDF shows `/E_C`; likely an export artifact only.
- Preprint/grey-lit citations (Jehn 2024, Williams 2025, Lamilla Cuellar 2025, etc.) — normal for GCR literature; none underpin the headline.
- The "90% CI from only 5 varied parameters" critique — the paper already hedges honestly (lines 166, 216). Optional caveat at most.
- Minor hyphenation ("non governmental", "non-piped" vs "nonpiped").

---

# APPENDIX — Suggested implementation order (single re-run)

1. Code: §1.1 + §1.2 + §1.3 (Lankford units + sign + clip, as one block) → §1.5 (velocity floor) → §1.4 (polarity decision) → §1.7 (parameters, pending sign-off) → §1.8 (de-dup constants, clear checkpoint, verify `calculate_distance=True`) → §1.10 (housekeeping) → §1.9 (tests).
2. Run the full 1000-iteration Monte Carlo via the **gcp** script (or local), fresh results dir.
3. Manuscript: §2.1 (delete worksheet) + §2.3–§2.7 (cheap text fixes) immediately; §2.2 (Eq 5 sign) after PDF check; §2.8–§2.13 (methods + all headline numbers + validation re-sourcing) from the new output.
4. Sanity-check, do NOT validate against any single median-parameter run (§1.6) — only the 1000-run medians/CIs are meaningful.

**Everything hinges on §1.0/§1.1: the paper's numbers are already stale vs the code, and the biggest remaining correctness issue (slope-blind walking) is coupled to a sign fix and slope clipping. Treat those three as one atomic change with the monotonicity test as the gate.**
