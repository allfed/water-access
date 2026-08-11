# Investigation memo: the ~348M "resilient by gap" population

**Date:** 2026-08-11 · **Branch:** `kr/unpiped-breakdown` · **Status:** analysis only — nothing applied, awaiting sign-off.

## TL;DR

The 347.7M figure and all four previously-diagnosed root causes verify exactly. But the strategic picture changes once the gap is decomposed:

1. **~148M of the 348M is genuinely "other unpiped"** — people JMP itself places on unimproved sources or surface water (same data vintage as the piped shares). Classifying them resilient is *correct and defensible*, not a bias. This is directly citable against review comment 2.
2. **~200M is unallocated for a knowable reason**: mostly *piped-definition drift* — the household survey reports more piped users than JMP's harmonized estimate (India urban: survey 81.3 vs JMP 66), so the difference falls out of every category — plus three newly-found parser losses (below). These people are mostly on piped/borehole/packaged-type sources, i.e. **plausibly grid-vulnerable**.
3. **Fixing the previously-listed mechanical bugs recovers ≈0 of the 348M** — normalization and reconciliation only ever scale *down*, and the >100 rows (USA, SLB) contribute zero gap because the residual clips at 0. However, **newly-found bugs** (total-only surveys dropping packaged/delivered; Niger name-mapping; 5 countries missing from the unified CSV) account for ~25–30M and *are* recoverable.
4. Quantified bracket on the affected side (baseline 5.99B affected): status quo +0 · ladder-informed pro-rata **+150M (+2.5%)** · full pro-rata +242M (+4.0%) · everything-vulnerable +348M (+5.8%).

**Recommendation (one path):** fix the new mechanical bugs, make the gap an explicit `Unallocated` column with the JMP-ladder share moved into `OtherUnpiped` (documented), and treat the remaining unknown's allocation as a *sampled model parameter* in the Monte Carlo — bracketing resilient↔pro-rata↔vulnerable. Central estimate ≈ +2.5% affected; the band is the reviewer-proof answer to comment 2. Details in §6.

---

## 1. Verification of the prior diagnosis

Every claim in the investigation brief reproduces against `data/processed/merged_data.csv` and the code:

| Claim | Verdict |
|---|---|
| Urban share-sum mean 96.93 (30.31–116.28), rural 92.36 (2.00–138.74) | ✅ exact |
| Gap population 347.7M = 4.42% of 7.87B; IND 89.0M, CHN 77.5M, PAK 29.8M, PHL 19.6M; 18 countries >20% | ✅ exact |
| Residual + clip at `gis_global_module.py:960–981`; `zone_pop_other_unpiped` (line 944) dead | ✅ |
| Parser takes latest WHO year unconditionally; piped "-" → normalisation silently skipped (`jmp_source_parser.py`, `load_piped_reference_from_water_csv`) | ✅ mechanism confirmed |
| Validator skips `piped + unpiped > 100` check when reference is None (`build_unified_source_breakdown.py`, the `elif piped_value is not None` branch) | ✅ |
| Reconciliation only touches `data_origin == "research"` rows (`imputation_script.py:605, 643`) | ✅ |
| `fillna(0)` asserts zero for missing categories (`imputation_script.py:686`) | ✅ |

**One important correction to the brief's framing:** the USA-rural (138.74) and SLB-urban (116.28) overshoot rows do **not** corrupt the headline arithmetic. The excess sits entirely in `OtherUnpiped` (USA rural: Piped 95 + OtherUnpiped 43.7), which the affected-side formula never reads; the resilient residual just clips to a too-small value. They are composition errors, not population errors.

### New root causes found in this pass (not in the brief)

5. **Niger → Nigeria name collision.** `pycountry.search_fuzzy("Niger")[0]` returns `NGA`. In the parser's reference builder Nigeria's row overwrites it (so NGA is fine), but **NER ends up with no piped reference at all** — silently unnormalised and invisible to the validator. NER, TWN, ESH have no WHO reference under the current mapping.
6. **Total-only surveys lose everything except borehole.** For surveys reporting only country totals (no urban/rural split), `resolve_borehole_urban_rural` falls back to the total for borehole — but there is no such fallback for packaged/delivered/other. Myanmar's MMR_2021_HFPS reports **packaged 22.7, surface 15.9, rain 3.5** (total); all shipped as 0. Same for Ukraine (packaged 29, tanker 6). This alone explains most of MMR's 18.2M and UKR's 4.8M gaps — and the dropped packaged share is *grid-vulnerable*, so this one is a real bias, mechanically fixable.
7. **Row-99 "Other improved sources" sub-`Other` rows (102–103) are never extracted** (only cart 100 / tanker 101 are). Pakistan: 2.8pp rural. Philippines: **45.5pp rural / 61.3pp urban** sit there (FIES "community water system" users) — most harmonizes into JMP's piped estimate, but the non-piped remainder is Level-I point sources, typically *motorized* deep wells, i.e. grid-relevant.
8. **Five merged countries are absent from the unified CSV entirely** (FLK, ISR, NIC, TWN, ESH) — all their subcategories become 0 via `fillna(0)`. NIC alone is 2.0M gap population.

---

## 2. Decomposition of the 347.7M

Stratum-level (country × urban/rural), using `merged_data.csv`, the unified CSV, and the WHO 2023 CSV:

| Bucket | Gap pop | Notes |
|---|---:|---|
| 1. No data at all (`data_origin == "none"`, 23 countries) | 3.3M | VEN 2.8M dominates |
| 2. Country missing from unified CSV (new cause #8) | 3.4M | NIC 2.0M, TWN 1.2M |
| 3. Parser reference None → unnormalised (brief's cause #3) | 5.2M | KAZ 2.9M, LBY 1.4M |
| 4. Research-origin rows with residual gap | 5.4M | ROU 2.5M, JPN 2.3M |
| 5. JMP-origin incomplete subcategories (the "honest gap") | **330.4M** | 95% of the total |

Bucket 5 splits further using the WHO ladder identity (`Piped + Non-piped + Unimproved + Surface ≈ 100`, same vintage as the shipped piped values):

| Component | Gap pop | Interpretation |
|---|---:|---|
| Unimproved + surface-water users | **146.7M** | JMP itself says these people are on unimproved/surface sources. Not grid-dependent → **"resilient" is the right call**, now documentable. |
| Improved non-piped shortfall | **183.7M** | Mass JMP says is on improved non-piped sources but the survey subcategories don't cover. Root causes: piped-definition drift, total-only drops, row-99 misses. **Plausibly grid-vulnerable in part.** |

(Across all buckets: 148.1M genuine-other + 199.6M unknown.)

## 3. Artifact vs genuine, top contributors

Verified against the raw JMP workbooks (downloaded fresh from washdata.org for IND/CHN/PAK/PHL/MMR/UKR):

| Country | Gap | Genuine other | Unknown | What the unknown actually is |
|---|---:|---:|---:|---|
| IND | 89.0M | 29.8M | 59.2M | **Piped-definition drift.** HCES-2023 reports piped 81.3 urban / 41.0 rural; shipped JMP piped is 66 / 33. The gap people are mostly counted as *piped* by the survey. Artifact → allocate pro-rata (piped-heavy → affected). |
| CHN | 77.5M | 32.1M | 45.4M | Census-2020 is **binary** (piped vs "other"); census piped 82.4 rural vs JMP 72. Borehole = 0 is a false zero — rural China's 25% improved non-piped includes many electric-pumped wells. Worst case for naive pro-rata (nothing to pro-rate over). |
| PAK | 29.8M | 11.9M | 17.9M | Drift (census piped 30.1 rural vs JMP 17) + 2.8pp in unextracted row-99 "Other". |
| PHL | 19.6M | 3.1M | 16.5M | FIES "community water system" (45–61pp) in unextracted rows; the non-piped part is Level-I point sources — often motorized. **Most grid-relevant unknown of the top four.** |
| MMR | 18.2M | 8.2M | 10.0M | Mechanical: total-only survey dropped packaged 22.7 / surface 15.9 (cause #6). Fixable. |
| NER | 10.2M | 7.8M | 2.4M | Mostly genuine (surface/unimproved); also no piped reference (cause #5). |
| TCD | 7.3M | 6.0M | 1.3M | Genuine — status quo already right here. |
| SSD | 4.4M | 2.4M | 2.0M | Mostly genuine. |
| CAF | 2.3M | 1.8M | 0.5M | Mostly genuine. |
| NRU | 0.01M | 0 | 0.01M | Negligible (packaged-dominant microstate). |

Pattern: **the Sahel/LDC "worst-relative" countries are genuinely other-unpiped** (status quo correct); **the Asian mega-contributors are artifacts** of survey-vs-JMP piped harmonization and parser extraction limits (status quo wrong-direction).

## 4. Option comparison (quantified)

Effect on global grid-affected population; baseline 5.99B (lose-fraction midpoints 0.716 urban / 0.287 rural). Headline pass-through is sub-linear (headline = affected × per-country can't-reach fraction) and needs the MC rerun to pin down.

| Option | Global affected Δ | IND / CHN / PAK / PHL Δ | Assumes | Breaks where |
|---|---:|---|---|---|
| (a) Mechanical fixes only | **≈ +0M** on the original bug list; **new causes #6/#8 recover ~25M** into real categories (MMR/UKR packaged → affected; NIC/TWN filled) | MMR +~5M affected | Bugs caused the gap | They mostly didn't — normalization/reconciliation only scale down |
| (b) Pro-rata over known categories | **+242.3M (+4.04%)** | +62.0 / +66.0 / +14.2 / +15.8M | Unknown mass mirrors known mix | CHN (binary census: pro-rata is piped-vs-other only); Sahel countries where gap is genuinely other (over-counts affected there) |
| (c) Regional source-mix priors | between f1 and b (not fully simulated) | — | Neighbors' mix transfers | Adds a modeling layer reviewers can attack; only materially helps CHN-type binary-census rows. Cost: highest. Skip unless demanded. |
| (d) Status quo, documented | +0M | +0 | Unknown = resilient | IND/CHN/PAK/PHL where the unknown is demonstrably piped/packaged/motorized-well-like; one-directional bias stands |
| (e) Scenario bracket (d ↔ b ↔ all-vulnerable) | 0 ↔ +242 ↔ **+347.7M (+5.80%)** | e.g. IND 0 ↔ +62 ↔ +89M | Truth is inside the bracket | Nothing — but a ±350M band unweighted by evidence is unnecessarily wide |
| **(f) NEW: JMP-ladder split + pro-rata unknown** | **+149.5M (+2.50%)** | +42.9 / +38.0 / +8.8 / +13.4M | 148M unimproved/surface (JMP's own numbers) stay resilient; only the 200M unknown is allocated pro-rata | Countries where survey *did* capture surface water in OtherUnpiped (slight double-credit to resilient; second-order) |

A pro-rata-over-non-piped-only variant of (f) was also simulated: +66.7M, but it sends CHN to exactly 0 (nothing to pro-rate over) — rejected.

## 5. Answer to "how much of the 348M survives fixing the bugs?"

Essentially all of it (~345M). The originally-listed bugs produce the >100 overshoots, which contribute zero gap by construction. The gap's real drivers are (i) JMP-vs-survey piped harmonization drift, (ii) the parser's incomplete row extraction, (iii) genuinely unimproved/surface populations. Only the *newly-found* total-only and missing-country bugs (~25M) convert into real categories when fixed.

## 6. Recommended path

**Layer 1 — mechanical fixes (do regardless, ~½ day + regeneration):**
1. Piped reference: latest year *with piped data*, plus `"Niger": "NER"` in the manual alpha3 map (and dedupe protection so a fuzzy collision can never overwrite silently).
2. Total→urban/rural fallback for packaged/delivered/other-unpiped, mirroring the existing borehole fallback (recovers MMR/UKR).
3. Extract row 99 remainder (rows 102–103) into `Delivered`… no — into a conservative bucket: add to `OtherUnpiped` *after* checking it doesn't double-count with piped harmonization; PAK/PHL cases suggest per-country review. Minimum: validator warning when row 99 total ≫ cart+tanker.
4. Close the validator blind spot (also check `unpiped_sum` vs the *shipped* piped when the WHO reference is None) and extend the reconciliation pass to JMP rows.
5. Add NIC/TWN/ESH/FLK/ISR to the unified build (or document their exclusion).

**Layer 2 — make the gap explicit in the dataset (~½ day):**
- In `imputation_script.py`, stop letting the gap vanish into a residual: emit `URBANUnallocated` / `RURALUnallocated` columns = `clip(100 − sum, 0)`, and move `min(gap, Unimproved + Surface)` (from the WHO CSV, same vintage) into `OtherUnpiped` with a provenance flag. After this, `OtherUnpiped` is honest and `Unallocated` is the truly-unknown ~180M.

**Layer 3 — allocation as a sampled parameter (respects the "scenario logic lives in parameters" rule, ~½ day given the lose-fraction wiring pattern):**
- In `calculate_population_water_access`, replace the silent residual with explicit accounting: `resilient = OtherUnpiped + (1−f)·Borehole + (1−g)·Unallocated`, `affected += g·Unallocated·(affected_mix)`, where `g` is a new model parameter (0 = status quo, pro-rata-equivalent ≈ 0.75 given the known-mix weights, 1 = all-vulnerable). Sample or scenario-sweep `g` in the Monte Carlo exactly like the lose fractions. Delete the dead `zone_pop_other_unpiped` or use it in the new accounting.
- Headline reporting: central run at the ladder-informed pro-rata equivalent (**≈ +2.5% affected**), band `g ∈ {0, 1}` giving 0 to +5.8% on the affected side.

This gives the paper a sentence reviewers can't argue with: *"148M people on unimproved or surface sources (JMP 2022) retain access by construction; for the remaining 1.8–2.6% of the world population whose source type is unallocated after harmonization, we bracket the outcome between fully-resilient and fully-grid-dependent, shifting the global affected population by 0–5.8% (central estimate +2.5%)."*

## 7. Reproduction

Analysis scripts are in `scripts/investigations/unallocated_gap/` (`decompose.py` → bucket table, `split_bucket5.py` → ladder split, `simulate_options.py` → option deltas); they run on system python (pure pandas on CSVs) and reproduce every number above from repo data alone. Note the scripts write intermediate CSVs to a session scratchpad path — point `SCRATCH` anywhere writable when re-running. Raw JMP workbooks: `https://washdata.org/data/country/{ISO3}/household/download`. Lose-fraction midpoints from `ELECTRIC_BOREHOLE_SHARE_ESTIMATE.md` (0.716 / 0.287). All results in `results/` remain stale; nothing here relies on them.
