# Electric vs hand-pumped boreholes: the lose_fraction estimate

Kevin, 10 Aug 2026. Branch: `kr/unpiped-breakdown`. This is the deliverable
named in `HANDOFF.md` §"Single next action": a defensible range for the share
of borehole users who rely on electric pumps vs handpumps, to parameterize
`urban_borehole_lose_fraction` / `rural_borehole_lose_fraction` in
`gis_global_module.calculate_population_water_access()`.

## Recommended Monte Carlo parameters

**HEMP scenario** (all motorized pumping fails — grid, diesel, solar):

| Parameter | Recommended range | Central | Current code default |
|---|---|---|---|
| `urban_borehole_lose_fraction` | **0.63 – 0.80** | ~0.72 | 0.5 (too low) |
| `rural_borehole_lose_fraction` | **0.23 – 0.35** | ~0.29 | 0.25 (inside range, optimistic end) |

**Loss-of-central-infrastructure scenario** (grid fails; private diesel
generators and solar pumps survive while fuel/equipment lasts): multiply the
HEMP lose fractions by **0.75–0.95 (urban)** and **0.85–1.0 (rural)**. Basis:
the motorized share is overwhelmingly grid-electric everywhere it was
measurable (India's NSS definition is "electric or diesel" but household
electrification is ~97–99% post-Saubhagya and diesel is a shrinking pocket
phenomenon; urban SSA is the exception — Nigerian urban boreholes are
routinely generator-backed, which is why the urban multiplier dips lower).
Survivorship is time-limited by fuel supply; for a multi-month scenario the
multipliers converge back toward 1.0.

Sampling note: the bounds above are the population-weighted envelope of
per-country low/high bounds, i.e. "every country simultaneously at its
optimistic (resp. pessimistic) edge". That is the right envelope for MC here
because the dominant uncertainties are *systematic across countries* (the
same inference method — pump-stock-to-user-share conversion — is used in many
of them, so errors correlate). Uniform sampling across the range, or
triangular with the central values above, is defensible.

## What the parameter means and why 1 − handpump share

`Borehole` in this model is exactly JMP row 57 "Tubewell, borehole"
(`jmp_source_parser.py:46`) — people whose **main drinking source** is a
drilled tubewell/borehole, whether hand-pumped or motorized. Handpump users
keep their source when electric pumping fails; motorized-borehole users are
displaced. So `lose_fraction = 1 − (handpump share among borehole users)`.

The denominator matters: a census saying "43.6% of rural households use
handpumps" is a share of *all* households; every number below is converted to
handpump / (handpump + motorized borehole), both as main drinking source, and
the adversarial verification pass re-did that arithmetic from primary tables.

## Method

Same playbook as the source-breakdown research
(`researched_source_breakdown.csv`): a 6-region research fan-out against
primary sources (censuses, DHS/MICS/LSMS/PSLM/NSS, national water-point
inventories), followed by adversarial verification of the highest-stakes
countries, weighted by where the borehole population actually is:

- ~1.148B people worldwide have a tubewell/borehole as main drinking source
  (from `merged_data.csv`: 290M urban, 858M rural).
- India (520M), Bangladesh (138M), Pakistan (104M) are 66% of the global
  total; Nigeria (81M) has the largest *urban* borehole population anywhere
  (52M). Sub-Saharan Africa is 20%, South-East Asia 8%; everything else <4%.

Research effort followed those weights: India got a dedicated agent; the
long tail got regional defaults. 46 countries have researched per-country
bounds covering **94.3% of the urban and 96.7% of the rural** borehole
population; the remaining ~5% use regional default ranges.

Adversarial verification (independent agents instructed to refute, re-derive
the arithmetic from primary sources they located themselves): IND and PAK
**CONFIRMED** digit-for-digit; BGD, IDN, UGA, NGA, UKR **PLAUSIBLE** (inference
chains verified, no refutations; NGA urban corrected 8–25 → 10–25). None
refuted.

## Key country results (handpump share among borehole users, %)

| Country | Urban | Rural | Evidence | Verification |
|---|---|---|---|---|
| India (45% of global) | 25–38 | 70–80 | Census 2011 H-series + NSS 76th (2018) measure the split *directly* (census defines Hand pump = manual, Tubewell/Borehole = electric/diesel pump). Measured: rural 84.1% (2011) → 79.7% (2018); urban 57.1% → 39.2%. 2024 range extrapolates that slope, constrained by NSS-MIS 2020-21 / CAMS 2022-23 combined-category trend under Jal Jeevan Mission | CONFIRMED (anchors reproduced exactly) |
| Bangladesh | 35–65 | 88–98 | "Tubewell" ≈ shallow suction handpumps: ~16M private hand tubewells + ~1.8M public handpump points vs a small motorized stock; pushing rural handpump below 88% would require >2M motorized drinking tubewells no source supports. Urban: peri-urban Dhaka is already ~97% motorized among borehole users; smaller towns are not | PLAUSIBLE |
| Pakistan | 21–26 | 46–53 | PSLM 2019-20 Table 7.1 splits "hand pump" vs "motor pump" directly: urban 7.32/31.12 = 23.5%, rural 33.44/67.36 = 49.6%. Strong provincial gradient (Punjab rural 40%, Sindh rural 90%) | CONFIRMED (digit-for-digit) |
| Nigeria | 10–25 | 35–60 | WASH-NORM 2021 water-point census: urban handpump/(handpump+motorized) points = 14.8%, rural 39.3%; urban Nigeria runs on private electric boreholes (frequently generator-backed → matters for the non-HEMP scenario) | PLAUSIBLE (urban corrected up: handpump boreholes are the *most* functional type nationally, 78% vs 64%) |
| Indonesia | 2–10 | 10–35 | BPS "sumur bor/pompa" doesn't split pump type; urban bore wells near-universally electric jet pumps (Genter 2023 self-supply work); rural bounded by appliance-ownership arithmetic | PLAUSIBLE |
| Uganda | 50–85 | 88–99 | MWE water atlas: ~41k hand-pumped deep boreholes are the only point-source fleet serving ~16M rural borehole users; motorized boreholes feed piped schemes (whose users report taps, i.e. leave the Borehole category) | PLAUSIBLE (two source-conflation defects found and repaired; conclusion survives via category structure) |
| Ukraine | 0–5 | 5–20 | No pump-type split exists post-MICS-2012; bracketed on the Bosnia census analogue (11:1 electric:manual) adjusted for documented Ukrainian street hand-pump columns / Abyssinian wells | PLAUSIBLE |

The full 226-row table (all countries, bounds, evidence kind, confidence,
source, URL, verification status) is
`data/original_data/borehole_pump_type_estimates.csv`; the raw research
findings and verification records (quotes, arithmetic, caveats) are
`data/original_data/borehole_pump_type_research.json`. Reproduce the weighted
aggregation with
`python scripts/Data_Manipulation_Scripts/aggregate_borehole_pump_estimates.py`.

## The structural pattern (why urban and rural diverge)

1. **Rural South Asia + rural SSA (the bulk of the 858M rural users) are
   handpump-dominated** — India Mark II/III, Afridev, suction tubewells —
   because community handpoints were the rural water program technology for
   40 years, and because motorized boreholes there typically feed small piped
   schemes whose users *report piped taps* and therefore sit outside the
   Borehole category entirely (a JMP-classification effect that pushes the
   remaining borehole category toward handpumps).
2. **Urban borehole users are mostly on electric pumps** — private electric
   borewells in Indian/Nigerian/Indonesian cities, submersibles in urban
   Bangladesh — with handpump-majority urban populations only in the poorest
   urban systems (Uganda, Malawi, Sahel towns, Chad).
3. **High-income + Europe/CIS borehole users are ~0–15% handpump** (US/Nordic
   private wells effectively 100% powered; Bosnia's census measures 11:1
   electric:manual), but they are a rounding error in the global weights
   (<4%). Rich-country boreholes should be assumed to fail with the grid, as
   the source-breakdown README already warns.

## Caveats and follow-ups

- **India dominates everything** (45% of global). Its post-2018 split is
  extrapolated (NSS-MIS 2020-21 and CAMS 2022-23 publish only the combined
  handpump+tubewell category). If Jal Jeevan Mission converts handpump belts
  (UP/Bihar) faster or slower than assumed, the rural range shifts. The next
  NSS drinking-water round, or the delayed census, resolves this.
- Pump *counts* ≠ user shares (motorized points serve more people each). Used
  only as bounds, flagged `inventory` in the CSV; this is the main systematic
  uncertainty in BGD/SSA rural numbers.
- Urban SSA bounds are wide (evidence is thin outside Nigeria); they carry
  little global weight but matter for country-level results.
- Not covered individually: Egypt/Northern Africa (~0.4% of global weight,
  regional default `europe-cis-row` is probably too electric for the rural
  Nile Delta hand-pump tradition — flag if North Africa results are ever
  quoted), Oceania tail, Gulf states.
- The model applies one global scalar per stratum. The weighted range is
  correct for global totals, but a single scalar misallocates *where* the
  displaced people are (urban Nigeria loses ~80% of borehole users, urban
  Uganda ~30%), which matters for the spatial distance-to-water results. The
  per-country CSV supports a future per-country
  `borehole_lose_fraction` column in `merged_data.csv` if that precision is
  wanted.
- Windmill/animal-drawn pumps (Sint Maarten's census G-12 windmill category,
  NSS's animal/wind tubewells) are motorized-classified here but are actually
  HEMP-resilient; negligible populations, conservative direction.
