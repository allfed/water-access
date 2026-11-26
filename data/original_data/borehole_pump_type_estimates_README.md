# borehole_pump_type_estimates.csv — hand-pumped vs motorised boreholes

Per-country estimates of the share of borehole users who rely on a hand pump
rather than a motorised (electric or diesel) pump. They set the Monte Carlo
ranges for `urban_borehole_lose_fraction` and `rural_borehole_lose_fraction`
(`src/monte_carlo_parameters.py`).

## Model parameters

The model assumes all motorised pumping fails (grid, diesel and solar), so
hand-pump users keep their source and motorised-borehole users are displaced:

`borehole_lose_fraction = 1 − (hand-pump share among borehole users)`

| Parameter | Range | Central |
|---|---|---|
| `urban_borehole_lose_fraction` | 0.63 – 0.80 | ~0.72 |
| `rural_borehole_lose_fraction` | 0.23 – 0.35 | ~0.29 |

Both are sampled uniformly. The bounds are the population-weighted envelope of
the per-country low and high bounds, i.e. every country at its optimistic
(respectively pessimistic) edge at once. This envelope suits the Monte Carlo
because the main uncertainties are shared across countries (the same
inference method, converting pump stock to user shares, is used in many of
them, so errors are correlated).

Reproduce the weighted ranges with:

```bash
python scripts/Data_Manipulation_Scripts/aggregate_borehole_pump_estimates.py
```

## What "borehole" means

`Borehole` in this model is JMP row 57 "Tubewell, borehole"
(`jmp_source_parser.py`): people whose **main drinking source** is a drilled
tubewell or borehole, hand-pumped or motorised. Every source figure is
converted to hand pump / (hand pump + motorised borehole), both as main
drinking source. A census stating that "43.6% of rural households use
handpumps" is a share of all households and is converted before use.

## Method

Estimates come from primary sources (censuses, DHS/MICS/LSMS/PSLM/NSS
household surveys, national water-point inventories), weighted by where the
borehole population lives:

- About 1.15 billion people have a tubewell or borehole as their main drinking
  source (from `merged_data.csv`: 290 million urban, 858 million rural).
- India (520 million), Bangladesh (138 million) and Pakistan (104 million) make
  up 66% of the global total. Nigeria (81 million) has the largest urban
  borehole population (52 million). Sub-Saharan Africa is 20%, South-East Asia
  8%, everything else under 4%.

46 countries have researched per-country bounds, covering 94.3% of the urban
and 96.7% of the rural borehole population. The remaining countries use
regional default ranges. The highest-weight countries were re-derived
independently from primary tables: India and Pakistan reproduce exactly;
Bangladesh, Indonesia, Uganda, Nigeria and Ukraine are consistent with their
evidence chains.

## Key country results (hand-pump share among borehole users, %)

| Country | Urban | Rural | Evidence |
|---|---|---|---|
| India (45% of global) | 25–38 | 70–80 | Census 2011 H-series and NSS 76th round (2018) measure the split directly (Hand pump = manual, Tubewell/Borehole = electric or diesel pump): rural 84.1% (2011) and 79.7% (2018), urban 57.1% and 39.2%. The 2024 range extends that trend, constrained by NSS-MIS 2020-21 and CAMS 2022-23 combined-category data |
| Bangladesh | 35–65 | 88–98 | "Tubewell" is mostly shallow suction hand pumps: ~16 million private hand tubewells and ~1.8 million public hand-pump points against a small motorised stock. Peri-urban Dhaka is ~97% motorised among borehole users; smaller towns are not |
| Pakistan | 21–26 | 46–53 | PSLM 2019-20 Table 7.1 splits hand pump and motor pump directly: urban 7.32/31.12 = 23.5%, rural 33.44/67.36 = 49.6% |
| Nigeria | 10–25 | 35–60 | WASH-NORM 2021 water-point census: urban hand pump/(hand pump + motorised) points 14.8%, rural 39.3%. Urban Nigeria relies on private electric boreholes |
| Indonesia | 2–10 | 10–35 | BPS "sumur bor/pompa" does not split pump type. Urban bore wells are almost all electric jet pumps; rural bounded by appliance-ownership arithmetic |
| Uganda | 50–85 | 88–99 | Ministry of Water and Environment water atlas: ~41,000 hand-pumped deep boreholes are the only point-source fleet serving ~16 million rural borehole users; motorised boreholes feed piped schemes, whose users report taps |
| Ukraine | 0–5 | 5–20 | No pump-type split since MICS 2012; bracketed on the Bosnia census analogue (11:1 electric to manual), adjusted for documented street hand-pump columns |

The full table (all countries, bounds, evidence kind, confidence, source, URL)
is `borehole_pump_type_estimates.csv`.

## Why urban and rural differ

1. Rural South Asia and rural sub-Saharan Africa (most of the 858 million rural
   users) are hand-pump dominated, because community hand pumps have been the
   rural water technology for decades. Motorised boreholes there usually feed
   small piped schemes whose users report piped taps and so sit outside the
   Borehole category.
2. Urban borehole users are mostly on electric pumps: private borewells in
   Indian, Nigerian and Indonesian cities, submersibles in urban Bangladesh.
   Hand-pump majorities in towns occur only in the poorest urban systems
   (Uganda, Malawi, Sahel towns, Chad).
3. High-income, European and CIS borehole users are ~0–15% hand pump (private
   wells in the US and Nordics are effectively all powered), but they are under
   4% of the global weight.

## Caveats

- India dominates (45% of global). Its post-2018 split is extrapolated, because
  NSS-MIS 2020-21 and CAMS 2022-23 only publish the combined hand pump plus
  tubewell category.
- Pump counts are not user shares (motorised points serve more people each).
  Counts are used only as bounds, flagged `inventory` in the CSV.
- Urban sub-Saharan bounds are wide outside Nigeria. They carry little global
  weight but matter for country-level results.
- Egypt and Northern Africa (~0.4% of global weight) use the
  `europe-cis-row` regional default, which is probably too electric for the
  rural Nile Delta.
- The model applies one global value per stratum. The weighted range is right
  for global totals but does not capture where displaced people are (urban
  Nigeria loses ~80% of borehole users, urban Uganda ~30%).
- Windmill and animal-drawn pumps are counted as motorised although they would
  keep working; their populations are negligible.
