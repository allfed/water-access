# cos-latitude cell-area weighting of the population split: OFF vs ON (5 arcmin)

> **Status: ADOPTED AS DEFAULT, 2026-08-09 (Kevin sign-off).** The
> recommendation below was written as "adopt, with the switch, default OFF";
> Kevin overrode it the same day on the grounds that leaving a genuine bug
> off-by-default is a trap for the next reader. `area_weighted_population` now
> defaults to **True** everywhere and the regression baseline has been
> re-frozen against the corrected model. The legacy density-only split stays
> fully reachable (`--no-area-weighted`, `--no-area-weighted-population`,
> `area_weighted_population=False`) and reproduces every pre-correction
> published figure; the pre-correction baseline stays in git history at
> `640dceb`. Re-frozen baseline (legacy 5-arcmin CSV, the input the baseline
> has always used): global percent_with_water **77.7075 → 77.6298 (−0.0777 pp)**,
> country pop-weighted mean |Δ| **0.110 pp**, max **NOR +1.328 pp**, districts
> >100k max **Alaska 1.194 pp** — consistent with the parquet-input numbers
> below. (Those figures diff the new baseline against the *pre-refactor* one,
> so they carry the pre-existing refactor drift too; isolating the correction by
> running the current code both ways on the same input gives −0.077676 pp
> against −0.077654 pp, i.e. the drift contributes ~2e−5 pp. `--no-area-weighted`
> is byte-identical to the pre-flip run of this code, and still clears Gate 2's
> tolerances against the pre-refactor baseline at max 0.0015 pp country /
> 0.049 pp district.)

**Date:** 2026-08-09 · **Params:** AC-2 fixed set (crr 0, time_gathering 2.84, practical limits 40/20, met 4, watts 40, downhill_uphill, urban 1.3 / rural 1.4, calculate_distance, no plot) · **Input:** `data/GIS/extracted/master_5min.parquet` · **Only variable:** `area_weighted_population` False → True.

## The defect

`merge_and_adjust_population` splits each country's UN population across its grid
cells in proportion to `pop_density` alone:

    pop_density_perc = pop_density / Σ pop_density   (per ISOCODE)
    pop_zone         = pop_density_perc × Population

On a lat/long grid a cell is a constant number of degrees on a side, so its
ground area falls as `cos(latitude)` — a 5-arcmin cell at 60° N covers half the
ground of one on the equator. People per cell is `density × area`, so weighting
by density alone systematically over-weights a country's **poleward** cells
relative to its equatorward ones. Country totals are untouched (the weights are
renormalised nationally); only the *placement inside* a country moves — and
placement is what the distance-to-water model consumes.

The fix weights by `pop_density × cos(cell-centre latitude)`, behind the new
trailing kwarg `area_weighted_population` (**default True** as of 2026-08-09; it
landed default-False and was flipped the same day — see the status note above.
Pass False to reproduce any pre-correction result).

## Results

| Level | Metric | Value |
|---|---|---|
| Global | percent_with_water | 77.7516 → 77.6682 (**−0.0833 pp**) |
| Global | percent_piped_with_cycling_access | 24.4821 → 24.4254 (−0.0567 pp) |
| Global | percent_piped_with_walking_access | 34.8892 → 34.8258 (−0.0634 pp) |
| Global | weighted median distance to water | 4.42719 → 4.42719 (unchanged) |
| Global | modelled population | 7.849190452e9 → 7.849190568e9 (**+1.5e−6 %**) |
| Country (≥1M pop, n=156) | pop-weighted mean \|Δ percent_with_water\| | **0.113 pp** |
| Country (≥1M pop) | median / p90 / max \|Δ\| | 0.016 / 0.213 / **1.324 pp** (NOR) |
| Country (≥1M pop) | how many move > 0.1 / 0.5 / 1.0 pp | 29 / 1 / 1 |
| Country (≥1M pop) | direction | 69 up, 85 down, 2 unchanged |
| Country | modelled population per country | preserved to 2.2e−5 % (renormalisation) |
| District (>100k pop, n=2498) | median / p90 / max \|Δ\| | 0.004 / 0.036 / **1.051 pp** (Alaska) |
| District (>100k) | how many move > 0.25 / 1.0 / 5.0 pp | 13 / 1 / 0 |
| Row sets | 180 countries, 2896 districts | identical OFF and ON |

### Top 15 country movers (≥1M pop) with latitude span

| ISO | Country | Pop (M) | OFF | ON | Δ pp | lat range | span ° |
|---|---|---|---|---|---|---|---|
| NOR | Norway | 5.5 | 75.331 | 76.655 | **+1.324** | 58.0 … 71.0 | 13.0 |
| ESP | Spain | 46.7 | 62.761 | 62.292 | −0.469 | 27.7 … 43.7 | 16.0 |
| CHL | Chile | 19.2 | 54.808 | 54.353 | −0.455 | −55.9 … −17.6 | 38.2 |
| AUS | Australia | 25.8 | 82.832 | 82.405 | −0.427 | −43.6 … −10.1 | 33.5 |
| NZL | New Zealand | 4.9 | 83.469 | 83.854 | +0.384 | −47.0 … −34.5 | 12.5 |
| PRT | Portugal | 10.2 | 50.822 | 50.457 | −0.365 | 32.7 … 42.1 | 9.4 |
| ITA | Italy | 60.4 | 70.554 | 70.197 | −0.357 | 36.7 … 47.0 | 10.3 |
| GRC | Greece | 10.4 | 25.921 | 25.629 | −0.293 | 35.0 … 41.7 | 6.7 |
| HRV | Croatia | 4.1 | 62.187 | 61.911 | −0.276 | 42.5 … 46.5 | 4.0 |
| CHN | China | 1444.2 | 78.064 | 77.805 | −0.259 | 18.3 … 53.5 | 35.2 |
| URY | Uruguay | 3.5 | 50.143 | 50.379 | +0.236 | −34.9 … −30.1 | 4.7 |
| RUS | Russian Federation | 145.9 | 72.005 | 71.777 | −0.228 | 41.2 … 81.8 | 40.6 |
| IRQ | Iraq | 41.2 | 71.465 | 71.684 | +0.219 | 29.1 … 37.4 | 8.2 |
| BRA | Brazil | 214.0 | 58.417 | 58.636 | +0.219 | −33.7 … 5.2 | 38.9 |
| IRN | Iran | 85.0 | 32.907 | 32.693 | −0.214 | 25.1 … 39.7 | 14.6 |

**Does the mechanism's prediction hold?** Partly, and the discrepancy is
instructive. Latitude span alone is a weak predictor of |Δ| (Spearman 0.33 over
the 156 countries) — the top-15 movers do skew wide (median span 13.0° vs 5.6°
for the rest, 10 of 15 over 10°), but wide *equatorial* countries barely move
(Brazil, span 38.9°, +0.22 pp) while narrow *high-latitude* ones move a lot
(Croatia, span 4.0°, −0.28 pp). That is exactly what the mechanism predicts once
you differentiate it: `d(cos φ)/dφ = −sin φ`, so what matters is not the span in
degrees but the **spread of `cos φ` across the country's populated cells**. The
population-density-weighted coefficient of variation of `cos φ` within a country
correlates with |Δ| at **Spearman 0.69 / Pearson 0.61** — against 0.33 / 0.30
for raw span. Mechanism confirmed; "high-latitude-span countries move most" is
the right intuition stated slightly wrong.

### Worst districts (>100k pop)

Alaska (USA) 87.593 → 86.542 (1.051 pp); Santa Cruz (ARG) 0.555; Nevada (USA)
0.518; Troms og Finnmark (NOR) 0.477; Tasmania (AUS) 0.428; Queensland (AUS)
0.368; Arkhangelsk Oblast (RUS) 0.367; Magadan Oblast (RUS) 0.338; Western
Australia (AUS) 0.336; Sakha Republic (RUS) 0.279. All high-latitude or
high-latitude-span; none exceeds 1.1 pp.

## Does it get worse at 30 arc-seconds?

No — measured, not assumed. Five countries (the four biggest 5-arcmin movers
plus LBY, which the country outlier cut is defined against) were carved out of
`data/GIS/extracted/master_30as.parquet` (5.18M cells) and run OFF and ON at the
same fixed parameters:

| ISO | Δ pp at 5 arcmin | Δ pp at 30 arcsec |
|---|---|---|
| NOR | +1.324 | +1.302 |
| ESP | −0.469 | −0.419 |
| CHL | −0.455 | −0.440 |
| NZL | +0.384 | +0.313 |
| LBY | +0.029 | +0.044 |

The effect is a property of the grid's geometry, not of its resolution, and it
transfers essentially unchanged. **The adopt/don't decision can be taken at
5 arcmin and applied to the production 30as run**; no expensive whole-world 30as
A/B is needed to size it. (Caveat: a 5-country subset, so the *global* row was
not re-measured at 30as.)

## Chunked ≡ monolithic with the flag ON

The chunk planner's pre-pass computes each country's normalisation divisor from
a 4-column read, so it has to reproduce the run's weight bit for bit; with the
flag on it now reads `top`/`bottom` too and calls the same
`population_split_weights` helper. `scripts/validate_chunking.py --area-weighted`
holds this to the same bar as the default path — every country and district row
bit-identical, `GLOBAL` and split countries within 1e−12 relative:

| Run | Verdict | Largest abs. difference (all rows except GLOBAL) | GLOBAL row, relative |
|---|---|---|---|
| OFF, legacy CSV, 350k-row chunks | PASS | 0.000e+00 | 0.000e+00 |
| ON, legacy CSV, 350k-row chunks | PASS | 0.000e+00 | 2.043e−16 |
| ON, `master_5min.parquet`, 350k-row chunks | PASS | 0.000e+00 | 4.161e−16 |

The Parquet run matters separately: the chunk reader drops the four coordinate
columns for memory, so it is the case where a missing `top`/`bottom` would have
silently changed the answer. It does not.

## Reading

**The effect is real, small in aggregate, and concentrated exactly where the
geometry says it should be.** Globally it is −0.083 pp on percent_with_water —
the same order as the documented legacy-vs-scripted input swap (+0.044 pp), and
far below anything that would move a headline conclusion. At country level the
population-weighted mean error is 0.113 pp, roughly a quarter of the input-swap
figure (0.436 pp), with a short tail: one country over 1 pp (Norway, +1.32), 29
of 156 over 0.1 pp, and those 29 are predominantly mid- and high-latitude (NOR,
RUS, CHL, AUS, NZL, ESP, ITA, GRC, POL, KAZ, USA, ZAF…) plus a handful of
wide-span subtropical ones (BRA, MOZ, NAM, DZA). Tropical
countries — where most of the world's water-access burden sits — barely move,
which is why the global number stays small.

**Recommendation: adopt, with the switch, and not retroactively.** The
area-weighted split is unambiguously the more correct model of a lat/long grid,
it is now proven chunk-safe and resolution-stable, and it costs one extra
multiply plus two columns in the chunked read. But it is not worth invalidating
the frozen baseline or any committed 30as evidence over 0.08 pp: keep the
default OFF so every existing artefact stays valid, and turn it on deliberately
for the next production run (`--area-weighted-population` on
`scripts/run_monte_carlo.py`, `--area-weighted` on `scripts/capture_baseline.py`),
re-freezing the baseline in the same commit. Until then, treat per-country
figures for the high-latitude tail — Norway above all — as carrying a ~1 pp
known bias, and note that switching it on would breach Gate 2's 0.1 pp country
tolerance for 29 countries and the 0.25 pp district tolerance for 13 districts,
by design rather than by regression.

Reproduce:

    # (flags as they are now, i.e. area weighting on by default)
    .venv/bin/python scripts/capture_baseline.py --input data/GIS/extracted/master_5min.parquet --no-area-weighted --out /tmp/coslat_off
    .venv/bin/python scripts/capture_baseline.py --input data/GIS/extracted/master_5min.parquet --out /tmp/coslat_on
    .venv/bin/python scripts/validate_chunking.py --chunk-rows 350000 --report /tmp/chunk_check_aw.md
    .venv/bin/python scripts/validate_chunking.py --chunk-rows 350000 --no-area-weighted --report /tmp/chunk_check.md
