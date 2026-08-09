# Input-swap comparison: legacy QGIS CSV vs scripted extraction (5 arcmin)

**Date:** 2026-08-09 · **Params:** AC-2 fixed set (time_gathering 2.84 h, met 4, watts 40, downhill_uphill, urban 1.3 / rural 1.4) · **Code:** identical (post-refactor, commit 1f8128a) · **Only variable:** input table — `results/regression_baseline/` (legacy `updated_GIS_output_cleaned.csv`) vs `data/GIS/extracted/master_5min.parquet` (scripted pipeline).

This quantifies the verifier's flagged residual: Gate 1 matched population *totals* per country but found within-country redistribution (`pop_share_l1_pct` median 12.3%, max 35.4%) of undocumented QGIS origin. This is the end-to-end effect of that redistribution on model output.

## Results

| Level | Metric | Value |
|---|---|---|
| Global | percent_with_water | 77.7075 → 77.7516 (**+0.044 pp**) |
| Global | modelled population | identical (−0.000%) |
| Country (≥1M pop) | pop-weighted mean \|Δ percent_with_water\| | **0.436 pp** |
| Country (≥1M pop) | median / p90 / max \|Δ\| | 0.557 / 3.094 / **19.354 pp** (DJI) |
| Country | modelled population per country | identical to machine precision (renormalisation) |
| District (>100k, n=2483) | median / p90 / p99 / max \|Δ\| | 0.669 / 4.715 / 12.599 / **28.611 pp** |
| Row sets | countries 180=180; districts 2894 matched | 1 legacy-only, 2 scripted-only |

Worst countries (≥1M pop): DJI +19.35, LBN +8.51, GAB +6.68, ARM +5.21, LVA −5.19, SGP −5.02, JOR +4.80, GNB −4.61, TWN +4.40, CRI +3.95 pp.

## Reading

- **Headline/global numbers are robust to the pipeline swap** (+0.04 pp) — the manuscript's top-line conclusions do not depend on which extraction produced the table.
- **Population-weighted country error is small** (0.44 pp mean) but the **tail is real**: small, coastal, or urban-concentrated countries (Djibouti, Lebanon, Singapore) move by 5–19 pp because the undocumented QGIS resampling placed population differently within the country than the scripted, total-conserving `average` resampling does.
- **District-level tables/maps should be treated as pipeline-sensitive** beyond ~p90: 1 in 10 large districts moves >4.7 pp.
- No gate bounds this quantity by design (ADR-9): the legacy placement is unreproducible and not demonstrably better; the scripted placement is documented and conservative. This report exists so the shift is *known*, not hidden.

Reproduce: `.venv/bin/python scripts/capture_baseline.py --input data/GIS/extracted/master_5min.parquet --out /tmp/inputswap` then diff against `results/regression_baseline/`.
