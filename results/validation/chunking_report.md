# Chunked vs monolithic execution (ADR-5)

- Input: `/Users/kevin/Documents/ProgrammingIsFun/ALLFED/water-access/data/GIS/updated_GIS_output_cleaned.csv`
- Area-weighted population (cos-latitude): ON (default)
- Chunk budget: 200,000 modelled cells
- Water-source parameters: urban_borehole_lose_fraction=0.72, rural_borehole_lose_fraction=0.29, unallocated_vulnerable_fraction='prorata' (deliberately off the code defaults)
- Plan: 8 chunks over 187 countries, 1,395,693 modelled cells, largest chunk 200,000 rows (largest single country 216,634 = RUS, split by district: RUS), cell 300 arcsec, cos-lat area weighted
- Global LUT ranges pinned to slope 0.000000-50.519913 deg, mass 47.472-91.045 kg
- Monolithic run: 41.6 s; chunked run: 51.0 s
- Peak RSS across both runs in one process: 2.61 GB (monolithic alone reached 2.61 GB)

Both runs use the AC-2 fixed parameters. The bar is **identity**, not a tolerance: chunking changes only the order of execution, so every value must match bit for bit.

Two kinds of row cannot be reproduced bit for bit, and in both cases the reason is arithmetic rather than structural: their population columns are sums over cells that chunking splits up, and floating-point addition is not associative, so adding the parts lands within an ULP of adding the whole but not exactly on it. Those rows are (a) the `GLOBAL` summary, which sums every cell on Earth, and (b) any country so large it had to be spread over several chunks. Both are held to a relative tolerance instead. **Every other country row, and every district row without exception, must be bit-identical** -- including the districts of a split country, because splitting is done on district boundaries precisely so that no district is ever divided.

## Checks

| check                                                          | observed                                     | pass |
| -------------------------------------------------------------- | -------------------------------------------- | ---- |
| countries: shape                                               | (181, 25) vs (181, 25)                       | PASS |
| countries: column order                                        | identical                                    | PASS |
| countries: ISOCODE order                                       | identical                                    | PASS |
| countries: every value identical (excluding GLOBAL, RUS)       | yes                                          | PASS |
| countries: largest absolute difference (excluding GLOBAL, RUS) | 0.000e+00 (-)                                | PASS |
| countries: GLOBAL/RUS row(s) within 1e-12 relative             | 1.837e-16 (percent_with_only_cycling_access) | PASS |
| districts: shape                                               | (2895, 21) vs (2895, 21)                     | PASS |
| districts: column order                                        | identical                                    | PASS |
| districts: shapeID order                                       | identical                                    | PASS |
| districts: every value identical                               | yes                                          | PASS |
| districts: largest absolute difference                         | 0.000e+00 (-)                                | PASS |
| chunked per-cell frame is empty                                | 0 rows                                       | PASS |

## Chunk plan

| chunk | selection           | modelled_cells |
| ----- | ------------------- | -------------- |
| 1     | 5 whole countries   | 200000         |
| 2     | 6 whole countries   | 200000         |
| 3     | 10 whole countries  | 199998         |
| 4     | 16 whole countries  | 199998         |
| 5     | 28 whole countries  | 200000         |
| 6     | 121 whole countries | 179063         |
| 7     | RUS (50 districts)  | 199999         |
| 8     | RUS (33 districts)  | 16635          |

## Verdict

**PASS**
