# Chunked vs monolithic execution (ADR-5)

- Input: `/Users/kevin/Documents/ProgrammingIsFun/ALLFED/water-access/.claude/worktrees/30as-feasibility/data/GIS/updated_GIS_output_cleaned.csv`
- Chunk budget: 350,000 modelled cells
- Plan: 5 chunks over 187 countries, 1,403,388 modelled cells, largest chunk 350,000 rows (largest single country 221,479 = RUS), cell 300 arcsec
- Global LUT ranges pinned to slope 0.000000-50.519913 deg, mass 47.472-91.045 kg
- Monolithic run: 118.7 s; chunked run: 227.3 s
- Peak RSS across both runs in one process: 2.22 GB (monolithic alone reached 2.22 GB)

Both runs use the AC-2 fixed parameters. The bar is **identity**, not a tolerance: chunking changes only the order of execution, so every value must match bit for bit.

Two kinds of row cannot be reproduced bit for bit, and in both cases the reason is arithmetic rather than structural: their population columns are sums over cells that chunking splits up, and floating-point addition is not associative, so adding the parts lands within an ULP of adding the whole but not exactly on it. Those rows are (a) the `GLOBAL` summary, which sums every cell on Earth, and (b) any country so large it had to be spread over several chunks. Both are held to a relative tolerance instead. **Every other country row, and every district row without exception, must be bit-identical** -- including the districts of a split country, because splitting is done on district boundaries precisely so that no district is ever divided.

## Checks

| check                                                     | observed                 | pass |
| --------------------------------------------------------- | ------------------------ | ---- |
| countries: shape                                          | (181, 25) vs (181, 25)   | PASS |
| countries: column order                                   | identical                | PASS |
| countries: ISOCODE order                                  | identical                | PASS |
| countries: every value identical (excluding GLOBAL)       | yes                      | PASS |
| countries: largest absolute difference (excluding GLOBAL) | 0.000e+00 (-)            | PASS |
| countries: GLOBAL row(s) within 1e-12 relative            | 0.000e+00 (-)            | PASS |
| districts: shape                                          | (2895, 21) vs (2895, 21) | PASS |
| districts: column order                                   | identical                | PASS |
| districts: shapeID order                                  | identical                | PASS |
| districts: every value identical                          | yes                      | PASS |
| districts: largest absolute difference                    | 0.000e+00 (-)            | PASS |
| chunked per-cell frame is empty                           | 0 rows                   | PASS |

## Chunk plan

| chunk | selection           | modelled_cells |
| ----- | ------------------- | -------------- |
| 1     | 6 whole countries   | 350000         |
| 2     | 11 whole countries  | 349998         |
| 3     | 25 whole countries  | 350000         |
| 4     | 113 whole countries | 350000         |
| 5     | 32 whole countries  | 3390           |

## Verdict

**PASS**
