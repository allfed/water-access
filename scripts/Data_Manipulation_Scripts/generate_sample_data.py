"""
Builds the small zone sample used by the tests and by quick test runs
(``use_sample_data=True``): up to 20 populated 30 arc-second cells per country,
drawn at random from the master grid.

    python scripts/Data_Manipulation_Scripts/generate_sample_data.py
"""

from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

repo_root = Path(__file__).resolve().parents[2]
input_file = repo_root / "data" / "GIS" / "extracted" / "master_30as.parquet"
output_file = repo_root / "data" / "GIS" / "GIS_data_zones_sample.csv"

sample_size_per_country = 20

# Pick the rows from two columns only; the full grid does not fit in memory.
selection_columns = pd.read_parquet(input_file, columns=["ISOCODE", "pop_density"])
populated = selection_columns[selection_columns["pop_density"] > 0]
picked_rows = set(
    populated.sample(frac=1, random_state=0)
    .groupby("ISOCODE", observed=True)
    .head(sample_size_per_country)
    .index
)
del selection_columns, populated

# Read the grid one row group at a time and keep the picked rows.
grid = pq.ParquetFile(input_file)
sample_parts = []
first_row = 0
for group in range(grid.num_row_groups):
    row_group = grid.read_row_group(group).to_pandas()
    row_group.index = range(first_row, first_row + len(row_group))
    first_row += len(row_group)
    sample_parts.append(row_group[row_group.index.isin(picked_rows)])

gis_sample = pd.concat(sample_parts)
for column in ["shapeName", "shapeID", "ISOCODE"]:
    gis_sample[column] = gis_sample[column].astype(str)
gis_sample.to_csv(output_file, index=False)

print(f"Sampled {len(gis_sample)} cells to {output_file}")
