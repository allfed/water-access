"""Merge a percentile zone Parquet with cell centroids for map rasterisation.

At 30 arc-seconds centroids come from ``master_30as.parquet`` (``left/top/
right/bottom``). The merged table is written as Parquet (CSV is impractical
at this cell count).

Default input is the median zone map; pass ``--label 5th_percentile`` or
``95th_percentile`` for the other draws.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

data_script_dir = Path(__file__).resolve().parent
repo_root = data_script_dir.parent.parent

parquet_data_dir = repo_root / "results" / "parquet_files"
results_dir = repo_root / "results"
master_30as = repo_root / "data" / "GIS" / "extracted" / "master_30as.parquet"
# Legacy 5-arc-minute centroid table (only used if --resolution 5min).
fid_csv_5min = repo_root / "data" / "GIS" / "GIS_country_points_fid.csv"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--label",
        default="median",
        choices=("median", "5th_percentile", "95th_percentile"),
        help="which zone_simulation_result_<label>.parquet to export",
    )
    p.add_argument(
        "--resolution",
        choices=("30as", "5min"),
        default="30as",
        help="grid used for centroids (default 30as)",
    )
    return p.parse_args()


def centroids_from_master(master_path: Path, fids: pd.Series) -> pd.DataFrame:
    """Load cell bounds for the fids present in the zone result only."""
    import pyarrow.parquet as pq

    wanted = pd.DataFrame({"fid": fids.unique()})
    pieces = []
    pf = pq.ParquetFile(master_path)
    for i in range(pf.num_row_groups):
        chunk = pf.read_row_group(
            i, columns=["fid", "left", "top", "right", "bottom"]
        ).to_pandas()
        chunk = chunk.merge(wanted, on="fid", how="inner")
        if not chunk.empty:
            pieces.append(chunk)
    if not pieces:
        raise SystemExit(f"No centroid matches in {master_path}")
    bounds = pd.concat(pieces, ignore_index=True)
    bounds["longitude_centroid"] = (bounds["left"] + bounds["right"]) / 2
    bounds["latitude_centroid"] = (bounds["top"] + bounds["bottom"]) / 2
    return bounds[["fid", "longitude_centroid", "latitude_centroid"]]


def centroids_from_fid_csv(csv_path: Path) -> pd.DataFrame:
    original_df = pd.read_csv(csv_path)
    original_df["longitude_centroid"] = (
        original_df["left"] + original_df["right"]
    ) / 2
    original_df["latitude_centroid"] = (
        original_df["top"] + original_df["bottom"]
    ) / 2
    return original_df[["fid", "longitude_centroid", "latitude_centroid"]]


def main() -> None:
    args = parse_args()
    zone_path = parquet_data_dir / f"zone_simulation_result_{args.label}.parquet"
    if not zone_path.exists():
        raise SystemExit(
            f"Missing {zone_path}. Run parquet_process.py first so the "
            "percentile zone files exist."
        )

    processed_df = pd.read_parquet(zone_path)
    if args.resolution == "30as":
        if not master_30as.exists():
            raise SystemExit(f"Missing {master_30as}")
        print(f"Loading centroids from {master_30as} ...")
        centroids = centroids_from_master(master_30as, processed_df["fid"])
        # Analysis cells only (left join). A full right-join to every master
        # cell is ~220M rows and is not needed for the GeoTIFF burn.
        how = "left"
    else:
        if not fid_csv_5min.exists():
            raise SystemExit(f"Missing {fid_csv_5min}")
        print(f"Loading centroids from {fid_csv_5min} ...")
        centroids = centroids_from_fid_csv(fid_csv_5min)
        how = "right"

    merged_df = processed_df.merge(centroids, on="fid", how=how)
    print(f"Merged shape: {merged_df.shape} (how={how})")
    print(f"NaN counts:\n{merged_df.isna().sum()}")

    results_dir.mkdir(parents=True, exist_ok=True)
    out_parquet = (
        results_dir / f"GIS_merged_output_processed_with_centroids_{args.label}.parquet"
    )
    merged_df.to_parquet(out_parquet, index=False)
    print(f"Wrote {out_parquet}")

    # Zero-filled copy for raster scripts that treat missing land as 0.
    filled = merged_df.fillna(0)
    for col in ("zone_pop_with_water", "zone_pop_without_water"):
        if col in filled.columns:
            filled[col] = filled[col].clip(lower=0)
    out_filled = (
        results_dir
        / f"GIS_merged_output_processed_with_centroids_{args.label}_no_nans.parquet"
    )
    filled.to_parquet(out_filled, index=False)
    print(f"Wrote {out_filled}")


if __name__ == "__main__":
    main()
