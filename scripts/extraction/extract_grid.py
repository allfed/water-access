#!/usr/bin/env python
"""Scripted replacement for the undocumented manual QGIS extraction.

Builds the per-cell master table that the water-access model consumes, on
either master lattice , straight from the public source rasters in
``data/GIS/source_rasters/``.  Output is Parquet with the *verbatim* legacy
column contract , optionally plus a legacy CSV.

    # 5 arc-minute (validation grid; ~2.2M rows, ~1 min, <3 GB RSS)
    .venv/bin/python scripts/extraction/extract_grid.py \
        --resolution 5min --out data/GIS/extracted/master_5min.parquet

    # 30 arc-second (production grid; ~140M rows)
    .venv/bin/python scripts/extraction/extract_grid.py \
        --resolution 30as --out data/GIS/extracted/master_30as.parquet

Memory model: the grid is processed in full-width row blocks and each block is
flushed to its own Parquet row group, so peak RSS is a function of
``--block-rows``, never of the grid size.  The full 43200x21600 stack is never
materialised.

Row selection (reproduces the legacy pipeline exactly):
  1. drop cells whose centre falls in no geoBoundaries CGAZ ADM1 polygon
     (legacy ``dropna(subset=["shapeGroup"])``);
  2. drop cells with a nodata value in any of dtw_1 / pop_density / GHS_SMOD /
     slope_1 (legacy ``drop_nan_rows``).  GRIP nodata is *not* a drop reason --
     the legacy CSV keeps 7,675 such rows.
Consequently the ``*_nan`` indicator columns are all-zero in the output, which
is exactly what the legacy CSV contains.  ``--keep-nan-rows`` disables step 2
for diagnostics and then the indicators become informative.
"""

from __future__ import annotations

import argparse
import os
import resource
import sys
import time

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from grid import (  # noqa: E402
    RASTER_DIR,
    REPO_ROOT,
    iter_row_blocks,
    load_master_grid,
)
from layers import (  # noqa: E402
    GRIP_COLUMNS,
    GRIP_TP_TO_COLUMN,
    RASTER_NAN_COLUMNS,
    open_readers,
)

#: The legacy column contract, in order .
OUTPUT_COLUMNS = [
    "fid",
    "left",
    "top",
    "right",
    "bottom",
    "shapeName",
    "shapeID",
    "ISOCODE",
    "GHS_SMOD",
    "pop_density",
    "dtw_1",
    "grip_1_1",
    "grip_2_1",
    "grip_3_1",
    "grip_4_1",
    "grip_5_1",
    "slope_1",
    "dtw_1_nan",
    "pop_density_nan",
    "GHS_SMOD_nan",
    "slope_1_nan",
    "any_nan",
]

DEFAULT_ADM1 = os.path.join(RASTER_DIR, "geoBoundariesCGAZ_ADM1.gpkg")


def peak_rss_gb() -> float:
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # Linux reports KiB, macOS bytes.
    if sys.platform != "darwin":
        rss *= 1024
    return rss / 1024**3


def build_schema(dictionaries: dict) -> pa.Schema:
    fields = [
        pa.field("fid", pa.int64()),
        pa.field("left", pa.float64()),
        pa.field("top", pa.float64()),
        pa.field("right", pa.float64()),
        pa.field("bottom", pa.float64()),
        pa.field("shapeName", pa.dictionary(pa.int32(), pa.string())),
        pa.field("shapeID", pa.dictionary(pa.int32(), pa.string())),
        pa.field("ISOCODE", pa.dictionary(pa.int32(), pa.string())),
    ]
    for name in ["GHS_SMOD", "pop_density", "dtw_1"] + GRIP_COLUMNS + ["slope_1"]:
        fields.append(pa.field(name, pa.float32()))
    for name in [
        "dtw_1_nan",
        "pop_density_nan",
        "GHS_SMOD_nan",
        "slope_1_nan",
        "any_nan",
    ]:
        fields.append(pa.field(name, pa.int8()))
    schema = pa.schema(fields)
    assert [f.name for f in schema] == OUTPUT_COLUMNS, "column contract drift"
    del dictionaries
    return schema


def _dict_array(codes: np.ndarray, categories: np.ndarray) -> pa.Array:
    return pa.DictionaryArray.from_arrays(
        pa.array(codes.astype("int32")), pa.array(categories.tolist(), pa.string())
    )


def extract(args) -> dict:
    t_start = time.time()
    grid = load_master_grid(args.resolution)
    print(
        f"master grid [{grid.resolution}] {grid.width}x{grid.height} "
        f"cell={grid.cell_size:.11f} origin=({grid.transform.c:.9f}, "
        f"{grid.transform.f:.9f}) crs={grid.crs}",
        flush=True,
    )

    readers = open_readers(grid, args)
    admin = readers["_admin"]
    grip = readers["_grip"]
    print(
        f"sources: slope={os.path.basename(readers['slope_1'].path)} "
        f"pop={os.path.basename(readers['pop_density'].path)} "
        f"(resampling={readers['pop_density'].resampling.name}) "
        f"adm1={os.path.basename(args.adm1)} ({len(admin.gdf)} polygons)",
        flush=True,
    )

    name_cats, name_codes = np.unique(admin.shape_name, return_inverse=True)
    id_cats, id_codes = np.unique(admin.shape_id, return_inverse=True)
    iso_cats, iso_codes = np.unique(admin.iso, return_inverse=True)

    # File-level provenance. `cell_size_degrees` in particular lets a consumer
    # apply the resolution-aware population filter resolution-aware population filter without reading the
    # left/right columns at all -- those four float64 coordinate columns are
    # ~37% of the in-memory footprint at 30 arc-second.
    schema = build_schema({}).with_metadata(
        {
            k.encode(): str(v).encode()
            for k, v in {
                "pipeline": "scripts/extraction/extract_grid.py",
                "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "resolution": grid.resolution,
                "grid_width": grid.width,
                "grid_height": grid.height,
                "cell_size_degrees": repr(grid.cell_size),
                "origin_x": repr(grid.transform.c),
                "origin_y": repr(grid.transform.f),
                "crs": grid.crs,
                "fid_encoding": "col * grid_height + row + 1 (column-major, 1-based)",
                "source_dtw": os.path.basename(readers["dtw_1"].path),
                "source_pop": os.path.basename(readers["pop_density"].path),
                "source_pop_resampling": readers["pop_density"].resampling.name,
                "source_slope": os.path.basename(readers["slope_1"].path),
                "source_smod": os.path.basename(readers["GHS_SMOD"].path),
                "source_adm1": os.path.basename(args.adm1),
                "source_roads": "GRIP4 tp1-tp5 density, 5 arcmin, broadcast",
            }.items()
        }
    )
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    writer = pq.ParquetWriter(args.out, schema, compression=args.compression)

    stats = {
        "cells_total": 0,
        "cells_no_admin": 0,
        "cells_dropped_nan": 0,
        "rows_written": 0,
        "blocks": 0,
        "nan_by_layer": {c: 0 for c in RASTER_NAN_COLUMNS},
    }
    csv_chunks = []

    blocks = list(iter_row_blocks(grid, args.block_rows))
    if args.limit_blocks:
        blocks = blocks[: args.limit_blocks]
        print(f"NOTE: --limit-blocks {args.limit_blocks} -> partial extraction")

    for bi, (row_off, n_rows) in enumerate(blocks):
        t0 = time.time()
        adm = admin.read_block(grid, row_off, n_rows)
        stats["cells_total"] += adm.size
        has_admin = adm > 0
        n_admin = int(has_admin.sum())
        stats["cells_no_admin"] += adm.size - n_admin
        if n_admin == 0:
            stats["blocks"] += 1
            continue

        values = {
            name: readers[name].read_block(grid, row_off, n_rows)
            for name in ("dtw_1", "pop_density", "GHS_SMOD", "slope_1")
        }
        for tp, col in GRIP_TP_TO_COLUMN.items():
            values[col] = grip.read_block(grid, row_off, n_rows, tp)

        keep = has_admin
        nan_masks = {}
        for col in RASTER_NAN_COLUMNS:
            m = ~np.isfinite(values[col])
            nan_masks[col] = m
            stats["nan_by_layer"][col] += int((m & has_admin).sum())
        if not args.keep_nan_rows:
            bad = np.zeros_like(keep)
            for col in RASTER_NAN_COLUMNS:
                bad |= nan_masks[col]
            keep = has_admin & ~bad
            stats["cells_dropped_nan"] += n_admin - int(keep.sum())

        rr, cc = np.nonzero(keep)
        if rr.size == 0:
            stats["blocks"] += 1
            continue
        poly = adm[rr, cc] - 1
        master_rows = rr.astype(np.int64) + row_off
        left, top, right, bottom = grid.cell_bounds(master_rows, cc.astype(np.int64))

        arrays = [
            pa.array(grid.fid(master_rows, cc.astype(np.int64))),
            pa.array(left),
            pa.array(top),
            pa.array(right),
            pa.array(bottom),
            _dict_array(name_codes[poly], name_cats),
            _dict_array(id_codes[poly], id_cats),
            _dict_array(iso_codes[poly], iso_cats),
        ]
        for name in ["GHS_SMOD", "pop_density", "dtw_1"] + GRIP_COLUMNS + ["slope_1"]:
            arrays.append(pa.array(values[name][rr, cc].astype("float32")))
        any_nan = np.zeros(rr.size, dtype="int8")
        for col in RASTER_NAN_COLUMNS:
            flag = nan_masks[col][rr, cc].astype("int8")
            arrays.append(pa.array(flag))
            any_nan |= flag
        arrays.append(pa.array(any_nan))

        table = pa.Table.from_arrays(arrays, schema=schema)
        # Bounded row groups keep the file streamable: Phase C / D can read it
        # a row group at a time instead of materialising ~300M rows at once.
        writer.write_table(
            table, row_group_size=max(min(rr.size, args.row_group_rows), 1)
        )
        stats["rows_written"] += rr.size
        stats["blocks"] += 1
        if args.format == "csv":
            csv_chunks.append(table.to_pandas())

        del values, nan_masks, adm, keep, has_admin, rr, cc, arrays, table
        print(
            f"  block {bi + 1}/{len(blocks)} rows {row_off}-{row_off + n_rows} "
            f"-> {stats['rows_written']:,} rows total "
            f"({time.time() - t0:.1f}s, peak RSS {peak_rss_gb():.2f} GB)",
            flush=True,
        )

    writer.close()
    for r in readers.values():
        r.close()

    if args.format == "csv":
        import pandas as pd

        csv_path = os.path.splitext(args.out)[0] + ".csv"
        pd.concat(csv_chunks, ignore_index=True)[OUTPUT_COLUMNS].to_csv(
            csv_path, index=False
        )
        print(f"legacy CSV -> {csv_path}", flush=True)

    stats["seconds"] = time.time() - t_start
    stats["peak_rss_gb"] = peak_rss_gb()
    stats["out_bytes"] = os.path.getsize(args.out)
    print(
        "\n".join(
            [
                "",
                f"resolution        : {args.resolution}",
                f"cells scanned     : {stats['cells_total']:,}",
                f"cells w/o ADM1    : {stats['cells_no_admin']:,}",
                f"cells dropped nan : {stats['cells_dropped_nan']:,}",
                "  nan per layer   : "
                + ", ".join(f"{k}={v:,}" for k, v in stats["nan_by_layer"].items()),
                f"rows written      : {stats['rows_written']:,}",
                f"output            : {args.out} "
                f"({stats['out_bytes'] / 1024**2:.1f} MB)",
                f"wall clock        : {stats['seconds']:.1f} s",
                f"peak RSS          : {stats['peak_rss_gb']:.2f} GB",
            ]
        ),
        flush=True,
    )
    return stats


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--resolution", choices=("5min", "30as"), required=True)
    p.add_argument("--out", required=True, help="output .parquet path")
    p.add_argument(
        "--format",
        choices=("parquet", "csv"),
        default="parquet",
        help="'csv' additionally writes the legacy CSV next to the parquet",
    )
    p.add_argument(
        "--block-rows",
        type=int,
        default=None,
        help="master rows per block (default 540 at 5min, 480 at 30as)",
    )
    p.add_argument(
        "--pop-agg",
        choices=("average", "nearest", "bilinear"),
        default="average",
        help="resampling for GPW density when the master grid is coarser than "
        "30 arc-second (ignored at 30as, where the grids coincide)",
    )
    p.add_argument(
        "--pop-raster",
        default=None,
        help="override the GPW density GeoTIFF (default: the UN WPP-adjusted "
        "product; pass data/GIS/source_rasters/gpw_v4_density_2020_30sec.tif "
        "for the unadjusted fallback)",
    )
    p.add_argument("--slope-raster", default=None, help="override the slope GeoTIFF")
    p.add_argument("--smod-raster", default=None, help="override the GHS-SMOD GeoTIFF")
    p.add_argument("--adm1", default=DEFAULT_ADM1)
    p.add_argument("--compression", default="snappy")
    p.add_argument(
        "--row-group-rows",
        type=int,
        default=2_000_000,
        help="max rows per Parquet row group (streaming granularity)",
    )
    p.add_argument(
        "--keep-nan-rows",
        action="store_true",
        help="keep cells with nodata in dtw/pop/SMOD/slope (diagnostics)",
    )
    p.add_argument(
        "--limit-blocks",
        type=int,
        default=0,
        help="stop after N row blocks (smoke test of the 30as path)",
    )
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.block_rows is None:
        args.block_rows = 540 if args.resolution == "5min" else 480
    # Resolve user paths against the *invocation* cwd before pinning the
    # process to the repo root, so relative --out works from anywhere.
    args.out = os.path.abspath(args.out)
    if args.pop_raster:
        args.pop_raster = os.path.abspath(args.pop_raster)
    if args.slope_raster:
        args.slope_raster = os.path.abspath(args.slope_raster)
    if args.smod_raster:
        args.smod_raster = os.path.abspath(args.smod_raster)
    args.adm1 = os.path.abspath(args.adm1)
    os.chdir(REPO_ROOT)
    extract(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
