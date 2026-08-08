#!/usr/bin/env python
"""Cross-resolution self-checks for the scripted extraction.

Gate 1 (``validate_extraction.py``) proves the 5-arcmin output reproduces the
legacy table. This script proves the *30 arc-second* path is internally sound
without needing a legacy artefact to compare against — the properties checked
here are ones the 30as table must satisfy by construction:

1.  the ``fid`` encoding round-trips to the cell's own ``left``/``top``;
2.  ``right``/``bottom`` describe exactly one master cell;
3.  every 30as cell carries the GRIP road densities of the 5-arcmin cell that
    contains it (the documented 10x10 broadcast — ADR-3's information floor);
4.  ``dtw_1`` equals the Kummu 30as raster value at that cell, bit for bit;
5.  the ``*_nan`` indicator columns agree with the data;
6.  ``ISOCODE``/``shapeID`` are non-null everywhere.

    .venv/bin/python scripts/extraction/check_consistency.py \
        --table data/GIS/extracted/master_30as.parquet --resolution 30as

Exits 0 if every check passes.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from grid import GRID_DEFINING_RASTER, REPO_ROOT, load_master_grid  # noqa: E402
from layers import GRIP_DIR, GRIP_TP_TO_COLUMN  # noqa: E402

GRIP_COLUMNS = ["grip_1_1", "grip_2_1", "grip_3_1", "grip_4_1", "grip_5_1"]


def sample(table: str, n: int, seed: int) -> pd.DataFrame:
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(table)
    rng = np.random.default_rng(seed)
    groups = rng.choice(
        pf.num_row_groups, size=min(pf.num_row_groups, 12), replace=False
    )
    frames = [pf.read_row_group(int(g)).to_pandas() for g in groups]
    df = pd.concat(frames, ignore_index=True)
    if len(df) > n:
        df = df.sample(n, random_state=seed).reset_index(drop=True)
    return df


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--table", required=True)
    p.add_argument("--resolution", choices=("5min", "30as"), required=True)
    p.add_argument("--samples", type=int, default=200_000)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)
    args.table = os.path.abspath(args.table)
    os.chdir(REPO_ROOT)

    grid = load_master_grid(args.resolution)
    df = sample(args.table, args.samples, args.seed)
    print(f"{args.table}: sampled {len(df):,} rows", flush=True)

    cs = grid.cell_size
    col = np.round((df["left"].values - grid.transform.c) / cs).astype(np.int64)
    row = np.round((grid.transform.f - df["top"].values) / cs).astype(np.int64)
    checks = []

    def check(name, ok, detail=""):
        checks.append((name, bool(ok), detail))
        print(f"  [{'PASS' if ok else 'FAIL'}] {name} {detail}", flush=True)

    check(
        "fid == col * nrows + row + 1",
        np.array_equal(df["fid"].values, col * grid.height + row + 1),
    )
    check(
        "cell indices in range",
        bool(
            (row >= 0).all()
            and (row < grid.height).all()
            and (col >= 0).all()
            and (col < grid.width).all()
        ),
    )
    w = df["right"].values - df["left"].values
    h = df["top"].values - df["bottom"].values
    check(
        "right-left == top-bottom == cell size",
        bool(np.allclose(w, cs, atol=1e-12) and np.allclose(h, cs, atol=1e-12)),
        f"(max dev {max(np.abs(w - cs).max(), np.abs(h - cs).max()):.2e})",
    )

    # dtw straight off the Kummu raster
    with rasterio.open(GRID_DEFINING_RASTER[args.resolution]) as src:
        vals = np.empty(len(df), dtype="float32")
        for i, (r, c) in enumerate(zip(row, col)):
            vals[i] = src.read(1, window=Window(int(c), int(r), 1, 1))[0, 0]
            if i > 20000:  # windowed point reads are slow; 20k is plenty
                vals = vals[: i + 1]
                break
    k = len(vals)
    check(
        "dtw_1 == Kummu raster value",
        bool(np.allclose(vals, df["dtw_1"].values[:k], rtol=0, atol=0, equal_nan=True)),
        f"(on {k:,} sampled cells)",
    )

    # GRIP broadcast
    factor = grid.width // 4320
    grip_ok = True
    for tp, colname in GRIP_TP_TO_COLUMN.items():
        with rasterio.open(os.path.join(GRIP_DIR, f"grip4_tp{tp}_dens_m_km2.asc")) as s:
            arr = s.read(1).astype("float32")
            nod = s.nodata
        if nod is not None:
            arr[arr == np.float32(nod)] = np.nan
        expect = arr[row // factor, col // factor]
        got = df[colname].values.astype("float32")
        same = (expect == got) | (np.isnan(expect) & np.isnan(got))
        grip_ok &= bool(same.all())
        if not same.all():
            print(f"      {colname}: {(~same).sum()} mismatches", flush=True)
    check(
        f"grip_*_1 == GRIP4 tp value of the containing 5-arcmin cell "
        f"(broadcast factor {factor})",
        grip_ok,
    )

    nan_ok = True
    for c in ("dtw_1", "pop_density", "GHS_SMOD", "slope_1"):
        nan_ok &= bool(
            np.array_equal(
                df[c + "_nan"].values.astype(bool), ~np.isfinite(df[c].values)
            )
        )
    check("*_nan indicators agree with the data", nan_ok)
    check(
        "any_nan == OR of the four indicators",
        np.array_equal(
            df["any_nan"].values.astype(bool),
            np.any(
                [
                    df[c + "_nan"].values.astype(bool)
                    for c in ("dtw_1", "pop_density", "GHS_SMOD", "slope_1")
                ],
                axis=0,
            ),
        ),
    )
    check(
        "ISOCODE / shapeID / shapeName all populated",
        bool(
            df[["ISOCODE", "shapeID", "shapeName"]].notna().all().all()
            and (df["ISOCODE"].astype(str).str.len() > 0).all()
        ),
    )

    bad = [n for n, ok, _ in checks if not ok]
    print(f"\n{len(checks) - len(bad)}/{len(checks)} checks passed", flush=True)
    if bad:
        print("FAILED: " + ", ".join(bad))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
