#!/usr/bin/env python
"""Mosaic the 8 GPW v4.11 ESRI-ASCII tiles into one master-lattice GeoTIFF.

SEDAC ships the 30 arc-second GPW grids as eight ``.asc`` tiles (a 4x2 layout of
10800x10800 blocks covering the global 43200x21600 lattice).  The extraction
pipeline wants a single tiled, compressed GeoTIFF that is *bit-for-bit on the
same lattice* as the GPW GeoTIFF the pipeline already uses
(``gpw_v4_density_2020_30sec.tif``), so that ``layers.WarpedReader`` produces
identical resampling behaviour for the adjusted and unadjusted products.

Usage
-----
    .venv/bin/python scripts/extraction/convert_gpw_asc.py \
        --zip data/GIS/gpw-v4-population-density-adjusted-to-2015-unwpp-country-totals-rev11_2020_30_sec_asc.zip \
        --out data/GIS/source_rasters/gpw_v4_density_adjusted_2020_30sec.tif

``--zip`` extracts to a scratch directory (~6.1 GB) and deletes the ``.asc``
files as soon as each one has been written into the mosaic, so peak extra disk
is roughly one tile plus the finished GeoTIFF.  Pass ``--asc-dir`` instead if
the tiles are already unpacked (nothing is deleted in that mode unless
``--delete-asc`` is given).

Why the transform is *stamped*, not recomputed
----------------------------------------------
The ``.asc`` headers carry the clean corners (``xllcorner -180``,
``yllcorner -90``, ``cellsize 0.0083333333333339``) with float noise of order
1e-11 degrees -- e.g. tile 2's ``xllcorner`` is ``-89.999999999994``.  The
reference GeoTIFF carries ``cellsize 0.00833333333333333`` and origin
``(-180.0, 89.99999999999991)``.  The two describe the *same* lattice to within
~1e-11 deg, i.e. about one millionth of a cell, so the correct action is to
stamp the reference transform rather than resample.  ``--max-origin-drift``
(default 1e-6 deg) is asserted per tile: if a real misalignment ever shows up
the script fails instead of silently shifting data.

Note this reference lattice is *not* the Kummu master lattice (origin
-179.999999007 / +90.002416350); GPW is off-lattice from Kummu by ~0.1% of a
cell and is read through ``rasterio.warp.reproject`` in ``layers.py``.  Matching
the existing GPW GeoTIFF exactly is what keeps that warp unchanged.

Sanity check
------------
After the mosaic is written the script runs two spot-checks over a handful of
countries, integrating ``density * spherical cell area`` inside the
geoBoundaries CGAZ ADM0 polygon:

1. **Structural** — adjusted / unadjusted must be a *constant* per country.
   The UN WPP adjustment is a single national scalar, so if the mosaic really
   is the adjusted sibling of the reference raster then essentially every
   populated cell in a country carries the same ratio.  This is the check that
   proves the file is the adjusted product and not some unrelated raster, and
   it needs no external constants.
2. **External anchor** — the adjusted country total must land near the UN WPP
   2015-revision 2020 estimate (the series GPW v4.11 adjusts to), while the
   unadjusted total does not.  The discrimination is large (ROU 19.2M adjusted
   vs 34.5M unadjusted against a UN 19.2M), so a loose tolerance is enough; it
   absorbs both revision rounding and the fact that this crude integration
   weights by *cell* area while GPW density is per unit *land* area.

It is a spot-check, not a validation: Gate 1 (``validate_extraction.py``) is
the real gate.  Note that Gate 1 currently *fails* with the adjusted product —
the legacy artefact was measured to be the unadjusted one — so this script
producing a clean bill of health does not mean the pipeline should switch to
it.  See scripts/extraction/README.md §6.
"""

from __future__ import annotations

import argparse
import math
import os
import shutil
import sys
import time
import zipfile

import numpy as np
import rasterio
from rasterio.windows import Window

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from grid import RASTER_DIR, REPO_ROOT  # noqa: E402

#: The GeoTIFF whose grid the mosaic must reproduce exactly.
DEFAULT_REFERENCE = os.path.join(RASTER_DIR, "gpw_v4_density_2020_30sec.tif")
DEFAULT_OUT = os.path.join(RASTER_DIR, "gpw_v4_density_adjusted_2020_30sec.tif")
DEFAULT_ADM0 = os.path.join(RASTER_DIR, "geoBoundariesCGAZ_ADM0.gpkg")

#: ISO3 -> UN WPP **2015 revision**, medium variant, 2020 population in
#: millions.  That is the series GPW v4.11's "adjusted to 2015 UNWPP country
#: totals" product is calibrated to, so the adjusted raster should reproduce
#: these and the unadjusted raster should not.  These four are chosen because
#: their adjustment factors are far from 1 (measured 0.557 ROU, 0.738 NPL,
#: 0.868 ESP, 0.955 NGA), which makes the check discriminating not decorative.
EXPECTED_UN_2020_M = {"ROU": 19.24, "NPL": 30.29, "ESP": 45.92, "NGA": 210.0}

#: Relative tolerance on the UN anchor.  Deliberately loose: it has to absorb
#: WPP revision rounding *and* the fact that this integration weights by whole
#: cell area while GPW density is persons per km^2 of *land*.  The unadjusted
#: totals miss by 10-79%, so 8% still separates the two products cleanly.
UN_TOLERANCE = 0.08

#: Share of a country's populated cells that must carry the modal
#: adjusted/unadjusted ratio for the "it is a per-country scalar" test to pass.
RATIO_CONSTANCY = 0.98

EARTH_RADIUS_KM = 6371.0087714


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def pick_blocksize(*dims: int, cap: int = 512) -> int:
    """Largest multiple of 16 that is <= ``cap`` and divides every ``dim``.

    GTiff requires tile dimensions to be multiples of 16.  Choosing a block
    size that divides the tile offsets *and* the tile extents means every
    output block is written exactly once, which is what makes a random-order
    write into a compressed tiled GeoTIFF safe.
    """
    g = 0
    for d in dims:
        g = math.gcd(g, int(d))
    best = 0
    for b in range(16, cap + 1, 16):
        if g % b == 0:
            best = b
    if not best:
        raise ValueError(f"no valid block size <= {cap} divides gcd {g} of {dims}")
    return best


def cell_area_km2(top: np.ndarray, bottom: np.ndarray, cell_deg: float) -> np.ndarray:
    """Spherical area of a lat/lon cell, km^2 (same formula as Gate 1)."""
    return (
        (EARTH_RADIUS_KM**2)
        * np.radians(cell_deg)
        * (np.sin(np.radians(top)) - np.sin(np.radians(bottom)))
    )


def find_tiles(asc_dir: str) -> list[str]:
    tiles = sorted(
        os.path.join(asc_dir, f)
        for f in os.listdir(asc_dir)
        if f.lower().endswith(".asc")
    )
    if len(tiles) != 8:
        raise ValueError(f"expected 8 .asc tiles in {asc_dir}, found {len(tiles)}")
    return tiles


def extract_zip(zip_path: str, dest: str) -> str:
    os.makedirs(dest, exist_ok=True)
    print(f"extracting {zip_path}\n        -> {dest}", flush=True)
    t0 = time.time()
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(dest)
    print(f"extracted in {time.time() - t0:.0f}s", flush=True)
    return dest


# --------------------------------------------------------------------------
# conversion
# --------------------------------------------------------------------------
def convert(
    asc_dir: str,
    out_path: str,
    reference: str,
    max_origin_drift: float,
    delete_asc: bool,
) -> None:
    tiles = find_tiles(asc_dir)
    with rasterio.open(reference) as ref:
        ref_transform = ref.transform
        ref_w, ref_h = ref.width, ref.height
        ref_crs = ref.crs
        ref_nodata = ref.nodata
    cell = ref_transform.a
    print(
        f"reference {os.path.relpath(reference, REPO_ROOT)}: "
        f"{ref_w}x{ref_h} cell {cell!r} origin "
        f"({ref_transform.c!r}, {ref_transform.f!r}) nodata {ref_nodata}",
        flush=True,
    )

    # ---- 1. read every header, derive the tile -> master offsets ----------
    placements = []
    for path in tiles:
        with rasterio.open(path) as src:
            t, w, h = src.transform, src.width, src.height
            if abs(t.a - cell) > 1e-9 or abs(-t.e - cell) > 1e-9:
                raise ValueError(
                    f"{os.path.basename(path)}: cell size {t.a!r}/{-t.e!r} differs "
                    f"from the reference {cell!r} by more than 1e-9 deg -- refusing "
                    "to stamp the reference transform.  Resampling is NOT "
                    "authorised; stop and report."
                )
            col_off = (t.c - ref_transform.c) / cell
            row_off = (ref_transform.f - t.f) / cell
            col_i, row_i = round(col_off), round(row_off)
            drift_x = abs(col_off - col_i) * cell
            drift_y = abs(row_off - row_i) * cell
            if max(drift_x, drift_y) > max_origin_drift:
                raise ValueError(
                    f"{os.path.basename(path)}: origin ({t.c!r}, {t.f!r}) is "
                    f"{drift_x:.3e}/{drift_y:.3e} deg off the reference lattice, "
                    f"beyond --max-origin-drift {max_origin_drift:g}.  Refusing to "
                    "stamp; stop and report."
                )
            if col_i < 0 or row_i < 0 or col_i + w > ref_w or row_i + h > ref_h:
                raise ValueError(
                    f"{os.path.basename(path)}: window "
                    f"({row_i}:{row_i + h}, {col_i}:{col_i + w}) falls outside the "
                    f"{ref_w}x{ref_h} reference grid"
                )
            placements.append((path, col_i, row_i, w, h, drift_x, drift_y))

    covered = sum(w * h for _p, _c, _r, w, h, _dx, _dy in placements)
    if covered != ref_w * ref_h:
        raise ValueError(
            f"the 8 tiles cover {covered:,} cells, the reference grid has "
            f"{ref_w * ref_h:,} -- the mosaic would be incomplete"
        )
    print("tile placement (verified from each header, not assumed):", flush=True)
    for path, col_i, row_i, w, h, dx, dy in placements:
        print(
            f"  {os.path.basename(path)[-9:]:>9}  {w}x{h} at "
            f"row {row_i:>5} col {col_i:>5}  drift {dx:.2e}/{dy:.2e} deg",
            flush=True,
        )

    # ---- 2. mosaic --------------------------------------------------------
    offsets = [v for _p, c, r, w, h, _dx, _dy in placements for v in (c, r, w, h)]
    block = pick_blocksize(*offsets)
    profile = {
        "driver": "GTiff",
        "dtype": "float32",
        "count": 1,
        "width": ref_w,
        "height": ref_h,
        "crs": ref_crs,
        "transform": ref_transform,
        "nodata": ref_nodata,
        "tiled": True,
        "blockxsize": block,
        "blockysize": block,
        "compress": "deflate",
        "predictor": 2,
        "zlevel": 6,
        "BIGTIFF": "IF_SAFER",
    }
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    tmp_out = out_path + ".partial"
    print(f"writing {out_path} (tiled {block}x{block}, DEFLATE)", flush=True)
    t0 = time.time()
    with rasterio.open(tmp_out, "w", **profile) as dst:
        for path, col_i, row_i, w, h, _dx, _dy in placements:
            with rasterio.open(path) as src:
                arr = src.read(1).astype("float32")
                src_nodata = src.nodata
            if src_nodata is not None and not np.isclose(src_nodata, ref_nodata):
                arr[arr == np.float32(src_nodata)] = np.float32(ref_nodata)
            window = Window(col_i, row_i, w, h)
            dst.write(arr, 1, window=window)
            # Read the block straight back out of the (compressed) destination
            # and require bit-equality: DEFLATE + PREDICTOR must be lossless,
            # and this is cheap insurance against a silent codec surprise.
            back = dst.read(1, window=window)
            if not np.array_equal(arr, back):
                bad = int(np.count_nonzero(arr != back))
                raise ValueError(
                    f"{os.path.basename(path)}: {bad:,} cells differ after the "
                    "GeoTIFF round-trip -- the compression is not lossless"
                )
            del back
            finite = arr != np.float32(ref_nodata)
            print(
                f"  wrote {os.path.basename(path)[-9:]:>9}  "
                f"{int(finite.sum()):,} data cells, max {float(arr[finite].max()):,.1f}"
                f", read-back bit-exact  [{time.time() - t0:.0f}s]",
                flush=True,
            )
            del arr
            if delete_asc:
                os.remove(path)
    os.replace(tmp_out, out_path)
    print(
        f"mosaic done in {time.time() - t0:.0f}s, "
        f"{os.path.getsize(out_path) / 1e6:.0f} MB",
        flush=True,
    )


# --------------------------------------------------------------------------
# sanity check
# --------------------------------------------------------------------------
def _read_country(src, geoms, bounds):
    """Windowed read of ``src`` masked to ``geoms``; returns (values, areas)."""
    from rasterio import features

    left, bottom, right, top = bounds
    win = rasterio.windows.from_bounds(
        left, bottom, right, top, transform=src.transform
    )
    win = win.round_offsets(op="floor").round_lengths(op="ceil")
    win = win.intersection(Window(0, 0, src.width, src.height))
    arr = src.read(1, window=win).astype("float64")
    t = src.window_transform(win)
    mask = features.geometry_mask(
        geoms, out_shape=arr.shape, transform=t, invert=True, all_touched=False
    )
    rows = np.arange(arr.shape[0])
    row_top = t.f + rows * t.e
    area = np.broadcast_to(
        cell_area_km2(row_top, row_top + t.e, t.a)[:, None], arr.shape
    )
    nodata = src.nodata if src.nodata is not None else -9999
    keep = mask & (arr != nodata) & (arr > 0)
    return arr, area, keep


def _country_population(src, geoms, bounds) -> float:
    """Sum(density * spherical cell area) inside ``geoms``, read windowed."""
    arr, area, keep = _read_country(src, geoms, bounds)
    return float(np.sum(arr[keep] * area[keep]))


def sanity_check(
    out_path: str, reference: str, adm0_path: str, tolerance: float
) -> int:
    """Prove the mosaic is the UN WPP-adjusted sibling of ``reference``."""
    import geopandas as gpd

    if not os.path.exists(adm0_path):
        print(f"SKIP sanity check: {adm0_path} not found", flush=True)
        return 0
    gdf = gpd.read_file(adm0_path).set_crs(4326, allow_override=True)
    key = "shapeGroup" if "shapeGroup" in gdf.columns else "shapeID"
    print("\nsanity check", flush=True)
    print(
        "| country | unadjusted (M) | adjusted (M) | factor | const. | "
        "UN WPP 2015rev 2020 (M) | adj err | ok |",
        flush=True,
    )
    print("|---|---|---|---|---|---|---|---|")
    bad = 0
    with rasterio.open(out_path) as adj, rasterio.open(reference) as una:
        if (adj.width, adj.height) != (una.width, una.height) or not np.allclose(
            list(adj.transform)[:6], list(una.transform)[:6], rtol=0, atol=1e-12
        ):
            raise ValueError(
                "converted raster is not grid-identical to the reference:\n"
                f"  adjusted   {adj.width}x{adj.height} {adj.transform}\n"
                f"  reference  {una.width}x{una.height} {una.transform}"
            )
        print(
            f"  grid-identical to {os.path.basename(reference)}: "
            f"{adj.width}x{adj.height}, transform matches to 1e-12 deg",
            flush=True,
        )
        for iso, un_m in sorted(EXPECTED_UN_2020_M.items()):
            sub = gdf[gdf[key] == iso]
            if sub.empty:
                print(f"| {iso} | — | — | — | — | {un_m} | — | NOT FOUND |", flush=True)
                bad += 1
                continue
            # Rasterise the polygon parts directly rather than dissolving them:
            # some CGAZ ADM0 geometries (NGA) are not valid enough for GEOS's
            # unary union, and geometry_mask happily takes the parts.
            geoms = list(sub.geometry.values)
            bounds = tuple(sub.total_bounds)
            a_u, area_u, keep_u = _read_country(una, geoms, bounds)
            a_a, _area_a, keep_a = _read_country(adj, geoms, bounds)
            p_una = float(np.sum(a_u[keep_u] * area_u[keep_u]))
            p_adj = float(np.sum(a_a[keep_a] * area_u[keep_a]))
            factor = p_adj / p_una if p_una else float("nan")

            # (1) structural: the adjustment must be a single national scalar.
            both = keep_u & keep_a
            per_cell = np.round(a_a[both] / a_u[both], 4)
            vals, counts = np.unique(per_cell, return_counts=True)
            share = float(counts.max() / counts.sum()) if counts.size else 0.0
            modal = float(vals[np.argmax(counts)]) if counts.size else float("nan")
            const_ok = share >= RATIO_CONSTANCY

            # (2) external anchor: adjusted total must match UN WPP.
            err = p_adj / (un_m * 1e6) - 1.0
            un_ok = abs(err) <= tolerance
            ok = const_ok and un_ok
            bad += not ok
            print(
                f"| {iso} | {p_una / 1e6:,.2f} | {p_adj / 1e6:,.2f} | {modal:.4f} | "
                f"{100 * share:.1f}% | {un_m:,.2f} | {100 * err:+.2f}% | "
                f"{'OK' if ok else ('RATIO-NOT-CONSTANT' if not const_ok else 'UN-MISMATCH')} |",
                flush=True,
            )
    if bad:
        print(
            f"\nFAIL: {bad} spot-check country/countries did not behave like the UN "
            "WPP-adjusted sibling of the reference raster (a constant per-country "
            f"rescaling landing within {100 * tolerance:.0f}% of UN WPP). The "
            "converted raster may not be the adjusted product.",
            flush=True,
        )
    else:
        print(
            "\nPASS: the mosaic is grid-identical to the unadjusted reference, is a "
            "constant per-country rescaling of it, and its country totals reproduce "
            "UN WPP 2015-revision 2020 (which the unadjusted totals do not) — it is "
            "the UN WPP-adjusted product.\nNOTE this does NOT mean the pipeline "
            "should use it: Gate 1 measured the legacy artefact to be the "
            "*unadjusted* product. See scripts/extraction/README.md §6.",
            flush=True,
        )
    return 1 if bad else 0


# --------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    src = p.add_mutually_exclusive_group()
    src.add_argument("--zip", default=None, help="the SEDAC *_asc.zip to unpack")
    src.add_argument("--asc-dir", default=None, help="directory of unpacked tiles")
    p.add_argument("--out", default=DEFAULT_OUT)
    p.add_argument("--reference", default=DEFAULT_REFERENCE)
    p.add_argument("--adm0", default=DEFAULT_ADM0)
    p.add_argument(
        "--scratch",
        default=None,
        help="where --zip is unpacked (default: <out dir>/_gpw_asc_scratch)",
    )
    p.add_argument(
        "--delete-asc",
        action="store_true",
        help="delete each .asc after it has been mosaicked (implied by --zip)",
    )
    p.add_argument("--keep-asc", action="store_true", help="never delete .asc files")
    p.add_argument("--max-origin-drift", type=float, default=1e-6)
    p.add_argument("--un-tolerance", type=float, default=UN_TOLERANCE)
    p.add_argument(
        "--check-only",
        action="store_true",
        help="skip conversion, only re-run the country-ratio sanity check",
    )
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    for attr in ("zip", "asc_dir", "out", "reference", "adm0", "scratch"):
        if getattr(args, attr):
            setattr(args, attr, os.path.abspath(getattr(args, attr)))
    os.chdir(REPO_ROOT)

    scratch = None
    if not args.check_only:
        if not args.zip and not args.asc_dir:
            raise SystemExit("one of --zip / --asc-dir is required")
        asc_dir = args.asc_dir
        delete_asc = args.delete_asc and not args.keep_asc
        if args.zip:
            scratch = args.scratch or os.path.join(
                os.path.dirname(args.out), "_gpw_asc_scratch"
            )
            asc_dir = extract_zip(args.zip, scratch)
            delete_asc = not args.keep_asc
        try:
            convert(
                asc_dir, args.out, args.reference, args.max_origin_drift, delete_asc
            )
        finally:
            if scratch and not args.keep_asc and os.path.isdir(scratch):
                shutil.rmtree(scratch, ignore_errors=True)
                print(f"removed scratch {scratch}", flush=True)

    return sanity_check(args.out, args.reference, args.adm0, args.un_tolerance)


if __name__ == "__main__":
    raise SystemExit(main())
