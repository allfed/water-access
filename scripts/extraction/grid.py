"""Master-grid definition for the scripted raster extraction (SPEC ADR-1).

The master grid is the Kummu *distance to surface water* lattice:

    30 arc-second : 43200 x 21600, EPSG:4326, cell 0.00833333 deg
    5 arc-minute  :  4320 x  2160, EPSG:4326, cell 0.08333333 deg  (the same
                     lattice divided by 10)

Both are taken *verbatim* from the corresponding Kummu GeoTIFF rather than
being reconstructed from constants, so the emitted ``left/top/right/bottom``
values are bit-identical to the legacy QGIS export
(``data/GIS/updated_GIS_output_cleaned.csv``).  Note the origin is
(-179.999999007, +90.002416350) -- *not* (-180, +90); every other raster in the
pipeline is therefore very slightly off-lattice and must be read through
``rasterio.warp.reproject`` (nearest) rather than by raw array indexing.

``fid`` encoding
----------------
The legacy CSV uses **column-major, 1-based** ids::

    fid = col * nrows + row + 1

verified to match on 100% of the 2,190,446 legacy rows (``nrows`` = 2160 at
5 arc-minute).  The obvious row-major ``row * ncols + col`` alternative matches
0% of them, so the legacy convention is reproduced here and extended to 30
arc-second with ``nrows`` = 21600.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import rasterio
from affine import Affine

RESOLUTIONS = ("5min", "30as")

#: Repository root, derived from this file's location (scripts/extraction/).
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RASTER_DIR = os.path.join(REPO_ROOT, "data", "GIS", "source_rasters")

#: The file that *defines* the master grid for each resolution (ADR-1).
GRID_DEFINING_RASTER = {
    "5min": os.path.join(RASTER_DIR, "distance2water_5arcmin.tif"),
    "30as": os.path.join(RASTER_DIR, "distance2water_30arcsec.tif"),
}


@dataclass(frozen=True)
class MasterGrid:
    """An immutable description of one of the two master lattices."""

    resolution: str
    width: int
    height: int
    transform: Affine
    crs: str = "EPSG:4326"

    @property
    def cell_size(self) -> float:
        return self.transform.a

    def block_transform(self, row_off: int) -> Affine:
        """Affine of a full-width block starting at master row ``row_off``."""
        t = self.transform
        return Affine(t.a, 0.0, t.c, 0.0, t.e, t.f + row_off * t.e)

    def cell_bounds(self, rows: np.ndarray, cols: np.ndarray):
        """Return ``(left, top, right, bottom)`` in degrees for cell indices.

        Phase A's resolution-aware population filter infers the cell size from
        ``right - left``, so these columns are load-bearing, not decoration.
        """
        t = self.transform
        left = t.c + cols * t.a
        top = t.f + rows * t.e
        return left, top, left + t.a, top + t.e

    def fid(self, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
        """Legacy column-major, 1-based cell id (see module docstring)."""
        return cols.astype(np.int64) * self.height + rows.astype(np.int64) + 1


def load_master_grid(resolution: str) -> MasterGrid:
    """Read the master grid straight off the Kummu raster for ``resolution``."""
    if resolution not in RESOLUTIONS:
        raise ValueError(f"resolution must be one of {RESOLUTIONS}, got {resolution!r}")
    path = GRID_DEFINING_RASTER[resolution]
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"grid-defining raster missing: {path}\n"
            "See scripts/extraction/README.md for the download commands."
        )
    with rasterio.open(path) as src:
        grid = MasterGrid(
            resolution=resolution,
            width=src.width,
            height=src.height,
            transform=src.transform,
            crs=str(src.crs),
        )
    expected = {"5min": (4320, 2160), "30as": (43200, 21600)}[resolution]
    if (grid.width, grid.height) != expected:
        raise ValueError(
            f"{path} is {grid.width}x{grid.height}, expected {expected} (ADR-1)"
        )
    return grid


def iter_row_blocks(grid: MasterGrid, block_rows: int):
    """Yield ``(row_off, n_rows)`` covering the grid in full-width strips."""
    for row_off in range(0, grid.height, block_rows):
        yield row_off, min(block_rows, grid.height - row_off)
