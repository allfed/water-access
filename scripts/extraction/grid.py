"""Master-grid definition for the scripted raster extraction.

The master grid is the Kummu *distance to surface water* lattice at
30 arc-seconds: 43200 x 21600 cells, EPSG:4326, cell 0.00833333 deg.

It is taken *verbatim* from the Kummu GeoTIFF rather than reconstructed from
constants. Note the origin is (-179.999999007, +90.002416350) -- *not*
(-180, +90); every other raster in the pipeline is therefore very slightly
off-lattice and must be read through ``rasterio.warp.reproject`` (nearest)
rather than by raw array indexing.

``fid`` encoding
----------------
Cell ids are **column-major, 1-based**::

    fid = col * nrows + row + 1

with ``nrows`` = 21600.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import rasterio
from affine import Affine

#: Repository root, derived from this file's location (scripts/extraction/).
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RASTER_DIR = os.path.join(REPO_ROOT, "data", "GIS", "source_rasters")

#: The file that *defines* the master grid.
GRID_DEFINING_RASTER = os.path.join(RASTER_DIR, "distance2water_30arcsec.tif")

#: Expected (width, height) of the master grid.
GRID_SHAPE = (43200, 21600)


@dataclass(frozen=True)
class MasterGrid:
    """An immutable description of the master lattice."""

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

        The model infers the cell size from ``right - left`` and the cell
        latitude from ``top``/``bottom``, so these columns are load-bearing.
        """
        t = self.transform
        left = t.c + cols * t.a
        top = t.f + rows * t.e
        return left, top, left + t.a, top + t.e

    def fid(self, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
        """Column-major, 1-based cell id (see module docstring)."""
        return cols.astype(np.int64) * self.height + rows.astype(np.int64) + 1


def load_master_grid() -> MasterGrid:
    """Read the master grid straight off the Kummu 30 arc-second raster."""
    path = GRID_DEFINING_RASTER
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"grid-defining raster missing: {path}\n"
            "See scripts/extraction/README.md for the download commands."
        )
    with rasterio.open(path) as src:
        grid = MasterGrid(
            width=src.width,
            height=src.height,
            transform=src.transform,
            crs=str(src.crs),
        )
    if (grid.width, grid.height) != GRID_SHAPE:
        raise ValueError(f"{path} is {grid.width}x{grid.height}, expected {GRID_SHAPE}")
    return grid


def iter_row_blocks(grid: MasterGrid, block_rows: int):
    """Yield ``(row_off, n_rows)`` covering the grid in full-width strips."""
    for row_off in range(0, grid.height, block_rows):
        yield row_off, min(block_rows, grid.height - row_off)
