"""Windowed, grid-aligned readers for every source layer (SPEC ADR-3).

Every reader exposes ``read_block(grid, row_off, n_rows) -> np.ndarray`` of
shape ``(n_rows, grid.width)`` with ``np.nan`` in place of the source nodata
value, so the extraction driver never has to know which layer needs
reprojection, which needs a CRS override and which is natively on-lattice.

Design notes
------------
* ``dtw`` is read natively off the grid-defining raster -- zero resampling,
  bit-exact Kummu values (this is the decision-driving layer, FEASIBILITY §2).
* Everything else goes through ``rasterio.warp.reproject`` onto the block's
  affine, or through exact integer index arithmetic where the lattices are
  provably coincident (GRIP4).  Categorical / coarse layers use
  nearest-neighbour, matching what QGIS evidently did.
* GRIP4 ``.asc`` grids ship without a ``.prj`` (``crs=None``); EPSG:4326 is
  asserted explicitly per the dataset ReadMe.
* GHS-SMOD is the only Mollweide (ESRI:54009) layer and gets a genuine
  reprojection.
"""

from __future__ import annotations

import os

import numpy as np
import rasterio
from affine import Affine
from rasterio.enums import Resampling
from rasterio.warp import reproject
from rasterio.windows import Window

from grid import RASTER_DIR, MasterGrid

# --------------------------------------------------------------------------
# Source paths
# --------------------------------------------------------------------------
GPW_DENSITY = os.path.join(RASTER_DIR, "gpw_v4_density_2020_30sec.tif")
#: GHS-SMOD R2023A 1 km Mollweide.  V2-0 is the default (newest, and the file
#: recorded in data/GIS/source_rasters/MANIFEST.md).  V1-0 -- the version named
#: in ``src/gis_global_module.py:171`` -- is also on disk and selectable with
#: ``--smod-raster``; it was measured at 91.9% vs 91.7% cell-level class
#: agreement with the legacy CSV, i.e. not materially different.
GHS_SMOD = os.path.join(RASTER_DIR, "GHS_SMOD_E2020_GLOBE_R2023A_54009_1000_V2_0.tif")
GRIP_DIR = os.path.join(RASTER_DIR, "grip4")

#: EarthEnv slope. The legacy CSV's ``slope_1`` was proven (100.00% bit-exact
#: on 5,000 random cells) to be ``slope_10KMmd_GMTEDmd.tif`` -- the *median*
#: aggregation published natively on the 5-arcmin grid, NOT the ``mn`` (mean)
#: variant SPEC ADR-3 guessed at.  The 30 arc-second analogue of that product
#: is ``slope_1KMmd_GMTEDmd.tif``; ``slope_1KMmn_GMTEDmd.tif`` is kept as a
#: fallback so the pipeline still runs on a checkout that only has the file
#: named in ADR-3.  See scripts/extraction/README.md.
SLOPE_CANDIDATES = {
    "5min": [
        "slope_10KMmd_GMTEDmd.tif",
        "slope_1KMmd_GMTEDmd.tif",
        "slope_1KMmn_GMTEDmd.tif",
    ],
    "30as": ["slope_1KMmd_GMTEDmd.tif", "slope_1KMmn_GMTEDmd.tif"],
}

#: GRIP4 road type -> legacy column.  The legacy QGIS export carried the raw
#: ``tp11..tp51`` names and ``pre_process_GIS_data.py:63-73`` renames them with
#: a deliberate *inversion* (``tp51 -> grip_1_1`` ... ``tp11 -> grip_5_1``).
#: Verified empirically against the legacy CSV: 99.9% of sampled cells.
GRIP_TP_TO_COLUMN = {
    5: "grip_1_1",  # tp5 = local roads
    4: "grip_2_1",  # tp4 = tertiary
    3: "grip_3_1",  # tp3 = secondary
    2: "grip_4_1",  # tp2 = primary
    1: "grip_5_1",  # tp1 = highways
}
GRIP_COLUMNS = [GRIP_TP_TO_COLUMN[tp] for tp in (5, 4, 3, 2, 1)]

RASTER_NAN_COLUMNS = ["dtw_1", "pop_density", "GHS_SMOD", "slope_1"]


def _require(path: str, what: str) -> str:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"missing {what}: {path}\n"
            "See scripts/extraction/README.md for the download commands."
        )
    return path


def resolve_slope_path(resolution: str, override: str | None = None) -> str:
    if override:
        return _require(override, "slope raster")
    for name in SLOPE_CANDIDATES[resolution]:
        candidate = os.path.join(RASTER_DIR, name)
        if os.path.exists(candidate):
            return candidate
    raise FileNotFoundError(
        "no EarthEnv slope raster found; expected one of "
        + ", ".join(SLOPE_CANDIDATES[resolution])
        + f" in {RASTER_DIR}"
    )


# --------------------------------------------------------------------------
# Readers
# --------------------------------------------------------------------------
class NativeReader:
    """Direct windowed read of a raster that *is* the master grid."""

    def __init__(self, path: str, name: str):
        self.path = _require(path, name)
        self.name = name
        self.src = rasterio.open(self.path)

    def read_block(self, grid: MasterGrid, row_off: int, n_rows: int) -> np.ndarray:
        if (self.src.width, self.src.height) != (grid.width, grid.height):
            raise ValueError(
                f"{self.name}: {self.src.width}x{self.src.height} is not the "
                f"master grid {grid.width}x{grid.height}"
            )
        arr = self.src.read(1, window=Window(0, row_off, grid.width, n_rows)).astype(
            "float32"
        )
        nodata = self.src.nodata
        if nodata is not None:
            arr[arr == np.float32(nodata)] = np.nan
        return arr

    def close(self):
        self.src.close()


class WarpedReader:
    """Reproject an arbitrary raster onto the master block affine."""

    def __init__(
        self,
        path: str,
        name: str,
        resampling: Resampling = Resampling.nearest,
        src_crs: str | None = None,
    ):
        self.path = _require(path, name)
        self.name = name
        self.resampling = resampling
        self.src = rasterio.open(self.path)
        self.src_crs = src_crs or self.src.crs
        if self.src_crs is None:
            raise ValueError(f"{name}: source CRS is undefined and no override given")
        self.nodata = self.src.nodata

    def read_block(self, grid: MasterGrid, row_off: int, n_rows: int) -> np.ndarray:
        dst = np.full((n_rows, grid.width), np.nan, dtype="float32")
        reproject(
            source=rasterio.band(self.src, 1),
            destination=dst,
            src_crs=self.src_crs,
            src_nodata=self.nodata,
            dst_transform=grid.block_transform(row_off),
            dst_crs=grid.crs,
            dst_nodata=np.nan,
            resampling=self.resampling,
            num_threads=2,
        )
        return dst

    def close(self):
        self.src.close()


class GripReader:
    """The five GRIP4 5-arcmin road-density grids.

    The GRIP lattice (origin -180/+90, cell 1/12 deg) and the master lattice
    (origin -179.999999007/+90.002416350) are offset by only 0.03 of a cell, so
    the nearest-neighbour mapping is exactly

        grip_row = master_row // f,   grip_col = master_col // f

    with ``f`` = 1 at 5 arc-minute and 10 at 30 arc-second (i.e. the documented
    10x10 broadcast -- the information floor of the road layer stays 5 arcmin).
    The mapping is asserted in :meth:`check_alignment` rather than assumed.
    """

    def __init__(self):
        self.arrays = {}
        for tp in (1, 2, 3, 4, 5):
            path = _require(
                os.path.join(GRIP_DIR, f"grip4_tp{tp}_dens_m_km2.asc"),
                f"GRIP4 tp{tp}",
            )
            with rasterio.open(path) as src:
                if src.crs is not None and src.crs.to_string() != "EPSG:4326":
                    raise ValueError(f"GRIP tp{tp} unexpected CRS {src.crs}")
                arr = src.read(1).astype("float32")
                if src.nodata is not None:
                    arr[arr == np.float32(src.nodata)] = np.nan
                self.arrays[tp] = arr
                self.transform = src.transform
                self.shape = arr.shape

    def check_alignment(self, grid: MasterGrid) -> int:
        factor = grid.width // self.shape[1]
        if grid.width % self.shape[1] or grid.height // self.shape[0] != factor:
            raise ValueError("GRIP grid is not an integer factor of the master grid")
        # Verify the floor-divide mapping against true nearest-neighbour lookup
        # at the extremes and at a few interior cells.
        probe_rows = np.array([0, 1, factor - 1, factor, grid.height - 1])
        probe_cols = np.array([0, 1, factor - 1, factor, grid.width - 1])
        cs = grid.cell_size
        lat = grid.transform.f + (probe_rows + 0.5) * grid.transform.e
        lon = grid.transform.c + (probe_cols + 0.5) * cs
        exp_r = np.floor((self.transform.f - lat) / -self.transform.e).astype(int)
        exp_c = np.floor((lon - self.transform.c) / self.transform.a).astype(int)
        if not (
            np.array_equal(exp_r, probe_rows // factor)
            and np.array_equal(exp_c, probe_cols // factor)
        ):
            raise ValueError(
                "GRIP <-> master nearest-neighbour mapping is not floor-divide; "
                f"rows {exp_r} vs {probe_rows // factor}, "
                f"cols {exp_c} vs {probe_cols // factor}"
            )
        return factor

    def read_block(self, grid: MasterGrid, row_off: int, n_rows: int, tp: int):
        factor = grid.width // self.shape[1]
        rows = (np.arange(row_off, row_off + n_rows) // factor).astype(np.int64)
        cols = (np.arange(grid.width) // factor).astype(np.int64)
        return self.arrays[tp][np.ix_(rows, cols)]

    def close(self):
        self.arrays.clear()


class AdminRasteriser:
    """Rasterise geoBoundaries CGAZ ADM1 onto the master grid, block by block.

    Cell value is ``polygon_index + 1``; 0 means "no ADM1 polygon covers this
    cell centre" and those cells are dropped from the output (count reported),
    reproducing the legacy ``dropna(subset=["shapeGroup"])`` step.

    ``ISOCODE`` comes from ADM1's ``shapeGroup`` (the ISO3 of the parent
    country).  Checked against the legacy CSV: the ADM1 layer's shapeGroup
    reproduces the legacy ISOCODE, and ADM0 supplies no additional cells --
    both CGAZ layers cover the same 218 countries.
    """

    def __init__(self, adm1_path: str, adm0_path: str | None = None):
        import geopandas as gpd
        from shapely.geometry import box

        _require(adm1_path, "geoBoundaries CGAZ ADM1")
        gdf = gpd.read_file(adm1_path)
        # CGAZ declares an "Undefined geographic SRS"; coordinates are WGS84.
        gdf = gdf.set_crs(4326, allow_override=True)
        missing = {"shapeName", "shapeID", "shapeGroup"} - set(gdf.columns)
        if missing:
            raise ValueError(f"ADM1 layer is missing columns {sorted(missing)}")
        self.gdf = gdf.reset_index(drop=True)
        self.geoms = self.gdf.geometry.values
        self.sindex = self.gdf.sindex
        self._box = box
        self.shape_name = np.asarray(self.gdf["shapeName"].astype(str))
        self.shape_id = np.asarray(self.gdf["shapeID"].astype(str))
        self.iso = np.asarray(self.gdf["shapeGroup"].astype(str))

    def read_block(self, grid: MasterGrid, row_off: int, n_rows: int) -> np.ndarray:
        from rasterio import features

        t = grid.block_transform(row_off)
        left = t.c
        right = t.c + grid.width * t.a
        top = t.f
        bottom = t.f + n_rows * t.e
        hits = self.sindex.query(
            self._box(left, bottom, right, top), predicate="intersects"
        )
        out = np.zeros((n_rows, grid.width), dtype="int32")
        if len(hits) == 0:
            return out
        shapes = ((self.geoms[i], int(i) + 1) for i in hits)
        features.rasterize(
            shapes,
            out=out,
            transform=t,
            fill=0,
            all_touched=False,
            dtype="int32",
        )
        return out

    def close(self):
        pass


def open_readers(grid: MasterGrid, args) -> dict:
    """Instantiate every layer reader for one extraction run."""
    from grid import GRID_DEFINING_RASTER

    pop_resampling = (
        Resampling.nearest
        if grid.resolution == "30as"
        else getattr(Resampling, args.pop_agg)
    )
    readers = {
        "dtw_1": NativeReader(GRID_DEFINING_RASTER[grid.resolution], "Kummu dtw"),
        "pop_density": WarpedReader(
            GPW_DENSITY, "GPW v4.11 density", resampling=pop_resampling
        ),
        "slope_1": WarpedReader(
            resolve_slope_path(grid.resolution, args.slope_raster),
            "EarthEnv slope",
            resampling=Resampling.nearest,
        ),
        "GHS_SMOD": WarpedReader(
            getattr(args, "smod_raster", None) or GHS_SMOD,
            "GHS-SMOD R2023A",
            resampling=Resampling.nearest,
        ),
    }
    readers["_grip"] = GripReader()
    readers["_grip"].check_alignment(grid)
    readers["_admin"] = AdminRasteriser(args.adm1)
    return readers
