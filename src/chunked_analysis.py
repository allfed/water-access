"""Chunked execution of the global analysis, for the 30 arc-second grid (ADR-5).

At 5 arc-minutes the model holds every populated cell in memory at once: 1.4M
rows, ~3 GB peak. At 30 arc-seconds there are ~140M populated cells and the same
approach needs ~57 GB -- more than twice the AC-6 budget of 24 GB.

The way out is to notice **what the model actually aggregates over**. Every
per-cell calculation is independent, and every aggregation is grouped by
country or by district -- and districts nest inside countries. So if a chunk
holds a set of *whole countries*, its aggregate rows are already final: nothing
has to be re-combined across chunks, and no non-additive statistic is ever
split. Chunking on country groups turns a 57 GB problem into ceil(57 / budget)
independent ones.

Four things do still couple across the whole world, and each is handled
explicitly:

1. **The solver LUT grid.** Its axes span the observed slope and body-mass
   range, which differs per chunk. The planning pre-pass measures the *global*
   ranges and :func:`src.solver_lut.pinned_grid_ranges` freezes them, so every
   chunk interpolates on the identical grid.
2. **The global summary row.** Its population columns are plain sums (chunk
   additive), but its seven population-weighted medians and percentiles are
   not. Those four columns are spilled to disk chunk by chunk and replayed once
   at the end -- the only place per-cell data outlives its chunk.
3. **The outlier cut**, which is defined relative to Libya's weighted median
   distance to water, and the district filter that follows it. Both run in
   :func:`src.gis_global_module.finalise_country_data` after every chunk is in.
4. **Countries too big to be a chunk.** At 30 arc-seconds Russia alone is ~16%
   of all modelled cells -- measured, not guessed -- so no budget that helps
   fits it whole. Such a country is split along *district* boundaries, which
   keeps every district intact; the price is that its own summary row has to be
   recombined (:func:`combine_partial_countries`) and, critically, that its
   national population-density normalisation has to be supplied from outside
   the chunk, or it would normalise against its own fragment and invent people.

``pop_density_perc``, worth calling out, is otherwise *not* on that list: it
normalises population density within a country, so whole-country chunks make it
exact for free. That is exactly why country groups, rather than an arbitrary row
range, are the unit of work.

The monolithic path is untouched and remains the reference;
``scripts/validate_chunking.py`` asserts the two produce identical country and
district frames on the same input at fixed parameters.
"""

import tempfile
import time
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

import src.gis_global_module as gis
import src.solver_lut as solver_lut

#: Default chunk budget, in *post-filter* (modelled) rows.
#:
#: Calibrated, not guessed. Peak RSS was measured against chunk size on the
#: 5-arcmin grid and fits
#:
#:     peak GB ~= 1.0 + 1.2 * (millions of modelled cells in the largest chunk)
#:
#: The 1.2 GB/M coefficient is ~3x the steady-state frame (~406 bytes/row)
#: because pandas' merge and per-column dtype conversions hold transient copies;
#: peak lives inside preprocessing, not in the model itself. 12M cells therefore
#: projects to ~15 GB, comfortably inside AC-6's 24 GB with room for the
#: finalisation pass. Raising it buys little: the LUT is cached across chunks,
#: so extra chunks cost I/O and vectorised arithmetic, not solver time.
DEFAULT_CHUNK_ROWS = 12_000_000

#: Columns the planning pre-pass reads. Deliberately tiny: at 30 arc-seconds
#: this is ~1.7 GB against ~40 GB for the full frame. ``shapeID`` is here so an
#: oversized country can be split along district boundaries.
PLANNING_COLUMNS = ("ISOCODE", "shapeID", "pop_density", "slope_1")

#: Columns a chunk does *not* need when the cell size is known from elsewhere.
#: Four float64 coordinate columns are ~37% of the frame's footprint.
COORDINATE_COLUMNS = ("left", "top", "right", "bottom")

#: The two of those a cos-latitude area weighting *does* need back (they give
#: the cell-centre latitude). Half the saving, and only when the flag is on.
LATITUDE_COLUMNS = tuple(gis.LATITUDE_COLUMNS)

#: Every column anything downstream of preprocessing reads. The preprocessed
#: frame carries roughly twice this many -- the GRIP road densities (consumed by
#: road_analysis and dead thereafter), the nodata indicators, the raw population
#: density, and the country table's geography columns. In the monolithic path
#: they are harmless passengers that get handed back to the caller; a chunked
#: run throws the per-cell frame away, so carrying them is pure cost.
#:
#: Dropping unread columns cannot change a number -- ``validate_chunking.py``
#: is what proves that, since it compares against the monolithic path which
#: keeps them.
CHUNK_KEEP_COLUMNS = (
    # identifiers and grouping keys
    "ISOCODE",
    "shapeID",
    "shapeName",
    "Entity",
    "region",
    "subregion",
    # model inputs
    "slope_1",
    "Crr",
    "Average Weight",
    "dtw_1",
    "urban_rural",
    # population and access inputs
    "pop_zone",
    "country_pop_raw",
    "NATPiped",
    "URBANPiped",
    "RURALPiped",
    "PBO",
    "Household_Size",
)


@dataclass
class ChunkSelection:
    """One unit of work: either whole countries, or a slice of one country.

    Whole-country selections are the normal case and are exactly decomposable.
    A ``districts`` selection exists only for a country too big to fit a chunk
    on its own -- at 30 arc-seconds Russia is ~16% of all modelled cells, which
    no sensible budget accommodates. Splitting on district boundaries keeps
    every district whole (districts nest inside countries), so only that one
    country's *own* aggregate row needs recombining.
    """

    isocodes: tuple
    #: District ids, when this is a slice of a single country; else None.
    shape_ids: tuple = None

    @property
    def is_partial_country(self):
        return self.shape_ids is not None


@dataclass
class ChunkPlan:
    """A partition of the world's countries into memory-sized work units."""

    #: One :class:`ChunkSelection` per chunk.
    chunks: list
    #: Post-filter row count per chunk, parallel to :attr:`chunks`.
    chunk_rows: list
    #: ``(min, max)`` slope in degrees over every surviving cell.
    slope_range: tuple
    #: ``(min, max)`` country average body mass over every surviving country.
    mass_range: tuple
    #: Grid cell edge length in degrees (ADR-8).
    cell_size_degrees: float
    #: Total surviving rows.
    total_rows: int
    #: Per-ISOCODE ``pop_density`` totals, for chunks holding a partial country.
    #: Totals of the *area-weighted* weight when :attr:`area_weighted_population`
    #: is set, since that is what the run will normalise against.
    density_totals: dict = field(default_factory=dict)
    #: Diagnostics for the run header.
    notes: dict = field(default_factory=dict)
    #: Whether the plan was built with cos-latitude area weighting. The survivor
    #: set (and hence the chunk sizes, the pinned LUT ranges and the density
    #: totals) depends on it, so a plan may only be reused by a run with the
    #: same setting.
    area_weighted_population: bool = True

    def __len__(self):
        return len(self.chunks)

    @property
    def split_countries(self):
        """ISOCODEs that had to be spread over more than one chunk."""
        return sorted(
            {
                selection.isocodes[0]
                for selection in self.chunks
                if selection.is_partial_country
            }
        )

    def describe(self):
        """A short human-readable summary for a run log."""
        largest = max(self.chunk_rows) if self.chunk_rows else 0
        split = self.split_countries
        return (
            f"{len(self.chunks)} chunks over {len(self.notes.get('countries', []))} "
            f"countries, {self.total_rows:,} modelled cells, "
            f"largest chunk {largest:,} rows "
            f"(largest single country {self.notes.get('largest_country_rows', 0):,} "
            f"= {self.notes.get('largest_country', '?')}"
            + (f", split by district: {', '.join(split)}" if split else "")
            + f"), cell {self.cell_size_degrees * 3600:.0f} arcsec"
            + (", cos-lat area weighted" if self.area_weighted_population else "")
        )


def read_parquet_cell_size(path):
    """Read ``cell_size_degrees`` from a Parquet footer, if it is there.

    ``scripts/extraction/extract_grid.py`` stamps the master-grid cell size into
    the file metadata precisely so a consumer can apply the ADR-8 population
    threshold without reading the coordinate columns.

    Returns:
        float or None.
    """
    try:
        import pyarrow.parquet as pq

        metadata = pq.ParquetFile(str(path)).schema_arrow.metadata or {}
    except Exception:
        return None
    raw = metadata.get(b"cell_size_degrees")
    if raw is None:
        return None
    try:
        return float(raw.decode())
    except (ValueError, AttributeError):
        return None


def _is_parquet(path):
    candidate = gis.Path(path)
    return candidate.is_dir() or candidate.suffix.lower() in gis.PARQUET_SUFFIXES


def _read_columns(path, columns, isocodes=None):
    """Read a column subset, optionally restricted to a set of countries."""
    if _is_parquet(path):
        filters = None
        if isocodes is not None:
            filters = [("ISOCODE", "in", list(isocodes))]
        frame = pd.read_parquet(path, columns=list(columns), filters=filters)
        return gis.optimise_zone_dtypes(frame)
    frame = pd.read_csv(path, usecols=list(columns))
    if isocodes is not None:
        frame = frame[frame["ISOCODE"].isin(list(isocodes))]
    return gis.optimise_zone_dtypes(frame)


def plan_chunks(
    zone_file,
    country_file=None,
    max_rows_per_chunk=DEFAULT_CHUNK_ROWS,
    cell_size_degrees=None,
    area_weighted_population=True,
):
    """Work out which countries go in which chunk, and the global LUT ranges.

    Reads three columns of the zone frame -- ISOCODE, pop_density, slope_1 --
    and replays just enough of :func:`gis_global_module.merge_and_adjust_population`
    (the per-country density normalisation and the ADR-8 population filter) to
    know how many cells each country actually contributes. Countries are then
    bin-packed largest-first into chunks under ``max_rows_per_chunk``; a country
    bigger than the budget on its own becomes its own chunk, which is the
    smallest unit this scheme can produce.

    Args:
        zone_file: CSV or Parquet zone frame.
        country_file: the country table. Defaults to the module's.
        max_rows_per_chunk (int): budget, in post-filter rows.
        cell_size_degrees (float, optional): overrides both the Parquet footer
            and the ``left``/``right`` inference.
        area_weighted_population (bool, optional): normalise population by
            ``pop_density x cos(latitude)`` rather than ``pop_density`` alone,
            matching :func:`gis_global_module.merge_and_adjust_population`'s
            argument of the same name. Must match the setting the run itself
            uses: it changes the survivor set, the chunk sizes and the density
            totals handed to split countries. Defaults to True.

    Returns:
        ChunkPlan
    """
    country_file = country_file or gis.COUNTRY_DATA_FILE
    df_input = pd.read_csv(country_file)

    if cell_size_degrees is None:
        cell_size_degrees = read_parquet_cell_size(zone_file)

    columns = list(PLANNING_COLUMNS)
    if area_weighted_population:
        # The pre-pass has to reproduce the run's weight exactly, so it needs
        # the cell-centre latitude too.
        columns += [c for c in LATITUDE_COLUMNS if c not in columns]
    if cell_size_degrees is None:
        # Fall back to the column-based inference (ADR-8) -- the CSV path, and
        # any Parquet without the footer stamp.
        columns += ["left", "right"]

    frame = _read_columns(zone_file, columns)
    if cell_size_degrees is None:
        cell_size_degrees = gis.infer_cell_size_degrees(frame)
    minimum_pop = gis.minimum_population_per_cell(cell_size_degrees)

    # Reproduce the inner merge's country selection and the per-country density
    # normalisation, without materialising the merge itself. The arithmetic is
    # deliberately the *verbatim* expression from merge_and_adjust_population,
    # not an equivalent one: the survivor set decides the global slope range,
    # which the LUT grid is pinned to, so a last-bit difference here would show
    # up as a chunked-vs-monolithic mismatch.
    if df_input["alpha3"].duplicated().any():
        raise ValueError(
            "country table has duplicate alpha3 codes; the chunk planner assumes "
            "the ISOCODE->country merge is one-to-one"
        )
    populations = df_input.set_index("alpha3")["Population"]
    weights = df_input.set_index("alpha3")["Average Weight"]
    # Everything here stays categorical. Calling .astype(object) on ISOCODE
    # would materialise one Python string pointer per raw grid cell -- 2+ GB at
    # 30 arc-seconds, in the one pass that is supposed to be cheap.
    keep = frame["ISOCODE"].isin(populations.index).to_numpy()
    frame = frame[keep].reset_index(drop=True)
    for column in ("ISOCODE", "shapeID"):
        if isinstance(frame[column].dtype, pd.CategoricalDtype):
            frame[column] = frame[column].cat.remove_unused_categories()
    iso = frame["ISOCODE"]

    # Capture the per-country divisor as well as using it. A chunk holding only
    # part of a country has to be handed this, or it would normalise against its
    # own fragment (see merge_and_adjust_population's density_totals). The numpy
    # scalar is kept as-is, *not* coerced to a Python float: pop_density is
    # float32, and dividing by a float64 would promote the result and stop it
    # matching a whole-world run.
    split_weights = gis.population_split_weights(
        frame, area_weighted_population=area_weighted_population
    )
    density_totals = {
        str(key): group.sum()
        for key, group in split_weights.groupby(iso, observed=True, sort=True)
    }

    def _share(x):
        total = density_totals[x.name]
        return x / total if total > 0 else x * 0

    density_share = split_weights.groupby(
        iso, group_keys=False, observed=True, sort=True
    ).apply(_share)
    # Look the national population up through the categorical codes rather than
    # Series.map, which on a categorical returns another categorical and cannot
    # then be multiplied. This also keeps the int64 dtype the merge would give,
    # so the float32 * int64 -> float64 promotion matches.
    population_by_category = populations.reindex(iso.cat.categories).to_numpy()
    pop_values = pd.Series(
        population_by_category[iso.cat.codes.to_numpy()], index=iso.index
    )
    pop_zone = density_share * pop_values

    survivors = (pop_zone > minimum_pop).to_numpy()
    surviving_iso = iso[survivors]
    surviving_slope = frame["slope_1"].to_numpy(dtype=float)[survivors]

    surviving_shape = frame["shapeID"][survivors]
    counts = surviving_iso.value_counts()
    counts = counts[counts > 0].sort_values(ascending=False)
    if counts.empty:
        raise ValueError("No cells survive the population filter; nothing to chunk.")

    slope_range = (float(np.nanmin(surviving_slope)), float(np.nanmax(surviving_slope)))
    surviving_weights = weights.reindex(list(counts.index)).dropna()
    mass_range = (float(surviving_weights.min()), float(surviving_weights.max()))

    # Greedy first-fit-decreasing bin packing. Countries are few (~200) and the
    # sizes are wildly uneven, so anything cleverer buys nothing.
    budget = max(int(max_rows_per_chunk), 1)
    oversized = [str(code) for code, rows in counts.items() if int(rows) > budget]

    chunks, chunk_rows = [], []
    for isocode, rows in counts.items():
        rows = int(rows)
        if rows > budget:
            continue  # handled below, district by district
        for index in range(len(chunks)):
            if chunks[index].is_partial_country:
                continue
            if chunk_rows[index] + rows <= budget:
                chunks[index] = ChunkSelection(
                    isocodes=chunks[index].isocodes + (isocode,)
                )
                chunk_rows[index] += rows
                break
        else:
            chunks.append(ChunkSelection(isocodes=(isocode,)))
            chunk_rows.append(rows)

    # A country bigger than a whole chunk gets split along district boundaries.
    # Districts are the finest key the aggregation uses, so they stay whole and
    # the district frame is still exact; only the country's own summary row is
    # recombined afterwards (see combine_partial_countries).
    district_counts = {}
    for isocode in oversized:
        mask = (surviving_iso == isocode).to_numpy()
        # value_counts on a categorical reports *every* category, so the zero
        # rows here are other countries' districts. Dropping them matters: a
        # foreign shapeID left in this country's bin would be pulled into the
        # chunk by the shapeID filter and counted twice.
        sizes = surviving_shape[mask].value_counts()
        sizes = sizes[sizes > 0].sort_values(ascending=False)
        district_counts[isocode] = {str(k): int(v) for k, v in sizes.items()}
        bins, bin_rows = [], []
        for shape_id, rows in sizes.items():
            rows = int(rows)
            for index in range(len(bins)):
                if bin_rows[index] + rows <= budget:
                    bins[index].append(str(shape_id))
                    bin_rows[index] += rows
                    break
            else:
                bins.append([str(shape_id)])
                bin_rows.append(rows)
        for shape_ids, rows in zip(bins, bin_rows):
            chunks.append(
                ChunkSelection(isocodes=(isocode,), shape_ids=tuple(shape_ids))
            )
            chunk_rows.append(rows)

    return ChunkPlan(
        chunks=chunks,
        chunk_rows=chunk_rows,
        slope_range=slope_range,
        mass_range=mass_range,
        cell_size_degrees=float(cell_size_degrees),
        total_rows=int(counts.sum()),
        density_totals=density_totals,
        area_weighted_population=bool(area_weighted_population),
        notes={
            "countries": list(counts.index),
            "largest_country": str(counts.index[0]),
            "largest_country_rows": int(counts.iloc[0]),
            "budget_rows": budget,
            "oversized_countries": oversized,
            "largest_district_rows": max(
                (max(sizes.values()) for sizes in district_counts.values()),
                default=0,
            ),
        },
    )


def _chunk_columns(zone_file, area_weighted_population=True):
    """Which zone columns a chunk needs to read.

    The cell size comes from the plan, so the four float64 coordinate columns
    can be skipped -- worth ~37% of the frame at 30 arc-seconds. Only possible
    for Parquet; a CSV read is all-or-nothing here because the column set has to
    survive :func:`gis_global_module.optimise_zone_dtypes` unchanged.

    Area weighting needs the cell-centre latitude, so ``top``/``bottom`` come
    back when it is on; ``left``/``right`` still do not.
    """
    if not _is_parquet(zone_file):
        return None  # everything
    import pyarrow.parquet as pq

    if gis.Path(zone_file).is_dir():
        return None
    dropped = set(COORDINATE_COLUMNS)
    if area_weighted_population:
        dropped -= set(LATITUDE_COLUMNS)
    available = [f.name for f in pq.ParquetFile(str(zone_file)).schema_arrow]
    return [c for c in available if c not in dropped]


def run_chunked_global_analysis(
    crr_adjustment,
    time_gathering_water,
    practical_limit_bicycle,
    practical_limit_buckets,
    met,
    watts,
    hill_polarity,
    urban_adjustment,
    rural_adjustment,
    calculate_distance=True,
    human_mass=62,
    use_sample_data=False,
    chunk_rows=DEFAULT_CHUNK_ROWS,
    plan=None,
    zone_file=None,
    progress=True,
    spill_dir=None,
    area_weighted_population=True,
):
    """Run the global analysis one country-group at a time.

    Produces country and district frames identical to
    :func:`gis_global_module.run_global_analysis` on the same input and
    parameters, at a peak memory set by ``chunk_rows`` rather than by the size
    of the grid.

    Args:
        crr_adjustment ... rural_adjustment: as ``run_global_analysis``.
        calculate_distance (bool): must be True; the cached-velocity branch is
            not chunk-aware.
        human_mass (float): as ``run_global_analysis``.
        use_sample_data (bool): as ``run_global_analysis``.
        chunk_rows (int): post-filter row budget per chunk.
        plan (ChunkPlan, optional): reuse a plan instead of recomputing it. The
            plan depends only on the input file, so a multi-draw run should
            build it once.
        zone_file (optional): overrides the module-level input path.
        progress (bool): print a line per chunk.
        spill_dir (optional): where to park the global summary row's four
            weighted-statistic columns between chunks. Defaults to a temporary
            directory. Keeping them in memory instead would add
            ``32 bytes x every modelled cell`` -- 4.5 GB at 30 arc-seconds --
            to the peak *while a chunk frame is also live*, which is the worst
            possible moment for it.
        area_weighted_population (bool): see
            :func:`gis_global_module.merge_and_adjust_population`. A supplied
            ``plan`` must agree with it.

    Returns:
        tuple: ``(df_countries, df_districts)``.
    """
    if not calculate_distance:
        raise ValueError(
            "run_chunked_global_analysis requires calculate_distance=True; the "
            "pre-computed-velocity branch reads whole-world CSVs."
        )

    zone_file = zone_file or gis.zone_input_file(use_sample_data)
    if plan is None:
        plan = plan_chunks(
            zone_file,
            max_rows_per_chunk=chunk_rows,
            area_weighted_population=area_weighted_population,
        )
    elif bool(plan.area_weighted_population) != bool(area_weighted_population):
        # The plan's density totals, survivor set and pinned LUT ranges are all
        # computed under one weighting or the other. Reusing the wrong one would
        # silently normalise split countries against the wrong divisor.
        raise ValueError(
            "chunk plan was built with area_weighted_population="
            f"{plan.area_weighted_population}, but the run asks for "
            f"{bool(area_weighted_population)}; rebuild the plan."
        )

    df_input = _compact_country_table(pd.read_csv(gis.COUNTRY_DATA_FILE))
    columns = _chunk_columns(
        zone_file, area_weighted_population=area_weighted_population
    )

    country_parts, district_parts = [], []
    weighted_files, country_weighted_files = [], []
    global_sums = {key: 0.0 for key in gis.GLOBAL_SUM_COLUMNS}

    if spill_dir is not None:
        Path(spill_dir).mkdir(parents=True, exist_ok=True)
        spill = nullcontext(str(spill_dir))
    else:
        spill = tempfile.TemporaryDirectory(prefix="water-access-chunks-")

    started = time.perf_counter()
    with spill as spill_root, solver_lut.pinned_grid_ranges(
        slope_range=plan.slope_range, mass_range=plan.mass_range
    ), solver_lut.force_lut(), solver_lut.lut_cache():
        split_countries = set(plan.split_countries)
        for index, selection in enumerate(plan.chunks):
            chunk_started = time.perf_counter()
            df_zones_input = _read_chunk(zone_file, columns, selection)

            df_zones = gis.preprocess_frame(
                df_zones_input,
                df_input,
                crr_adjustment=crr_adjustment,
                urban_adjustment=urban_adjustment,
                rural_adjustment=rural_adjustment,
                cell_size_degrees=plan.cell_size_degrees,
                # Only a partial country needs the national divisor supplied;
                # a whole-country chunk derives the identical value itself, and
                # leaving it to do so keeps that path byte-for-byte the one the
                # monolithic run takes.
                density_totals=(
                    plan.density_totals if selection.is_partial_country else None
                ),
                area_weighted_population=area_weighted_population,
            )
            del df_zones_input
            df_zones = df_zones[list(CHUNK_KEEP_COLUMNS)].copy()

            df_zones = gis.calculate_and_merge_bicycle_distance(
                df_zones,
                calculate_distance=True,
                export_file_location=gis.EXPORT_FILE_LOCATION,
                practical_limit_bicycle=practical_limit_bicycle,
                watts=watts,
                human_mass=human_mass,
                hill_polarity=hill_polarity,
            )
            df_zones = gis.calculate_and_merge_walking_distance(
                df_zones,
                calculate_distance=True,
                export_file_location=gis.EXPORT_FILE_LOCATION,
                practical_limit_buckets=practical_limit_buckets,
                met=met,
                human_mass=human_mass,
                hill_polarity=hill_polarity,
            )
            df_zones = gis.process_zones_for_water_access(
                df_zones, time_gathering_water=time_gathering_water
            )

            district_parts.append(
                gis.process_district_data(
                    df_zones[list(gis.DISTRICT_ZONE_COLUMNS)].copy()
                )
            )
            country_parts.append(gis.country_partials(df_zones))

            for key, column in gis.GLOBAL_SUM_COLUMNS.items():
                global_sums[key] += float(df_zones[column].sum())

            spill_path = Path(spill_root) / f"global_weighted_{index:05d}.parquet"
            df_zones[list(gis.GLOBAL_WEIGHTED_COLUMNS)].reset_index(
                drop=True
            ).to_parquet(spill_path, index=False)
            weighted_files.append(spill_path)

            # Split countries need their own cells replayed too, so their
            # weighted statistics can be recomputed over the whole country.
            if selection.is_partial_country:
                country_path = (
                    Path(spill_root) / f"country_weighted_{index:05d}.parquet"
                )
                df_zones[["ISOCODE", *gis.GLOBAL_WEIGHTED_COLUMNS]].reset_index(
                    drop=True
                ).to_parquet(country_path, index=False)
                country_weighted_files.append(country_path)

            rows = len(df_zones)
            del df_zones
            if progress:
                label = (
                    f"{selection.isocodes[0]} "
                    f"({len(selection.shape_ids)} districts)"
                    if selection.is_partial_country
                    else f"{len(selection.isocodes)} countries"
                )
                print(
                    f"  chunk {index + 1}/{len(plan.chunks)}: {label}, "
                    f"{rows:,} cells, "
                    f"{time.perf_counter() - chunk_started:.1f}s",
                    flush=True,
                )

        # The global row's weighted statistics, replayed from the spill. This
        # runs while no chunk frame is alive, so it gets the whole budget to
        # itself -- which it needs, because weighted_median sorts what it is
        # given.
        df_weighted = pd.concat(
            [pd.read_parquet(path) for path in weighted_files], ignore_index=True
        )
        df_global = gis.aggregate_global(df_weighted, sums=global_sums)
        del df_weighted

        df_country_weighted = (
            pd.concat(
                [pd.read_parquet(path) for path in country_weighted_files],
                ignore_index=True,
            )
            if country_weighted_files
            else None
        )

    # ---- global finalisation ------------------------------------------------
    # groupby sorts, so a whole-world run emits countries in ISOCODE order and
    # districts in shapeID order; a stable sort of the concatenated chunks
    # reproduces that exactly.
    df_countries = combine_partial_countries(
        pd.concat(country_parts, ignore_index=True), df_country_weighted
    )
    del country_parts

    df_countries = gis.finalise_country_data(df_countries, df_global)

    df_districts = (
        pd.concat(district_parts, ignore_index=True)
        .sort_values("shapeID", kind="stable")
        .reset_index(drop=True)
    )
    del district_parts
    df_districts = df_districts[df_districts["ISOCODE"].isin(df_countries["ISOCODE"])]

    if progress:
        print(
            f"  chunked analysis complete in {time.perf_counter() - started:.1f}s",
            flush=True,
        )
    return gis.decategorise(df_countries), gis.decategorise(df_districts)


def _compact_country_table(df_input):
    """Categorise the country table's text columns before it is merged.

    Measured: this is the single biggest lever on peak memory. The country merge
    broadcasts Entity/region/subregion/Continent/the alpha codes onto every cell,
    and ``compact_zone_frame`` only categorises them at the *end* of
    preprocessing -- so for the duration of the merge the frame carries nine
    object columns, one Python string pointer per cell. Categorising the right
    side first means the merge produces categoricals directly and that spike
    never happens.

    ``alpha3`` is left alone: it is the join key, and the merge is already
    handling a categorical-vs-object key mismatch on the other side.
    """
    for column in df_input.columns:
        if df_input[column].dtype == object and column != "alpha3":
            df_input[column] = df_input[column].astype("category")
    return df_input


def _read_chunk(zone_file, columns, selection):
    """Read one chunk's rows, restricted to ``columns`` where possible.

    Parquet reads push the selection down into Arrow, so peak memory is the
    *filtered* result plus a row group of decode buffers. A CSV has to be read
    whole and filtered afterwards -- fine for the 5 arc-minute grid, where the
    whole file fits comfortably, and the reason memory-bounded chunking needs
    the Parquet input (ADR-2).
    """
    if selection.is_partial_country:
        column, values = "shapeID", list(selection.shape_ids)
    else:
        column, values = "ISOCODE", list(selection.isocodes)

    if _is_parquet(zone_file):
        frame = pd.read_parquet(
            zone_file,
            columns=list(columns) if columns else None,
            filters=[(column, "in", values)],
        )
        return gis.optimise_zone_dtypes(frame.reset_index(drop=True))

    frame = pd.read_csv(zone_file, usecols=list(columns) if columns else None)
    frame = frame[frame[column].isin(values)].reset_index(drop=True)
    return gis.optimise_zone_dtypes(frame)


def combine_partial_countries(country_parts, weighted_by_country):
    """Fold per-chunk country rows into one row per country.

    Countries that fitted in a single chunk contribute exactly one row, and
    summing a single value or taking the first of a single value returns it
    unchanged -- so those rows come through bit-identical. Countries that had to
    be split contribute several rows, whose population columns add. Their seven
    population-weighted statistics cannot be added, so they are recomputed from
    the country's replayed cells in ``weighted_by_country``.

    Args:
        country_parts (DataFrame): concatenated :func:`gis.country_partials`
            output from every chunk.
        weighted_by_country (DataFrame or None): the split countries' cells,
            carrying ISOCODE plus :data:`gis.GLOBAL_WEIGHTED_COLUMNS`.

    Returns:
        DataFrame: one row per country, ISOCODE-sorted.
    """
    sum_columns = [
        "country_pop_raw",
        "zone_pop_with_water",
        "zone_pop_without_water",
        "population_piped_with_access",
        "population_piped_with_cycling_access",
        "population_piped_with_walking_access",
        "population_piped_with_only_cycling_access",
    ]
    aggregation = {
        column: ("sum" if column in sum_columns else "first")
        for column in country_parts.columns
        if column != "ISOCODE"
    }
    combined = (
        country_parts.groupby("ISOCODE", observed=True, sort=True)
        .agg(aggregation)
        .reset_index()
    )
    combined = combined[list(country_parts.columns)]

    if weighted_by_country is not None and not weighted_by_country.empty:
        recomputed = gis.calculate_weighted_results(weighted_by_country)
        recomputed = recomputed.set_index("ISOCODE")
        target = combined["ISOCODE"].astype(object)
        for column in recomputed.columns:
            replacement = target.map(recomputed[column])
            combined[column] = replacement.where(replacement.notna(), combined[column])

    return combined
