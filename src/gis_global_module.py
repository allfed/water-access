import pathlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px

# import weightedstats as ws
import sys
from pathlib import Path
import pdb
import os
import time
import warnings

import src.solver_lut as solver_lut

# ## Import Data from CSVs.
# CSVs created in previous script, which did the cycling mobility on a per country basis
#
# ### GIS Data From QGIS Export
# https://ghsl.jrc.ec.europa.eu/download.php?ds=smod <- urbanisation
#
# https://www.earthenv.org/topography <- topography
#
# https://www.globio.info/download-grip-dataset <- roads


##############################################################################################################################
#
# PRE-PROCESSING
#
##############################################################################################################################

# Assuming the script is in the src directory, set the script_dir accordingly
script_dir = Path(__file__).resolve().parent
repo_root = script_dir.parent
results_dir = repo_root / "results"

# Define paths relative to the src directory
URB_DATA_FILE = repo_root / "data" / "GIS" / "updated_GIS_output_cleaned.csv"
URB_DATA_FILE_SAMPLE = repo_root / "data" / "GIS" / "GIS_data_zones_sample.csv"

COUNTRY_DATA_FILE = repo_root / "data" / "processed" / "merged_data.csv"
EXPORT_FILE_LOCATION = repo_root / "data" / "processed"
CRR_FILE = repo_root / "data" / "lookup tables" / "Crr.csv"
FILE_PATH_PARAMS = (
    repo_root / "data" / "lookup tables" / "mobility-model-parameters.csv"
)

# Results File Path
DISTRICT_RESULTS_FILE_PATH = results_dir / "district_results_single_run.csv"
COUNTRY_RESULTS_FILE_PATH = results_dir / "country_results_single_run.csv"


def weighted_mean(var, wts):
    if len(var) == 0 or len(wts) == 0:
        return 0
    elif len(var) != len(wts):
        raise TypeError("Shapes of var and wts must be the same.")
    elif np.sum(wts) == 0:
        raise ZeroDivisionError("Weights sum to zero, can't be normalized.")
    else:
        return np.average(var, weights=wts)


def weighted_median_series(val, weight):
    """Calculates the weighted median
    ArithmeticError
    If the sum of the weights is zero, or if the weights are not positive.
    """
    try:
        if len(val) == 0 or len(weight) == 0:
            return np.nan
        if np.sum(weight) == 0:
            return np.nan

        df = pd.DataFrame({"val": val, "weight": weight})
        df_sorted = df.sort_values("val")
        cumsum = df_sorted["weight"].cumsum()
        cutoff = df_sorted["weight"].sum() / 2.0
        result = df_sorted[cumsum >= cutoff]["val"].iloc[0]
        # return just the value
    except (ValueError, IndexError, KeyError) as e:
        print(f"Warning: weighted_median_series failed with error: {e}")
        result = np.nan
    return result


def weighted_median(df, val_column, weight_column):
    """Calculates the weighted median
    ArithmeticError
    If the sum of the weights is zero, or if the weights are not positive.
    """
    df_sorted = df.sort_values(val_column)
    cumsum = df_sorted[weight_column].cumsum()
    cutoff = df_sorted[weight_column].sum() / 2.0
    return df_sorted[cumsum >= cutoff][val_column].iloc[0]


def run_weighted_median_on_grouped_df(df, groupby_column, value_column, weight_column):
    """Calculate the weighted median of a dataframe grouped by a column.
    Args:
        df (pandas.DataFrame): DataFrame to calculate weighted median on.
        groupby_column (str): Column to group by.
        value_column (str): Column to calculate weighted median on.
        weight_column (str): Column to use as weight.
    Returns:
        pandas.DataFrame: DataFrame with weighted median for each group.
    """
    # initialize empty list
    d = []
    # loop through each group
    for i in df[groupby_column].unique():
        df_group = df[df[groupby_column] == i]
        # if rows in dataframe are more than 1, calculate weighted median
        if len(df_group) > 1:
            median = weighted_median(df_group, value_column, weight_column)
        else:
            median = df_group[value_column].values[0]
        d.append(
            {
                groupby_column: i,
                "median": median,
            }
        )
    return pd.DataFrame(d)


#: Wall-clock seconds the most recent :func:`load_data` call spent reading the
#: zone frame off disk. Records load time so benchmarks can measure solver/refactor time excluding the
#: initial data load, and the load happens several frames down inside
#: ``run_global_analysis``, so it is recorded here rather than plumbed through
#: every signature.
LAST_LOAD_SECONDS = 0.0

#: File extensions that :func:`load_data` reads as Parquet . Anything
#: else is read as CSV, byte-for-byte as before.
PARQUET_SUFFIXES = (".parquet", ".parq", ".pq")


#: Zone columns stored as float32 rather than float64 . These are raster
#: samples with at most ~4 significant digits of real information, so float32 is
#: lossless in any meaningful sense and halves the frame. Population *counts*,
#: which get summed over millions of cells, stay float64.
FLOAT32_ZONE_COLUMNS = (
    "left",
    "top",
    "right",
    "bottom",
    "GHS_SMOD",
    "pop_density",
    "dtw_1",
    "grip_1_1",
    "grip_2_1",
    "grip_3_1",
    "grip_4_1",
    "grip_5_1",
    "slope_1",
)

#: Zone columns stored as categoricals . At 30 arc-seconds these are
#: ~140M repetitions of a few thousand short strings; as objects they alone cost
#: more than the whole numeric frame.
CATEGORICAL_ZONE_COLUMNS = ("shapeName", "shapeID", "ISOCODE")


def optimise_zone_dtypes(df_zones_input):
    """Shrink a freshly loaded zone frame in place .

    float64 rasters become float32, repeated identifier strings become
    categoricals, and the NaN-indicator flags become ``int8``. At 5 arc-minutes
    this is a convenience; at 30 arc-seconds (~140M rows) it is the difference
    between fitting in 32 GB and not.

    Args:
        df_zones_input (DataFrame): the zone frame as read off disk.

    Returns:
        DataFrame: the same frame, with narrowed dtypes.
    """
    for column in FLOAT32_ZONE_COLUMNS:
        if (
            column in df_zones_input.columns
            and df_zones_input[column].dtype == "float64"
        ):
            df_zones_input[column] = df_zones_input[column].astype("float32")

    for column in CATEGORICAL_ZONE_COLUMNS:
        if column in df_zones_input.columns and not isinstance(
            df_zones_input[column].dtype, pd.CategoricalDtype
        ):
            df_zones_input[column] = df_zones_input[column].astype("category")

    for column in df_zones_input.columns:
        if str(column).endswith("_nan") or column == "any_nan":
            if pd.api.types.is_numeric_dtype(df_zones_input[column]):
                df_zones_input[column] = df_zones_input[column].astype("int8")

    return df_zones_input


def _read_zone_frame(path):
    """Read a zone frame from Parquet or CSV, dispatching on the file extension.

    Parquet/CSV dispatch: the 30 arc-second extract is Parquet (a CSV would be ~40 GB) but
    carries the identical column contract, so everything downstream is unchanged.
    A directory is treated as a partitioned Parquet dataset.
    """
    candidate = Path(path)
    if candidate.is_dir() or candidate.suffix.lower() in PARQUET_SUFFIXES:
        return optimise_zone_dtypes(pd.read_parquet(path))
    return optimise_zone_dtypes(pd.read_csv(path))


def load_data(urb_data_file, country_data_file):
    """
    Load data from CSV (or Parquet) files.

    Parameters:
    - urb_data_file (str): The file path of the urban data file. CSV by default;
      a ``.parquet``/``.parq``/``.pq`` file, or a directory holding a partitioned
      Parquet dataset, is read as Parquet instead .
    - country_data_file (str): The file path of the country data CSV file.

    Returns:
    - df_zones_input (pandas.DataFrame): The loaded urban data as a DataFrame.
    - df_input (pandas.DataFrame): The loaded country data as a DataFrame.
    """
    global LAST_LOAD_SECONDS
    started = time.perf_counter()
    try:
        df_zones_input = _read_zone_frame(urb_data_file)
        df_input = pd.read_csv(country_data_file)
    except (FileNotFoundError, pd.errors.EmptyDataError) as e:
        # Try relative path as fallback
        try:
            df_zones_input = _read_zone_frame(Path(".") / urb_data_file)
            df_input = pd.read_csv(Path(".") / country_data_file)
        except Exception as fallback_e:
            raise FileNotFoundError(
                f"Could not load data files: {e}, fallback error: {fallback_e}"
            )
    LAST_LOAD_SECONDS = time.perf_counter() - started

    return df_zones_input, df_input


# Function for managing urban/rural data
def manage_urban_rural(df_zones_input):
    """
    Converts the 'dtw_1' column from meters to kilometers and creates a new binary column
    'urban_rural' based on the 'GHS_SMOD' column. Rural values are below 15, while urban values
    are above 15.

    Args:
        df_zones_input (pandas.DataFrame): Input DataFrame containing the 'dtw_1' and 'GHS_SMOD' columns.

    Returns:
        pandas.DataFrame: DataFrame with the 'dtw_1' column converted to kilometers and a new 'urban_rural'
        column indicating whether a zone is urban (1) or rural (0).
    """
    # Convert dtw_1 from meters to kilometers
    df_zones_input["dtw_1"] /= 1000
    # Manage Urban / Rural Data
    # Use the GHS_SMOD_E2020_GLOBE_R2023A_54009_1000_V1_0 dataset.
    # from here: https://ghsl.jrc.ec.europa.eu/download.php?ds=smod
    # create new binary column for urban / rural. Rural is below 15, Urban above 15
    df_zones_input["urban_rural"] = np.where(df_zones_input["GHS_SMOD"] > 15, 1, 0)
    return df_zones_input


def adjust_euclidean(df_zones_input, urban_adjustment, rural_adjustment):
    """
    Adjusts distance to water to account for people having to travel on roads and paths instead of
    in straight lines (Euclidean distance), with different adjustments for urban and rural areas.

    Parameters:
    - df_zones_input (pd.DataFrame): DataFrame of GIS zones data
    - urban_adjustment (float): Adjustment ratio for urban areas
    - rural_adjustment (float): Adjustment ratio for rural areas
    """
    # Boolean indexing for urban areas
    urban_mask = df_zones_input["urban_rural"] == 1
    df_zones_input.loc[urban_mask, "dtw_1"] *= urban_adjustment

    # Boolean indexing for rural areas
    rural_mask = df_zones_input["urban_rural"] == 0
    df_zones_input.loc[rural_mask, "dtw_1"] *= rural_adjustment

    # Optionally, raise an error if there are invalid values
    if not df_zones_input["urban_rural"].isin([0, 1]).all():
        raise ValueError("Invalid value (not 0 or 1) in 'urban_rural' column")

    return df_zones_input


# Function for managing slope
def manage_slope(df_zones_input):
    """
    Perform slope management on the input DataFrame.

    Args:
        df_zones_input (pandas.DataFrame): The input DataFrame containing the slope data.

    Returns:
        pandas.DataFrame: The modified DataFrame after performing slope management.
    """
    axes = df_zones_input["slope_1"].hist(bins=100, log=False)
    df_zones_input["slope_1"].quantile(np.arange(0, 1, 0.05))
    # The histogram is a diagnostic side effect nobody collects. Closing its
    # figure matters once this runs hundreds of times in one process, as the
    # chunked driver does (chunks x draws): matplotlib keeps every unclosed
    # figure alive forever.
    try:
        plt.close(axes.get_figure())
    except AttributeError:  # pragma: no cover - only if hist() returns an array
        plt.close("all")
    return df_zones_input


# -----------------------------------------------------------------------------
# Resolution-aware population filter: the "does this cell hold anybody?" filter is a DENSITY threshold, not a
# headcount.
#
# The model has always dropped cells holding <= 10 people. At 5 arc-minutes a
# cell is roughly 85 km^2 at the equator, so 10 people is about 0.1 person/km^2
# -- an "effectively uninhabited" bar. Carried over verbatim to a 30 arc-second
# grid, where cells are 100x smaller in area, the same 10 people would mean
# ~10 people/km^2 and would silently delete most of the rural tail, which is
# precisely the population this model exists to describe.
#
# So the threshold scales with cell area: 10 people at 5 arc-minutes (i.e. the
# historical behaviour, bit-for-bit), 0.1 people at 30 arc-seconds.
# -----------------------------------------------------------------------------

# Cell edge length of the reference (5 arc-minute) grid, in degrees.
FIVE_ARCMIN_DEGREES = 300 / 3600

# People per cell below which a 5-arc-minute cell is treated as uninhabited.
MIN_POP_PER_CELL_5_ARCMIN = 10


def infer_cell_size_degrees(df_zones_input, default=FIVE_ARCMIN_DEGREES):
    """Infer the grid cell edge length, in degrees, from a zone DataFrame.

    Uses the ``left``/``right`` cell-bound columns that the GIS extract carries.
    The measured width is snapped to whole arc-seconds, both because every grid
    this model consumes is defined in whole arc-seconds and because snapping
    makes the 5-arc-minute case land on exactly ``FIVE_ARCMIN_DEGREES`` -- which
    is what keeps :func:`minimum_population_per_cell` bit-identical to the old
    hard-coded threshold despite the CSV's truncated coordinate text.

    Args:
        df_zones_input (DataFrame): zone frame, ideally with ``left``/``right``.
        default (float): value to return when the bounds are unavailable or
            degenerate (e.g. the minimal frames used in unit tests).

    Returns:
        float: cell edge length in degrees.
    """
    if not {"left", "right"}.issubset(df_zones_input.columns):
        return default

    widths = (df_zones_input["right"] - df_zones_input["left"]).abs()
    widths = widths[widths > 0]
    if widths.empty:
        return default

    arcseconds = round(float(widths.median()) * 3600)
    if arcseconds <= 0:
        return default

    return arcseconds / 3600


def minimum_population_per_cell(cell_size_degrees):
    """People-per-cell threshold below which a cell counts as uninhabited.

    Scales the historical 5-arc-minute threshold by the cell AREA ratio, so the
    implied population density is the same at every resolution . At
    ``cell_size_degrees == FIVE_ARCMIN_DEGREES`` the ratio is exactly 1.0 and
    the result is exactly ``MIN_POP_PER_CELL_5_ARCMIN``.

    Args:
        cell_size_degrees (float): cell edge length in degrees.

    Returns:
        float: threshold in people per cell. 10 at 5 arc-minutes, 0.1 at
        30 arc-seconds.
    """
    area_ratio = (cell_size_degrees / FIVE_ARCMIN_DEGREES) ** 2
    return MIN_POP_PER_CELL_5_ARCMIN * area_ratio


#: Columns the cos-latitude area weight needs. The 22-column input contract
#: guarantees them; only the minimal frames used in unit tests lack them.
LATITUDE_COLUMNS = ("top", "bottom")


def cell_centre_latitude(df):
    """Latitude of each cell's centre, from the ``top``/``bottom`` bounds.

    Args:
        df (DataFrame): a zone frame carrying the ``top`` and ``bottom`` cell
            bounds (both are part of the 22-column input contract).

    Returns:
        Series: cell-centre latitude in degrees, in the frame's own dtype
        (``float32`` after :func:`optimise_zone_dtypes`).

    Raises:
        ValueError: if either bound column is missing. There is deliberately no
            fallback: silently assuming a latitude would silently reweight the
            world's population, which is the exact failure this is here to fix.
    """
    missing = [column for column in LATITUDE_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(
            "area-weighted population needs the cell-bound columns "
            f"{list(LATITUDE_COLUMNS)}; missing {missing}. The 22-column zone "
            "contract carries them, so this frame is either a test stub or a "
            "column-trimmed read that needs 'top' and 'bottom' added back."
        )
    return (df["top"] + df["bottom"]) / 2


def population_split_weights(df, area_weighted_population=True):
    """The per-cell weight a country's population is split in proportion to.

    Historically this was population *density* alone. On a lat/long grid that is
    wrong by a factor of the cell's ground area: cells are a constant number of
    degrees wide, so their east-west extent shrinks as ``cos(latitude)`` toward
    the poles, and people-per-cell is ``density x area``. Weighting by density
    alone therefore over-weights a country's poleward cells relative to its
    equatorward ones. National totals are unaffected (the weights are
    renormalised per country); the *placement* within a country is not.

    The area-weighted form is the default as of 2026-08-09 (Kevin sign-off);
    see ``results/validation/cos_lat_diff.md`` for the size of the correction.

    Args:
        df (DataFrame): zone frame carrying ``pop_density`` and ``top``/
            ``bottom`` (the latter two only when area weighting is on).
        area_weighted_population (bool): True (default) weights by
            ``pop_density x cos(latitude)``. False reproduces the historical
            density-only weight exactly, dtype included, and is what the
            pre-2026-08-09 published results were produced with.

    Returns:
        Series: the weight, aligned to ``df``'s index.
    """
    density = df["pop_density"]
    if not area_weighted_population:
        return density
    # float32 in, float32 out (pop_density and the bounds are both float32 after
    # optimise_zone_dtypes), so the per-country sums below stay in the same
    # precision regime as the density-only path -- which is what lets the
    # chunked planner's pre-pass reproduce them bit for bit.
    return density * np.cos(np.radians(cell_centre_latitude(df)))


# Function for merging dataframes and adjusting populations
def merge_and_adjust_population(
    df_zones_input,
    df_input,
    cell_size_degrees=None,
    density_totals=None,
    area_weighted_population=True,
):
    """
    Merge and adjust population data based on zone and country information.

    Args:
        df_zones_input (DataFrame): Input DataFrame containing zone information.
        df_input (DataFrame): Input DataFrame containing country information.
        cell_size_degrees (float, optional): grid cell edge length, for the
            resolution-aware population threshold. Defaults to None, i.e. infer it from the
            frame's ``left``/``right`` columns as before. The 30 arc-second
            Parquet records it in its footer metadata, so a chunked reader can
            supply it here and skip reading the four float64 coordinate columns
            entirely -- they are ~37% of the frame's footprint.
        density_totals (mapping, optional): per-ISOCODE totals of
            ``pop_density`` over the *whole* country. Defaults to None, i.e.
            compute them from the rows present. Only a chunked run that has had
            to split a country across chunks needs this: population density is
            normalised nationally, so a partial country would otherwise
            normalise against its own fragment and invent people. Values must be
            the same dtype the column carries, or the arithmetic will promote
            and stop matching a whole-world run bit for bit. When
            ``area_weighted_population`` is set these must be totals of the
            *area-weighted* weight, i.e. the same quantity
            :func:`population_split_weights` returns.
        area_weighted_population (bool, optional): split each country's
            population in proportion to ``pop_density x cos(latitude)`` rather
            than ``pop_density`` alone -- see
            :func:`population_split_weights`. **Defaults to True** as of
            2026-08-09: weighting by density alone mis-places population within
            any country spanning latitudes, and leaving the correction off by
            default was judged a trap. Pass False for the legacy behaviour,
            which is bit-identical to the code before this argument existed and
            is what every pre-2026-08-09 published figure used.

    Returns:
        DataFrame: DataFrame with merged and adjusted population data.

    Raises:
        None
    """
    assert not df_zones_input.empty, "df_zones_input is empty"
    assert not df_input.empty, "df_input is empty"
    # this analysis loses some data as the overlap between the rasters is not perfect. To reduce this error, use the 30 arc second data. Too much heavy lifting for my computer to do this at the moment.
    # merge df_input and df_zones on ISO_CC. This assigns all the country data to each zone.
    # join inner will remove some of the data that is not in both datasets
    df_zones = df_zones_input.merge(
        df_input, left_on="ISOCODE", right_on="alpha3", how="inner"
    )

    # convert population density to percent of national population on a per country basis, grouped by ISO_CC
    # Prevent division by zero when country population sum is zero
    # NOTE: group_keys=False is required on pandas >= 2.0. Pandas 1.x returned the
    # original (flat) index here; pandas 2.x prepends the group keys, producing a
    # MultiIndex that cannot be assigned back onto df_zones. group_keys=False
    # restores the pandas 1.x result exactly -- values are unchanged.
    # The weight is pop_density by default and pop_density * cos(latitude) when
    # area weighting is on; everything below is indifferent to which.
    split_weights = population_split_weights(
        df_zones, area_weighted_population=area_weighted_population
    )
    grouped = split_weights.groupby(
        df_zones["ISOCODE"], group_keys=False, observed=True, sort=True
    )
    if density_totals is None:
        df_zones["pop_density_perc"] = grouped.apply(
            lambda x: x / x.sum() if x.sum() > 0 else x * 0
        )
    else:
        # Same expression, but with the divisor supplied rather than derived
        # from the rows in hand -- see the density_totals argument.
        def _share(x):
            total = density_totals[x.name]
            return x / total if total > 0 else x * 0

        df_zones["pop_density_perc"] = grouped.apply(_share)
    # multiply population density by population on a per country basis
    df_zones["pop_zone"] = df_zones["pop_density_perc"] * df_zones["Population"]

    # sum the population in each zone
    df_zones["country_pop_raw"] = df_zones.groupby("ISOCODE", observed=True)[
        "pop_zone"
    ].transform("sum")

    # trim the dataframe to only include rows where there is a population.
    # The threshold is resolution-aware : exactly 10 people per cell at
    # 5 arc-minutes, 0.1 at 30 arc-seconds -- the same ~0.1 person/km^2 bar.
    if cell_size_degrees is None:
        cell_size_degrees = infer_cell_size_degrees(df_zones_input)
    min_pop_per_cell = minimum_population_per_cell(cell_size_degrees)
    df_zones["any_pop"] = df_zones["pop_zone"].apply(
        lambda x: 1 if x > min_pop_per_cell else 0
    )
    df_zones = df_zones[df_zones["any_pop"] == 1]

    return df_zones


def decategorise(df):
    """Turn a frame's categorical columns back into plain object columns.

    Categoricals are a *per-cell* memory optimisation . The ~200-row
    country and ~3000-row district aggregates gain nothing from them and lose
    something important: a categorical keeps its full category list after rows
    are filtered out, so a later ``groupby`` on it (as the Monte Carlo result
    processors do) resurrects the dropped countries as empty groups. Handing the
    aggregates back with the dtypes they have always had keeps every downstream
    consumer -- groupbys, pickles, notebooks -- working unchanged.

    Args:
        df (DataFrame): an aggregate frame.

    Returns:
        DataFrame: the frame with no categorical columns.
    """
    categorical = [
        column
        for column in df.columns
        if isinstance(df[column].dtype, pd.CategoricalDtype)
    ]
    if not categorical:
        return df
    return df.astype({column: object for column in categorical})


def compact_zone_frame(df_zones):
    """Shrink a merged zone frame in place .

    Two jobs:

    1. The country merge broadcasts the country table's text columns (Entity,
       region, subregion, Continent, the alpha codes...) onto every single cell,
       and de-categorises ISOCODE on the way through. Measured at 5 arc-minutes
       those object columns are ~60% of the whole frame; at 30 arc-seconds they
       would be tens of GB of pointers to a couple of hundred distinct strings.
    2. pandas groups a categorical over *every* category by default, including
       ones the merge and the population filter just deleted -- which would
       inject empty groups into every downstream groupby. Pruning the category
       lists makes a categorical group exactly like the object column it
       replaced.

    Args:
        df_zones (DataFrame): the merged, filtered zone frame.

    Returns:
        DataFrame: the same frame, with text columns categorised and category
        lists pruned to what is actually present.
    """
    for column in df_zones.columns:
        if df_zones[column].dtype == object:
            df_zones[column] = df_zones[column].astype("category")
        elif isinstance(df_zones[column].dtype, pd.CategoricalDtype):
            df_zones[column] = df_zones[column].cat.remove_unused_categories()
    return df_zones


# Function for road analysis
# def road_analysis_old(df_zones):
#     """
#     Perform road analysis on the given DataFrame.

#     Args:
#         df_zones (pandas.DataFrame): The DataFrame containing the zones data.

#     Returns:
#         pandas.DataFrame: The DataFrame with additional columns for road analysis.

#     """
#     # Define the columns to work with for GRIP data
#     columns = ["grip_1_1", "grip_2_1", "grip_3_1", "grip_4_1", "grip_5_1"]

#     # Create a numpy array from the DataFrame for faster processing
#     data = df_zones[columns].to_numpy()

#     # Find the index of the first non-zero value in each row
#     non_zero_indices = np.argmax(data != 0, axis=1)

#     # Handle rows with all zeros
#     all_zeros = np.all(data == 0, axis=1)
#     non_zero_indices[all_zeros] = -1  # Set a distinct value for rows with all zeros

#     # Map indices to road types
#     road_types = np.array(
#         [
#             "No Roads",
#             "Highways",
#             "Primary Roads",
#             "Secondary Roads",
#             "Tertiary Roads",
#             "Local Roads",
#         ]
#     )

#     # Assign the dominant road type
#     df_zones["dominant_road_type"] = road_types[
#         non_zero_indices + 1
#     ]  # +1 to adjust for 'No Roads'

#     # Load the Crr mapping table
#     df_crr = pd.read_csv(CRR_FILE)

#     # Create a dictionary from the DataFrame
#     crr_mapping = pd.Series(
#         df_crr.Crr.values, index=df_crr["Assigned Zone Surface"]
#     ).to_dict()

#     # Map the road types in df_zones to Crr values
#     df_zones["Crr"] = df_zones["dominant_road_type"].map(crr_mapping)

#     return df_zones


def crr_add_uncertainty(road_type, adjustment):
    """
    The road type is adjusted by the given integer amount, unless the change would go out of bounds.
    In these cases, the adjustment stops at the best or worst road type.

    Parameters:
    road_type (str): The current road type.
    adjustment (int): The amount of adjustment to be made.

    Returns:
    str: The adjusted road type.

    Raises:
    None
    """

    # Define the road types in their logical order from best to worst
    road_types_logical_order = [
        "Highways",
        "Primary Roads",
        "Secondary Roads",
        "Tertiary Roads",
        "Local Roads",
        "No Roads",
    ]

    # Find the current index of the road type
    current_index = road_types_logical_order.index(road_type)

    # Adjust the index within the bounds
    new_index = np.clip(
        current_index + adjustment, 0, len(road_types_logical_order) - 1
    )

    # Return the adjusted road type
    return road_types_logical_order[new_index]


def road_analysis(df_zones, crr_adjustment=0):
    """
    Maps CRR values to road types in df_zones. Adjusts the CRR up or down to add uncertainty.

    Args:
        df_zones (pandas.DataFrame): The DataFrame containing zone data.
        crr_adjustment (int, optional): The adjustment value for Crr mapping. Defaults to 0.

    Returns:
        pandas.DataFrame: The DataFrame with additional columns for dominant road type and Crr values.
    """

    # Define the columns to work with for GRIP data
    columns = ["grip_1_1", "grip_2_1", "grip_3_1", "grip_4_1", "grip_5_1"]

    # Create a numpy array from the DataFrame for faster processing
    data = df_zones[columns].to_numpy()

    # Find the index of the first non-zero value in each row
    non_zero_indices = np.argmax(data != 0, axis=1)

    # Handle rows with all zeros
    all_zeros = np.all(data == 0, axis=1)
    non_zero_indices[all_zeros] = -1  # Set a distinct value for rows with all zeros

    # Map indices to road types
    road_types = np.array(
        [
            "No Roads",
            "Highways",
            "Primary Roads",
            "Secondary Roads",
            "Tertiary Roads",
            "Local Roads",
        ]
    )

    # Assign the dominant road type
    df_zones["dominant_road_type"] = road_types[
        non_zero_indices + 1
    ]  # +1 to adjust for 'No Roads'

    # Load the Crr mapping table
    df_crr = pd.read_csv(CRR_FILE)

    # Add uncertainty (make road type 1 better or 1 worse, unless road type is best (highway) or worst (no roads))
    df_crr["Crr"] = df_crr["Crr"].shift(-crr_adjustment)
    df_crr["Crr"] = df_crr["Crr"].ffill().bfill()

    # Create a series from the DataFrame
    crr_mapping = pd.Series(df_crr.Crr.values, index=df_crr["Assigned Zone Surface"])

    # Create a dictionary from the series
    crr_mapping = crr_mapping.to_dict()

    # Map the road types in df_zones to Crr values
    df_zones["Crr"] = df_zones["dominant_road_type"].map(crr_mapping)

    return df_zones


def calculate_nat_piped(df_zones):
    """
    Calculates National piped using the formula
    [ URBANPiped ×‘% urban’ ÷100+(100−‘% urban’ )÷100×RURALPiped ]

    """
    df_zones["NATPiped"] = (
        df_zones["URBANPiped"] * df_zones["% urban"] / 100
        + (100 - df_zones["% urban"]) / 100 * df_zones["RURALPiped"]
    )
    return df_zones


def preprocess_frame(
    df_zones_input,
    df_input,
    crr_adjustment,
    urban_adjustment,
    rural_adjustment,
    cell_size_degrees=None,
    density_totals=None,
    area_weighted_population=True,
):
    """Run the preprocessing pipeline over frames that are already in memory.

    This is the body of :func:`preprocess_data` with the file loading lifted
    out, so the chunked driver (:mod:`src.chunked_analysis`) can feed it one
    country-group at a time and be running *exactly* the same code as the
    monolithic path.

    Args:
        df_zones_input (DataFrame): raw zone frame, straight off disk.
        df_input (DataFrame): the country table.
        crr_adjustment (int): road-class adjustment for :func:`road_analysis`.
        urban_adjustment, rural_adjustment (float): Euclidean-distance factors.
        cell_size_degrees (float, optional): see
            :func:`merge_and_adjust_population`.
        density_totals (mapping, optional): see
            :func:`merge_and_adjust_population`.
        area_weighted_population (bool, optional): see
            :func:`merge_and_adjust_population`. Defaults to True; pass False
            for the legacy density-only split.

    Returns:
        pandas.DataFrame: the preprocessed zone frame.
    """
    df_zones_input = manage_urban_rural(df_zones_input)
    df_zones_input = adjust_euclidean(
        df_zones_input,
        urban_adjustment=urban_adjustment,
        rural_adjustment=rural_adjustment,
    )
    df_zones_input = manage_slope(df_zones_input)
    df_zones = merge_and_adjust_population(
        df_zones_input,
        df_input,
        cell_size_degrees=cell_size_degrees,
        density_totals=density_totals,
        area_weighted_population=area_weighted_population,
    )
    df_zones = road_analysis(df_zones, crr_adjustment=crr_adjustment)
    df_zones = calculate_nat_piped(df_zones)
    # Memory optimization. Done here, at the end of the pipeline, rather than inside
    # merge_and_adjust_population / road_analysis, so that those functions'
    # returned dtypes -- and the unit tests pinning them -- are untouched.
    df_zones = compact_zone_frame(df_zones)
    return df_zones


def zone_input_file(use_sample_data=False):
    """The zone frame the model reads, honouring the sample-data switch."""
    if use_sample_data:
        warnings.warn(
            "Using sample data. This should only be done for testing, and not for generating real model results!"
        )
        return URB_DATA_FILE_SAMPLE
    return URB_DATA_FILE


# Main function to run all steps
def preprocess_data(
    crr_adjustment,
    urban_adjustment,
    rural_adjustment,
    use_sample_data=False,
    area_weighted_population=True,
):
    """
    Preprocesses data by loading, managing, merging, and adjusting population data.

    Args:
        crr_adjustment (float): The adjustment factor for road analysis.
        area_weighted_population (bool, optional): see
            :func:`merge_and_adjust_population`. Defaults to True; pass False
            for the legacy density-only split.

    Returns:
        pandas.DataFrame: The preprocessed data.

    """

    urban_data_file = zone_input_file(use_sample_data)
    df_zones_input, df_input = load_data(urban_data_file, COUNTRY_DATA_FILE)
    return preprocess_frame(
        df_zones_input,
        df_input,
        crr_adjustment=crr_adjustment,
        urban_adjustment=urban_adjustment,
        rural_adjustment=rural_adjustment,
        area_weighted_population=area_weighted_population,
    )


##############################################################################################################################
#
# ANALYSIS I: Models
#
##############################################################################################################################


def load_hpv_parameters(file_path_params, hpv_name):
    """
    Load HPV parameters from a CSV file and return the subset of parameters for a specific HPV name.

    Parameters:
    - file_path_params (str): The file path to the CSV file containing the HPV parameters.
    - hpv_name (str): The name of the HPV for which to retrieve the parameters.

    Returns:
    - pandas.DataFrame: A DataFrame containing the subset of parameters for the specified HPV name.
    """
    allHPV_param_df = pd.read_csv(file_path_params)
    return allHPV_param_df[allHPV_param_df["Name"] == hpv_name]


def extract_slope_crr(df_zones):
    """
    Extracts slope and Crr values from the given DataFrame.

    Args:
        df_zones (pandas.DataFrame): The DataFrame containing slope and Crr values.

    Returns:
        tuple: A tuple containing two pandas.Series objects - slope_zones and Crr_values.
               slope_zones: The series containing slope values.
               Crr_values: The series containing Crr values.
    """
    df_zones["Crr"] = df_zones["Crr"].astype(float)
    # TODO update test to use country specific body mass
    country_average_weights = df_zones["Average Weight"]
    slope_zones = df_zones["slope_1"]
    Crr_values = df_zones["Crr"]
    return slope_zones, Crr_values, country_average_weights


def run_bicycle_model(
    mv, mo, hpv, slope_zones, Crr_values, country_average_weights, load_attempt
):
    """
    Runs a bicycle model for different slope zones and Crr values.

    Args:
        mv: Model variables.
        mo: Model options.
        hpv (object): Object representing the human-powered vehicle.
        slope_zones (list): List of slope zones.
        Crr_values (list): List of Crr values.
        load_attempt (int): Load attempt number.

    Returns:
        numpy.ndarray: Array of results for each slope zone and Crr value.
    """
    # Adjust the project root and import mobility module as needed
    project_root = Path().resolve().parent
    sys.path.append(str(project_root))
    import src.mobility_module as mm

    n_runs = len(slope_zones)

    # Lookup table: for anything bigger than a toy frame, solve the model once on a
    # (Crr x slope x mass) grid and interpolate, instead of once per cell. The
    # per-cell loop below stays as the ground truth and is still used for small
    # inputs.
    if solver_lut.should_use_lut(n_runs, Crr_values):
        return solver_lut.bicycle_results_via_lut(
            mm,
            mv,
            mo,
            hpv,
            np.asarray(slope_zones, dtype=float),
            np.asarray(Crr_values, dtype=float),
            np.asarray(country_average_weights, dtype=float),
            load_attempt,
        )

    results = np.zeros((n_runs, 3))
    # TODO update test to include country_average_weight
    for i, (slope, crr, country_average_weight) in enumerate(
        zip(slope_zones, Crr_values, country_average_weights)
    ):
        hpv.Crr = crr
        mv.m1 = country_average_weight
        results[i] = mm.mobility_models.single_bike_run(
            # need to modify single_bike_run function to take country_average_weight instead of m1
            # (done by updated mv to take country_average_weight as m1)
            mv,
            mo,
            hpv,
            slope,
            load_attempt,
        )
    # for i, (slope, crr) in enumerate(zip(slope_zones, Crr_values)):
    #     hpv.Crr = crr
    #     results[i] = mm.mobility_models.single_bike_run(
    #         mv, mo, hpv, slope, load_attempt
    #     )
    return results


def process_and_save_results(
    df_zones, results, export_file_location, velocity_type, save_csv=False
):
    """
    Process the results and optionally save them to a DataFrame.
    The results should only be saved for single runs, as saving all monte carlo csvs will be too large.

    Args:
        df_zones (pandas.DataFrame): The DataFrame containing the zones data.
        results (numpy.ndarray): The results array containing the velocity vectors.
        export_file_location (str): The file location where the output CSV will be saved.
        velocity_type (str): The type of velocity (e.g., walk/bicycle).
        save_csv (bool, optional): Whether to save the results as a CSV file. Defaults to False.

    Returns:
        pandas.DataFrame: The updated DataFrame with the new columns.

    """
    # Unpack results
    loaded_velocity_vec, unloaded_velocity_vec, max_load_vec = results.T

    # Harmonic mean of leg velocities (same distance each way). Use 2ab/(a+b)
    # rather than 2/(1/a+1/b) so a zero/unreachable leg yields 0 without
    # divide-by-zero warnings.
    denom = loaded_velocity_vec + unloaded_velocity_vec
    average_velocity = np.divide(
        2.0 * loaded_velocity_vec * unloaded_velocity_vec,
        denom,
        out=np.zeros_like(denom, dtype=float),
        where=denom != 0,
    )

    # Customizing column names based on the type of velocity
    loaded_velocity_col = f"loaded_velocity_{velocity_type}"
    unloaded_velocity_col = f"unloaded_velocity_{velocity_type}"
    average_velocity_col = f"average_velocity_{velocity_type}"
    max_load_col = f"max_load_{velocity_type}"

    # Updating the DataFrame with the new columns
    df_zones[loaded_velocity_col] = loaded_velocity_vec
    df_zones[unloaded_velocity_col] = unloaded_velocity_vec
    df_zones[average_velocity_col] = average_velocity
    df_zones[max_load_col] = max_load_vec

    if save_csv:
        # Customizing the output CSV filename based on the type of velocity
        output_csv_filename = os.path.join(
            export_file_location, f"{velocity_type}_velocity_by_zone.csv"
        )

        # only save the newly created cols in the csv
        df_zones[
            [
                "fid",
                loaded_velocity_col,
                unloaded_velocity_col,
                average_velocity_col,
                max_load_col,
            ]
        ].to_csv(output_csv_filename)

    return df_zones


# function to map hill polarity strings to integers for lhillpo and ulhillpo, using a dictionary for the mapping
def map_hill_polarity(hill_polarity):
    """
    Maps the hill polarity string to integers for the unloaded trip (ulhillpo) and the loaded trip back (lhillpo).

    Args:
        hill_polarity (str): The hill polarity string.

    Returns:
        int, int: The integer values for ulhillpo and lhillpo.
    """
    hill_polarity_mapping = {
        "uphill_downhill": (1, -1),
        "downhill_uphill": (-1, 1),
        "uphill_flat": (1, 0),
        "flat_uphill": (0, 1),
        "downhill_flat": (-1, 0),
        "flat_downhill": (0, -1),
        "flat_flat": (0, 0),
    }
    ulhillpo, lhillpo = hill_polarity_mapping[hill_polarity]

    return ulhillpo, lhillpo


def calculate_and_merge_bicycle_distance(
    df_zones,
    calculate_distance,
    export_file_location,
    practical_limit_bicycle=40,
    watts=75,
    human_mass=62,
    hill_polarity="flat_uphill",
):
    """
    Calculates and merges bicycle distance for each zone in the given dataframe.

    Args:
        df_zones (pandas.DataFrame): The dataframe containing zone information.
        calculate_distance (bool): Flag indicating whether to calculate the distance or not.
        export_file_location (str): The file location to export the results.
        practical_limit_bicycle (int, optional): The practical limit for bicycle distance in kg. Defaults to 40.

    Returns:
        pandas.DataFrame: The dataframe with bicycle distance merged.

    Raises:
        FileNotFoundError: If the mobility model parameters file is not found.
    """
    if calculate_distance:
        # Adjust the project root and import mobility module as needed
        project_root = Path().resolve().parent
        sys.path.append(str(project_root))
        import src.mobility_module as mm

        file_path_params = FILE_PATH_PARAMS
        param_df = load_hpv_parameters(file_path_params, "Bicycle")

        # Overwrite practical limit with the function inputs
        param_df["PracticalLimit"] = practical_limit_bicycle

        # Get integer values for hill polarity
        ulhillpo, lhillpo = map_hill_polarity(hill_polarity)

        # Initialise mobility model options, using updated hill polarities
        mo = mm.model_options(ulhillpo=ulhillpo, lhillpo=lhillpo)
        mo.model_selection = 2  # Cycling model

        # Extract values, initalise vars, and run model
        slope_zones, Crr_values, country_average_weights = extract_slope_crr(df_zones)
        mv = mm.model_variables(P_t=watts)
        hpv = mm.HPV_variables(param_df, mv)

        results = run_bicycle_model(
            mv,
            mo,
            hpv,
            slope_zones,
            Crr_values,
            country_average_weights,
            load_attempt=15,
        )
        process_and_save_results(df_zones, results, export_file_location, "bicycle")
    else:
        try:
            df_zones_bicycle = pd.read_csv(
                export_file_location / "bicycle_velocity_by_zone.csv"
            )
        except TypeError:
            df_zones_bicycle = pd.read_csv(
                export_file_location + "bicycle_velocity_by_zone.csv"
            )
        df_zones = df_zones.merge(df_zones_bicycle, on="fid", how="left")
    return df_zones


def run_walking_model(
    mv, mo, met, hpv, slope_zones, country_average_weights, load_attempt
):
    """
    Run the walking model for multiple slope zones.

    Args:
        mv: Model variables.
        mo: Model options.
        met: Metabolic equivalent of task.
        hpv (object): Object representing the human-powered vehicle.
        slope_zones (list): List of slope zones.
        Crr_values (list): List of Crr values.
        load_attempt (float): Load attempt number.

    Returns:
        numpy.ndarray: An array of results for each slope zone.

    """
    # Adjust the project root and import mobility module as needed
    project_root = Path().resolve().parent
    sys.path.append(str(project_root))
    import src.mobility_module as mm

    n_runs = len(slope_zones)

    # Lookup table: see run_bicycle_model. The walking table has no Crr axis -- buckets
    # are carried, not rolled -- so it is a (slope x mass) grid.
    if solver_lut.should_use_lut(n_runs):
        return solver_lut.walking_results_via_lut(
            mm,
            mv,
            mo,
            met,
            hpv,
            np.asarray(slope_zones, dtype=float),
            np.asarray(country_average_weights, dtype=float),
            load_attempt,
        )

    results = np.zeros((n_runs, 3))

    # Run model for each zone with country-specific weights
    for i, (slope, country_average_weight) in enumerate(
        zip(slope_zones, country_average_weights)
    ):
        mv.m1 = country_average_weight
        met_values = mm.MET_values(
            mv,
            country_weight=country_average_weight,
            met=met,
            use_country_specific_weights=True,
        )
        results[i] = mm.mobility_models.single_lankford_run(
            mv, mo, met_values, hpv, slope, load_attempt
        )
    return results


def calculate_and_merge_walking_distance(
    df_zones,
    calculate_distance,
    export_file_location,
    practical_limit_buckets=20,
    met=4.5,
    human_mass=62,
    hill_polarity="flat_uphill",
):
    """
    Calculate and merge walking distance for zones.

    Args:
        df_zones (pandas.DataFrame): The input DataFrame containing zone data.
        calculate_distance (bool): Flag indicating whether to calculate the walking distance.
        export_file_location (str): The file location to export the results.
        practical_limit_buckets (int, optional): The practical limit buckets. Defaults to 20.
        met (float, optional): The MET value. Defaults to 3.3.

    Returns:
        pandas.DataFrame: The updated DataFrame with walking distance information.
    """
    if calculate_distance:
        # Adjust the project root and import mobility module as needed
        project_root = Path().resolve().parent
        sys.path.append(str(project_root))
        import src.mobility_module as mm

        file_path_params = FILE_PATH_PARAMS
        param_df = load_hpv_parameters(file_path_params, "Buckets")

        # Overwrite practical limit with the function inputs
        param_df["PracticalLimit"] = practical_limit_buckets

        # Get integer values for hill polarity
        ulhillpo, lhillpo = map_hill_polarity(hill_polarity)

        # Initialise mobility model options, using updated hill polarities
        mo = mm.model_options(ulhillpo=ulhillpo, lhillpo=lhillpo)
        mo.model_selection = 3  # Lankford model
        mv = mm.model_variables(m1=human_mass)
        hpv = mm.HPV_variables(param_df, mv)

        # Run model for each zone with country-specific weights
        slope_zones, Crr_values, country_average_weights = extract_slope_crr(df_zones)
        results = run_walking_model(
            mv, mo, met, hpv, slope_zones, country_average_weights, load_attempt=15
        )
        process_and_save_results(df_zones, results, export_file_location, "walk")
    else:
        try:
            df_zones_walking = pd.read_csv(
                export_file_location / "walk_velocity_by_zone.csv"
            )
        except TypeError:
            df_zones_walking = pd.read_csv(
                export_file_location + "walk_velocity_by_zone.csv"
            )
        df_zones = df_zones.merge(df_zones_walking, on="fid", how="left")
    return df_zones


##############################################################################################################################
#
# ANALYSIS II: Population Water Access
#
##############################################################################################################################


def calculate_max_distances(df_zones, time_gathering_water):
    """
    Calculate the maximum distances achievable for gathering water in each zone.

    Args:
        df_zones (pandas.DataFrame): DataFrame containing zone information.
        time_gathering_water (float): Time taken to gather water in hours.

    Returns:
        pandas.DataFrame: DataFrame with additional columns for max distances and water ration.

    """
    # Max distance achievable (not round trip, just the distance from home to water source)
    # Convert velocity from m/s to km/h: multiply by 3600/1000
    df_zones["max distance cycling"] = (
        df_zones["average_velocity_bicycle"] * time_gathering_water * 3600 / 2 / 1000
    )
    df_zones["max distance walking"] = (
        df_zones["average_velocity_walk"] * time_gathering_water * 3600 / 2 / 1000
    )
    # Use water_ration_kms to calculate the water ration achievable per bike per zone
    df_zones["water_ration_kms"] = (
        df_zones["max distance cycling"] * df_zones["max_load_bicycle"]
    )
    return df_zones


# Tolerance on the per-stratum share sum. The regenerated dataset guarantees
# the five source shares plus Unallocated sum to <= 100.5 (rounding slack);
# anything above that means the frame predates the Unallocated columns.
SHARE_SUM_TOLERANCE = 100.5

REGENERATE_MESSAGE = (
    "merged_data.csv predates the Unallocated columns — regenerate it "
    "(scripts/Data_Manipulation_Scripts)."
)


def _validate_unallocated_vulnerable_fraction(unallocated_vulnerable_fraction):
    """Validate the unallocated-allocation parameter.

    Accepts a float in [0, 1]. Returns it as a float so callers can rely on a
    normalised value.
    """
    if isinstance(unallocated_vulnerable_fraction, (bool, np.bool_)):
        raise ValueError(
            "unallocated_vulnerable_fraction must be a float in [0, 1], got "
            f"{unallocated_vulnerable_fraction!r}"
        )
    if isinstance(
        unallocated_vulnerable_fraction, (int, float, np.integer, np.floating)
    ):
        value = float(unallocated_vulnerable_fraction)
        if not 0 <= value <= 1:
            raise ValueError(
                "unallocated_vulnerable_fraction must be between 0 and 1, got "
                f"{unallocated_vulnerable_fraction!r}"
            )
        return value
    raise ValueError(
        "unallocated_vulnerable_fraction must be a float in [0, 1], got "
        f"{unallocated_vulnerable_fraction!r}"
    )


def _check_share_sums(df_zones):
    """Raise if any stratum's five source shares plus Unallocated exceed 100.5.

    After the Unallocated columns exist, the six shares are a partition of the
    stratum and must sum to (at most) 100 up to rounding. A violation means the
    frame is stale — mixing new-style Unallocated with old-style unnormalised
    source shares — and every downstream population figure would be wrong.
    """
    offenders = pd.Series(False, index=df_zones.index)
    for stratum in ("URBAN", "RURAL"):
        share_sum = (
            df_zones[f"{stratum}Piped"].fillna(0)
            + df_zones[f"{stratum}Packaged"]
            + df_zones[f"{stratum}Delivered"]
            + df_zones[f"{stratum}Borehole"]
            + df_zones[f"{stratum}UnpipedUnaffected"]
            + df_zones[f"{stratum}UnpipedUnknown"]
            + df_zones[f"{stratum}Unallocated"]
        )
        offenders = offenders | (share_sum > SHARE_SUM_TOLERANCE)

    if offenders.any():
        id_col = next(
            (col for col in ("alpha3", "ISOCODE") if col in df_zones.columns), None
        )
        if id_col is None:
            offending = sorted(str(i) for i in df_zones.index[offenders].unique())
            label = "row"
        else:
            offending = sorted(df_zones.loc[offenders, id_col].astype(str).unique())
            label = id_col
        raise ValueError(
            f"Source shares plus Unallocated exceed {SHARE_SUM_TOLERANCE} for "
            f"{label}(s): {', '.join(offending)}. {REGENERATE_MESSAGE}"
        )


def calculate_population_water_access(
    df_zones,
    urban_borehole_lose_fraction=0.5,
    rural_borehole_lose_fraction=0.25,
    unallocated_vulnerable_fraction=0.0,
):
    """
    Calculates the population with and without access to water for each zone.

    Piped, packaged, delivered, and (a configurable fraction of) borehole users
    lose their usual supply and must reach an alternative source by walking or
    cycling. Other-unpiped users, the surviving share of borehole users, and
    the unaffected share of the *unallocated* population retain access.

    The accounting is explicit, not residual. Per stratum s (urban/rural), with
    borehole lose fraction f_s and unallocated vulnerable fraction g:

        affected_s   = Piped + Packaged + Delivered + f_s*Borehole + g*(Unallocated + UnpipedUnknown)
        unaffected_s  = UnpipedUnaffected + (1-f_s)*Borehole + (1-g)*(Unallocated + UnpipedUnknown)

    The two strata are blended by ``urban_rural`` exactly as the individual
    source categories are, so ``zone_pop_affected + zone_pop_unaffected``
    equals ``pop_zone`` whenever the six shares sum to 100 in both strata.

    ``Unallocated`` is the share of a stratum whose source type is unknown
    after harmonisation (see UNALLOCATED_GAP_INVESTIGATION.md). Its allocation
    is a model parameter rather than a silent assumption: the previous code
    computed the unaffected population as a clipped residual, which implicitly
    treated 100% of the unallocated mass as unaffected (g = 0).

    Args:
        df_zones (pandas.DataFrame): DataFrame containing zone information.
            Must contain URBANUnallocated / RURALUnallocated.
        urban_borehole_lose_fraction (float): Share of urban borehole users
            displaced from their usual source.
        rural_borehole_lose_fraction (float): Share of rural borehole users
            displaced from their usual source.
        unallocated_vulnerable_fraction (float): Share of the unallocated
            population treated as grid-vulnerable (i.e. added to the affected
            population). A float g in [0, 1] — 0.0 reproduces the historical
            residual behaviour on clean data, 1.0 is the all-vulnerable
            bracket. Defaults to 0.0.

    Returns:
        pandas.DataFrame: DataFrame with additional columns representing the population
        with and without access to water for each zone.

    Raises:
        KeyError: if URBANUnallocated / RURALUnallocated are missing (the input
            predates the Unallocated columns and must be regenerated).
        ValueError: if any row's five source shares plus Unallocated exceed
            100.5 in either stratum, or if unallocated_vulnerable_fraction is
            not a float in [0, 1].
    """
    g = _validate_unallocated_vulnerable_fraction(
        unallocated_vulnerable_fraction
    )

    for col in (
        "URBANPackaged",
        "RURALPackaged",
        "URBANDelivered",
        "RURALDelivered",
        "URBANBorehole",
        "RURALBorehole",
        "URBANUnpipedUnaffected",
        "RURALUnpipedUnaffected",
        "URBANUnpipedUnknown",
        "RURALUnpipedUnknown",
    ):
        if col not in df_zones.columns:
            df_zones[col] = 0
        else:
            df_zones[col] = df_zones[col].fillna(0)

    # The Unallocated columns are NOT defaulted to zero: silently assuming
    # zero would reinstate the residual behaviour on stale data and hide the
    # ~348M-person share-sum gap the parameter exists to bracket.
    missing_unallocated = [
        col
        for col in ("URBANUnallocated", "RURALUnallocated")
        if col not in df_zones.columns
    ]
    if missing_unallocated:
        raise KeyError(
            f"Missing required column(s) {missing_unallocated}: {REGENERATE_MESSAGE}"
        )
    for col in ("URBANUnallocated", "RURALUnallocated"):
        df_zones[col] = df_zones[col].fillna(0)

    _check_share_sums(df_zones)

    df_zones["zone_pop_piped"] = 0

    # If urban use urban shares, if rural use rural shares (urban_rural column)
    df_zones["zone_pop_piped"] = (
        df_zones["pop_zone"] * df_zones["urban_rural"] * df_zones["URBANPiped"] / 100
        + df_zones["pop_zone"]
        * (1 - df_zones["urban_rural"])
        * df_zones["RURALPiped"]
        / 100
    )
    df_zones["zone_pop_packaged"] = (
        df_zones["pop_zone"] * df_zones["urban_rural"] * df_zones["URBANPackaged"] / 100
        + df_zones["pop_zone"]
        * (1 - df_zones["urban_rural"])
        * df_zones["RURALPackaged"]
        / 100
    )
    df_zones["zone_pop_delivered"] = (
        df_zones["pop_zone"]
        * df_zones["urban_rural"]
        * df_zones["URBANDelivered"]
        / 100
        + df_zones["pop_zone"]
        * (1 - df_zones["urban_rural"])
        * df_zones["RURALDelivered"]
        / 100
    )
    df_zones["zone_pop_borehole"] = (
        df_zones["pop_zone"] * df_zones["urban_rural"] * df_zones["URBANBorehole"] / 100
        + df_zones["pop_zone"]
        * (1 - df_zones["urban_rural"])
        * df_zones["RURALBorehole"]
        / 100
    )
    df_zones["zone_pop_borehole_affected"] = (
        df_zones["pop_zone"]
        * df_zones["urban_rural"]
        * df_zones["URBANBorehole"]
        * urban_borehole_lose_fraction
        / 100
        + df_zones["pop_zone"]
        * (1 - df_zones["urban_rural"])
        * df_zones["RURALBorehole"]
        * rural_borehole_lose_fraction
        / 100
    )
    # Now used in the unaffected accounting below (it used to be dead code).
    df_zones["zone_pop_unpiped_unaffected"] = (
        df_zones["pop_zone"]
        * df_zones["urban_rural"]
        * df_zones["URBANUnpipedUnaffected"]
        / 100
        + df_zones["pop_zone"]
        * (1 - df_zones["urban_rural"])
        * df_zones["RURALUnpipedUnaffected"]
        / 100
    )
    df_zones["zone_pop_unpiped_unknown"] = (
        df_zones["pop_zone"]
        * df_zones["urban_rural"]
        * df_zones["URBANUnpipedUnknown"]
        / 100
        + df_zones["pop_zone"]
        * (1 - df_zones["urban_rural"])
        * df_zones["RURALUnpipedUnknown"]
        / 100
    )

    # Unallocated and UnpipedUnknown: split by g (vulnerable) vs (1-g) (unaffected).
    stratum_weights = {
        "URBAN": df_zones["urban_rural"],
        "RURAL": 1 - df_zones["urban_rural"],
    }

    unallocated_total = 0
    unallocated_affected = 0
    unallocated_unaffected = 0
    unpiped_unknown_affected = 0
    unpiped_unknown_unaffected = 0
    for stratum, weight in stratum_weights.items():
        unallocated_share = df_zones[f"{stratum}Unallocated"]
        unknown_share = df_zones[f"{stratum}UnpipedUnknown"]
        stratum_pop_unallocated = (
            df_zones["pop_zone"] * weight * unallocated_share / 100
        )
        stratum_pop_unknown = df_zones["pop_zone"] * weight * unknown_share / 100
        unallocated_total = unallocated_total + stratum_pop_unallocated
        unallocated_affected = unallocated_affected + stratum_pop_unallocated * g
        unallocated_unaffected = (
            unallocated_unaffected + stratum_pop_unallocated * (1 - g)
        )
        unpiped_unknown_affected = unpiped_unknown_affected + stratum_pop_unknown * g
        unpiped_unknown_unaffected = (
            unpiped_unknown_unaffected + stratum_pop_unknown * (1 - g)
        )

    df_zones["zone_pop_unallocated"] = unallocated_total
    df_zones["zone_pop_unallocated_affected"] = unallocated_affected
    df_zones["zone_pop_unallocated_unaffected"] = unallocated_unaffected
    df_zones["zone_pop_unpiped_unknown_affected"] = unpiped_unknown_affected
    df_zones["zone_pop_unpiped_unknown_unaffected"] = unpiped_unknown_unaffected

    df_zones["zone_pop_affected"] = (
        df_zones["zone_pop_piped"]
        + df_zones["zone_pop_packaged"]
        + df_zones["zone_pop_delivered"]
        + df_zones["zone_pop_borehole_affected"]
        + df_zones["zone_pop_unallocated_affected"]
        + df_zones["zone_pop_unpiped_unknown_affected"]
    )
    # Explicit sum of the categories that retain access, not a residual: the
    # surviving borehole users, everyone on other unpiped sources, and the
    # unaffected share of the unallocated population. No clip is needed because
    # every term is non-negative by construction.
    df_zones["zone_pop_borehole_unaffected"] = (
        df_zones["zone_pop_borehole"] - df_zones["zone_pop_borehole_affected"]
    )
    df_zones["zone_pop_unaffected"] = (
        df_zones["zone_pop_unpiped_unaffected"]
        + df_zones["zone_pop_borehole_unaffected"]
        + df_zones["zone_pop_unallocated_unaffected"]
        + df_zones["zone_pop_unpiped_unknown_unaffected"]
    )

    # Determine if it's possible to reach water with walking/cycling
    df_zones["zone_cycling_okay"] = (
        df_zones["dtw_1"] < df_zones["max distance cycling"]
    ) * 1
    df_zones["zone_walking_okay"] = (
        df_zones["dtw_1"] < df_zones["max distance walking"]
    ) * 1

    # Calculate the fraction of the zone that can collect water by cycling
    df_zones["fraction_of_zone_with_cycling_access"] = df_zones["zone_cycling_okay"] * (
        df_zones["PBO"] / 100
    )
    df_zones["fraction_of_zone_with_walking_access"] = df_zones["zone_walking_okay"] * 1

    # Calculate the affected population that can reach water by cycling and walking
    df_zones["population_affected_with_cycling_access"] = (
        df_zones["fraction_of_zone_with_cycling_access"] * df_zones["zone_pop_affected"]
    )
    df_zones["population_affected_with_walking_access"] = (
        df_zones["fraction_of_zone_with_walking_access"] * df_zones["zone_pop_affected"]
    )
    # Select the maximum between the two, if walkable, max will always be walking
    df_zones["population_affected_with_access"] = df_zones[
        [
            "population_affected_with_cycling_access",
            "population_affected_with_walking_access",
        ]
    ].max(axis=1)

    # Calculate the population that can access water only by cycling
    # Equals population_affected_with_cycling_access if zone_walking_okay is 0
    df_zones["population_affected_with_only_cycling_access"] = df_zones[
        "population_affected_with_cycling_access"
    ] * (1 - df_zones["zone_walking_okay"])

    # Calculate zone population with and without water access
    df_zones["zone_pop_with_water"] = (
        df_zones["population_affected_with_access"]
        + df_zones["zone_pop_unaffected"]
    )
    df_zones["zone_pop_without_water"] = (
        df_zones["pop_zone"] - df_zones["zone_pop_with_water"]
    )
    return df_zones


def calculate_water_rations(df_zones):
    """
    Calculate the water rations achievable utilizing all the bikes in each zone.

    Parameters:
    - df_zones (pandas.DataFrame): A DataFrame containing the zone data.

    Returns:
    - df_zones (pandas.DataFrame): The input DataFrame with additional columns for water rations.

    The function calculates the water rations achievable by dividing the water ration distance (water_ration_kms)
    by the distance to water source (dtw_1) for each zone. It then calculates the number of bikes in each zone
    by dividing the population of the zone by the average household size and multiplying it by the PBO (Percent Bike Ownership) factor.
    Finally, it calculates the total water rations achievable in each zone by multiplying the number of bikes in the zone
    by the water rations per bike.

    Example usage:
    >>> df = calculate_water_rations(df_zones)
    """

    # Prevent division by zero when distance to water is zero
    df_zones["water_rations_per_bike"] = np.where(
        df_zones["dtw_1"] > 0,
        df_zones["water_ration_kms"] / df_zones["dtw_1"],
        np.inf,  # Infinite water access when distance is zero
    )
    df_zones["bikes_in_zone"] = (
        df_zones["pop_zone"]
        / df_zones["Household_Size"]
        * df_zones["PBO"]
        / 100  # must divide by 100 as PBO is percentage.
    )
    df_zones["water_rations_achievable"] = (
        df_zones["bikes_in_zone"] * df_zones["water_rations_per_bike"]
    )
    return df_zones


def process_zones_for_water_access(
    df_zones,
    time_gathering_water=16,
    urban_borehole_lose_fraction=1,
    rural_borehole_lose_fraction=1,
    unallocated_vulnerable_fraction=0.0,
):
    df_zones = calculate_max_distances(df_zones, time_gathering_water)
    df_zones = calculate_population_water_access(
        df_zones,
        urban_borehole_lose_fraction=urban_borehole_lose_fraction,
        rural_borehole_lose_fraction=rural_borehole_lose_fraction,
        unallocated_vulnerable_fraction=unallocated_vulnerable_fraction,
    )
    df_zones = calculate_water_rations(df_zones)
    return df_zones


##############################################################################################################################
#
# ANALYSIS III: Aggregating Country Data
#
##############################################################################################################################


def aggregate_country_level_data(df_zones):
    """
    Aggregate zone data into country level summaries.

    Parameters:
        df_zones (DataFrame): The input DataFrame containing zone-level data.

    Returns:
        df_countries (DataFrame): The aggregated DataFrame with country-level summaries.
    """
    df_countries = (
        df_zones.groupby("ISOCODE", observed=True)
        .agg(
            {
                "Entity": "first",
                "country_pop_raw": "first",
                "zone_pop_with_water": "sum",
                "zone_pop_without_water": "sum",
                "population_affected_with_access": "sum",
                "population_affected_with_cycling_access": "sum",
                "population_affected_with_walking_access": "sum",
                "population_affected_with_only_cycling_access": "sum",
                "NATPiped": "first",
                "region": "first",
                "subregion": "first",
                # Additional aggregations can be added here
            }
        )
        .reset_index()
    )
    # Rename the columns
    df_countries.columns = [
        "ISOCODE",
        "Entity",
        "country_pop_raw",
        "zone_pop_with_water",
        "zone_pop_without_water",
        "population_affected_with_access",
        "population_affected_with_cycling_access",
        "population_affected_with_walking_access",
        "population_affected_with_only_cycling_access",
        "NATPiped",
        "region",
        "subregion",
    ]

    return df_countries


def weighted_percentile(df, val_column, weight_column, percentile):
    """Calculates the weighted percentile
    ArithmeticError
    If the sum of the weights is zero, or if the weights are not positive.
    """
    df_sorted = df.sort_values(val_column)
    cumsum = df_sorted[weight_column].cumsum()
    cutoff = df_sorted[weight_column].sum() * percentile / 100.0
    return df_sorted[cumsum >= cutoff][val_column].iloc[0]


def calculate_weighted_results(df_zones):
    """
    Calculate the weighted median for each country group.
    """
    # return error if the input DataFrame is empty
    if df_zones.empty:
        raise ValueError("Input DataFrame is empty")

    # Only the four columns the lambda reads are passed to it. That keeps the
    # per-group frames small at 30 arc-seconds  and, incidentally,
    # silences the pandas 2.x FutureWarning about `apply` operating on the
    # grouping column -- the grouping column is no longer in the selection.
    df_median_group = (
        df_zones.groupby("ISOCODE", observed=True)[
            [
                "dtw_1",
                "pop_zone",
                "max distance cycling",
                "max distance walking",
            ]
        ]
        .apply(
            lambda x: pd.Series(
                {
                    "weighted_med": weighted_median(x, "dtw_1", "pop_zone"),
                    "weighted_med_cycling": weighted_percentile(
                        x, "max distance cycling", "pop_zone", 50
                    ),
                    "weighted_5th_cycling": weighted_percentile(
                        x, "max distance cycling", "pop_zone", 5
                    ),
                    "weighted_95th_cycling": weighted_percentile(
                        x, "max distance cycling", "pop_zone", 95
                    ),
                    "weighted_med_walking": weighted_percentile(
                        x, "max distance walking", "pop_zone", 50
                    ),
                    "weighted_5th_walking": weighted_percentile(
                        x, "max distance walking", "pop_zone", 5
                    ),
                    "weighted_95th_walking": weighted_percentile(
                        x, "max distance walking", "pop_zone", 95
                    ),
                }
            )
        )
        .reset_index()
    )

    return df_median_group


#: Columns :func:`aggregate_global` sums, and the global-row key each lands
#: under. All are plain sums, so a chunked run can accumulate them chunk by
#: chunk and hand the totals back via ``aggregate_global(..., sums=...)``.
GLOBAL_SUM_COLUMNS = {
    "country_pop_raw": "pop_zone",
    "zone_pop_with_water": "zone_pop_with_water",
    "zone_pop_without_water": "zone_pop_without_water",
    "population_affected_with_access": "population_affected_with_access",
    "population_affected_with_cycling_access": "population_affected_with_cycling_access",
    "population_affected_with_walking_access": "population_affected_with_walking_access",
    "population_affected_with_only_cycling_access": "population_affected_with_only_cycling_access",
}

#: The only zone columns the global row's weighted statistics read. A chunked
#: run spills just these to disk and replays them in the finaliser, because a
#: population-weighted median is the one quantity in the whole pipeline that is
#: neither per-country nor additive.
GLOBAL_WEIGHTED_COLUMNS = (
    "dtw_1",
    "max distance cycling",
    "max distance walking",
    "pop_zone",
)


def aggregate_global(df_zones, sums=None):
    """
    Get global data to be added to the country-level data.

    Parameters:
        df_zones (DataFrame): The input DataFrame containing zone-level data.
            A chunked caller may pass a frame holding only
            :data:`GLOBAL_WEIGHTED_COLUMNS`, provided it also passes ``sums``.
        sums (dict, optional): pre-computed totals keyed as
            :data:`GLOBAL_SUM_COLUMNS`. Defaults to None, i.e. sum ``df_zones``.

    Returns:
        df_global (DataFrame): One row of global data
    """
    # Calculate the sum for the specified columns
    if sums is None:
        sum_data = {
            key: df_zones[column].sum() for key, column in GLOBAL_SUM_COLUMNS.items()
        }
    else:
        sum_data = dict(sums)

    # Calculate the weighted median and the 5th and 95th percentiles
    weighted_data = {
        "weighted_med": weighted_median(df_zones, "dtw_1", "pop_zone"),
        "weighted_med_cycling": weighted_median(
            df_zones, "max distance cycling", "pop_zone"
        ),
        "weighted_5th_cycling": weighted_percentile(
            df_zones, "max distance cycling", "pop_zone", 5
        ),
        "weighted_95th_cycling": weighted_percentile(
            df_zones, "max distance cycling", "pop_zone", 95
        ),
        "weighted_med_walking": weighted_median(
            df_zones, "max distance walking", "pop_zone"
        ),
        "weighted_5th_walking": weighted_percentile(
            df_zones, "max distance walking", "pop_zone", 5
        ),
        "weighted_95th_walking": weighted_percentile(
            df_zones, "max distance walking", "pop_zone", 95
        ),
    }

    # Create a new DataFrame with the global row
    df_global = pd.DataFrame(
        [{"Entity": "Global", "ISOCODE": "GLOBAL", **sum_data, **weighted_data}]
    )

    return df_global


def clean_up_data(df_countries):
    """
    Clean up data from spurious country values.

    Parameters:
    - df_countries (pandas.DataFrame): The input dataframe containing country data.

    Returns:
    - df_countries (pandas.DataFrame): The cleaned dataframe with spurious country values removed.
    - countries_further_than_libya (pandas.DataFrame): The dataframe containing countries with distances greater than the maximum distance to water.
    - list_of_countries_to_remove (list): The list of specific countries manually removed from the dataframe.
    """
    # df_countries = df_countries.dropna()  # Remove any NaN rows

    # Remove outliers based on max possible distance to water
    libya_weighted_med = df_countries.loc[
        df_countries["ISOCODE"] == "LBY", "weighted_med"
    ].values

    if len(libya_weighted_med) > 0:
        max_distance = libya_weighted_med[0] + 1
        countries_further_than_libya = df_countries[
            df_countries["weighted_med"] > max_distance
        ]
        df_countries = df_countries[df_countries["weighted_med"] < max_distance]
    else:
        countries_further_than_libya = pd.DataFrame()
        print(
            "Warning: No data for Libya (ISOCODE 'LBY'). Skipping outlier removal based on Libya's weighted_med."
        )

    # Manually remove specific countries
    list_of_countries_to_remove = [
        "GUM",
        "ASM",
        "TON",
        "MNP",
        "ATG",
        "DMA",
        "ABW",
        "BRB",
    ]
    df_countries = df_countries[
        ~df_countries["ISOCODE"].isin(list_of_countries_to_remove)
    ]

    # Return the cleaned dataframe and lists of removed countries for logging or review
    return df_countries, countries_further_than_libya, list_of_countries_to_remove


def country_partials(df_zones):
    """Aggregate zone data to one row per country, before any global step.

    This is the chunk-decomposable half of :func:`process_country_data`: every
    calculation in it is grouped by ISOCODE, so running it over a subset of
    *whole* countries gives exactly the rows that a whole-world run would give
    for those countries. :func:`finalise_country_data` does the rest, which is
    the half that genuinely needs every country at once.

    Args:
        df_zones (pandas.DataFrame): zone-level data.

    Returns:
        pandas.DataFrame: one row per country, with the weighted statistics
        merged in.
    """
    assert not df_zones.empty, "Input dataframe is empty"
    warnings.warn("Input dataframe contains NaN values")

    df_countries = aggregate_country_level_data(df_zones)
    assert not df_countries.empty, "Country-level dataframe is empty"
    # TODO check this out (assert raises error, but probably not an issue as some NaNs are expected?)
    # assert not df_countries.isnull().values.any(), "Country-level dataframe contains NaN values"

    df_median_group = calculate_weighted_results(df_zones)
    assert not df_median_group.empty, "Weighted median dataframe is empty"
    # TODO check this out (assert raises error, but probably not an issue as some NaNs are expected?)
    # assert not df_median_group.isnull().values.any(), "Weighted median dataframe contains NaN values"

    return df_countries.merge(df_median_group, on="ISOCODE")  # Merge weighted median


def finalise_country_data(df_countries, df_global):
    """Turn per-country aggregates into the final country table.

    The half of :func:`process_country_data` that needs the whole world: the
    global row, and the outlier cut that is defined relative to Libya's
    population-weighted median distance to water.

    Args:
        df_countries (pandas.DataFrame): output of :func:`country_partials`,
            covering every country (concatenated across chunks if need be).
        df_global (pandas.DataFrame): the one-row output of
            :func:`aggregate_global`.

    Returns:
        pandas.DataFrame: the processed country-level summaries.
    """
    # replace country_pop_raw with sum of countries
    df_global = df_global.copy()
    df_global["country_pop_raw"] = df_countries["country_pop_raw"].sum()

    df_countries = pd.concat([df_countries, df_global], ignore_index=True)

    # Rename and create percentage columns
    df_countries = df_countries.rename(
        columns={
            "zone_pop_with_water": "country_pop_with_water",
            "zone_pop_without_water": "country_pop_without_water",
        }
    )

    # Check if 'country_pop_with_water' exists in df_countries
    if "country_pop_with_water" not in df_countries.columns:
        raise KeyError(
            "Column 'country_pop_with_water' does not exist in df_countries. Available columns: {}".format(
                df_countries.columns
            )
        )

    df_countries["percent_with_water"] = (
        df_countries["country_pop_with_water"] / df_countries["country_pop_raw"] * 100
    )
    df_countries["percent_without_water"] = (
        df_countries["country_pop_without_water"]
        / df_countries["country_pop_raw"]
        * 100
    )

    # add percentage columns for cycling and water access
    df_countries["percent_affected_with_cycling_access"] = (
        df_countries["population_affected_with_cycling_access"]
        / df_countries["country_pop_raw"]
        * 100
    )

    df_countries["percent_affected_with_walking_access"] = (
        df_countries["population_affected_with_walking_access"]
        / df_countries["country_pop_raw"]
        * 100
    )

    df_countries["proportion_affected_access_from_cycling"] = (
        df_countries["population_affected_with_only_cycling_access"]
        / df_countries["population_affected_with_access"]
        * 100
    )

    df_countries["percent_with_only_cycling_access"] = (
        df_countries["population_affected_with_only_cycling_access"]
        / df_countries["country_pop_raw"]
        * 100
    )

    df_countries, removed_further_than_libya, removed_countries_list = clean_up_data(
        df_countries
    )

    # Log or print summary of removed countries
    if not removed_further_than_libya.empty:
        print(
            "Countries removed from analysis due to being further than Libya's median:",
            removed_further_than_libya["Entity"].tolist(),
        )
    print("Countries removed manually:", removed_countries_list)

    return df_countries


def process_country_data(df_zones):
    """
    Orchestrate the processing of zone data into country-level summaries and cleanup.

    Args:
        df_zones (pandas.DataFrame): The input dataframe containing zone-level data.

    Returns:
        pandas.DataFrame: The processed dataframe containing country-level summaries.
    """
    return finalise_country_data(country_partials(df_zones), aggregate_global(df_zones))


##############################################################################################################################
#
# ANALYSIS IV: Aggregating District Data
#
##############################################################################################################################


#: The only zone columns :func:`process_district_data` reads. ``run_global_analysis``
#: hands it a copy of exactly these rather than of the whole frame .
DISTRICT_ZONE_COLUMNS = (
    "shapeID",
    "Entity",
    "ISOCODE",
    "shapeName",
    "pop_zone",
    "dtw_1",
    "zone_pop_with_water",
    "zone_pop_without_water",
    "population_affected_with_access",
    "population_affected_with_cycling_access",
    "population_affected_with_walking_access",
    "population_affected_with_only_cycling_access",
    "NATPiped",
    "region",
    "subregion",
)


def aggregate_district_level_data(df_zones):
    """
    Aggregate zone data into district level summaries.

    Parameters:
        df_zones (DataFrame): The input DataFrame containing zone-level data.

    Returns:
        df_countries (DataFrame): The aggregated DataFrame with district-level summaries.
    """

    df_zones["district_pop_raw"] = df_zones.groupby("shapeID", observed=True)[
        "pop_zone"
    ].transform("sum")

    df_districts = (
        df_zones.groupby("shapeID", observed=True)
        .agg(
            {
                "Entity": "first",
                "ISOCODE": "first",
                "shapeName": "first",
                "district_pop_raw": "first",
                "zone_pop_with_water": "sum",
                "zone_pop_without_water": "sum",
                "population_affected_with_access": "sum",
                "population_affected_with_cycling_access": "sum",
                "population_affected_with_walking_access": "sum",
                "population_affected_with_only_cycling_access": "sum",
                "NATPiped": "first",
                "region": "first",
                "subregion": "first",
            }
        )
        .reset_index()
    )
    # Rename the columns
    df_districts.columns = [
        "shapeID",
        "Entity",
        "ISOCODE",
        "shapeName",
        "district_pop_raw",
        "zone_pop_with_water",
        "zone_pop_without_water",
        "population_affected_with_access",
        "population_affected_with_cycling_access",
        "population_affected_with_walking_access",
        "population_affected_with_only_cycling_access",
        "NATPiped",
        "region",
        "subregion",
    ]

    return df_districts


def process_district_data(df_zones):
    """
    Orchestrate the processing of zone data into district-level summaries and cleanup.

    Args:
        df_zones (pandas.DataFrame): The input dataframe containing zone-level data.

    Returns:
        pandas.DataFrame: The processed dataframe containing district-level summaries.
    """
    assert not df_zones.empty, "Input dataframe is empty"

    df_districts = aggregate_district_level_data(df_zones)

    # use groupby to create weighted median, needs to be speerate from the above groupby as it uses apply, which can't be used in the same groupby
    # needs to use apply because the function required two columns as input
    # Selecting the two columns the lambda reads keeps the per-group frames
    # small  and silences the pandas 2.x grouping-column FutureWarning.
    df_median_group = df_zones.groupby(["shapeID"], observed=True)[
        ["dtw_1", "pop_zone"]
    ].apply(
        lambda x: pd.Series({"weighted_med": weighted_median(x, "dtw_1", "pop_zone")})
    )

    # merge the weighted median back into the df_districts dataframe
    df_districts = df_districts.merge(df_median_group, on="shapeID")

    # drop rows from the dataframe that have Nan in pop_zone and dtw_1
    df_zones = df_zones.dropna(subset=["pop_zone", "dtw_1"])

    # create summary columns
    # rename zone columns to country
    df_districts = df_districts.rename(
        columns={
            "zone_pop_with_water": "district_pop_with_water",
            "zone_pop_without_water": "district_pop_without_water",
        }
    )

    # create percent
    df_districts["percent_with_water"] = (
        df_districts["district_pop_with_water"] / df_districts["district_pop_raw"] * 100
    )
    df_districts["percent_without_water"] = (
        df_districts["district_pop_without_water"]
        / df_districts["district_pop_raw"]
        * 100
    )

    # add percentage columns for cycling and water access
    df_districts["percent_affected_with_cycling_access"] = (
        df_districts["population_affected_with_cycling_access"]
        / df_districts["district_pop_raw"]
        * 100
    )

    df_districts["percent_affected_with_walking_access"] = (
        df_districts["population_affected_with_walking_access"]
        / df_districts["district_pop_raw"]
        * 100
    )

    df_districts["proportion_affected_access_from_cycling"] = (
        df_districts["population_affected_with_only_cycling_access"]
        / df_districts["population_affected_with_access"]
        * 100
    )

    df_districts["percent_with_only_cycling_access"] = (
        df_districts["population_affected_with_only_cycling_access"]
        / df_districts["district_pop_raw"]
        * 100
    )

    list_of_countries_to_remove = [
        "GUM",
        "ASM",
        "TON",
        "MNP",
        "ATG",
        "DMA",
        "ABW",
        "BRB",
    ]
    df_districts = df_districts[
        ~df_districts["ISOCODE"].isin(list_of_countries_to_remove)
    ]

    return df_districts


##############################################################################################################################
#
# VISUALISATIONS
#
##############################################################################################################################


def plot_chloropleth(df_countries):
    hover_data_list = [
        "ISOCODE",
        "country_pop_raw",
        "country_pop_with_water",
        "country_pop_without_water",
        "population_affected_with_access",
        "population_affected_with_cycling_access",
        "population_affected_with_walking_access",
        "percent_without_water",
        "percent_with_water",
        "NATPiped",
        "region",
        "subregion",
        "weighted_med",
    ]

    choro = px.choropleth(
        title="Percent of Population Has to Relocate",
        data_frame=df_countries,
        locations="ISOCODE",
        height=600,
        color="percent_without_water",
        # use constant colorbar grading (not relative)
        color_continuous_scale="ylorbr",
        # color_continuous_scale="YlGnBu_r",
        # color_continuous_scale="PuRd",
        # color_continuous_scale="Greys",
        range_color=(0, 100),
        scope="world",
        hover_name="ISOCODE",
        hover_data=hover_data_list,
    )
    choro.layout.coloraxis.colorbar.title = ""
    choro.show()


##############################################################################################################################
#
# MAIN Function
#
##############################################################################################################################


def run_global_analysis(
    crr_adjustment,
    time_gathering_water,
    practical_limit_bicycle,
    practical_limit_buckets,
    met,
    watts,
    hill_polarity,
    urban_adjustment,
    rural_adjustment,
    urban_borehole_lose_fraction=0.5,
    rural_borehole_lose_fraction=0.25,
    unallocated_vulnerable_fraction=0.0,
    calculate_distance=True,
    plot=False,
    human_mass=62,  # gets overridden by country specific weight
    use_sample_data=False,  # only change to test functionality
    zone_columns=None,  # Memory optimization: trim the returned per-cell frame
    chunk_rows=None,  # Memory optimization: run country-group by country-group
    chunk_plan=None,  # reuse a plan across draws
    area_weighted_population=True,  # cos-latitude cell-area weighting; False = legacy
):
    """
    Runs one run of the global analysis for water access.

    Args:
        crr_adjustment (float): The adjustment factor for calculating the Coefficient of Rolling Resistance (CRR).
        time_gathering_water (float): The time taken to gather water in minutes.
        practical_limit_bicycle (float): The practical limit for distance traveled by bicycle in kilometers.
        practical_limit_buckets (float): The practical limit for distance traveled by carrying buckets in kilometers.
        met (str): metabolic equivalent of task (energy for walking).
        watts (float): The power output in watts (energy for cycling).
        urban_borehole_lose_fraction (float): Share of urban borehole users who lose access
            when electric pumping fails (scenario parameter, see ELECTRIC_BOREHOLE_SHARE_ESTIMATE.md).
        rural_borehole_lose_fraction (float): Share of rural borehole users who lose access
            when electric pumping fails.
        unallocated_vulnerable_fraction (float): Share of the population whose
            source type is unallocated that is treated as grid-vulnerable.
            Float in [0, 1]. See calculate_population_water_access.
        calculate_distance (bool, optional): Whether to calculate distance or not. Defaults to True, otherwise new results are not generated.
        human_mass (float): The mass of an average human carrying water in kilograms.
        plot (bool, optional): Whether to plot the chloropleth map or not. Defaults to False.
        use_sample_data (bool, optional): Whether to use sample data or not (only for testing). Defaults to False.
        zone_columns (list, optional): if given, only these columns of the
            per-cell zone frame are returned; an empty list returns an empty
            frame (no rows), meaning "I do not need the per-cell output".
            Defaults to None (every column, the historical behaviour). Memory optimization: at
            30 arc-seconds the full zone frame is tens of GB and callers keep
            three columns of it, so handing back the whole thing is what
            actually blows the memory budget.
        chunk_rows (int, optional): if set, run the analysis one group of whole
            countries at a time, each group holding at most this many modelled
            cells . Peak memory then
            follows this number instead of the size of the grid. The returned
            per-cell frame is empty in this mode -- per-cell output does not
            outlive its chunk, which is the whole point. Defaults to None, i.e.
            the monolithic path, unchanged.
        chunk_plan (ChunkPlan, optional): a plan from
            :func:`src.chunked_analysis.plan_chunks` to reuse instead of
            recomputing it. The plan depends only on the input file, so a
            multi-draw run should build it once.
        area_weighted_population (bool, optional): split each country's
            population by ``pop_density x cos(latitude)`` rather than
            ``pop_density`` alone (see :func:`population_split_weights`).
            **Defaults to True** as of 2026-08-09; pass False for the
            historical, bit-identical behaviour. A supplied ``chunk_plan`` must
            have been built with the same setting -- the planner's pre-pass
            normalises with the same weight.

    Returns:
        pandas.DataFrame: The processed data for each country.
    """
    if chunk_rows or chunk_plan is not None:
        import src.chunked_analysis as chunked

        df_countries, df_districts = chunked.run_chunked_global_analysis(
            crr_adjustment=crr_adjustment,
            time_gathering_water=time_gathering_water,
            practical_limit_bicycle=practical_limit_bicycle,
            practical_limit_buckets=practical_limit_buckets,
            met=met,
            watts=watts,
            hill_polarity=hill_polarity,
            urban_adjustment=urban_adjustment,
            rural_adjustment=rural_adjustment,
            urban_borehole_lose_fraction=urban_borehole_lose_fraction,
            rural_borehole_lose_fraction=rural_borehole_lose_fraction,
            unallocated_vulnerable_fraction=unallocated_vulnerable_fraction,
            calculate_distance=calculate_distance,
            human_mass=human_mass,
            use_sample_data=use_sample_data,
            chunk_rows=chunk_rows or chunked.DEFAULT_CHUNK_ROWS,
            plan=chunk_plan,
            area_weighted_population=area_weighted_population,
        )
        if plot:
            plot_chloropleth(df_countries)
        return df_countries, df_districts, pd.DataFrame()

    df_zones = preprocess_data(
        crr_adjustment=crr_adjustment,
        urban_adjustment=urban_adjustment,
        rural_adjustment=rural_adjustment,
        use_sample_data=use_sample_data,
        area_weighted_population=area_weighted_population,
    )
    df_zones = calculate_and_merge_bicycle_distance(
        df_zones,
        calculate_distance=calculate_distance,
        export_file_location=EXPORT_FILE_LOCATION,
        practical_limit_bicycle=practical_limit_bicycle,
        watts=watts,
        human_mass=human_mass,
        hill_polarity=hill_polarity,
    )
    df_zones = calculate_and_merge_walking_distance(
        df_zones,
        calculate_distance=calculate_distance,
        export_file_location=EXPORT_FILE_LOCATION,
        practical_limit_buckets=practical_limit_buckets,
        met=met,
        human_mass=human_mass,
        hill_polarity=hill_polarity,
    )
    df_zones = process_zones_for_water_access(
        df_zones,
        time_gathering_water=time_gathering_water,
        urban_borehole_lose_fraction=urban_borehole_lose_fraction,
        rural_borehole_lose_fraction=rural_borehole_lose_fraction,
        unallocated_vulnerable_fraction=unallocated_vulnerable_fraction,
    )

    # print("\n\n 1 \n\n")
    # print(len(df_zones["ISOCODE"].unique()))

    # add df_districts.
    # Memory optimization: this used to copy the whole ~60-column zone frame purely because
    # process_district_data adds a column to what it is given. Copying just the
    # columns it reads is behaviourally identical and ~4x smaller.
    df_zones_districts = df_zones[list(DISTRICT_ZONE_COLUMNS)].copy()
    df_districts = process_district_data(df_zones_districts)
    del df_zones_districts
    df_countries = process_country_data(df_zones)

    # drop countries in df_distciricts if they don't appear in df_copuntries
    df_districts = df_districts[df_districts["ISOCODE"].isin(df_countries["ISOCODE"])]

    # Memory optimization: the aggregates leave with their historical dtypes; only the
    # per-cell frame keeps the categorical/float32 treatment.
    df_countries = decategorise(df_countries)
    df_districts = decategorise(df_districts)

    if plot:
        plot_chloropleth(df_countries)

    if zone_columns is not None:
        zone_columns = list(zone_columns)
        # An explicitly empty selection means "I do not need the per-cell frame
        # at all" -- return the schema with no rows, so a 140M-row index does
        # not get pickled back to the parent process for nothing .
        df_zones = df_zones.iloc[:0] if not zone_columns else df_zones[zone_columns]

    # print("\n\n 5 \n\n")
    # print(len(df_districts["ISOCODE"].unique()))
    # print(len(df_countries["ISOCODE"].unique()))

    # TODO consider retuyrning df_zones here as well
    return df_countries, df_districts, df_zones


if __name__ == "__main__":

    df_countries, df_districts, zones_results = run_global_analysis(
        crr_adjustment=0,
        time_gathering_water=2.84,
        practical_limit_bicycle=40,
        practical_limit_buckets=20,
        met=4,
        watts=40,
        hill_polarity="downhill_uphill",
        urban_adjustment=1.3,
        rural_adjustment=1.4,
        calculate_distance=True,
        plot=True,
        human_mass=62,  # gets overridden by country specific weight
        use_sample_data=False,
    )

    df_countries.to_csv(COUNTRY_RESULTS_FILE_PATH, index=False)
    df_districts.to_csv(DISTRICT_RESULTS_FILE_PATH, index=False)
