"""Canonical Monte-Carlo sampling parameters (ADR-7).

Single source of truth for the distributions the Monte-Carlo driver scripts draw
their per-iteration parameters from. Every driver
(``scripts/run_monte_carlo.py``, ``scripts/run_monte_carlo_gcp.py``,
``scripts/run_monte_carlo_test.py``) imports from here so that the three copies
can never drift apart again.

History: the local and GCP drivers had silently diverged on three parameters --
TIME_GATHERING (normal 4-7 h locally vs lognormal 2-4 h on GCP), METS (3-6 vs
3-5) and WATTS (20-80 vs 20-60). The **GCP values are canonical** (Kevin's
sign-off, 2026-08-08): a lognormal 2-4 h water-gathering budget is what matches
the manuscript's ~6.5 km median walking limit. The local driver's previous
values were non-canonical and have been retired.

Note that the ``*_LOWER_ESTIMATE`` / ``*_UPPER_ESTIMATE`` pairs are **90%
confidence bounds**, not hard limits -- see ``src.gis_monte_carlo.sample_normal``
and ``sample_lognormal``, which map them onto a distribution whose central 90%
lies between them. The three water-source parameters below
(``*_BOREHOLE_LOSE_FRACTION_*`` and ``UNALLOCATED_VULNERABLE_FRACTION_*``) are
the exception: they are sampled *uniformly* between hard bounds, so those pairs
are the support of the distribution, not a 90% interval.

The water-source parameters moved here from ``scripts/run_monte_carlo_gcp.py``
when ``kr/unpiped-breakdown`` merged into the 30 arc-second work (2026-08-11),
for the same ADR-7 reason the mobility ones did. Scenario logic lives in these
parameter ranges, not in ``merged_data.csv``: see
``UNALLOCATED_VULNERABLE_FRACTION_FIXED`` for the deterministic scenario runs.
"""

# -----------------------------------------------------------------------------
# WALKING AND CYCLING PARAMETERS FOR THE MONTE CARLO SIMULATIONS
# -----------------------------------------------------------------------------

# CRR adjustments (sampled as a uniform integer over the inclusive range).
# 1 means one road type better, -1 means one road type worse.
CRR_LOWER_ESTIMATE = -1
CRR_UPPER_ESTIMATE = 1

# Time gathering water, in hours. Sampled LOGNORMALLY (strictly positive,
# right-skewed): a 2-4 h 90% interval, per ADR-7.
TIME_GATHERING_LOWER_ESTIMATE = 2
TIME_GATHERING_UPPER_ESTIMATE = 4

# Practical load limits for cycling in kg.
PRACTICAL_LIMITS_BICYCLE_LOWER_ESTIMATE = 30
PRACTICAL_LIMITS_BICYCLE_UPPER_ESTIMATE = 45

# Practical load limits for walking with buckets in kg.
PRACTICAL_LIMITS_BUCKET_LOWER_ESTIMATE = 15
PRACTICAL_LIMITS_BUCKET_UPPER_ESTIMATE = 25

# Average METS available for walking with buckets to and from the water source.
METS_LOWER_ESTIMATE = 3
METS_UPPER_ESTIMATE = 5

# Average watts available for cycling to and from the water source.
WATTS_LOWER_ESTIMATE = 20
WATTS_UPPER_ESTIMATE = 60

# Polarity options (one chosen at random per simulation run).
# The first word defines the trip TO the water source, the second the trip FROM
# it. Other admissible options, deliberately not sampled: "downhill_flat",
# "flat_downhill", "flat_flat".
POLARITY_OPTIONS = [
    "uphill_downhill",
    "uphill_flat",
    "flat_uphill",
    "downhill_uphill",
]

# Adjustments for euclidean distance, to account for paths taken to water not
# being straight lines.
URBAN_ADJUSTMENT_LOWER_ESTIMATE = 1.2
URBAN_ADJUSTMENT_UPPER_ESTIMATE = 1.5

# Parameters for the generalised Pareto distribution used for rural adjustments
# (shape, scale, loc). Obtained from scripts/create_pareto_distribution.py.
RURAL_PDR_PARETO_SHAPE = 0.20007812499999994
RURAL_PDR_PARETO_SCALE = 0.19953125000000005
RURAL_PDR_PARETO_LOC = 1.0

# -----------------------------------------------------------------------------
# WATER-SOURCE ACCOUNTING PARAMETERS
# -----------------------------------------------------------------------------

# Share of borehole users who lose access when electric pumping fails, sampled
# uniformly over the HEMP-scenario ranges (population-weighted envelopes of the
# per-country bounds) from ELECTRIC_BOREHOLE_SHARE_ESTIMATE.md. Scenario logic
# lives in these parameters, not in the dataset: a loss-of-central-
# infrastructure run is the same code with different bounds.
URBAN_BOREHOLE_LOSE_FRACTION_LOWER_ESTIMATE = 0.63
URBAN_BOREHOLE_LOSE_FRACTION_UPPER_ESTIMATE = 0.80

RURAL_BOREHOLE_LOSE_FRACTION_LOWER_ESTIMATE = 0.23
RURAL_BOREHOLE_LOSE_FRACTION_UPPER_ESTIMATE = 0.35

# Share of the population whose water source type is UNALLOCATED after
# harmonisation (171.8M people in the shipped merged_data.csv) that is treated
# as grid-vulnerable. Sampled uniformly over the full 0-1 bracket by default:
# 0 = the pre-2026-08 assumption that all of them retain access,
# 1 = all of them are grid-dependent. See UNALLOCATED_GAP_INVESTIGATION.md.
UNALLOCATED_VULNERABLE_FRACTION_LOWER_ESTIMATE = 0.0
UNALLOCATED_VULNERABLE_FRACTION_UPPER_ESTIMATE = 1.0

# Set to a float in [0, 1] or the string "prorata" to pin g for a deterministic
# SCENARIO RUN instead of sampling it. Every driver that imports
# :func:`sample_parameters` then holds g constant across all iterations:
#   * "prorata" -- the recommended CENTRAL scenario: the unallocated population
#     mirrors each stratum's known source mix (~ +2.5% affected).
#   * 0.0 -- lower bracket, all unallocated people retain access (the
#     pre-2026-08 behaviour).
#   * 1.0 -- upper bracket, all unallocated people are grid-vulnerable
#     (~ +5.8% affected).
# None (the default) samples g ~ Uniform(0, 1) per iteration.
UNALLOCATED_VULNERABLE_FRACTION_FIXED = None

# Keys of the parameter dict returned by :func:`sample_parameters`, in the order
# ``src.gis_monte_carlo.run_simulation`` expects its positional arguments. The
# three water-source parameters sit last because they are the last three
# positional arguments of that signature, immediately after
# ``rural_adjustment`` -- a driver may therefore zip these names straight into a
# starmap.
PARAMETER_NAMES = (
    "crr_adjustments",
    "time_gatherings",
    "practical_limits_bicycle",
    "practical_limits_buckets",
    "mets",
    "watts_values",
    "hill_polarities",
    "urban_adjustments",
    "rural_adjustments",
    "urban_borehole_lose_fractions",
    "rural_borehole_lose_fractions",
    "unallocated_vulnerable_fractions",
)


def sample_parameters(num_iterations):
    """Draw ``num_iterations`` Monte-Carlo parameter sets.

    Owning the *distribution family* here as well as the bounds is deliberate:
    the historical divergence between the drivers was partly a bounds
    disagreement and partly a normal-vs-lognormal disagreement, and constants
    alone would only have fixed the first half.

    Args:
        num_iterations (int): number of simulation iterations to draw for.

    Returns:
        dict: maps each name in :data:`PARAMETER_NAMES` to a length
        ``num_iterations`` ``numpy.ndarray``. Zip the values together to get one
        argument tuple per iteration.
    """
    # Imported lazily so that this module stays cheap to import for callers that
    # only want the constants (e.g. the solver-LUT validation harness).
    import numpy as np

    import src.gis_monte_carlo as mc

    if UNALLOCATED_VULNERABLE_FRACTION_FIXED is None:
        unallocated_vulnerable_fractions = np.random.uniform(
            UNALLOCATED_VULNERABLE_FRACTION_LOWER_ESTIMATE,
            UNALLOCATED_VULNERABLE_FRACTION_UPPER_ESTIMATE,
            size=num_iterations,
        )
    else:
        # Deterministic scenario run: the same g for every iteration. dtype=object
        # keeps "prorata" a plain Python string through .tolist(), JSON and CSV.
        unallocated_vulnerable_fractions = np.full(
            num_iterations, UNALLOCATED_VULNERABLE_FRACTION_FIXED, dtype=object
        )

    return {
        "crr_adjustments": np.random.randint(
            CRR_LOWER_ESTIMATE, CRR_UPPER_ESTIMATE + 1, size=num_iterations
        ),
        "time_gatherings": mc.sample_lognormal(
            TIME_GATHERING_LOWER_ESTIMATE,
            TIME_GATHERING_UPPER_ESTIMATE,
            num_iterations,
        ),
        "practical_limits_bicycle": mc.sample_normal(
            PRACTICAL_LIMITS_BICYCLE_LOWER_ESTIMATE,
            PRACTICAL_LIMITS_BICYCLE_UPPER_ESTIMATE,
            num_iterations,
        ),
        "practical_limits_buckets": mc.sample_normal(
            PRACTICAL_LIMITS_BUCKET_LOWER_ESTIMATE,
            PRACTICAL_LIMITS_BUCKET_UPPER_ESTIMATE,
            num_iterations,
        ),
        "mets": mc.sample_normal(
            METS_LOWER_ESTIMATE, METS_UPPER_ESTIMATE, num_iterations
        ),
        "watts_values": mc.sample_normal(
            WATTS_LOWER_ESTIMATE, WATTS_UPPER_ESTIMATE, num_iterations
        ),
        "hill_polarities": np.random.choice(POLARITY_OPTIONS, num_iterations),
        "urban_adjustments": mc.sample_normal(
            URBAN_ADJUSTMENT_LOWER_ESTIMATE,
            URBAN_ADJUSTMENT_UPPER_ESTIMATE,
            num_iterations,
        ),
        "rural_adjustments": mc.sample_gpd(
            RURAL_PDR_PARETO_SHAPE,
            RURAL_PDR_PARETO_SCALE,
            RURAL_PDR_PARETO_LOC,
            num_iterations,
        ),
        "urban_borehole_lose_fractions": np.random.uniform(
            URBAN_BOREHOLE_LOSE_FRACTION_LOWER_ESTIMATE,
            URBAN_BOREHOLE_LOSE_FRACTION_UPPER_ESTIMATE,
            size=num_iterations,
        ),
        "rural_borehole_lose_fractions": np.random.uniform(
            RURAL_BOREHOLE_LOSE_FRACTION_LOWER_ESTIMATE,
            RURAL_BOREHOLE_LOSE_FRACTION_UPPER_ESTIMATE,
            size=num_iterations,
        ),
        "unallocated_vulnerable_fractions": unallocated_vulnerable_fractions,
    }
