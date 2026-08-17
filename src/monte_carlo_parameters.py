"""Monte-Carlo sampling parameters.

Set your estimates for lower and upper bounds for the scenario you want to run.

Monte-Carlo driver scripts draw their per-iteration parameters from here
(``scripts/run_monte_carlo.py``, ``scripts/run_monte_carlo_gcp.py``,
``scripts/run_monte_carlo_test.py``)
"""

# -----------------------------------------------------------------------------
# WALKING AND CYCLING PARAMETERS FOR THE MONTE CARLO SIMULATIONS
# -----------------------------------------------------------------------------

# CRR adjustments (sampled as a uniform integer over the inclusive range).
# 1 means one road type better, -1 means one road type worse.
CRR_LOWER_ESTIMATE = -1
CRR_UPPER_ESTIMATE = 1

# Time gathering water, in hours. Sampled LOGNORMALLY (strictly positive,
# right-skewed): a 2-4 h 90% interval.
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

# Share of borehole users who lose access when electric pumping fails
URBAN_BOREHOLE_LOSE_FRACTION_LOWER_ESTIMATE = 0.63
URBAN_BOREHOLE_LOSE_FRACTION_UPPER_ESTIMATE = 0.80

RURAL_BOREHOLE_LOSE_FRACTION_LOWER_ESTIMATE = 0.23
RURAL_BOREHOLE_LOSE_FRACTION_UPPER_ESTIMATE = 0.35

# Share of the population whose water source type is UNALLOCATED/unknown who
# lose access when electric pumping fails. Sampled uniformly over the
# full 0-1 bracket: 0 = all unaffected, 1 = all grid-vulnerable.
UNALLOCATED_VULNERABLE_FRACTION_LOWER_ESTIMATE = 0.0
UNALLOCATED_VULNERABLE_FRACTION_UPPER_ESTIMATE = 1.0

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

    Args:
        num_iterations (int): number of simulation iterations to draw for.

    Returns:
        dict: maps each name in :data:`PARAMETER_NAMES` to a length
        ``num_iterations`` ``numpy.ndarray``. Zip the values together to get one
        argument tuple per iteration.
    """
    # Imported lazily so that this module stays cheap to import for callers that
    # only want the parameter bounds.
    import numpy as np

    import src.gis_monte_carlo as mc

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
        "unallocated_vulnerable_fractions": np.random.uniform(
            UNALLOCATED_VULNERABLE_FRACTION_LOWER_ESTIMATE,
            UNALLOCATED_VULNERABLE_FRACTION_UPPER_ESTIMATE,
            size=num_iterations,
        ),
    }
