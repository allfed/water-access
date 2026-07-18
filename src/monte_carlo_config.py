"""Shared Monte Carlo parameter configuration.

Single source of truth for the walking/cycling sensitivity parameters used by
the three Monte Carlo run scripts:

    scripts/run_monte_carlo.py
    scripts/run_monte_carlo_gcp.py       (this is what the cloud Spot VM runs)
    scripts/run_monte_carlo_test.py

Previously each script held its own copy-pasted copy of these constants, which
meant editing one script silently left the others (including the cloud script)
running stale parameters. Import from here instead so the three scripts can
never drift apart.

NUM_ITERATIONS / MAX_WORKERS are intentionally NOT defined here -- they
legitimately differ per script (the test script uses 2), so each script keeps
its own local values.

⚠️  CHECKPOINT WARNING: run_monte_carlo_gcp.py writes a ``checkpoint.json`` that
serialises the *sampled* parameter arrays and restores them on resume. If you
change any parameter value or distribution below, you MUST delete the existing
``results/parquet_files/checkpoint.json`` before the next run, otherwise the
stale checkpoint will override the new parameters on resume.

⚠️  PROVISIONAL VALUES: the METS, TIME_GATHERING and WATTS-derivation choices
below reflect the 2026-07 model review recommendations and are PROVISIONAL,
pending co-author (James) sign-off. See REVIEW_AND_IMPLEMENTATION_PLAN.md §1.7.
"""

import numpy as np

# -------------------------------------------------------------------------------
# DEFINE WALKING AND CYCLING PARAMETERS FOR MONTE CARLO SIMULATIONS
# -------------------------------------------------------------------------------

# CRR adjustments.
# 1 means one road type better, -1 means one road type worse
# Sampled via np.random.randint (integers), not the CI samplers.
CRR_LOWER_ESTIMATE = -1
CRR_UPPER_ESTIMATE = 1

# Time gathering water in hours.
# PROVISIONAL: sampled as a LOGNORMAL distribution (was normal 4-7). low/high
# are treated as a 90% CI, giving a median of ~sqrt(2*5) ≈ 3.16 h. Manuscript
# sources support up to ~4 h/day. Distance is LINEAR in time, so this is the
# single biggest sensitivity lever -- flag in Methods.
# NOTE: this parameter must be sampled with mc.sample_lognormal (NOT
# sample_normal). sample_lognormal asserts low > 0, which holds here.
TIME_GATHERING_LOWER_ESTIMATE = 2
TIME_GATHERING_UPPER_ESTIMATE = 5

# Practical load limits for cycling in kg
PRACTICAL_LIMITS_BICYCLE_LOWER_ESTIMATE = 30
PRACTICAL_LIMITS_BICYCLE_UPPER_ESTIMATE = 45

# Practical load limits for walking with buckets in kg
PRACTICAL_LIMITS_BUCKET_LOWER_ESTIMATE = 15
PRACTICAL_LIMITS_BUCKET_UPPER_ESTIMATE = 25

# Average METS available for walking with buckets to and from water source.
# PROVISIONAL: normal, low=3, high=5 (median 4). Was 3-6 (median 4.5).
# Evidence centers water-carrying at 4.3-5.0 METs (Mozambique 4.3, Baka 7.28
# outlier); median 4 is a defensible compromise -- floor at empty-walking cost
# (~3.1), cap below the Baka outlier. Pending co-author sign-off.
METS_LOWER_ESTIMATE = 3
METS_UPPER_ESTIMATE = 5

# Reference body mass (kg) used for the ACSM watts-from-METs derivation.
# Country-specific weights are used in the full model; 62 kg is the
# sensitivity/analytic reference mass.
REFERENCE_MASS_KG = 62.0

# Polarity options (randomly chosen from list each simulation run)
# The first word defines the trip to the water source
# The second word defines the trip from the water source
# Options to include: "uphill_downhill", "downhill_uphill", "uphill_flat",
# "flat_uphill", "downhill_flat", "flat_downhill", "flat_flat"

POLARITY_OPTIONS = [
    "uphill_downhill",
    "uphill_flat",
    "flat_uphill",
    "downhill_uphill",
]

# Adjustments for euclidean distance to account for paths taken to water
# not being straight lines
URBAN_ADJUSTMENT_LOWER_ESTIMATE = 1.2
URBAN_ADJUSTMENT_UPPER_ESTIMATE = 1.5

# Set the parameters for the GPD distribution for rural adjustments.
# Shape, scale, and loc
# These values were obtained from the scripts/create_pareto_distribution.py
RURAL_PDR_PARETO_SHAPE = 0.20007812499999994
RURAL_PDR_PARETO_SCALE = 0.19953125000000005
RURAL_PDR_PARETO_LOC = 1.0

# -------------------------------------------------------------------------------


def derive_watts_from_mets(mets, reference_mass_kg=REFERENCE_MASS_KG):
    """Derive mechanical cycling watts from the sampled walking METs.

    PROVISIONAL (pending co-author sign-off) -- REVIEW_AND_IMPLEMENTATION_PLAN §1.7,
    Option A. Replaces the previous independent WATTS sampling (was normal
    20-80 W). Tying the cyclist and walker to one shared metabolic budget kills
    the impossible "strong-walker + feeble-cyclist" combinations that inflate
    the CI, and makes the efficiency assumption auditable.

    Inverts the ACSM cycle-ergometry equation
        VO2 = 1.8 * (W * 6.12) / mass + 7      (mL/kg/min)
        VO2 = MET * 3.5
    to solve for mechanical power W at the drivetrain, at a reference mass of
    62 kg. Floored at ~5 W (≈ the 2-MET unloaded pedalling cost) rather than an
    arbitrary constant.

    Check values (62 kg): MET 3 -> ~20 W, 4 -> ~39 W, 4.5 -> ~49 W, 5 -> ~59 W;
    MET 2 -> floored at 5 W.

    CAVEAT: the ACSM equation is validated >= 50 W; the fetching regime is below
    that, but its fixed +7 intercept makes downward extrapolation stable.

    Parameters:
    - mets (float or numpy.ndarray): sampled MET value(s).
    - reference_mass_kg (float): reference body mass in kg.

    Returns:
    - float or numpy.ndarray: mechanical watts, floored at 5 W.
    """
    return np.maximum(
        (mets * 3.5 - 7.0) * reference_mass_kg / (1.8 * 6.12),
        5.0,
    )
