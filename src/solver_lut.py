"""Per-draw solver lookup tables for the walking and cycling mobility models.

The global analysis used to call ``fsolve`` once per grid cell per model
(~2.19M cells at 5 arc-minutes, ~140M at 30 arc-seconds). That is ~4 minutes of
solver time today and would be ~7 hours per draw at 30 arc-seconds.

The velocities, however, only depend on three per-cell quantities -- terrain
``slope``, the road-class rolling resistance ``Crr``, and the country's average
body ``mass`` -- plus a handful of scalars that are constant for a whole Monte
Carlo draw (rider power, MET, hill polarity, load attempted, vehicle mass). So
for each draw we solve the models once on a coarse grid of
``(slope x Crr x mass)`` and then interpolate ``(slope, mass)`` per Crr class
onto the cells -- linearly in slope, cubically in mass. The cost becomes ~10^5
solver calls per draw *independent of the number of cells*.

Fidelity notes (why this reproduces ``single_bike_run``/``single_lankford_run``
to well inside the LUT validation tolerance):

* The bike LUT stores the **uncapped** root of the power equation. The 7 m/s
  ceiling in ``single_bike_run`` is a kink; interpolating across it would smear
  it over one grid interval. Interpolating the smooth uncapped root and then
  applying ``min(v, 7)`` reproduces the kink exactly.
* ``max_load`` is not interpolated at all. ``max_safe_load`` plus its two clamps
  is closed-form arithmetic, so it is evaluated exactly, vectorised.
* The slope grid is densest near zero (that is where ``dv/dslope`` is largest
  for both models) and always contains the two slopes at which the Lankford
  gradient clip ``clip(tan(s) * 100, -18, 40)`` engages, so that kink also lands
  on a node rather than inside an interval.
* ``single_bike_run`` walks an initial-guess ladder of ``[1, 10, 50]`` m/s and
  accepts the first non-negative answer *without checking whether fsolve
  converged*. In razor-thin bands of the parameter space -- where the initial
  guess sits on the cubic's turning point, i.e. where
  ``3 * a * guess**2 + beta == 0`` -- fsolve bails out with ``ier == 5`` and
  returns a non-negative non-root, which the model then uses. That is a
  discontinuous artefact no smooth interpolant can reproduce, so cells whose
  ``beta`` lands near one of those turning points are simply handed back to
  ``single_bike_run`` and solved exactly. It is a tiny fraction of cells, and it
  keeps the refactor bit-faithful to the frozen model instead of silently
  "fixing" it (see the non-goals in the spec).

``single_bike_run`` and ``single_lankford_run`` are untouched and remain the
ground truth; ``scripts/validate_lut.py`` (LUT validation) checks this layer against them.
"""

import contextlib

import numpy as np

#: When set, :func:`build_slope_grid` / :func:`build_mass_grid` span these
#: ranges instead of the range of the values they are handed. Managed by
#: :func:`pinned_grid_ranges`.
_PINNED_RANGES = {"slope": None, "mass": None}


@contextlib.contextmanager
def pinned_grid_ranges(slope_range=None, mass_range=None):
    """Pin the LUT axes to a fixed span for the duration of the block.

    The chunked driver processes one group of countries at a time. Left to
    itself each chunk would build its grid over *its own* observed slope and
    mass range, so the same cell would be interpolated on a different grid
    depending on which chunk it landed in, and the chunked and monolithic runs
    would disagree in the last few decimal places. Pinning both axes to the
    globally observed ranges (measured once, in the planning pre-pass) makes the
    LUT a pure function of the draw -- identical in every chunk, and identical
    to the one a whole-world run builds.

    Args:
        slope_range (tuple, optional): ``(min, max)`` slope in degrees.
        mass_range (tuple, optional): ``(min, max)`` body mass in kg.
    """
    previous = dict(_PINNED_RANGES)
    if slope_range is not None:
        _PINNED_RANGES["slope"] = (float(slope_range[0]), float(slope_range[1]))
    if mass_range is not None:
        _PINNED_RANGES["mass"] = (float(mass_range[0]), float(mass_range[1]))
    try:
        yield
    finally:
        _PINNED_RANGES.update(previous)


#: When true, :func:`should_use_lut` ignores its row-count floor.
_FORCE_LUT = [False]


@contextlib.contextmanager
def force_lut():
    """Take the LUT path regardless of row count, for the duration of the block.

    :func:`should_use_lut` normally sends small frames down the original
    per-cell solver loop, which is *more* accurate than the LUT. That is the
    right default, but it would make a chunked run disagree with a monolithic
    one whenever a chunk happened to fall under the threshold. Chunked runs pin
    the choice so every cell is solved the same way wherever it lands.
    """
    previous = _FORCE_LUT[0]
    _FORCE_LUT[0] = True
    try:
        yield
    finally:
        _FORCE_LUT[0] = previous


#: Active memo for built LUTs, or None when caching is off.
_LUT_CACHE = [None]


@contextlib.contextmanager
def lut_cache():
    """Memoise built lookup tables for the duration of the block.

    A chunked run solves the same draw over and over, once per country group,
    and with the axes pinned the table it builds is *bit-identical* every time.
    Caching turns an O(chunks) solver cost back into O(1) -- measured at 5
    arc-minutes with seven chunks, it takes the chunked run from 145 s to 45 s.

    Off by default, because a cached table is a few MB that would otherwise
    live for the whole process, and the monolithic path builds each table
    exactly once anyway.
    """
    previous = _LUT_CACHE[0]
    _LUT_CACHE[0] = {}
    try:
        yield
    finally:
        _LUT_CACHE[0] = previous


def _cached(key, build):
    """Return ``build()``, memoised under ``key`` when caching is enabled."""
    cache = _LUT_CACHE[0]
    if cache is None:
        return build()
    if key not in cache:
        cache[key] = build()
    return cache[key]


def _grid_key(*grids):
    """A hashable identity for a set of grids: their exact bytes."""
    return tuple(np.asarray(grid, dtype=float).tobytes() for grid in grids)


def _scalar(value):
    """Collapse a possibly-array model attribute to a plain float."""
    return float(np.asarray(value).flatten()[0])


def _span(values, axis):
    """The (lo, hi) span an axis should cover: pinned if set, else observed."""
    pinned = _PINNED_RANGES[axis]
    if pinned is not None:
        return pinned
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return None
    return float(finite.min()), float(finite.max())


# ---------------------------------------------------------------------------
# Grid sizing 
# ---------------------------------------------------------------------------

#: Number of slope nodes. The LUT grid budgets 60k-130k solver calls per draw; with the
#: six road classes and 21 mass nodes, 1000 slope nodes lands at 126k, i.e. the
#: top of that budget. The slope axis is where the residual interpolation error
#: lives once the mass axis is cubic, so it gets the nodes.
DEFAULT_N_SLOPE = 1000

#: Number of body-mass nodes. ~21 for the LUT grid.
DEFAULT_N_MASS = 21

#: Slope nodes are placed at ``s_max * u**SLOPE_GRID_POWER`` for uniform ``u``,
#: which concentrates them near zero where the velocity curves bend hardest.
SLOPE_GRID_POWER = 3.0

#: Below this many cells the lookup table costs more than it saves, so the
#: original per-cell solver loop is used instead. This also keeps the unit
#: tests that monkeypatch ``single_bike_run``/``single_lankford_run`` and count
#: the calls on the exact legacy code path.
LUT_MIN_ROWS = 1000

#: Cells interpolated per pass. Interpolation gathers sixteen temporaries the
#: length of its input, so an unchunked 140M-cell frame would need ~18 GB of
#: scratch on its own .
EVAL_CHUNK_SIZE = 4_000_000

#: A LUT is built per distinct Crr value. Real data has six road classes; if a
#: caller passes something pathological we fall back to the legacy loop rather
#: than build a huge table.
MAX_CRR_CLASSES = 32

#: Velocity ceiling applied by ``single_bike_run`` after solving.
BIKE_MAX_VELOCITY = 7.0

#: Walking velocity ceiling (7 km/h), applied after LUT interpolation.
WALK_MAX_VELOCITY = 7.0 / 3.6

#: Gradient percentages at which ``Lankford_solution`` clips ``G``. The model is
#: only validated over -18%..+40% grade.
LANKFORD_CLIP_GRADES = (-18.0, 40.0)

#: The initial-guess ladder ``single_bike_run`` walks.
BIKE_VELOCITY_GUESSES = (1, 10, 50)

#: Guesses whose stall band actually changes the answer. Measured over
#: ``P_t`` in 20..80 W: the stall bands around the 10 m/s and 50 m/s guesses sit
#: entirely inside the region where the true root is well above the 7 m/s
#: ceiling, so the reference and the LUT both return exactly 7 there and the
#: artefact is invisible. Only the 1 m/s guess produces a visible artefact.
BIKE_CRITICAL_GUESSES = (1,)

#: Half-width, as a fraction of the critical value, of the band around
#: ``beta_crit = -3 * a * guess**2`` in which ``single_bike_run``'s fsolve call
#: returns a non-converged answer. Measured bands are 0.7%-1.5% wide over
#: ``P_t`` in 20..80 W; 5% is a >3x margin and still flags well under 0.1% of
#: real cells.
CRITICAL_BETA_REL_WINDOW = 0.05


def _lankford_clip_slopes_deg():
    """Slopes (degrees, positive) at which the Lankford gradient clip engages.

    ``Lankford_solution`` uses ``G = clip(tan(s) * 100, -18, 40)`` where ``s``
    is the slope in radians multiplied by the hill polarity (+1/-1/0). The clip
    is a kink in slope, so both breakpoints are forced onto the slope grid.
    """
    return tuple(
        float(np.degrees(np.arctan(abs(grade) / 100.0)))
        for grade in LANKFORD_CLIP_GRADES
    )


def build_slope_grid(slope_values, n_slope=DEFAULT_N_SLOPE):
    """Build the slope axis of a LUT, spanning the observed slope range.

    Nodes are cubically spaced so they bunch up near zero -- both models change
    fastest there -- and the Lankford clip breakpoints are inserted exactly.

    Args:
        slope_values (array-like): the per-cell slopes the LUT must cover.
        n_slope (int): approximate number of nodes.

    Returns:
        numpy.ndarray: sorted, strictly increasing slope nodes (degrees).
    """
    span = _span(slope_values, "slope")
    if span is None:
        return np.array([0.0, 1.0])

    lo, hi = span
    if hi <= lo:
        # Degenerate (single slope): a two-node grid still interpolates exactly.
        return np.array([lo, lo + 1.0])

    u = np.linspace(0.0, 1.0, max(int(n_slope), 2))
    grid = lo + (hi - lo) * u**SLOPE_GRID_POWER

    breakpoints = [b for b in _lankford_clip_slopes_deg() if lo < b < hi]
    if breakpoints:
        grid = np.concatenate([grid, np.asarray(breakpoints, dtype=float)])

    grid = np.unique(grid)
    return grid


def build_mass_grid(mass_values, n_mass=DEFAULT_N_MASS):
    """Build the body-mass axis of a LUT, spanning the observed country weights.

    Evenly spaced, because the mass axis is sampled with a Catmull-Rom cubic
    (see :func:`_interpolate_mass`) which assumes a uniform grid.

    Args:
        mass_values (array-like): the per-cell average body masses (kg).
        n_mass (int): number of nodes.

    Returns:
        numpy.ndarray: sorted, strictly increasing mass nodes (kg).
    """
    span = _span(mass_values, "mass")
    if span is None:
        return np.array([62.0, 63.0])

    lo, hi = span
    if hi <= lo:
        return np.array([lo, lo + 1.0])

    return np.linspace(lo, hi, max(int(n_mass), 2))


# ---------------------------------------------------------------------------
# Vectorised bilinear interpolation
# ---------------------------------------------------------------------------


def _axis_weights(grid, values):
    """Return (lower index, upper index, upper weight) for linear interpolation.

    Values outside the grid are clamped to the end nodes. Non-finite values
    produce non-finite weights, which propagate NaN through the interpolation.
    """
    values = np.asarray(values, dtype=float)
    idx = np.searchsorted(grid, values, side="right") - 1
    idx = np.clip(idx, 0, grid.size - 2)
    lower = grid[idx]
    upper = grid[idx + 1]
    with np.errstate(invalid="ignore", divide="ignore"):
        frac = (values - lower) / (upper - lower)
    frac = np.clip(frac, 0.0, 1.0)
    # Preserve NaN inputs (np.clip turns NaN into NaN, but searchsorted has
    # already produced a valid index, so re-apply the mask explicitly).
    frac = np.where(np.isfinite(values), frac, np.nan)
    return idx, frac


def _interpolate_mass(nodes, mass_frac):
    """Sample along the mass axis from four neighbouring nodes.

    Uses a Catmull-Rom cubic. Velocity is an analytic function of body mass in
    both models (mass enters the bike model only through the total mass in a
    smooth cubic, and the Lankford model only through ``budget_VO2 / m_load``),
    so there is no kink for a cubic to ring against -- unlike the slope axis,
    where the Lankford gradient clip and the bike's 7 m/s ceiling both bite.

    Measured: switching this axis from linear to cubic drops the LUT's peak
    velocity error by ~30x at no extra solver cost, because the mass axis is
    where the bulk of the interpolation error lived.

    Args:
        nodes: sequence of four arrays ``(p_-1, p_0, p_1, p_2)``, the table
            values at the mass nodes surrounding each cell.
        mass_frac: position within the ``p_0 .. p_1`` interval, in ``[0, 1]``.
    """
    p0, p1, p2, p3 = nodes
    t = mass_frac
    return p1 + 0.5 * t * (
        (p2 - p0)
        + t * ((2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) + t * (3.0 * (p1 - p2) + p3 - p0))
    )


def extend_mass_axis(table):
    """Pad the trailing (mass) axis with one cubically extrapolated ghost node
    at each end, so the Catmull-Rom stencil stays four-wide in the boundary
    intervals instead of degrading to a clamped (quadratic) endpoint.

    ``3*T[0] - 3*T[1] + T[2]`` is the value a cubic through the first three
    nodes would take one step before them, so the boundary stencil still
    reproduces cubics exactly.
    """
    if table.shape[-1] < 3:
        return table
    leading = 3.0 * table[..., 0] - 3.0 * table[..., 1] + table[..., 2]
    trailing = 3.0 * table[..., -1] - 3.0 * table[..., -2] + table[..., -3]
    return np.concatenate([leading[..., None], table, trailing[..., None]], axis=-1)


def _sample_table(table, slope_idx, slope_frac, mass_idx, mass_frac, crr_idx=None):
    """Sample a mass-extended ``table`` at the supplied cell coordinates.

    Cubic along the mass axis, linear along the slope axis (see
    :func:`_interpolate_mass` for why the two axes differ). ``table`` must
    already have been through :func:`extend_mass_axis`.

    Args:
        table: ``(n_slope, n_mass + 2)`` or ``(n_crr, n_slope, n_mass + 2)``.
        slope_idx, slope_frac: output of :func:`_axis_weights` on the slope axis.
        mass_idx, mass_frac: output of :func:`_axis_weights` on the mass axis.
        crr_idx: per-cell index into the leading axis, when the table has one.

    Returns:
        numpy.ndarray: interpolated values, one per cell.
    """
    padded = table.shape[-1] >= 4
    offsets = (0, 1, 2, 3) if padded else (0, 0, 1, 1)

    def column(slope_offset):
        rows = slope_idx + slope_offset
        if crr_idx is None:
            gathered = [table[rows, mass_idx + offset] for offset in offsets]
        else:
            gathered = [table[crr_idx, rows, mass_idx + offset] for offset in offsets]
        if padded:
            return _interpolate_mass(gathered, mass_frac)
        return gathered[1] + (gathered[2] - gathered[1]) * mass_frac

    low = column(0)
    high = column(1)
    return low + (high - low) * slope_frac


# ---------------------------------------------------------------------------
# Bicycle
# ---------------------------------------------------------------------------


def bike_max_load(mm, mv, hpv, slope_deg, load_attempted):
    """Vectorised, exact replica of the ``max_load_HPV`` logic in ``single_bike_run``.

    ``max_safe_load`` followed by the clamps to ``load_capacity`` and to
    ``load_attempted`` is closed-form, so it is computed directly rather than
    interpolated.

    Args:
        mm: the ``src.mobility_module`` module (passed in to avoid an import
            cycle and to match the lazy-import style of the callers).
        mv: model variables (supplies ``F_max`` and ``g``).
        hpv: HPV variables (supplies ``m_HPV_only`` and ``load_capacity``).
        slope_deg (numpy.ndarray): per-cell slope in degrees.
        load_attempted (float): the load the user tries to carry.

    Returns:
        numpy.ndarray: per-cell maximum safe load in kg.
    """
    del mm  # kept for signature symmetry with the rest of the module
    slope_deg = np.asarray(slope_deg, dtype=float)
    s = (slope_deg / 360.0) * (2.0 * np.pi)

    m_hpv_only = float(np.asarray(hpv.m_HPV_only).flatten()[0])
    load_capacity = float(np.asarray(hpv.load_capacity).flatten()[0])

    sin_s = np.sin(s)
    with np.errstate(divide="ignore", invalid="ignore"):
        max_pushable = mv.F_max / (sin_s * mv.g) - m_hpv_only
    # ``max_safe_load`` leaves the load capacity untouched when the slope is
    # exactly zero (it guards the division), so mirror that branch.
    max_load = np.where(s != 0, max_pushable, load_capacity)

    max_load = np.minimum(max_load, load_capacity)
    max_load = np.minimum(max_load, float(load_attempted))
    return max_load


def _solve_bike_node(model, data):
    """Solve the bike power equation at one LUT node, reliably.

    Walks the same guess ladder as ``single_bike_run`` but *checks convergence*
    and falls back to a bracketed Brent solve. A non-converged answer baked into
    a LUT node would smear across a whole grid interval, so nodes must always
    carry the true root; the reference's non-converged answers are reproduced
    separately, per cell, by :func:`critical_beta_mask`.
    """
    from scipy.optimize import brentq, fsolve

    for guess in BIKE_VELOCITY_GUESSES:
        solution = fsolve(model, guess, args=data, full_output=True)
        velocity = np.asarray(solution[0]).item()
        if solution[2] == 1 and velocity >= 0:
            return velocity

    # The residual is a v**3 + beta v - eta P_t (up to the 1/eta scaling) and
    # always has exactly one positive real root, so bracket and bisect.
    def residual(v):
        return float(np.asarray(model(np.array([v]), *data)).ravel()[0])

    upper = 1.0
    for _ in range(60):
        if residual(upper) > 0:
            break
        upper *= 2.0
    else:  # pragma: no cover - would need a degenerate parameter set
        raise ValueError("No valid solution found.")
    return brentq(residual, 0.0, upper, xtol=1e-12, rtol=1e-14)


def _bike_uncapped_run(mm, mv, mo, hpv, slope, load_attempted):
    """``single_bike_run`` without the 7 m/s ceiling, for one grid node.

    Mirrors ``mobility_models.single_bike_run`` (same residual, same load
    clamps, same argument packing) but returns the raw roots so the ceiling can
    be applied after interpolation.
    """
    model = mm.mobility_models.bike_power_solution
    s = (slope / 360) * (2 * np.pi)

    max_load_HPV = mm.max_safe_load(
        hpv.m_HPV_only, hpv.load_capacity, mv.F_max, s, mv.g
    )
    if max_load_HPV > hpv.load_capacity:
        max_load_HPV = hpv.load_capacity.flatten()
    if max_load_HPV > load_attempted:
        max_load_HPV = load_attempted

    velocities = []
    for extra_mass, polarity in ((0.0, mo.ulhillpo), (max_load_HPV, mo.lhillpo)):
        data = (
            mv.ro,
            mv.C_d,
            mv.A,
            mv.m1 + hpv.m_HPV_only.flatten() + extra_mass,
            hpv.Crr,
            mv.eta,
            mv.P_t,
            mv.g,
            s * polarity,
        )
        velocities.append(_solve_bike_node(model, data))

    unloaded_velocity, loaded_velocity = velocities
    return loaded_velocity, unloaded_velocity


def critical_beta_mask(mv, mo, hpv, slope_deg, crr, mass, max_load):
    """Cells where ``single_bike_run``'s fsolve ladder may not have converged.

    The residual seen by fsolve depends on the per-cell inputs only through
    ``beta = m_t * g * (Crr + s * polarity)``. The solver stalls when an initial
    guess sits on the cubic's turning point, ``3 * a * guess**2 + beta == 0``.
    This flags cells within :data:`CRITICAL_BETA_REL_WINDOW` of any of those
    three turning points, for either leg of the trip.

    Args:
        mv, mo, hpv: the per-draw model objects.
        slope_deg (numpy.ndarray): per-cell slope in degrees.
        crr (numpy.ndarray): per-cell rolling resistance.
        mass (numpy.ndarray): per-cell average body mass, kg.
        max_load (numpy.ndarray): per-cell max safe load, kg.

    Returns:
        numpy.ndarray: boolean mask of cells needing an exact solve.
    """
    a_coefficient = 0.5 * mv.ro * mv.C_d * mv.A
    s = (np.asarray(slope_deg, dtype=float) / 360.0) * (2.0 * np.pi)
    m_hpv_only = float(np.asarray(hpv.m_HPV_only).flatten()[0])
    crr = np.asarray(crr, dtype=float)
    mass = np.asarray(mass, dtype=float)

    betas = (
        (mass + m_hpv_only) * mv.g * (crr + s * mo.ulhillpo),
        (mass + m_hpv_only + np.asarray(max_load, dtype=float))
        * mv.g
        * (crr + s * mo.lhillpo),
    )

    mask = np.zeros(np.shape(s), dtype=bool)
    for guess in BIKE_CRITICAL_GUESSES:
        beta_critical = -3.0 * a_coefficient * guess**2
        window = abs(beta_critical) * CRITICAL_BETA_REL_WINDOW
        for beta in betas:
            mask |= np.abs(beta - beta_critical) <= window
    return mask


def build_bicycle_lut(
    mm,
    mv,
    mo,
    hpv,
    crr_grid,
    slope_grid,
    mass_grid,
    load_attempt,
):
    """Solve the bike model on the ``(Crr, slope, mass)`` grid for one draw.

    Returns:
        dict: ``crr_grid``/``slope_grid``/``mass_grid`` plus ``loaded`` and
        ``unloaded`` arrays of shape ``(n_crr, n_slope, n_mass)`` holding the
        *uncapped* roots.
    """
    n_crr, n_slope, n_mass = len(crr_grid), len(slope_grid), len(mass_grid)
    loaded = np.empty((n_crr, n_slope, n_mass), dtype=float)
    unloaded = np.empty((n_crr, n_slope, n_mass), dtype=float)

    original_crr = hpv.Crr
    original_m1 = mv.m1
    try:
        for ic, crr in enumerate(crr_grid):
            hpv.Crr = crr
            for im, mass in enumerate(mass_grid):
                mv.m1 = mass
                for isl, slope in enumerate(slope_grid):
                    loaded[ic, isl, im], unloaded[ic, isl, im] = _bike_uncapped_run(
                        mm, mv, mo, hpv, slope, load_attempt
                    )
    finally:
        hpv.Crr = original_crr
        mv.m1 = original_m1

    return {
        "crr_grid": np.asarray(crr_grid, dtype=float),
        "slope_grid": np.asarray(slope_grid, dtype=float),
        "mass_grid": np.asarray(mass_grid, dtype=float),
        "loaded": loaded,
        "unloaded": unloaded,
    }


def evaluate_bicycle_lut(lut, slope, crr, mass, chunk_size=EVAL_CHUNK_SIZE):
    """Interpolate a bicycle LUT onto cells and apply the 7 m/s ceiling.

    Evaluated in chunks: every gather allocates one temporary the length of the
    input, and there are sixteen of them, which at 140M cells would be ~18 GB of
    scratch in one go .

    Args:
        lut (dict): output of :func:`build_bicycle_lut`.
        slope, crr, mass (numpy.ndarray): per-cell inputs.
        chunk_size (int): cells per interpolation chunk.

    Returns:
        tuple: ``(loaded_velocity, unloaded_velocity)`` arrays.
    """
    crr_grid = lut["crr_grid"]
    slope_grid = lut["slope_grid"]
    mass_grid = lut["mass_grid"]
    loaded_table = extend_mass_axis(lut["loaded"])
    unloaded_table = extend_mass_axis(lut["unloaded"])

    n_cells = len(slope)
    loaded = np.empty(n_cells, dtype=float)
    unloaded = np.empty(n_cells, dtype=float)

    for start in range(0, n_cells, chunk_size):
        stop = min(start + chunk_size, n_cells)
        crr_idx = np.clip(
            np.searchsorted(crr_grid, np.asarray(crr[start:stop], dtype=float)),
            0,
            crr_grid.size - 1,
        )
        slope_idx, slope_frac = _axis_weights(slope_grid, slope[start:stop])
        mass_idx, mass_frac = _axis_weights(mass_grid, mass[start:stop])

        loaded[start:stop] = _sample_table(
            loaded_table, slope_idx, slope_frac, mass_idx, mass_frac, crr_idx
        )
        unloaded[start:stop] = _sample_table(
            unloaded_table, slope_idx, slope_frac, mass_idx, mass_frac, crr_idx
        )

    return (
        np.minimum(loaded, BIKE_MAX_VELOCITY),
        np.minimum(unloaded, BIKE_MAX_VELOCITY),
    )


def bicycle_results_via_lut(
    mm,
    mv,
    mo,
    hpv,
    slope,
    crr,
    mass,
    load_attempt,
    n_slope=DEFAULT_N_SLOPE,
    n_mass=DEFAULT_N_MASS,
):
    """Full LUT path for ``run_bicycle_model``.

    Returns:
        numpy.ndarray: ``(n_cells, 3)`` of loaded velocity, unloaded velocity,
        max load -- the same layout the per-cell loop produces.
    """
    crr_grid = np.unique(np.asarray(crr, dtype=float))
    crr_grid = crr_grid[np.isfinite(crr_grid)]
    slope_grid = build_slope_grid(slope, n_slope=n_slope)
    mass_grid = build_mass_grid(mass, n_mass=n_mass)

    # Everything the table's contents depend on. If two calls agree on all of
    # this, they would solve the identical grid of identical equations.
    key = (
        "bicycle",
        _grid_key(crr_grid, slope_grid, mass_grid),
        (mv.ro, mv.C_d, mv.A, mv.eta, mv.P_t, mv.g, mv.F_max),
        (mo.ulhillpo, mo.lhillpo),
        float(load_attempt),
        _scalar(hpv.m_HPV_only),
        _scalar(hpv.load_capacity),
    )
    lut = _cached(
        key,
        lambda: build_bicycle_lut(
            mm, mv, mo, hpv, crr_grid, slope_grid, mass_grid, load_attempt
        ),
    )
    loaded, unloaded = evaluate_bicycle_lut(lut, slope, crr, mass)
    max_load = bike_max_load(mm, mv, hpv, slope, load_attempt)

    results = np.empty((len(slope), 3), dtype=float)
    results[:, 0] = loaded
    results[:, 1] = unloaded
    results[:, 2] = max_load

    exact = np.flatnonzero(critical_beta_mask(mv, mo, hpv, slope, crr, mass, max_load))
    if exact.size:
        original_crr, original_m1 = hpv.Crr, mv.m1
        try:
            for i in exact:
                hpv.Crr = crr[i]
                mv.m1 = mass[i]
                results[i] = mm.mobility_models.single_bike_run(
                    mv, mo, hpv, slope[i], load_attempt
                )
        finally:
            hpv.Crr, mv.m1 = original_crr, original_m1

    return results


# ---------------------------------------------------------------------------
# Walking (Lankford)
# ---------------------------------------------------------------------------


def walking_max_load(hpv, load_attempted):
    """Exact replica of the ``max_load_HPV`` line in ``single_lankford_run``."""
    load_capacity = float(np.asarray(hpv.load_capacity).flatten()[0])
    return min(load_capacity, float(load_attempted))


def build_walking_lut(mm, mv, mo, met, hpv, slope_grid, mass_grid, load_attempt):
    """Solve the Lankford model on the ``(slope, mass)`` grid for one draw.

    The walking model has no Crr dependence (buckets are carried, not rolled),
    so the table is two-dimensional. There is no velocity ceiling either, so the
    stored values are the raw ``single_lankford_run`` outputs -- which means the
    NaN it returns on a non-converged solve is stored and propagated verbatim.
    """
    n_slope, n_mass = len(slope_grid), len(mass_grid)
    loaded = np.empty((n_slope, n_mass), dtype=float)
    unloaded = np.empty((n_slope, n_mass), dtype=float)

    original_m1 = mv.m1
    try:
        for im, mass in enumerate(mass_grid):
            mv.m1 = mass
            met_values = mm.MET_values(
                mv,
                country_weight=mass,
                met=met,
                use_country_specific_weights=True,
            )
            for isl, slope in enumerate(slope_grid):
                (
                    loaded[isl, im],
                    unloaded[isl, im],
                    _,
                ) = mm.mobility_models.single_lankford_run(
                    mv, mo, met_values, hpv, slope, load_attempt
                )
    finally:
        mv.m1 = original_m1

    return {
        "slope_grid": np.asarray(slope_grid, dtype=float),
        "mass_grid": np.asarray(mass_grid, dtype=float),
        "loaded": loaded,
        "unloaded": unloaded,
    }


def evaluate_walking_lut(lut, slope, mass, chunk_size=EVAL_CHUNK_SIZE):
    """Interpolate a walking LUT onto cells, in memory-bounded chunks.

    Returns:
        tuple: ``(loaded_velocity, unloaded_velocity)`` arrays.
    """
    slope_grid = lut["slope_grid"]
    mass_grid = lut["mass_grid"]
    loaded_table = extend_mass_axis(lut["loaded"])
    unloaded_table = extend_mass_axis(lut["unloaded"])

    n_cells = len(slope)
    loaded = np.empty(n_cells, dtype=float)
    unloaded = np.empty(n_cells, dtype=float)

    for start in range(0, n_cells, chunk_size):
        stop = min(start + chunk_size, n_cells)
        slope_idx, slope_frac = _axis_weights(slope_grid, slope[start:stop])
        mass_idx, mass_frac = _axis_weights(mass_grid, mass[start:stop])
        loaded[start:stop] = _sample_table(
            loaded_table, slope_idx, slope_frac, mass_idx, mass_frac
        )
        unloaded[start:stop] = _sample_table(
            unloaded_table, slope_idx, slope_frac, mass_idx, mass_frac
        )

    return (
        np.minimum(loaded, WALK_MAX_VELOCITY),
        np.minimum(unloaded, WALK_MAX_VELOCITY),
    )


def walking_results_via_lut(
    mm,
    mv,
    mo,
    met,
    hpv,
    slope,
    mass,
    load_attempt,
    n_slope=DEFAULT_N_SLOPE,
    n_mass=DEFAULT_N_MASS,
):
    """Full LUT path for ``run_walking_model``.

    Returns:
        numpy.ndarray: ``(n_cells, 3)`` of loaded velocity, unloaded velocity,
        max load.
    """
    slope_grid = build_slope_grid(slope, n_slope=n_slope)
    mass_grid = build_mass_grid(mass, n_mass=n_mass)

    # The Lankford path reads only these: the grids, the MET budget, the hill
    # polarities, the attempted load and the bucket capacity.
    key = (
        "walking",
        _grid_key(slope_grid, mass_grid),
        float(met),
        (mo.ulhillpo, mo.lhillpo),
        float(load_attempt),
        _scalar(hpv.load_capacity),
    )
    lut = _cached(
        key,
        lambda: build_walking_lut(
            mm, mv, mo, met, hpv, slope_grid, mass_grid, load_attempt
        ),
    )
    loaded, unloaded = evaluate_walking_lut(lut, slope, mass)

    results = np.empty((len(slope), 3), dtype=float)
    results[:, 0] = loaded
    results[:, 1] = unloaded
    results[:, 2] = walking_max_load(hpv, load_attempt)
    return results


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------


def should_use_lut(n_rows, crr=None, min_rows=LUT_MIN_ROWS):
    """Whether the LUT path is worth it (and safe) for this input.

    Small inputs go down the original per-cell loop: the LUT would cost more
    than it saves, and the unit tests that monkeypatch the single-run solvers
    depend on that loop being used.
    """
    if n_rows < min_rows and not _FORCE_LUT[0]:
        return False
    if crr is not None:
        distinct = np.unique(np.asarray(crr, dtype=float))
        distinct = distinct[np.isfinite(distinct)]
        if distinct.size == 0 or distinct.size > MAX_CRR_CLASSES:
            return False
    return True
