"""Check the solver lookup table (LUT) against the direct per-cell solvers.

Draws random (slope, Crr, body-mass) tuples -- slope from the *real* 5-arcmin
slope distribution, Crr from the six road classes, mass from the country
average-weight table -- across several Monte Carlo draws whose parameters come
from the canonical ranges in ``src/monte_carlo_parameters.py``, then compares the
LUT-interpolated loaded/unloaded walking and cycling velocities against direct
``single_lankford_run`` / ``single_bike_run`` calls.

Pass bar: ``|dv| <= min(0.005 m/s, 0.5% of v)`` and NaN / no-solve
cases agree exactly. Exits 0 on pass, 1 on failure.

Usage:
    python scripts/validate_lut.py --samples 10000 --report results/validation/lut_report.md
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import src.mobility_module as mm  # noqa: E402
import src.solver_lut as solver_lut  # noqa: E402
from src.gis_global_module import (  # noqa: E402
    CRR_FILE,
    COUNTRY_DATA_FILE,
    FILE_PATH_PARAMS,
    URB_DATA_FILE,
    load_hpv_parameters,
    map_hill_polarity,
)
from src.monte_carlo_parameters import POLARITY_OPTIONS, sample_parameters  # noqa: E402

#: ``sample_parameters`` returns plural keys; map them to the singular names
#: ``run_global_analysis`` (and this script) use.
PARAMETER_ALIASES = {
    "crr_adjustments": "crr_adjustment",
    "time_gatherings": "time_gathering_water",
    "practical_limits_bicycle": "practical_limit_bicycle",
    "practical_limits_buckets": "practical_limit_buckets",
    "mets": "met",
    "watts_values": "watts",
    "hill_polarities": "hill_polarity",
    "urban_adjustments": "urban_adjustment",
    "rural_adjustments": "rural_adjustment",
}

#: Absolute floor of the tolerance, in m/s.
ABS_TOLERANCE = 0.005

#: Relative tolerance, as a fraction of the reference velocity.
REL_TOLERANCE = 0.005

#: The load both models attempt to carry (hard-coded at the call sites in
#: ``calculate_and_merge_bicycle_distance`` / ``calculate_and_merge_walking_distance``).
LOAD_ATTEMPT = 15


def markdown_table(frame):
    """Render a DataFrame as a GitHub markdown table (no ``tabulate`` needed)."""
    columns = [str(column) for column in frame.columns]
    rows = [
        [str(value) for value in record] for record in frame.itertuples(index=False)
    ]
    widths = [
        (
            max(len(columns[i]), *(len(row[i]) for row in rows))
            if rows
            else len(columns[i])
        )
        for i in range(len(columns))
    ]
    header = "| " + " | ".join(c.ljust(w) for c, w in zip(columns, widths)) + " |"
    divider = "| " + " | ".join("-" * w for w in widths) + " |"
    body = [
        "| " + " | ".join(v.ljust(w) for v, w in zip(row, widths)) + " |"
        for row in rows
    ]
    return "\n".join([header, divider, *body])


def crr_values_for_adjustment(crr_adjustment):
    """The six road-class Crr values under a given ``crr_adjustment``.

    Mirrors the shift/ffill/bfill mapping in ``gis_global_module.road_analysis``.
    """
    df_crr = pd.read_csv(CRR_FILE)
    shifted = df_crr["Crr"].shift(-int(crr_adjustment)).ffill().bfill()
    return np.unique(shifted.to_numpy(dtype=float))


def load_sample_pools(zone_file, country_file):
    """Slope values from the real GIS extract and body masses from the country table."""
    slopes = pd.read_csv(zone_file, usecols=["slope_1"])["slope_1"].to_numpy(
        dtype=float
    )
    slopes = slopes[np.isfinite(slopes)]
    masses = pd.read_csv(country_file)["Average Weight"].to_numpy(dtype=float)
    masses = masses[np.isfinite(masses)]
    return slopes, masses


def compare(name, lut_values, reference_values, records):
    """Score one quantity against the pass bar and append a row to ``records``."""
    lut_values = np.asarray(lut_values, dtype=float)
    reference_values = np.asarray(reference_values, dtype=float)

    lut_nan = np.isnan(lut_values)
    ref_nan = np.isnan(reference_values)
    nan_mismatch = int(np.count_nonzero(lut_nan != ref_nan))

    both_finite = ~lut_nan & ~ref_nan
    delta = np.abs(lut_values[both_finite] - reference_values[both_finite])
    reference_finite = np.abs(reference_values[both_finite])
    tolerance = np.minimum(ABS_TOLERANCE, REL_TOLERANCE * reference_finite)
    breaches = int(np.count_nonzero(delta > tolerance))

    with np.errstate(divide="ignore", invalid="ignore"):
        relative = np.where(reference_finite > 0, delta / reference_finite, 0.0)

    records.append(
        {
            "quantity": name,
            "n": int(both_finite.sum()),
            "max_abs_error": f"{delta.max():.3e}" if delta.size else "0",
            "max_rel_error_pct": (f"{relative.max() * 100:.5f}" if delta.size else "0"),
            "tolerance_breaches": breaches,
            "nan_mismatches": nan_mismatch,
        }
    )
    return breaches == 0 and nan_mismatch == 0


def run_one_draw(rng, params, slope_pool, mass_pool, samples_per_draw):
    """Build LUTs for one parameter draw and score them against the solvers."""
    slope = rng.choice(slope_pool, samples_per_draw)
    mass = rng.choice(mass_pool, samples_per_draw)
    # Pin the grid endpoints to the true data range, exactly as a production run
    # would (the grid spans every cell in the frame, not just these samples).
    slope[0], slope[1] = slope_pool.min(), slope_pool.max()
    mass[0], mass[1] = mass_pool.min(), mass_pool.max()

    crr_pool = crr_values_for_adjustment(params["crr_adjustment"])
    crr = rng.choice(crr_pool, samples_per_draw)
    crr[: len(crr_pool)] = crr_pool

    ulhillpo, lhillpo = map_hill_polarity(params["hill_polarity"])

    # ---- cycling -----------------------------------------------------------
    bike_params = load_hpv_parameters(FILE_PATH_PARAMS, "Bicycle")
    bike_params["PracticalLimit"] = params["practical_limit_bicycle"]
    mo = mm.model_options(ulhillpo=ulhillpo, lhillpo=lhillpo)
    mo.model_selection = 2
    mv = mm.model_variables(P_t=params["watts"])
    hpv = mm.HPV_variables(bike_params, mv)

    lut_bike = solver_lut.bicycle_results_via_lut(
        mm, mv, mo, hpv, slope, crr, mass, LOAD_ATTEMPT
    )
    reference_bike = np.zeros_like(lut_bike)
    for i in range(samples_per_draw):
        hpv.Crr = crr[i]
        mv.m1 = mass[i]
        reference_bike[i] = mm.mobility_models.single_bike_run(
            mv, mo, hpv, slope[i], LOAD_ATTEMPT
        )

    # ---- walking -----------------------------------------------------------
    bucket_params = load_hpv_parameters(FILE_PATH_PARAMS, "Buckets")
    bucket_params["PracticalLimit"] = params["practical_limit_buckets"]
    mo_walk = mm.model_options(ulhillpo=ulhillpo, lhillpo=lhillpo)
    mo_walk.model_selection = 3
    mv_walk = mm.model_variables(m1=62)
    hpv_walk = mm.HPV_variables(bucket_params, mv_walk)

    lut_walk = solver_lut.walking_results_via_lut(
        mm, mv_walk, mo_walk, params["met"], hpv_walk, slope, mass, LOAD_ATTEMPT
    )
    reference_walk = np.zeros_like(lut_walk)
    for i in range(samples_per_draw):
        mv_walk.m1 = mass[i]
        met_values = mm.MET_values(
            mv_walk,
            country_weight=mass[i],
            met=params["met"],
            use_country_specific_weights=True,
        )
        reference_walk[i] = mm.mobility_models.single_lankford_run(
            mv_walk, mo_walk, met_values, hpv_walk, slope[i], LOAD_ATTEMPT
        )

    return {
        "bike": (lut_bike, reference_bike),
        "walk": (lut_walk, reference_walk),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=10000)
    parser.add_argument(
        "--draws",
        type=int,
        default=10,
        help="Number of Monte Carlo parameter draws to spread the samples over.",
    )
    parser.add_argument("--seed", type=int, default=20260808)
    parser.add_argument(
        "--report",
        type=Path,
        default=PROJECT_ROOT / "results" / "validation" / "lut_report.md",
    )
    parser.add_argument("--zone-file", type=Path, default=URB_DATA_FILE)
    parser.add_argument("--country-file", type=Path, default=COUNTRY_DATA_FILE)
    args = parser.parse_args()

    draws = max(1, args.draws)
    samples_per_draw = max(1, args.samples // draws)

    rng = np.random.default_rng(args.seed)
    np.random.seed(args.seed)  # sample_parameters uses the legacy global RNG

    print(f"Loading slope / mass pools from {args.zone_file} ...", flush=True)
    slope_pool, mass_pool = load_sample_pools(args.zone_file, args.country_file)
    parameter_draws = sample_parameters(draws)

    started = time.perf_counter()
    collected = {
        "bike": [[], []],
        "walk": [[], []],
    }
    draw_rows = []
    for draw in range(draws):
        params = {
            PARAMETER_ALIASES[name]: values[draw]
            for name, values in parameter_draws.items()
            if name in PARAMETER_ALIASES
        }
        # Hill polarity is stratified across draws rather than left to the i.i.d.
        # sample. It is the one parameter that changes the *shape* of the problem
        # -- a downhill leg can drive the bike solver into the regime where its
        # initial-guess ladder stalls -- and an i.i.d. draw of 10 routinely
        # misses one of the four options entirely.
        params["hill_polarity"] = POLARITY_OPTIONS[draw % len(POLARITY_OPTIONS)]
        result = run_one_draw(rng, params, slope_pool, mass_pool, samples_per_draw)
        for model in ("bike", "walk"):
            collected[model][0].append(result[model][0])
            collected[model][1].append(result[model][1])
        draw_rows.append(
            {
                "draw": draw,
                "hill_polarity": params["hill_polarity"],
                "watts": round(float(params["watts"]), 2),
                "met": round(float(params["met"]), 3),
                "crr_adjustment": int(params["crr_adjustment"]),
                "practical_limit_bicycle": int(params["practical_limit_bicycle"]),
                "practical_limit_buckets": int(params["practical_limit_buckets"]),
            }
        )
        print(f"  draw {draw + 1}/{draws} done", flush=True)
    elapsed = time.perf_counter() - started

    records = []
    passed = True
    for model, labels in (
        (
            "bike",
            (
                "bicycle loaded velocity",
                "bicycle unloaded velocity",
                "bicycle max load",
            ),
        ),
        (
            "walk",
            (
                "walking loaded velocity",
                "walking unloaded velocity",
                "walking max load",
            ),
        ),
    ):
        lut_all = np.vstack(collected[model][0])
        reference_all = np.vstack(collected[model][1])
        for column, label in enumerate(labels):
            passed &= compare(
                label, lut_all[:, column], reference_all[:, column], records
            )

    summary = pd.DataFrame(records)
    total_samples = samples_per_draw * draws

    args.report.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Solver lookup-table (LUT) fidelity",
        "",
        f"- Samples: **{total_samples}** ({draws} parameter draws x {samples_per_draw})",
        f"- Seed: `{args.seed}`",
        f"- Slope pool: `{args.zone_file}` (`slope_1`, {slope_pool.size} cells, "
        f"{slope_pool.min():.4f}-{slope_pool.max():.4f} deg)",
        f"- Mass pool: `{args.country_file}` (`Average Weight`, {mass_pool.size} countries, "
        f"{mass_pool.min():.2f}-{mass_pool.max():.2f} kg)",
        f"- Draw parameters: canonical ranges from `src/monte_carlo_parameters.py`",
        f"- LUT grid: {solver_lut.DEFAULT_N_SLOPE} slope nodes x "
        f"{solver_lut.DEFAULT_N_MASS} mass nodes x 6 Crr classes",
        f"- Wall clock: {elapsed:.1f} s",
        "",
        "## Pass bar",
        "",
        f"|dv| <= min({ABS_TOLERANCE} m/s, {REL_TOLERANCE * 100:.1f}% of v)`; "
        "NaN / no-solve cases must agree exactly.",
        "",
        "## Results",
        "",
        markdown_table(summary),
        "",
        "## Parameter draws sampled",
        "",
        markdown_table(pd.DataFrame(draw_rows)),
        "",
        "## Verdict",
        "",
        f"**{'PASS' if passed else 'FAIL'}**",
        "",
    ]
    args.report.write_text("\n".join(lines))

    print(summary.to_string(index=False))
    print(f"\nReport written to {args.report}")
    print("PASS" if passed else "FAIL")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
