"""Prove the chunked path and the monolithic path agree exactly (chunked vs monolithic validation).

Runs ``run_global_analysis`` twice on the same input at the same fixed
parameters -- once whole-world, once one country-group at a time -- and requires
the country and district frames to be *identical*, not merely close. That is a
much stronger bar than Gate 2's tolerances, and it is the right one: chunking is
supposed to be a pure execution-strategy change, so any numeric difference at
all means a real coupling was missed.

    python scripts/validate_chunking.py --input data/GIS/updated_GIS_output_cleaned.csv \
        --chunk-rows 200000 --report results/validation/chunking_report.md

Exits 0 on agreement, 1 otherwise.
"""

import argparse
import resource
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import src.chunked_analysis as chunked  # noqa: E402
import src.gis_global_module as gis  # noqa: E402

#: Same fixed parameters as fixed regression parameters, so this runs against a known-good point.
FIXED_PARAMETERS = dict(
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
    plot=False,
)


def markdown_table(frame):
    """Render a DataFrame as a GitHub markdown table (no ``tabulate`` needed)."""
    columns = [str(c) for c in frame.columns]
    rows = [[str(v) for v in record] for record in frame.itertuples(index=False)]
    widths = [
        max([len(columns[i])] + [len(r[i]) for r in rows]) for i in range(len(columns))
    ]
    header = "| " + " | ".join(c.ljust(w) for c, w in zip(columns, widths)) + " |"
    divider = "| " + " | ".join("-" * w for w in widths) + " |"
    body = [
        "| " + " | ".join(v.ljust(w) for v, w in zip(r, widths)) + " |" for r in rows
    ]
    return "\n".join([header, divider, *body])


def peak_rss_gb():
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform != "darwin":  # Linux reports KiB, macOS bytes
        rss *= 1024
    return rss / 1e9


#: Relative bar for the one row that cannot be bit-identical. See
#: ``GLOBAL_ROW_NOTE``.
GLOBAL_ROW_RTOL = 1e-12

GLOBAL_ROW_NOTE = (
    "Two kinds of row cannot be reproduced bit for bit, and in both cases the "
    "reason is arithmetic rather than structural: their population columns are "
    "sums over cells that chunking splits up, and floating-point addition is "
    "not associative, so adding the parts lands within an ULP of adding the "
    "whole but not exactly on it. Those rows are (a) the `GLOBAL` summary, "
    "which sums every cell on Earth, and (b) any country so large it had to be "
    "spread over several chunks. Both are held to a relative tolerance instead. "
    "**Every other country row, and every district row without exception, must "
    "be bit-identical** -- including the districts of a split country, because "
    "splitting is done on district boundaries precisely so that no district is "
    "ever divided."
)


def compare(label, left, right, key, checks, exempt_keys=(), rtol=GLOBAL_ROW_RTOL):
    """Compare two frames exactly; record one row per failure mode.

    Rows whose ``key`` is in ``exempt_keys`` are held to ``rtol`` instead of
    bit-identity, and reported separately.
    """
    left = left.reset_index(drop=True)
    right = right.reset_index(drop=True)

    same_shape = left.shape == right.shape
    checks.append(
        {
            "check": f"{label}: shape",
            "observed": f"{left.shape} vs {right.shape}",
            "pass": same_shape,
        }
    )

    same_columns = list(left.columns) == list(right.columns)
    checks.append(
        {
            "check": f"{label}: column order",
            "observed": "identical" if same_columns else "DIFFERENT",
            "pass": same_columns,
        }
    )
    if not (same_shape and same_columns):
        return

    same_keys = left[key].tolist() == right[key].tolist()
    checks.append(
        {
            "check": f"{label}: {key} order",
            "observed": "identical" if same_keys else "DIFFERENT",
            "pass": same_keys,
        }
    )

    exempt = left[key].isin(list(exempt_keys)).to_numpy()
    strict = ~exempt

    worst_column, worst_delta = "-", 0.0
    worst_exempt_column, worst_exempt_rel = "-", 0.0
    mismatched = []
    for column in left.columns:
        a, b = left[column], right[column]
        if pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b):
            av, bv = a.to_numpy(dtype=float), b.to_numpy(dtype=float)
            delta = np.abs(np.nan_to_num(av - bv, nan=0.0))
            # NaN must line up too.
            if not a.isna().equals(b.isna()):
                mismatched.append(f"{column} (NaN pattern)")
            strict_delta = delta[strict]
            if strict_delta.size and strict_delta.max() > worst_delta:
                worst_delta, worst_column = float(strict_delta.max()), column
            if strict_delta.size and strict_delta.max() != 0.0:
                mismatched.append(f"{column} (max |d| {strict_delta.max():.3e})")
            if exempt.any():
                scale = np.where(np.abs(av) > 0, np.abs(av), 1.0)
                relative = float(np.nanmax((delta / scale)[exempt]))
                if relative > worst_exempt_rel:
                    worst_exempt_rel, worst_exempt_column = relative, column
        else:
            if not a[strict].astype(object).equals(b[strict].astype(object)):
                mismatched.append(f"{column} (values)")

    scope = f" (excluding {', '.join(exempt_keys)})" if exempt.any() else ""
    checks.append(
        {
            "check": f"{label}: every value identical{scope}",
            "observed": (
                "yes"
                if not mismatched
                else f"{len(mismatched)} column(s) differ: {', '.join(mismatched[:6])}"
            ),
            "pass": not mismatched,
        }
    )
    checks.append(
        {
            "check": f"{label}: largest absolute difference{scope}",
            "observed": f"{worst_delta:.3e} ({worst_column})",
            "pass": worst_delta == 0.0,
        }
    )
    if exempt.any():
        checks.append(
            {
                "check": (
                    f"{label}: {'/'.join(exempt_keys)} row(s) within "
                    f"{GLOBAL_ROW_RTOL:g} relative"
                ),
                "observed": f"{worst_exempt_rel:.3e} ({worst_exempt_column})",
                "pass": worst_exempt_rel <= rtol,
            }
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="zone frame (CSV or Parquet). Defaults to the 5 arc-minute CSV.",
    )
    parser.add_argument(
        "--chunk-rows",
        type=int,
        default=200_000,
        help=(
            "post-filter row budget per chunk. Small on purpose here: the point "
            "is to force many chunks, not to save memory."
        ),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=PROJECT_ROOT / "results" / "validation" / "chunking_report.md",
    )
    parser.add_argument(
        "--area-weighted",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "run both paths with cos-latitude area weighting of the population "
            "split. On by default, matching the model default. The equivalence "
            "bar is the same either way: the planner's pre-pass has to "
            "reproduce the run's weight exactly. --no-area-weighted checks the "
            "legacy path still chunks identically too."
        ),
    )
    args = parser.parse_args()

    if args.input is not None:
        gis.URB_DATA_FILE = args.input
    zone_file = gis.URB_DATA_FILE
    parameters = dict(
        FIXED_PARAMETERS, area_weighted_population=bool(args.area_weighted)
    )

    print(f"Input: {zone_file}")
    print(f"Area-weighted population: {bool(args.area_weighted)}")
    print("Planning chunks ...", flush=True)
    plan = chunked.plan_chunks(
        zone_file,
        max_rows_per_chunk=args.chunk_rows,
        area_weighted_population=bool(args.area_weighted),
    )
    print(f"  {plan.describe()}", flush=True)

    print("Monolithic run ...", flush=True)
    started = time.perf_counter()
    mono_countries, mono_districts, _ = gis.run_global_analysis(**parameters)
    mono_seconds = time.perf_counter() - started
    mono_rss = peak_rss_gb()

    print("Chunked run ...", flush=True)
    started = time.perf_counter()
    chunk_countries, chunk_districts, chunk_zones = gis.run_global_analysis(
        **parameters, chunk_rows=args.chunk_rows, chunk_plan=plan
    )
    chunk_seconds = time.perf_counter() - started

    checks = []
    compare(
        "countries",
        mono_countries,
        chunk_countries,
        "ISOCODE",
        checks,
        exempt_keys=("GLOBAL", *plan.split_countries),
    )
    compare("districts", mono_districts, chunk_districts, "shapeID", checks)
    checks.append(
        {
            "check": "chunked per-cell frame is empty",
            "observed": f"{len(chunk_zones)} rows",
            "pass": len(chunk_zones) == 0,
        }
    )

    summary = pd.DataFrame(checks)
    summary["pass"] = summary["pass"].map({True: "PASS", False: "FAIL"})
    passed = not (summary["pass"] == "FAIL").any()

    plan_rows = pd.DataFrame(
        {
            "chunk": range(1, len(plan.chunks) + 1),
            "selection": [
                (
                    f"{selection.isocodes[0]} "
                    f"({len(selection.shape_ids)} districts)"
                    if selection.is_partial_country
                    else f"{len(selection.isocodes)} whole countries"
                )
                for selection in plan.chunks
            ],
            "modelled_cells": plan.chunk_rows,
        }
    )

    args.report.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Chunked vs monolithic execution ",
        "",
        f"- Input: `{zone_file}`",
        f"- Area-weighted population (cos-latitude): "
        f"{'ON (default)' if args.area_weighted else 'OFF (legacy split)'}",
        f"- Chunk budget: {args.chunk_rows:,} modelled cells",
        f"- Plan: {plan.describe()}",
        f"- Global LUT ranges pinned to slope "
        f"{plan.slope_range[0]:.6f}-{plan.slope_range[1]:.6f} deg, mass "
        f"{plan.mass_range[0]:.3f}-{plan.mass_range[1]:.3f} kg",
        f"- Monolithic run: {mono_seconds:.1f} s; chunked run: {chunk_seconds:.1f} s",
        f"- Peak RSS across both runs in one process: {peak_rss_gb():.2f} GB "
        f"(monolithic alone reached {mono_rss:.2f} GB)",
        "",
        "Both runs use the fixed regression parameters. The bar is **identity**, not a "
        "tolerance: chunking changes only the order of execution, so every value "
        "must match bit for bit.",
        "",
        GLOBAL_ROW_NOTE,
        "",
        "## Checks",
        "",
        markdown_table(summary),
        "",
        "## Chunk plan",
        "",
        markdown_table(plan_rows),
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
    sys.exit(main())
