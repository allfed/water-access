"""AC-5 (Gate 2): diff a post-refactor run against the frozen pre-refactor baseline.

Compares ``results/regression_new/`` (produced by ``scripts/capture_baseline.py
--out results/regression_new`` with the refactored solver) against the committed
``results/regression_baseline/`` (the same fixed parameters on the pre-refactor
code). Exits 0 only if every one of these holds:

* identical row sets -- same countries, same shapeIDs;
* per-country ``percent_with_water`` within **0.1 pp**;
* every per-country population-count column within **0.1%**;
* per-district ``percent_with_water`` within **0.25 pp**, for districts over
  100k people.

It also reports the AC-7 wall clock (<= 90 s excluding the initial data load),
read from the ``timing.json`` ``capture_baseline.py`` writes beside the CSVs.

Usage:
    python scripts/validate_regression.py --report results/validation/gate2_report.md
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent

#: Tolerance on ``percent_with_water`` at country level, percentage points.
COUNTRY_PERCENT_TOLERANCE_PP = 0.1

#: Tolerance on every population-count column, as a fraction.
POPULATION_TOLERANCE = 0.001

#: Tolerance on ``percent_with_water`` at district level, percentage points.
DISTRICT_PERCENT_TOLERANCE_PP = 0.25

#: Districts below this population are exempt from the district tolerance.
DISTRICT_POPULATION_FLOOR = 100_000

#: AC-7: seconds the fixed-parameter run may take, excluding the data load.
SPEED_BUDGET_SECONDS = 90.0

#: Population-count columns checked at country level.
COUNTRY_POPULATION_COLUMNS = (
    "country_pop_raw",
    "country_pop_with_water",
    "country_pop_without_water",
    "population_piped_with_access",
    "population_piped_with_cycling_access",
    "population_piped_with_walking_access",
    "population_piped_with_only_cycling_access",
)


def markdown_table(frame):
    """Render a DataFrame as a GitHub markdown table (no ``tabulate`` needed)."""
    columns = [str(column) for column in frame.columns]
    rows = [
        [str(value) for value in record] for record in frame.itertuples(index=False)
    ]
    widths = [
        max([len(columns[i])] + [len(row[i]) for row in rows])
        for i in range(len(columns))
    ]
    header = "| " + " | ".join(c.ljust(w) for c, w in zip(columns, widths)) + " |"
    divider = "| " + " | ".join("-" * w for w in widths) + " |"
    body = [
        "| " + " | ".join(v.ljust(w) for v, w in zip(row, widths)) + " |"
        for row in rows
    ]
    return "\n".join([header, divider, *body])


def compare_row_sets(baseline, new, key, label, checks):
    """Record whether the two frames cover exactly the same keys."""
    missing = sorted(set(baseline[key]) - set(new[key]))
    added = sorted(set(new[key]) - set(baseline[key]))
    checks.append(
        {
            "check": f"{label} row set identical",
            "observed": f"{len(missing)} missing, {len(added)} added",
            "bar": "0 missing, 0 added",
            "pass": not missing and not added,
        }
    )
    return missing, added


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline",
        type=Path,
        default=PROJECT_ROOT / "results" / "regression_baseline",
    )
    parser.add_argument(
        "--new", type=Path, default=PROJECT_ROOT / "results" / "regression_new"
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=PROJECT_ROOT / "results" / "validation" / "gate2_report.md",
    )
    parser.add_argument(
        "--timing-note",
        default=None,
        help=(
            "free text appended to the AC-7 section, for recording measurement "
            "conditions (e.g. a loaded machine)."
        ),
    )
    args = parser.parse_args()

    baseline_countries = pd.read_csv(args.baseline / "countries.csv")
    new_countries = pd.read_csv(args.new / "countries.csv")
    baseline_districts = pd.read_csv(args.baseline / "districts.csv")
    new_districts = pd.read_csv(args.new / "districts.csv")

    checks = []
    detail_rows = []

    # ---- row sets ----------------------------------------------------------
    compare_row_sets(baseline_countries, new_countries, "ISOCODE", "countries", checks)
    compare_row_sets(baseline_districts, new_districts, "shapeID", "districts", checks)

    # ---- country percent_with_water ---------------------------------------
    countries = baseline_countries.merge(
        new_countries, on="ISOCODE", suffixes=("_base", "_new")
    )
    percent_delta = (
        countries["percent_with_water_new"] - countries["percent_with_water_base"]
    ).abs()
    worst = int(percent_delta.idxmax())
    checks.append(
        {
            "check": "country percent_with_water",
            "observed": f"max {percent_delta.max():.5f} pp ({countries['ISOCODE'][worst]})",
            "bar": f"<= {COUNTRY_PERCENT_TOLERANCE_PP} pp",
            "pass": bool(percent_delta.max() <= COUNTRY_PERCENT_TOLERANCE_PP),
        }
    )

    # ---- country population columns ---------------------------------------
    for column in COUNTRY_POPULATION_COLUMNS:
        base = countries[f"{column}_base"].astype(float)
        new = countries[f"{column}_new"].astype(float)
        denominator = base.abs().replace(0.0, np.nan)
        relative = ((new - base).abs() / denominator).fillna(0.0)
        worst = int(relative.idxmax())
        passed = bool(relative.max() <= POPULATION_TOLERANCE)
        checks.append(
            {
                "check": f"country {column}",
                "observed": f"max {relative.max() * 100:.5f}% ({countries['ISOCODE'][worst]})",
                "bar": f"<= {POPULATION_TOLERANCE * 100:.1f}%",
                "pass": passed,
            }
        )
        detail_rows.append(
            {
                "column": column,
                "max_rel_error_pct": round(float(relative.max() * 100), 6),
                "worst_country": countries["ISOCODE"][worst],
                "baseline_value": f"{base[worst]:.6g}",
                "new_value": f"{new[worst]:.6g}",
            }
        )

    # ---- district percent_with_water --------------------------------------
    districts = baseline_districts.merge(
        new_districts, on="shapeID", suffixes=("_base", "_new")
    )
    large = districts[districts["district_pop_raw_base"] > DISTRICT_POPULATION_FLOOR]
    district_delta = (
        large["percent_with_water_new"] - large["percent_with_water_base"]
    ).abs()
    worst = int(district_delta.idxmax())
    checks.append(
        {
            "check": f"district percent_with_water (>{DISTRICT_POPULATION_FLOOR:,} people)",
            "observed": (
                f"max {district_delta.max():.5f} pp "
                f"({districts['shapeID'][worst]}, n={len(large)})"
            ),
            "bar": f"<= {DISTRICT_PERCENT_TOLERANCE_PP} pp",
            "pass": bool(district_delta.max() <= DISTRICT_PERCENT_TOLERANCE_PP),
        }
    )

    # ---- AC-7 speed --------------------------------------------------------
    timing = {}
    timing_file = args.new / "timing.json"
    if timing_file.exists():
        timing = json.loads(timing_file.read_text())
        seconds = float(timing["seconds_excluding_load"])
        checks.append(
            {
                "check": "AC-7 run time excluding data load",
                "observed": f"{seconds:.1f} s",
                "bar": f"<= {SPEED_BUDGET_SECONDS:.0f} s",
                "pass": seconds <= SPEED_BUDGET_SECONDS,
            }
        )
    else:
        checks.append(
            {
                "check": "AC-7 run time excluding data load",
                "observed": "timing.json missing",
                "bar": f"<= {SPEED_BUDGET_SECONDS:.0f} s",
                "pass": False,
            }
        )

    summary = pd.DataFrame(checks)
    summary["pass"] = summary["pass"].map({True: "PASS", False: "FAIL"})
    passed = not (summary["pass"] == "FAIL").any()

    args.report.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# AC-5 (Gate 2): refactor regression at 5 arc-minutes",
        "",
        f"- Baseline (pre-refactor, committed): `{args.baseline}`",
        f"- New (post-refactor): `{args.new}`",
        f"- Countries compared: {len(countries)}; districts compared: {len(districts)} "
        f"({len(large)} over {DISTRICT_POPULATION_FLOOR:,} people)",
        "",
        "Both runs use the AC-2 fixed parameters: `crr_adjustment=0, "
        "time_gathering_water=2.84, practical_limit_bicycle=40, "
        "practical_limit_buckets=20, met=4, watts=40, "
        'hill_polarity="downhill_uphill", urban_adjustment=1.3, '
        "rural_adjustment=1.4, calculate_distance=True, plot=False`.",
        "",
        "## Checks",
        "",
        markdown_table(summary),
        "",
        "## AC-7 timing",
        "",
    ]
    if timing:
        lines += [
            f"- Input: `{timing['input']}`",
            f"- Total wall clock: **{timing['total_seconds']:.1f} s**",
            f"- Initial data load: {timing['load_seconds']:.1f} s",
            f"- **Excluding data load: {timing['seconds_excluding_load']:.1f} s** "
            f"(bar: <= {SPEED_BUDGET_SECONDS:.0f} s; the pre-refactor solver alone "
            "took ~4 minutes)",
            "",
        ]
        if timing.get("peak_rss_gb") is not None:
            cells = timing.get("modelled_cells")
            lines += [
                "## Memory (ADR-5 datum for Phase D)",
                "",
                f"- Peak RSS of the single fixed-parameter run: "
                f"**{timing['peak_rss_gb']:.2f} GB**",
                (
                    f"- Modelled cells: {cells:,}"
                    if cells
                    else "- Modelled cells: not recorded"
                ),
                "",
                "This is the 5-arcmin datum; AC-6's 24 GB budget is measured at "
                "30 arc-seconds by Phase D.",
                "",
            ]
    else:
        lines += ["- No `timing.json` found next to the new results.", ""]

    if args.timing_note:
        lines += [args.timing_note, ""]

    lines += [
        "## Per-column population detail",
        "",
        markdown_table(pd.DataFrame(detail_rows)),
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
