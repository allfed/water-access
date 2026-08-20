#!/usr/bin/env python
"""Compare one deterministic model run on legacy QGIS CSV vs extracted grid.

Runs :func:`src.gis_global_module.run_global_analysis` twice with identical
fixed parameters — only the zone input file changes — and writes a markdown
report of global and per-country deltas.

    python scripts/compare_grid_inputs.py

    python scripts/compare_grid_inputs.py \\
        --legacy data/GIS/updated_GIS_output_cleaned.csv \\
        --new data/GIS/extracted/master_5min.parquet \\
        --report results/validation/grid_input_comparison.md

Exit code is 0 unless ``--strict`` is set and the global ``percent_with_water``
gap exceeds ``--tolerance-pp``.
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import src.gis_global_module as gis  # noqa: E402

DEFAULT_LEGACY = PROJECT_ROOT / "data" / "GIS" / "updated_GIS_output_cleaned.csv"
DEFAULT_NEW = PROJECT_ROOT / "data" / "GIS" / "extracted" / "master_5min.parquet"
DEFAULT_REPORT = PROJECT_ROOT / "results" / "validation" / "grid_input_comparison.md"

#: Same fixed draw as ``validate_chunking.py`` and ``gis_global_module``'s
#: ``__main__`` block so the comparison isolates the zone input swap.
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

#: Population-share metrics reported in percentage points (new − legacy).
PERCENT_METRICS = (
    "percent_with_water",
    "percent_without_water",
    "percent_affected_with_cycling_access",
    "percent_affected_with_walking_access",
    "percent_with_only_cycling_access",
)

#: Absolute population columns (reported as counts and % of legacy total).
POPULATION_COLUMNS = (
    "country_pop_raw",
    "country_pop_with_water",
    "country_pop_without_water",
    "population_affected_with_access",
    "population_affected_with_cycling_access",
    "population_affected_with_walking_access",
    "population_affected_with_only_cycling_access",
)


def markdown_table(frame: pd.DataFrame) -> str:
    columns = [str(column) for column in frame.columns]
    rows = [
        [str(value) for value in record] for record in frame.itertuples(index=False)
    ]
    widths = [
        max(len(columns[i]), *(len(row[i]) for row in rows)) if rows else len(columns[i])
        for i in range(len(columns))
    ]
    header = "| " + " | ".join(c.ljust(w) for c, w in zip(columns, widths)) + " |"
    divider = "| " + " | ".join("-" * w for w in widths) + " |"
    body = [
        "| " + " | ".join(v.ljust(w) for v, w in zip(row, widths)) + " |" for row in rows
    ]
    return "\n".join([header, divider, *body])


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--legacy",
        type=Path,
        default=DEFAULT_LEGACY,
        help="legacy QGIS zone table (default: updated_GIS_output_cleaned.csv)",
    )
    parser.add_argument(
        "--new",
        type=Path,
        default=DEFAULT_NEW,
        help="scripted extraction output (default: master_5min.parquet)",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=DEFAULT_REPORT,
        help="markdown report path",
    )
    parser.add_argument(
        "--tolerance-pp",
        type=float,
        default=2.0,
        help=(
            "reference band for the global percent_with_water gap when using "
            "--strict (default 2.0 pp, same as compare_smoke_runs.py)"
        ),
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="exit 1 if global percent_with_water gap exceeds --tolerance-pp",
    )
    parser.add_argument(
        "--area-weighted",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="cos-latitude population split (default on, matching the model)",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=25,
        help="number of largest per-country movers to tabulate (default 25)",
    )
    return parser.parse_args(argv)


def require_input(path: Path) -> Path:
    if not path.is_file():
        raise SystemExit(
            f"missing zone input: {path}\n"
            "Build the extracted table with scripts/extraction/extract_grid.py first."
        )
    return path


def run_once(zone_file: Path, area_weighted_population: bool) -> tuple[pd.DataFrame, float]:
    gis.URB_DATA_FILE = zone_file
    parameters = dict(
        FIXED_PARAMETERS,
        area_weighted_population=area_weighted_population,
    )
    started = time.perf_counter()
    countries, _districts, _zones = gis.run_global_analysis(**parameters)
    elapsed = time.perf_counter() - started
    return countries, elapsed


def global_row(frame: pd.DataFrame) -> pd.Series:
    match = frame.loc[frame["ISOCODE"] == "GLOBAL"]
    if match.empty:
        raise SystemExit("country frame has no GLOBAL summary row")
    return match.iloc[0]


def country_frame(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.loc[frame["ISOCODE"] != "GLOBAL"].reset_index(drop=True)


def compare_countries(legacy: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    columns = ["ISOCODE", "Entity", "country_pop_raw", *PERCENT_METRICS]
    merged = country_frame(legacy)[columns].merge(
        country_frame(new)[columns],
        on="ISOCODE",
        suffixes=("_legacy", "_new"),
        how="inner",
    )
    rows = []
    for record in merged.itertuples(index=False):
        rec = {
            "ISOCODE": record.ISOCODE,
            "Entity": record.Entity_legacy,
            "country_pop_raw_legacy": record.country_pop_raw_legacy,
            "country_pop_raw_new": record.country_pop_raw_new,
        }
        if record.country_pop_raw_legacy:
            rec["country_pop_raw_delta_pct"] = 100.0 * (
                record.country_pop_raw_new - record.country_pop_raw_legacy
            ) / record.country_pop_raw_legacy
        else:
            rec["country_pop_raw_delta_pct"] = np.nan
        for metric in PERCENT_METRICS:
            legacy_value = getattr(record, f"{metric}_legacy")
            new_value = getattr(record, f"{metric}_new")
            rec[f"{metric}_delta_pp"] = float(new_value - legacy_value)
        rows.append(rec)
    return pd.DataFrame(rows)


def global_metric_table(legacy: pd.Series, new: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame]:
    percent_rows = []
    for metric in PERCENT_METRICS:
        legacy_value = float(legacy[metric])
        new_value = float(new[metric])
        percent_rows.append(
            {
                "metric": metric,
                "legacy": legacy_value,
                "new": new_value,
                "delta_pp": new_value - legacy_value,
            }
        )

    population_rows = []
    for column in POPULATION_COLUMNS:
        legacy_value = float(legacy[column])
        new_value = float(new[column])
        population_rows.append(
            {
                "metric": column,
                "legacy": legacy_value,
                "new": new_value,
                "delta": new_value - legacy_value,
                "delta_pct": (
                    100.0 * (new_value - legacy_value) / legacy_value
                    if legacy_value
                    else np.nan
                ),
            }
        )
    return pd.DataFrame(percent_rows), pd.DataFrame(population_rows)


def write_report(
    path: Path,
    *,
    legacy_path: Path,
    new_path: Path,
    area_weighted: bool,
    legacy_seconds: float,
    new_seconds: float,
    global_percent: pd.DataFrame,
    global_population: pd.DataFrame,
    country_diff: pd.DataFrame,
    tolerance_pp: float,
    top_n: int,
) -> float:
    path.parent.mkdir(parents=True, exist_ok=True)
    headline = "percent_with_water"
    global_delta = float(
        global_percent.loc[global_percent["metric"] == headline, "delta_pp"].iloc[0]
    )

    percent_diff = country_diff.dropna(subset=[f"{headline}_delta_pp"]).copy()
    percent_diff["abs_delta_pp"] = percent_diff[f"{headline}_delta_pp"].abs()
    movers = percent_diff.sort_values("abs_delta_pp", ascending=False).head(top_n)

    lines = []
    add = lines.append
    add("# Grid input comparison — legacy QGIS CSV vs scripted extraction")
    add("")
    add(
        f"*Generated {date.today().isoformat()} by "
        "`scripts/compare_grid_inputs.py`.*"
    )
    add("")
    add("| | |")
    add("|---|---|")
    add(f"| legacy zone input | `{legacy_path}` |")
    add(f"| new zone input | `{new_path}` |")
    add(f"| area-weighted population split | **{area_weighted}** |")
    add(f"| legacy run time | {legacy_seconds:.1f} s |")
    add(f"| new run time | {new_seconds:.1f} s |")
    add(f"| countries compared | {len(country_diff)} |")
    add("")
    add("## Global summary (`ISOCODE == GLOBAL`)")
    add("")
    add(
        f"**`{headline}` gap: {global_delta:+.4f} pp** "
        f"(reference band ±{tolerance_pp:.1f} pp when using `--strict`)."
    )
    add("")
    add("Population-share metrics (percentage points):")
    add("")
    add(markdown_table(global_percent.round(6)))
    add("")
    add("Population counts (absolute):")
    add("")
    add(markdown_table(global_population.round(6)))
    add("")
    add("## Per-country distribution of `percent_with_water` deltas")
    add("")
    deltas = percent_diff[f"{headline}_delta_pp"]
    add(
        f"- median: **{deltas.median():+.4f} pp**\n"
        f"- mean: **{deltas.mean():+.4f} pp**\n"
        f"- p95 |delta|: **{deltas.abs().quantile(0.95):.4f} pp**\n"
        f"- max |delta|: **{deltas.abs().max():.4f} pp** "
        f"({percent_diff.loc[deltas.abs().idxmax(), 'ISOCODE']})\n"
        f"- countries with |delta| > 1 pp: **{(deltas.abs() > 1.0).sum()}**\n"
        f"- countries with |delta| > 2 pp: **{(deltas.abs() > 2.0).sum()}**"
    )
    add("")
    add(f"## Largest movers (top {top_n} by |`percent_with_water` delta|)")
    add("")
    mover_cols = [
        "ISOCODE",
        "Entity",
        "country_pop_raw_delta_pct",
        "percent_with_water_delta_pp",
        "percent_affected_with_walking_access_delta_pp",
        "percent_affected_with_cycling_access_delta_pp",
    ]
    add(markdown_table(movers[mover_cols].round(4)))
    add("")
    add("## Notes")
    add("")
    add(
        "- This compares **zone inputs only** at 5 arc-minute with fixed mobility "
        "parameters; it is not a Monte Carlo draw.\n"
        "- Gate 1 (`validate_extraction.py`) validates the **input layers** against "
        "the legacy CSV; this script validates **downstream model aggregates**.\n"
        "- Differences in `pop_density` spatial placement and coarse-grid GHS-SMOD "
        "tie-breaks are the usual drivers; `dtw_1` and `slope_1` match the legacy "
        "CSV exactly at 5 arc-minute in Gate 1."
    )
    add("")
    path.write_text("\n".join(lines))
    return global_delta


def main(argv=None) -> int:
    args = parse_args(argv)
    legacy_path = require_input(args.legacy)
    new_path = require_input(args.new)

    print(f"Legacy input: {legacy_path}", flush=True)
    print("Running legacy grid ...", flush=True)
    legacy_countries, legacy_seconds = run_once(legacy_path, args.area_weighted)
    print(f"  done in {legacy_seconds:.1f}s", flush=True)

    print(f"New input: {new_path}", flush=True)
    print("Running extracted grid ...", flush=True)
    new_countries, new_seconds = run_once(new_path, args.area_weighted)
    print(f"  done in {new_seconds:.1f}s", flush=True)

    global_percent, global_population = global_metric_table(
        global_row(legacy_countries),
        global_row(new_countries),
    )
    country_diff = compare_countries(legacy_countries, new_countries)
    global_delta = write_report(
        args.report,
        legacy_path=legacy_path,
        new_path=new_path,
        area_weighted=bool(args.area_weighted),
        legacy_seconds=legacy_seconds,
        new_seconds=new_seconds,
        global_percent=global_percent,
        global_population=global_population,
        country_diff=country_diff,
        tolerance_pp=args.tolerance_pp,
        top_n=args.top,
    )

    print(f"Report written to {args.report}", flush=True)
    print(f"Global percent_with_water delta: {global_delta:+.4f} pp", flush=True)

    if args.strict and abs(global_delta) > args.tolerance_pp:
        print(
            f"STRICT FAIL: |delta| > {args.tolerance_pp:.1f} pp",
            file=sys.stderr,
            flush=True,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
