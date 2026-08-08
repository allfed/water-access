"""Compare a 30 arc-second Monte Carlo run against its 5 arc-minute twin (AC-8).

Both runs must have been driven from the same ``--seed``, so that the ten
parameter tuples are identical and resolution is the only difference between
them. The run logs carry ``PARAM_DRAW`` lines; this script diffs them and
refuses to report unless they match.

    python scripts/compare_smoke_runs.py \
        --run-30as results/validation/smoke30 \
        --run-5min results/validation/twin5min \
        --log-30as results/validation/smoke30_run.log \
        --log-5min results/validation/twin5min_run.log \
        --out results/validation/smoke_comparison.md

Metrics are global population-weighted aggregates rebuilt from the per-draw
country tables, not means of country percentages.
"""

import argparse
from pathlib import Path
import re
import sys

import pandas as pd

#: Global metrics, each a (numerator, denominator) pair of country columns.
#: Every one is a population share, reported in percentage points.
METRICS = {
    "percent_with_water": ("country_pop_with_water", "country_pop_raw"),
    "percent_piped_with_walking_access": (
        "population_piped_with_walking_access",
        "country_pop_raw",
    ),
    "percent_piped_with_cycling_access": (
        "population_piped_with_cycling_access",
        "country_pop_raw",
    ),
    "percent_with_only_cycling_access": (
        "population_piped_with_only_cycling_access",
        "country_pop_raw",
    ),
}

PARAM_DRAW_RE = re.compile(r"^PARAM_DRAW \d+ ")


def read_param_draws(log_path):
    """Pull the ``PARAM_DRAW`` lines out of a run log."""
    lines = [
        line.rstrip("\n")
        for line in Path(log_path).read_text().splitlines()
        if PARAM_DRAW_RE.match(line)
    ]
    if not lines:
        raise SystemExit(
            f"{log_path}: no PARAM_DRAW lines. Was the run driven with --seed, "
            "on a build that logs the draws?"
        )
    return lines


def per_draw_globals(run_dir):
    """Global population-weighted metrics, one row per draw.

    Reads ``parquet_files/countries_simulation_result_*.parquet``. Note the
    files are named in *completion* order, so with more than one worker the
    index is not the draw index -- which is why every comparison downstream is
    over the order-invariant distribution (median / percentiles), never a
    row-by-row pairing.
    """
    files = sorted(
        (run_dir / "parquet_files").glob("countries_simulation_result_*.parquet"),
        key=lambda p: int(p.stem.rsplit("_", 1)[1]),
    )
    if not files:
        raise SystemExit(f"{run_dir}: no per-draw country parquets found")

    rows = []
    for path in files:
        frame = pd.read_parquet(path)
        row = {"file": path.name}
        for metric, (numerator, denominator) in METRICS.items():
            row[metric] = 100.0 * frame[numerator].sum() / frame[denominator].sum()
        row["country_pop_raw"] = frame["country_pop_raw"].sum()
        row["n_countries"] = len(frame)
        rows.append(row)
    return pd.DataFrame(rows)


def summarise(frame):
    """Median / 5th / 95th of each metric across the draws."""
    return {
        metric: {
            "median": frame[metric].median(),
            "p5": frame[metric].quantile(0.05),
            "p95": frame[metric].quantile(0.95),
            "min": frame[metric].min(),
            "max": frame[metric].max(),
        }
        for metric in METRICS
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--run-30as", type=Path, required=True)
    parser.add_argument("--run-5min", type=Path, required=True)
    parser.add_argument("--log-30as", type=Path, required=True)
    parser.add_argument("--log-5min", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--run-5min-paired",
        type=Path,
        default=None,
        help=(
            "a 5 arc-minute twin run made with --workers 1, so its per-draw "
            "parquet index equals the draw index and can be paired against the "
            "30as run draw by draw. Enables the paired section, which is the "
            "statistically meaningful comparison once both runs share a seed."
        ),
    )
    parser.add_argument(
        "--log-5min-paired",
        type=Path,
        default=None,
        help="run log for --run-5min-paired (checked for identical PARAM_DRAWs)",
    )
    parser.add_argument(
        "--preamble",
        type=Path,
        default=None,
        help=(
            "markdown file inserted under the title -- run provenance and "
            "interpretation, which this script cannot derive from the outputs"
        ),
    )
    parser.add_argument(
        "--tolerance-pp",
        type=float,
        default=2.0,
        help="AC-8's allowed |30as - 5min| gap on percent_with_water (default 2.0)",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    draws_30as = read_param_draws(args.log_30as)
    draws_5min = read_param_draws(args.log_5min)
    params_identical = draws_30as == draws_5min

    frame_30as = per_draw_globals(args.run_30as)
    frame_5min = per_draw_globals(args.run_5min)

    stats_30as = summarise(frame_30as)
    stats_5min = summarise(frame_5min)

    headline = "percent_with_water"
    delta = stats_30as[headline]["median"] - stats_5min[headline]["median"]
    within_tolerance = abs(delta) <= args.tolerance_pp

    walking = "percent_piped_with_walking_access"
    walking_delta = stats_30as[walking]["median"] - stats_5min[walking]["median"]
    # AC-8 pins the direction test on the WALKING-access shift specifically:
    # Phase 0 measured +0.46 pp more population inside the ~6.5 km walking
    # margin at 30as, so the finer grid should show at least as much walking
    # access as the coarse one.
    direction_matches = walking_delta >= 0

    paired = None
    if args.run_5min_paired is not None:
        paired_frame = per_draw_globals(args.run_5min_paired)
        if args.log_5min_paired is not None:
            if read_param_draws(args.log_5min_paired) != draws_30as:
                raise SystemExit(
                    f"{args.log_5min_paired}: PARAM_DRAW lines differ from the "
                    "30as run -- the paired comparison would be meaningless."
                )
        if len(paired_frame) != len(frame_30as):
            raise SystemExit("paired run has a different number of draws")
        paired = pd.DataFrame(
            {metric: frame_30as[metric] - paired_frame[metric] for metric in METRICS}
        )
        paired["draw"] = range(len(paired))

    lines = []
    lines.append("# AC-8 -- 30 arc-second vs 5 arc-minute smoke comparison")
    lines.append("")
    if args.preamble is not None:
        lines.append(args.preamble.read_text().rstrip("\n"))
        lines.append("")
    lines.append(
        f"| parameters identical (PARAM_DRAW diff) | "
        f"{'**YES**' if params_identical else '**NO**'} |"
    )
    lines.append("|---|---|")
    lines.append(
        f"| global `percent_with_water` gap | **{delta:+.3f} pp** "
        f"({'PASS' if within_tolerance else 'FAIL'}, tolerance "
        f"±{args.tolerance_pp:.1f} pp) |"
    )
    lines.append(
        f"| walking-access direction (30as >= 5min, per Phase 0) | "
        f"**{walking_delta:+.3f} pp** "
        f"({'PASS' if direction_matches else 'FAIL'}) |"
    )
    lines.append(
        f"| draws compared | {len(frame_30as)} at 30as, {len(frame_5min)} at 5min |"
    )
    lines.append("")

    lines.append("## Global medians across the draws (population-weighted)")
    lines.append("")
    lines.append("| metric | 30 arc-second | 5 arc-minute | delta (pp) |")
    lines.append("|---|---|---|---|")
    for metric in METRICS:
        a = stats_30as[metric]["median"]
        b = stats_5min[metric]["median"]
        lines.append(f"| `{metric}` | {a:.3f} | {b:.3f} | {a - b:+.3f} |")
    lines.append("")

    lines.append("## Spread of the headline metric")
    lines.append("")
    lines.append("| run | median | 5th | 95th | min | max |")
    lines.append("|---|---|---|---|---|---|")
    for label, stats in (("30 arc-second", stats_30as), ("5 arc-minute", stats_5min)):
        s = stats[headline]
        lines.append(
            f"| {label} | {s['median']:.3f} | {s['p5']:.3f} | {s['p95']:.3f} | "
            f"{s['min']:.3f} | {s['max']:.3f} |"
        )
    lines.append("")

    lines.append(f"## Per-draw global `{headline}` (sorted, order-invariant)")
    lines.append("")
    lines.append("| rank | 30 arc-second | 5 arc-minute | delta (pp) |")
    lines.append("|---|---|---|---|")
    sorted_30as = frame_30as[headline].sort_values().to_list()
    sorted_5min = frame_5min[headline].sort_values().to_list()
    for rank, (a, b) in enumerate(zip(sorted_30as, sorted_5min), start=1):
        lines.append(f"| {rank} | {a:.3f} | {b:.3f} | {a - b:+.3f} |")
    lines.append("")

    if paired is not None:
        lines.append("## Paired per-draw deltas (30as minus 5min, same draw)")
        lines.append("")
        lines.append(
            "Both runs ran `--workers 1` from the same seed, so draw *i* on one "
            "side is the same parameter tuple as draw *i* on the other. This is "
            "the comparison that actually isolates resolution: parameter noise "
            "cancels within each pair."
        )
        lines.append("")
        lines.append(
            "| draw | `percent_with_water` delta (pp) | "
            "`percent_piped_with_walking_access` delta (pp) |"
        )
        lines.append("|---|---|---|")
        for _, row in paired.iterrows():
            lines.append(
                f"| {int(row['draw'])} | {row['percent_with_water']:+.3f} | "
                f"{row['percent_piped_with_walking_access']:+.3f} |"
            )
        lines.append("")
        lines.append("| metric | mean delta | median delta | draws with 30as higher |")
        lines.append("|---|---|---|---|")
        for metric in METRICS:
            column = paired[metric]
            lines.append(
                f"| `{metric}` | {column.mean():+.3f} pp | "
                f"{column.median():+.3f} pp | {int((column > 0).sum())}/{len(column)} |"
            )
        lines.append("")

    lines.append("## Modelled population")
    lines.append("")
    lines.append(
        "Ranges are across the ten draws: a country drops out of a draw when no "
        "cell in it clears that draw's filters, so the count is not fixed."
    )
    lines.append("")
    lines.append("| run | countries (min-max) | total `country_pop_raw` (min-max) |")
    lines.append("|---|---|---|")
    for label, frame in (("30 arc-second", frame_30as), ("5 arc-minute", frame_5min)):
        counts = frame["n_countries"]
        pops = frame["country_pop_raw"]
        lines.append(
            f"| {label} | {int(counts.min())}-{int(counts.max())} | "
            f"{pops.min():,.0f} - {pops.max():,.0f} |"
        )
    lines.append("")

    if not params_identical:
        lines.append("> **Parameter draws differ between the runs.** The comparison")
        lines.append("> below is therefore NOT a clean resolution contrast.")
        lines.append("")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nwrote {args.out}")

    ok = params_identical and within_tolerance and direction_matches
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
