#!/usr/bin/env python
"""Gate 1 -- extraction fidelity of the scripted pipeline at 5 arc-minute.

Compares a freshly extracted master table against the legacy manual-QGIS
artefact ``data/GIS/updated_GIS_output_cleaned.csv`` and writes a markdown
report that *is* the gate artefact (SPEC AC-3).

    .venv/bin/python scripts/extraction/validate_extraction.py \
        --new data/GIS/extracted/master_5min.parquet \
        --old data/GIS/updated_GIS_output_cleaned.csv \
        --report results/validation/gate1_report.md

Exit code is 0 only if every per-country failure is covered by an entry in the
committed exception list (``scripts/extraction/gate1_exceptions.json``) whose
``max_magnitude`` is not exceeded -- per ADR-9 the bar is statistical, not
cell-exact, but an exception can never be a blank cheque.

Gating population: countries with an estimated population >= 1,000,000, where
population is ``sum(pop_density * cell_area_km2)`` over the legacy rows.
Smaller countries are reported but not gated.

Weighting: metrics computed on the *joined* (spatially aligned) cell set use a
single common weight -- the legacy ``pop_density * cell_area`` -- so that a
difference in a metric reflects the layer under test and not a difference in
the population layer.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import date

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from grid import REPO_ROOT  # noqa: E402

EARTH_RADIUS_KM = 6371.0087714
GATING_POP = 1_000_000.0
GRIP_COLUMNS = ["grip_1_1", "grip_2_1", "grip_3_1", "grip_4_1", "grip_5_1"]
DEFAULT_EXCEPTIONS = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "gate1_exceptions.json"
)

#: metric -> (human label, tolerance, unit)
THRESHOLDS = {
    "cell_count_pct": ("modelled-cell count", 2.0, "%"),
    "dtw_exact_pct": ("dtw_1 cell-exact match", 99.0, "% (floor)"),
    "dtw_mean_pct": ("dtw_1 pop-weighted mean", 2.0, "%"),
    "slope_mean_pct": ("slope_1 pop-weighted mean", 2.0, "%"),
    "smod_urban_pp": ("GHS_SMOD urban share", 2.0, "pp"),
    "smod_mode_match": ("GHS_SMOD modal class", 0.0, "bool"),
    "road_class_match": ("dominant road class", 0.0, "bool"),
    "pop_total_pct": ("pop_density total", 1.0, "%"),
}
FLOOR_METRICS = {"dtw_exact_pct"}
BOOL_METRICS = {"smod_mode_match", "road_class_match"}


def cell_area_km2(top: np.ndarray, bottom: np.ndarray, cell_deg: np.ndarray):
    """Spherical area of a lat/lon cell, km^2."""
    return (
        (EARTH_RADIUS_KM**2)
        * np.radians(cell_deg)
        * (np.sin(np.radians(top)) - np.sin(np.radians(bottom)))
    )


def load_frame(path: str, columns) -> pd.DataFrame:
    if path.endswith(".parquet"):
        df = pd.read_parquet(path, columns=list(columns))
    else:
        df = pd.read_csv(path, usecols=list(columns))
    for c in ("shapeName", "shapeID", "ISOCODE"):
        if c in df.columns:
            df[c] = df[c].astype(str)
    df["area_km2"] = cell_area_km2(
        df["top"].values, df["bottom"].values, df["right"].values - df["left"].values
    )
    df["pop"] = df["pop_density"].values * df["area_km2"].values
    return df


def _wmean(values: np.ndarray, weights: np.ndarray) -> float:
    ok = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not ok.any():
        return np.nan
    return float(np.sum(values[ok] * weights[ok]) / np.sum(weights[ok]))


def _pct_diff(new: float, old: float) -> float:
    if not np.isfinite(new) or not np.isfinite(old) or old == 0:
        return np.nan
    return 100.0 * (new - old) / abs(old)


def global_agreement(joined: pd.DataFrame) -> list[tuple]:
    """Cell-level agreement per layer over the whole spatially aligned set."""
    out = []
    for col in ["dtw_1", "slope_1", "pop_density", "GHS_SMOD"] + GRIP_COLUMNS:
        a = joined[col + "_o"].values.astype("float64")
        b = joined[col + "_n"].values.astype("float64")
        both_nan = ~np.isfinite(a) & ~np.isfinite(b)
        same = np.isclose(a, b, rtol=1e-5, atol=1e-3) | both_nan
        fin = np.isfinite(a) & np.isfinite(b)
        ratio = float(np.mean(b[fin]) / np.mean(a[fin])) if fin.any() else np.nan
        out.append(
            (
                col,
                100.0 * float(same.mean()),
                ratio,
                int((~np.isfinite(a)).sum()),
                int((~np.isfinite(b)).sum()),
            )
        )
    return out


def per_country_metrics(old: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    joined = old.merge(new, on="fid", suffixes=("_o", "_n"), how="inner")
    joined["w"] = joined["pop_o"].clip(lower=0)

    old_g = old.groupby("ISOCODE", observed=True)
    new_g = new.groupby("ISOCODE", observed=True)
    rows = []
    countries = sorted(set(old["ISOCODE"]) | set(new["ISOCODE"]))
    jg = dict(list(joined.groupby("ISOCODE_o", observed=True)))
    for iso in countries:
        o = old_g.get_group(iso) if iso in old_g.groups else old.iloc[:0]
        n = new_g.get_group(iso) if iso in new_g.groups else new.iloc[:0]
        j = jg.get(iso, joined.iloc[:0])
        pop_old = float(np.nansum(o["pop"].values))
        rec = {
            "ISOCODE": iso,
            "pop_old_M": pop_old / 1e6,
            "cells_old": len(o),
            "cells_new": len(n),
            "cells_joined": len(j),
            "cell_count_pct": _pct_diff(len(n), len(o)),
            "pop_total_pct": _pct_diff(float(np.nansum(n["pop"].values)), pop_old),
        }
        if len(j):
            do, dn = j["dtw_1_o"].values, j["dtw_1_n"].values
            both_nan = ~np.isfinite(do) & ~np.isfinite(dn)
            same = np.isclose(do, dn, rtol=1e-5, atol=1e-3, equal_nan=False)
            rec["dtw_exact_pct"] = 100.0 * float(np.mean(same | both_nan))
            rec["dtw_mean_pct"] = _pct_diff(
                _wmean(dn, j["w"].values), _wmean(do, j["w"].values)
            )
            rec["slope_mean_pct"] = _pct_diff(
                _wmean(j["slope_1_n"].values, j["w"].values),
                _wmean(j["slope_1_o"].values, j["w"].values),
            )
            urb_o = 100.0 * _wmean(
                (j["GHS_SMOD_o"].values > 15).astype(float), j["w"].values
            )
            urb_n = 100.0 * _wmean(
                (j["GHS_SMOD_n"].values > 15).astype(float), j["w"].values
            )
            rec["smod_urban_pp"] = urb_n - urb_o
            rec["smod_urban_old"] = urb_o
            mo = pd.Series(j["GHS_SMOD_o"].values).mode()
            mn = pd.Series(j["GHS_SMOD_n"].values).mode()
            rec["smod_mode_match"] = float(
                len(mo) and len(mn) and mo.iloc[0] == mn.iloc[0]
            )
            ro = [_wmean(j[c + "_o"].values, j["w"].values) for c in GRIP_COLUMNS]
            rn = [_wmean(j[c + "_n"].values, j["w"].values) for c in GRIP_COLUMNS]
            rec["road_class_old"] = GRIP_COLUMNS[int(np.nanargmax(ro))]
            rec["road_class_new"] = GRIP_COLUMNS[int(np.nanargmax(rn))]
            rec["road_class_match"] = float(
                rec["road_class_old"] == rec["road_class_new"]
            )
            # Diagnostic: what the model actually consumes is the *within
            # country normalised* density (see merge_and_adjust_population),
            # so a uniform per-country scale factor is cancelled downstream.
            so = j["pop_density_o"].values
            sn = j["pop_density_n"].values
            ok = np.isfinite(so) & np.isfinite(sn) & (np.nansum(so) > 0)
            if ok.any() and np.nansum(sn[ok]) > 0:
                share_o = so[ok] / np.nansum(so[ok])
                share_n = sn[ok] / np.nansum(sn[ok])
                rec["pop_share_l1_pct"] = 100.0 * float(
                    0.5 * np.nansum(np.abs(share_o - share_n))
                )
        rows.append(rec)
    df = (
        pd.DataFrame(rows)
        .set_index("ISOCODE")
        .sort_values("pop_old_M", ascending=False)
    )
    return df, joined


def failures(df: pd.DataFrame) -> list[dict]:
    out = []
    gating = df[df["pop_old_M"] >= GATING_POP / 1e6]
    for iso, row in gating.iterrows():
        for metric, (_, tol, _u) in THRESHOLDS.items():
            v = row.get(metric, np.nan)
            if not np.isfinite(v):
                continue
            if metric in BOOL_METRICS:
                bad, mag = v < 1.0, 1.0 - v
            elif metric in FLOOR_METRICS:
                bad, mag = v < tol, tol - v
            else:
                bad, mag = abs(v) > tol, abs(v)
            if bad:
                out.append(
                    {
                        "ISOCODE": iso,
                        "metric": metric,
                        "value": float(v),
                        "magnitude": float(mag),
                        "pop_M": float(row["pop_old_M"]),
                    }
                )
    return out


def load_exceptions(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    with open(path) as fh:
        return json.load(fh).get("exceptions", [])


def cover(fail: dict, exceptions: list[dict]) -> dict | None:
    for e in exceptions:
        if e["metric"] != fail["metric"]:
            continue
        if e["country"] not in ("*", fail["ISOCODE"]):
            continue
        if fail["magnitude"] <= float(e["max_magnitude"]) + 1e-9:
            return e
    return None


def write_report(path, df, fails, covered, uncovered, meta, agreement, ids):
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    gating = df[df["pop_old_M"] >= GATING_POP / 1e6]
    L = []
    a = L.append
    a("# Gate 1 report — scripted extraction vs legacy QGIS output (5 arc-minute)")
    a("")
    a(
        f"*Generated {date.today().isoformat()} by "
        "`scripts/extraction/validate_extraction.py` (SPEC AC-3, ADR-9).*"
    )
    a("")
    a("| | |")
    a("|---|---|")
    for k, v in meta.items():
        a(f"| {k} | {v} |")
    a("")
    a("## Verdict")
    a("")
    a(f"- Countries gated (pop >= 1M): **{len(gating)}** of {len(df)} present.")
    a(f"- Per-(country, metric) failures against the raw thresholds: **{len(fails)}**.")
    a(f"- Covered by the committed exception list: **{len(covered)}**.")
    a(
        f"- **Uncovered failures: {len(uncovered)}** "
        f"→ gate {'**FAIL**' if uncovered else '**PASS**'}."
    )
    a("")
    a("### Thresholds")
    a("")
    a("| metric | bar |")
    a("|---|---|")
    for m, (label, tol, unit) in THRESHOLDS.items():
        bar = (
            "exact match"
            if m in BOOL_METRICS
            else f">= {tol}{unit}" if m in FLOOR_METRICS else f"within {tol} {unit}"
        )
        a(f"| `{m}` — {label} | {bar} |")
    a("")
    a("## Cell-level layer agreement (all spatially aligned cells, ungated)")
    a("")
    a(
        "`agreement` = share of joined cells where the two tables carry the same "
        "value (rtol 1e-5) or are both nodata. `mean ratio` = new/legacy over "
        "cells finite in both."
    )
    a("")
    a("| layer | agreement % | mean ratio | legacy nodata | new nodata |")
    a("|---|---|---|---|---|")
    for col, pct, ratio, na_o, na_n in agreement:
        a(f"| `{col}` | {pct:.4f} | {ratio:.6f} | {na_o:,} | {na_n:,} |")
    a("")
    a("| identifier | agreement % | note |")
    a("|---|---|---|")
    a(f"| `ISOCODE` | {ids['ISOCODE']:.4f} | geoBoundaries CGAZ ADM1 " "`shapeGroup` |")
    a(f"| `shapeID` | {ids['shapeID']:.4f} | CGAZ ADM1 id |")
    a(
        f"| `shapeName` | {ids['shapeName']:.4f} | the residual is **legacy "
        "mojibake**: the legacy CSV stores non-ASCII names with `?` "
        "substitutions (`R?jasth?n`, `Khanty-Mansiysk Autonomous Okrug ? Ugra`). "
        "The scripted pipeline emits correct UTF-8, so this difference is a fix, "
        "not a regression. |"
    )
    a("")
    a("## Global summary (every country present, gated and ungated)")
    a("")
    a(
        "`p95 abs` / `worst` are taken over *all* countries and territories, "
        "including the sub-1M ones that AC-3 does not gate — small islands "
        "dominate the tails."
    )
    a("")
    a("| metric | median | p95 abs | worst country |")
    a("|---|---|---|---|")
    for m in THRESHOLDS:
        if m in BOOL_METRICS or m not in df:
            continue
        v = df[m].dropna()
        if v.empty:
            continue
        worst = v.abs().idxmax() if m not in FLOOR_METRICS else v.idxmin()
        a(
            f"| `{m}` | {v.median():.3f} | {np.percentile(v.abs(), 95):.3f} | "
            f"{worst} ({v.loc[worst]:.3f}) |"
        )
    a("")
    a("## Diagnostic: within-country population distribution")
    a("")
    a(
        "`pop_share_l1_pct` is the total-variation distance (half the L1 "
        "distance) between the legacy and new *within-country normalised* "
        "`pop_density` vectors — i.e. the share of a country's population that "
        "sits in different cells. This is the form in which the model actually "
        "consumes population (`merge_and_adjust_population`, "
        "`src/gis_global_module.py:323-327`, renormalises `pop_density` to a "
        "within-country share and multiplies by an external country total), so "
        "it is the population statistic that can move results — any uniform "
        "per-country scale factor is cancelled exactly and cannot."
    )
    a("")
    if "pop_share_l1_pct" in df:
        v = gating["pop_share_l1_pct"].dropna()
        a(
            f"Over the {len(v)} gated countries: median **{v.median():.2f}%**, "
            f"p95 **{np.percentile(v, 95):.2f}%**, max **{v.max():.2f}%** "
            f"({v.idxmax()}). It is **not** one of AC-3's gated metrics (AC-3 "
            "gates `pop_density` *totals*), but it is the residual most likely "
            "to move downstream numbers and is recorded here deliberately."
        )
    a("")
    a("## Per-country detail — 30 largest gated countries")
    a("")
    cols = [
        "pop_old_M",
        "cells_old",
        "cells_new",
        "cell_count_pct",
        "dtw_exact_pct",
        "dtw_mean_pct",
        "slope_mean_pct",
        "smod_urban_pp",
        "pop_total_pct",
        "pop_share_l1_pct",
    ]
    cols = [c for c in cols if c in df.columns]
    head = gating[cols].head(30)
    a("| " + " | ".join(["ISOCODE"] + cols) + " |")
    a("|" + "---|" * (len(cols) + 1))
    for iso, r in head.iterrows():
        a("| " + " | ".join([iso] + [f"{r[c]:,.3f}" for c in cols]) + " |")
    a("")
    a("## Exception list")
    a("")
    if not fails:
        a("No per-country threshold failures — the exception list was not needed.")
    else:
        a(
            "Every row below is a threshold failure that the committed exception "
            "list (`scripts/extraction/gate1_exceptions.json`) accepts as a known "
            "artefact of the undocumented manual QGIS provenance or of a "
            "source-variant substitution (ADR-9). `max_magnitude` bounds each "
            "exception so it cannot silently absorb a regression."
        )
        a("")
        a("| ISOCODE | pop (M) | metric | value | magnitude | max allowed | cause |")
        a("|---|---|---|---|---|---|---|")
        for f, e in covered:
            a(
                f"| {f['ISOCODE']} | {f['pop_M']:.1f} | `{f['metric']}` | "
                f"{f['value']:.3f} | {f['magnitude']:.3f} | "
                f"{float(e['max_magnitude']):.3f} | {e['cause']} |"
            )
    if uncovered:
        a("")
        a("### UNCOVERED failures (gate fails)")
        a("")
        a("| ISOCODE | pop (M) | metric | value | magnitude |")
        a("|---|---|---|---|---|")
        for f in uncovered:
            a(
                f"| {f['ISOCODE']} | {f['pop_M']:.1f} | `{f['metric']}` | "
                f"{f['value']:.3f} | {f['magnitude']:.3f} |"
            )
    a("")
    a("## Countries below the 1M gating population (reported, not gated)")
    a("")
    small = df[df["pop_old_M"] < GATING_POP / 1e6]
    a(
        f"{len(small)} countries/territories, "
        f"{small['cells_old'].sum():,} legacy cells. "
        f"Worst |cell_count_pct| = "
        f"{small['cell_count_pct'].abs().max():.2f}% "
        f"({small['cell_count_pct'].abs().idxmax() if len(small) else 'n/a'})."
    )
    a("")
    with open(path, "w") as fh:
        fh.write("\n".join(L) + "\n")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--new", required=True)
    p.add_argument("--old", required=True)
    p.add_argument("--report", required=True)
    p.add_argument("--exceptions", default=DEFAULT_EXCEPTIONS)
    p.add_argument("--per-country-csv", default=None)
    args = p.parse_args(argv)
    for attr in ("new", "old", "report", "exceptions"):
        setattr(args, attr, os.path.abspath(getattr(args, attr)))
    if args.per_country_csv:
        args.per_country_csv = os.path.abspath(args.per_country_csv)
    os.chdir(REPO_ROOT)

    cols = [
        "fid",
        "left",
        "top",
        "right",
        "bottom",
        "ISOCODE",
        "shapeID",
        "shapeName",
        "GHS_SMOD",
        "pop_density",
        "dtw_1",
        "slope_1",
    ] + GRIP_COLUMNS
    print(f"loading legacy  {args.old}", flush=True)
    old = load_frame(args.old, cols)
    print(f"loading new     {args.new}", flush=True)
    new = load_frame(args.new, cols)
    print(f"legacy {len(old):,} rows | new {len(new):,} rows", flush=True)

    df, joined = per_country_metrics(old, new)
    agreement = global_agreement(joined)
    ids = {
        c: 100.0
        * float((joined[c + "_o"].astype(str) == joined[c + "_n"].astype(str)).mean())
        for c in ("ISOCODE", "shapeID")
    }
    if "shapeName_o" in joined:
        ids["shapeName"] = 100.0 * float(
            (
                joined["shapeName_o"].astype(str) == joined["shapeName_n"].astype(str)
            ).mean()
        )
    else:
        ids["shapeName"] = np.nan
    fails = failures(df)
    exceptions = load_exceptions(args.exceptions)
    covered, uncovered = [], []
    for f in fails:
        e = cover(f, exceptions)
        (covered.append((f, e)) if e else uncovered.append(f))

    n_join = int(df["cells_joined"].sum())
    meta = {
        "new table": os.path.relpath(args.new, REPO_ROOT),
        "legacy table": os.path.relpath(args.old, REPO_ROOT),
        "legacy rows": f"{len(old):,}",
        "new rows": f"{len(new):,}",
        "spatially aligned (fid-joined) rows": f"{n_join:,}",
        "legacy-only rows": f"{len(old) - n_join:,}",
        "new-only rows": f"{len(new) - n_join:,}",
        "exception list": os.path.relpath(args.exceptions, REPO_ROOT),
    }
    write_report(args.report, df, fails, covered, uncovered, meta, agreement, ids)
    if args.per_country_csv:
        df.to_csv(args.per_country_csv)
    print(f"report -> {os.path.relpath(args.report, REPO_ROOT)}")
    print(f"failures={len(fails)} covered={len(covered)} uncovered={len(uncovered)}")
    if uncovered:
        for f in uncovered[:25]:
            print(
                f"  UNCOVERED {f['ISOCODE']} {f['metric']} "
                f"value={f['value']:.3f} magnitude={f['magnitude']:.3f}"
            )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
