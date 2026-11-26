"""Build a unified drinking-water source breakdown (JMP + researched fills).

Merges the JMP-parsed breakdown (jmp_source_breakdown_latest.csv) with the
manually curated research dataset (researched_source_breakdown.csv, built from
national censuses, DHS/EPHS reports, regulator statistics and statistical-
office publications for the countries whose JMP country files carry no
source-type data).

Priority: the best source wins. JMP-parsed values beat research fills wherever
the JMP row is *informative* — i.e. it carries a positive value in at least one
of packaged / delivered / borehole, the categories a real household survey
measures. A JMP row whose only content is the UnpipedUnaffected catch-all (or
nothing at all) is not informative: sparse EU-SILC/HBS blocks like
ROU_2012_EUSILC and BIH_2015_HBS say nothing about boreholes, and letting them
demote a national census fill deleted ROU's 37% rural borehole share. Those
rows fall back to research (data_origin="research", with the demotion recorded
in research_notes and a WARNING printed). Research otherwise only fills rows
where the JMP extract is entirely blank. Provenance is carried per row.

Research-filled rows are then reconciled against the WHO/JMP piped share at
build time (trust_categories_over_piped=False): known categories scale down to
the unpiped budget when they exceed 100 - piped; unknown categories stay null.
Imputation reruns the same scale-down rule, then, for any row whose piped share
had to be spatially interpolated (neighbour / continent), keeps the measured
categories and lowers piped instead: interpolation is the bottom of the source
hierarchy and never scales real data down. See researched_source_breakdown_README.md.

Run from repo root:
  python scripts/Data_Manipulation_Scripts/build_unified_source_breakdown.py
      [--exclude-low-confidence]

Outputs:
  data/processed/semi-processed/jmp_source_breakdown_unified.csv
  data/processed/semi-processed/jmp_source_breakdown_unified_validation.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from jmp_source_parser import (  # noqa: E402
    DEFAULT_MANUAL_ALPHA3_MAPPING,
    load_piped_reference_from_water_csv,
    reconcile_categories_over_100_pct,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
SEMI_PROCESSED = REPO_ROOT / "data" / "processed" / "semi-processed"
JMP_BREAKDOWN_CSV = SEMI_PROCESSED / "jmp_source_breakdown_latest.csv"
JMP_ERRORS_CSV = SEMI_PROCESSED / "jmp_source_breakdown_errors.csv"
RESEARCH_CSV = REPO_ROOT / "data" / "original_data" / "researched_source_breakdown.csv"
WATER_CSV = (
    REPO_ROOT / "data" / "original_data" / "WHO Household Water Data - 2023 Data.csv"
)
MERGED_DATA_CSV = REPO_ROOT / "data" / "processed" / "merged_data.csv"
OUTPUT_CSV = SEMI_PROCESSED / "jmp_source_breakdown_unified.csv"
VALIDATION_CSV = SEMI_PROCESSED / "jmp_source_breakdown_unified_validation.csv"

VALUE_COLS = [
    "URBANPackaged",
    "RURALPackaged",
    "URBANDelivered",
    "RURALDelivered",
    "URBANBorehole",
    "RURALBorehole",
    "URBANUnpipedUnaffected",
    "RURALUnpipedUnaffected",
    "URBANUnpipedUnknown",
    "RURALUnpipedUnknown",
]
PROVENANCE_COLS = [
    "data_origin",
    "research_tier",
    "research_confidence",
    "research_source",
    "research_url",
    "research_flags",
    "research_notes",
    "research_bounds",
    "research_verification",
]

# Sum of piped + our four categories per stratum may exceed 100 by this much
# before it is reported (rounding and reference-year mismatch tolerance).
SUM_TOLERANCE = 2.0

# Percentage points of the JMP row-99 "other improved sources" remainder that
# may sit outside every model category before the build reports it. The
# remainder is not reallocated (its composition is ambiguous, so it belongs in
# the sampled Unallocated bucket) — this is a visibility threshold only.
UNCLASSIFIED_REMAINDER_TOLERANCE = 2.0

# Columns that make a JMP row "informative": the ones only a household survey
# that actually asked about source type can fill. UnpipedUnaffected is excluded on
# purpose — it is the catch-all every ladder-style block can populate without
# measuring anything the model cares about.
INFORMATIVE_COLS = [
    "URBANPackaged",
    "RURALPackaged",
    "URBANDelivered",
    "RURALDelivered",
    "URBANBorehole",
    "RURALBorehole",
]

CATEGORIES = (
    "Packaged",
    "Delivered",
    "Borehole",
    "UnpipedUnaffected",
    "UnpipedUnknown",
)


def _category_pct(row, stratum: str, category: str) -> float | None:
    value = row[f"{stratum}{category}"]
    return None if pd.isna(value) else value


def jmp_row_is_informative(row) -> bool:
    """Whether a JMP row measured anything the research fill could not.

    "Best source wins": a JMP row only outranks a curated
    research row if it is informative, meaning it carries a positive value in
    at least one of packaged / delivered / borehole. A row whose entire content
    is the UnpipedUnaffected catch-all — the shape sparse EU-SILC / HBS blocks
    produce — measured none of the model's source types and must not demote a
    census-grade fill.

    Note on zeros: the parser writes 0.0, not NaN, for every category a
    selected survey did not report (apply_piped_reference_and_normalize), so
    "non-null" cannot distinguish measured from absent here. A positive value
    is therefore the only available evidence of content — the same convention
    the displaced-research warning below already uses.
    """
    for col in INFORMATIVE_COLS:
        value = row.get(col)
        if value is not None and pd.notna(value) and float(value) > 0:
            return True
    return False


def load_research(path: Path, exclude_low_confidence: bool) -> pd.DataFrame:
    research = pd.read_csv(path)
    for col in VALUE_COLS:
        if col not in research.columns:
            research[col] = pd.NA
        research[col] = pd.to_numeric(research[col], errors="coerce")

    has_values = research[VALUE_COLS].notna().any(axis=1)
    research = research[has_values].copy()

    if exclude_low_confidence:
        research = research[research["confidence"].str.lower() != "low"].copy()

    dupes = research[research.duplicated(subset=["alpha3"], keep=False)]
    if not dupes.empty:
        raise ValueError(
            f"Duplicate alpha3 rows in research CSV: {sorted(dupes['alpha3'])}"
        )
    return research


def validate(
    unified: pd.DataFrame,
    piped_reference: dict,
    merged_alpha3: "list[str] | None" = None,
) -> pd.DataFrame:
    """Range, sum and coverage checks; returns warnings (may be empty).

    Plus a diagnostic check: when the JMP row-99 "other improved sources"
    remainder (the part cart + tanker do not explain, rows 102-103) exceeds
    UNCLASSIFIED_REMAINDER_TOLERANCE in a stratum, it is reported. That mass is
    deliberately left out of the five model categories, so it lands in the
    sampled Unallocated bucket; the warning exists so a 45-61pp case like the
    Philippines cannot pass unnoticed.

    Three checks per stratum:
      1. every category value inside [0, 100];
      2. categories alone, and categories + piped, inside 100 + tolerance;
      3. when NEITHER a PipedEffective nor a WHO reference piped share exists,
         the sum cannot be verified at all — that is reported explicitly rather
         than passing silently, so a missing reference cannot look like a
         clean row.

    When ``merged_alpha3`` is given, every country the model ships must have a
    row here; the ones that do not are reported so they cannot vanish into a
    downstream ``fillna(0)``.
    """
    problems = []
    for _, row in unified.iterrows():
        alpha3 = row["alpha3"]
        for col in VALUE_COLS:
            value = row[col]
            if pd.notna(value) and not (0 <= value <= 100):
                problems.append((alpha3, col, value, "value outside [0, 100]"))

        piped = piped_reference.get(alpha3, {})
        for stratum in ("URBAN", "RURAL"):
            piped_value = row.get(f"{stratum}PipedEffective")
            if piped_value is None or pd.isna(piped_value):
                piped_value = piped.get(f"{stratum}Piped")
            if piped_value is not None and pd.isna(piped_value):
                piped_value = None
            parts = [
                row[f"{stratum}{cat}"]
                for cat in (
                    "Packaged",
                    "Delivered",
                    "Borehole",
                    "UnpipedUnaffected",
                    "UnpipedUnknown",
                )
            ]
            known = [p for p in parts if pd.notna(p)]
            if not known:
                continue
            unpiped_sum = sum(known)
            if unpiped_sum > 100 + SUM_TOLERANCE:
                problems.append(
                    (
                        alpha3,
                        stratum,
                        round(unpiped_sum, 2),
                        "unpiped categories sum > 100",
                    )
                )
            elif piped_value is None:
                problems.append(
                    (
                        alpha3,
                        stratum,
                        round(unpiped_sum, 2),
                        "no piped reference - sum unverifiable",
                    )
                )
            elif piped_value + unpiped_sum > 100 + SUM_TOLERANCE:
                problems.append(
                    (
                        alpha3,
                        stratum,
                        round(piped_value + unpiped_sum, 2),
                        f"piped ({piped_value}) + unpiped categories > 100",
                    )
                )

        for stratum in ("URBAN", "RURAL"):
            remainder = row.get(f"OtherImprovedUnclassified_{stratum}")
            if remainder is None or pd.isna(remainder):
                continue
            if float(remainder) > UNCLASSIFIED_REMAINDER_TOLERANCE:
                problems.append(
                    (
                        alpha3,
                        f"{stratum}OtherImprovedUnclassified",
                        round(float(remainder), 2),
                        (
                            f"{round(float(remainder), 1)}pp of JMP 'other improved "
                            "sources' is not cart or tanker and matches no model "
                            "category; left unallocated by design (diagnostic)"
                        ),
                    )
                )

    if merged_alpha3 is not None:
        present = set(unified["alpha3"])
        for alpha3 in sorted(set(merged_alpha3) - present):
            problems.append(
                (
                    alpha3,
                    "coverage",
                    pd.NA,
                    "in merged_data.csv but absent from unified breakdown; "
                    "placeholder row (data_origin=none) added",
                )
            )
    return pd.DataFrame(problems, columns=["alpha3", "field", "value", "problem"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--exclude-low-confidence",
        action="store_true",
        help="Drop research fills whose confidence is 'low'.",
    )
    args = parser.parse_args()

    jmp = pd.read_csv(JMP_BREAKDOWN_CSV)
    research = load_research(RESEARCH_CSV, args.exclude_low_confidence)

    unified = jmp.copy()
    for col in PROVENANCE_COLS:
        unified[col] = pd.NA

    jmp_has_data = unified[VALUE_COLS].notna().any(axis=1)
    unified.loc[jmp_has_data, "data_origin"] = "jmp"
    unified.loc[~jmp_has_data, "data_origin"] = "none"

    research_by_alpha3 = research.set_index("alpha3")
    filled = []
    demoted = []

    def apply_research(idx, r, extra_note: str | None = None) -> None:
        for col in VALUE_COLS:
            unified.at[idx, col] = r[col]
        unified.at[idx, "survey"] = r.get("survey")
        unified.at[idx, "survey_year"] = r.get("survey_year")
        unified.at[idx, "data_origin"] = "research"
        unified.at[idx, "research_tier"] = r.get("tier")
        unified.at[idx, "research_confidence"] = r.get("confidence")
        unified.at[idx, "research_source"] = r.get("source")
        unified.at[idx, "research_url"] = r.get("url")
        unified.at[idx, "research_flags"] = r.get("flags")
        notes = r.get("mapping_notes")
        if extra_note:
            notes = f"{notes} | {extra_note}" if pd.notna(notes) else extra_note
        unified.at[idx, "research_notes"] = notes
        unified.at[idx, "research_bounds"] = r.get("bounds_notes")
        unified.at[idx, "research_verification"] = r.get("verification")

    # Rows with no JMP data at all: straight research fill.
    for idx in unified.index[~jmp_has_data]:
        alpha3 = unified.at[idx, "alpha3"]
        if alpha3 not in research_by_alpha3.index:
            continue
        apply_research(idx, research_by_alpha3.loc[alpha3])
        filled.append(alpha3)

    # Rows where BOTH sources exist: best source wins. JMP keeps the row only
    # when it is informative (see jmp_row_is_informative); a catch-all-only JMP
    # row is demoted in favour of the research fill.
    for idx in unified.index[jmp_has_data]:
        alpha3 = unified.at[idx, "alpha3"]
        if alpha3 not in research_by_alpha3.index:
            continue
        row = unified.loc[idx]
        if jmp_row_is_informative(row):
            continue
        jmp_survey = row.get("survey")
        jmp_survey = str(jmp_survey) if pd.notna(jmp_survey) else "unnamed survey"
        note = (
            f"JMP row demoted: {jmp_survey} reported no packaged, delivered or "
            "borehole value (at most a catch-all UnpipedUnaffected), so the research "
            "fill is the better source"
        )
        r = research_by_alpha3.loc[alpha3]
        research_source = r.get("source")
        research_source = (
            str(research_source) if pd.notna(research_source) else "research fill"
        )
        apply_research(idx, r, extra_note=note)
        filled.append(alpha3)
        demoted.append((alpha3, jmp_survey, research_source))

    # Research rows for territories absent from the JMP extract entirely.
    missing = research_by_alpha3.index.difference(unified["alpha3"])
    for alpha3 in missing:
        r = research_by_alpha3.loc[alpha3]
        new_row = {
            "alpha3": alpha3,
            "Entity": r.get("Entity"),
            "data_origin": "research",
        }
        for col in VALUE_COLS:
            new_row[col] = r[col]
        new_row["survey"] = r.get("survey")
        new_row["survey_year"] = r.get("survey_year")
        new_row["research_tier"] = r.get("tier")
        new_row["research_confidence"] = r.get("confidence")
        new_row["research_source"] = r.get("source")
        new_row["research_url"] = r.get("url")
        new_row["research_flags"] = r.get("flags")
        new_row["research_notes"] = r.get("mapping_notes")
        new_row["research_bounds"] = r.get("bounds_notes")
        new_row["research_verification"] = r.get("verification")
        unified = pd.concat([unified, pd.DataFrame([new_row])], ignore_index=True)
        filled.append(alpha3)

    # Research rows still outranked by an informative JMP row. Read off
    # data_origin rather than the pre-demotion mask, so demoted countries are
    # not double-reported as both "ignored" and "used".
    research_available = set(research_by_alpha3.index)
    skipped_jmp_present = sorted(
        research_available & set(unified.loc[unified["data_origin"] == "jmp", "alpha3"])
    )

    # Normalize garbled Entity names in the JMP extract (Arabic-script rows for
    # BHR/KWT/QAT, literal '0' for MAC) using the research dataset's names.
    def _entity_is_garbled(value) -> bool:
        if pd.isna(value):
            return True
        text = str(value)
        return text == "0" or not any(c.isascii() and c.isalpha() for c in text)

    research_full = (
        pd.read_csv(RESEARCH_CSV).drop_duplicates("alpha3").set_index("alpha3")
    )
    for idx, row in unified.iterrows():
        if (
            _entity_is_garbled(row.get("Entity"))
            and row["alpha3"] in research_full.index
        ):
            unified.at[idx, "Entity"] = research_full.loc[row["alpha3"], "Entity"]

    piped_reference = {}
    if WATER_CSV.exists():
        piped_reference = load_piped_reference_from_water_csv(
            WATER_CSV, manual_alpha3_mapping=DEFAULT_MANUAL_ALPHA3_MAPPING
        )

    # Reconcile research-origin rows with the WHO piped share (scale-down rule).
    # Imputation re-reconciles, and lowers piped instead on rows whose piped
    # share was spatially interpolated (trust_categories_over_piped=True).
    for col in (
        "URBANPipedEffective",
        "RURALPipedEffective",
        "URBAN_budget_adjustment",
        "RURAL_budget_adjustment",
    ):
        unified[col] = pd.NA
    adjustments = []
    for idx, row in unified.iterrows():
        if row["data_origin"] != "research":
            continue
        piped = piped_reference.get(row["alpha3"], {})
        for stratum in ("URBAN", "RURAL"):
            (
                packaged,
                delivered,
                borehole,
                unaffected,
                unknown,
                piped_effective,
                note,
            ) = reconcile_categories_over_100_pct(
                piped_pct=piped.get(f"{stratum}Piped"),
                packaged_pct=_category_pct(row, stratum, "Packaged"),
                delivered_pct=_category_pct(row, stratum, "Delivered"),
                borehole_pct=_category_pct(row, stratum, "Borehole"),
                unpiped_unaffected_pct=_category_pct(row, stratum, "UnpipedUnaffected"),
                unpiped_unknown_pct=_category_pct(row, stratum, "UnpipedUnknown"),
                trust_categories_over_piped=False,
            )
            for cat, value in zip(
                CATEGORIES, (packaged, delivered, borehole, unaffected, unknown)
            ):
                if value is not None:
                    unified.at[idx, f"{stratum}{cat}"] = value
            if piped_effective is not None:
                unified.at[idx, f"{stratum}PipedEffective"] = piped_effective
            if note:
                unified.at[idx, f"{stratum}_budget_adjustment"] = note
                adjustments.append((row["alpha3"], stratum, note))

    # Coverage: every country the model ships must have a row here, even if it
    # is an empty one. Otherwise an absent country would silently become
    # all-zero categories via the downstream fillna(0) with no trace in the
    # provenance columns.
    merged_alpha3 = None
    if MERGED_DATA_CSV.exists():
        merged_alpha3 = sorted(
            pd.read_csv(MERGED_DATA_CSV, usecols=["alpha3"])["alpha3"].dropna().unique()
        )

    validation = validate(unified, piped_reference, merged_alpha3=merged_alpha3)

    # Provenance for the demoted-JMP rows, so the swap is auditable from the
    # validation CSV alone.
    if demoted:
        validation = pd.concat(
            [
                validation,
                pd.DataFrame(
                    [
                        {
                            "alpha3": alpha3,
                            "field": "data_origin",
                            "value": "jmp -> research",
                            "problem": (
                                f"JMP row ({jmp_survey}) is catch-all only - no "
                                "packaged/delivered/borehole value - so the research "
                                f"fill ({research_source}) wins"
                            ),
                        }
                        for alpha3, jmp_survey, research_source in demoted
                    ]
                ),
            ],
            ignore_index=True,
        )

    # "JMP wins" can demote a curated research row to nothing when the JMP
    # survey reports a category as an explicit zero. Report those, because a
    # zero from a survey that never asked the question is not the same datum as
    # a national census figure (BIH/ROU boreholes are the live example).
    displaced = []
    for alpha3 in skipped_jmp_present:
        jmp_row = unified.loc[unified["alpha3"] == alpha3]
        if jmp_row.empty:
            continue
        jmp_row = jmp_row.iloc[0]
        r = research_by_alpha3.loc[alpha3]
        lost = [
            col
            for col in VALUE_COLS
            if pd.notna(r[col]) and r[col] > 0 and not (jmp_row[col] > 0)
        ]
        if lost:
            displaced.append((alpha3, lost))
            validation = pd.concat(
                [
                    validation,
                    pd.DataFrame(
                        [
                            {
                                "alpha3": alpha3,
                                "field": ",".join(lost),
                                "value": pd.NA,
                                "problem": (
                                    "research fill ignored (JMP wins) but the JMP "
                                    "survey reports zero where research has a "
                                    "measured value"
                                ),
                            }
                        ]
                    ),
                ],
                ignore_index=True,
            )

    # Why the parser produced no row, verbatim, so a placeholder says which
    # kind of gap it is (ISR/FLK are a classified, documented exclusion; other
    # gaps are simply unfilled).
    parse_errors: dict[str, str] = {}
    if JMP_ERRORS_CSV.exists():
        errors_df = pd.read_csv(JMP_ERRORS_CSV)
        if {"iso3", "error"}.issubset(errors_df.columns):
            parse_errors = dict(zip(errors_df["iso3"], errors_df["error"]))

    coverage_gaps: list[str] = []
    if merged_alpha3 is not None:
        coverage_gaps = sorted(set(merged_alpha3) - set(unified["alpha3"]))
        placeholders = []
        for alpha3 in coverage_gaps:
            placeholder = {col: pd.NA for col in unified.columns}
            placeholder["alpha3"] = alpha3
            placeholder["data_origin"] = "none"
            note = (
                "no JMP source data and no research fill; placeholder row so the "
                "country is visible in provenance instead of silently missing"
            )
            if alpha3 in parse_errors:
                note = f"{note}; JMP parse status: {parse_errors[alpha3]}"
            placeholder["research_notes"] = note
            placeholders.append(placeholder)
        if placeholders:
            # Grown by reindex rather than concat: an all-NA frame would change
            # every column's dtype (and warn about it) on pandas 2.x.
            start = len(unified)
            unified = unified.reset_index(drop=True).reindex(
                range(start + len(placeholders))
            )
            for offset, placeholder in enumerate(placeholders):
                for col, value in placeholder.items():
                    if value is not pd.NA:
                        unified.at[start + offset, col] = value

    unified = unified.sort_values("alpha3").reset_index(drop=True)
    unified.to_csv(OUTPUT_CSV, index=False)
    validation.to_csv(VALIDATION_CSV, index=False)

    origin_counts = unified["data_origin"].value_counts(dropna=False)
    print("data_origin counts:")
    for origin, count in origin_counts.items():
        print(f"  {origin}: {count}")
    print(f"research fills applied: {len(filled)} -> {sorted(set(filled))}")
    if coverage_gaps:
        print(
            f"coverage: {len(coverage_gaps)} merged_data countries had no row "
            f"(placeholders added): {coverage_gaps}"
        )
    if skipped_jmp_present:
        print(
            "research rows ignored because an informative JMP row exists "
            f"(best source wins): {skipped_jmp_present}"
        )
    for alpha3, jmp_survey, research_source in demoted:
        print(
            f"  WARNING {alpha3}: JMP row demoted, data_origin jmp -> research. "
            f"JMP source: {jmp_survey} (catch-all UnpipedUnaffected only, no "
            f"packaged/delivered/borehole). Research source: {research_source}."
        )
    for alpha3, lost in displaced:
        print(
            f"  WARNING {alpha3}: JMP reports zero for {lost} where the research "
            "row has a measured value"
        )
    if adjustments:
        print(f"piped-budget reconciliation applied to {len(adjustments)} strata:")
        for alpha3, stratum, note in adjustments:
            print(f"  {alpha3} {stratum}: {note}")
    if validation.empty:
        print("validation: no problems found")
    else:
        print(f"validation: {len(validation)} problems -> {VALIDATION_CSV}")
        print(validation.to_string(index=False))
    print(f"wrote {len(unified)} rows to {OUTPUT_CSV}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
