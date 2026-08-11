"""Parse JMP household country files for drinking-water source breakdown.

Adapted from the well-coverage repository (parse_jmp.py). Extracts urban/rural
percentages of population using piped, packaged, delivered, borehole/tubewell,
and other unpiped improved sources from the 'Water Data' sheet of each country
xlsx.

JMP template row indices (0-based, verified across GHA/BGD/IDN/KEN/MEX/NLD):
  6   All piped
  7   Non-piped (improved unpiped total)
  57  Tubewell, borehole
  61  Traditional wells
  73  All springs
  85  Rainwater
  88  Packaged water
  91  Surface water
  99  Other improved sources     (parent of 100-103; diagnostic only)
  100 Cart with small tank/drum  (delivered water)
  101 Tanker truck provided      (delivered water)
  102 Other                      (unclassifiable remainder of row 99)
  103 Other                      (unclassifiable remainder of row 99)
  104 Other non-improved
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import openpyxl

DEFAULT_MAX_YEAR_GAP = 5

DEFAULT_MANUAL_ALPHA3_MAPPING = {
    "Channel Islands": "CHI",
    "United States Virgin Islands": "VIR",
    "Wallis and Futuna Islands": "WLF",
    "Democratic Republic of the Congo": "COD",
    "China, Hong Kong SAR": "HKG",
    "China, Macao SAR": "MAC",
    "Republic of Korea": "KOR",
    "Curaçao": "CUW",
    "Sint Maarten (Dutch part)": "SXM",
    "Democratic People's Republic of Korea": "PRK",
    # pycountry's fuzzy search resolves "Niger" to Nigeria (NGA); without this
    # entry Niger silently inherits no piped reference at all.
    "Niger": "NER",
}

# Top-level JMP rows used for scaling (avoid hierarchical double-count with piped/sub-rows)
BOREHOLE_ROWS = [57]
PACKAGED_ROWS = [88]
DELIVERED_ROWS = [100, 101]
OTHER_UNPIPED_ROWS = [61, 73, 85, 91, 104]

ROW_PIPED = 6
ROW_NON_PIPED = 7
# Parent row of the "Other improved sources" block. Its children are cart (100),
# tanker (101) and one or two unlabelled "Other" rows (102-103) that no model
# category can absorb without guessing (see UNALLOCATED_GAP_INVESTIGATION.md
# §"row-99"). The remainder is extracted as a DIAGNOSTIC only: it stays out of
# the five model categories and therefore lands in the sampled Unallocated
# bucket, which is where ambiguous mass belongs.
ROW_OTHER_IMPROVED = 99

# Leaf rows that only a household survey can fill in (wells, springs,
# rainwater, packaged, surface water, cart, tanker, other non-improved). A
# workbook that lists survey blocks but leaves every one of these empty carries
# no subcategory data to recover (see parse_country / VARIANT_LADDER_ONLY_ERROR).
# ROW_OTHER_IMPROVED is deliberately NOT in this set: it is a parent aggregate,
# and admin-source ladder blocks populate it with a harmonized "100" while
# measuring no source type at all (ISR_2018_PWH does exactly that).
SUBCATEGORY_ROWS = sorted(
    set(BOREHOLE_ROWS + PACKAGED_ROWS + DELIVERED_ROWS + OTHER_UNPIPED_ROWS)
)

VARIANT_LADDER_ONLY_ERROR = (
    "variant admin-source workbook: ladder estimates only, no household "
    "subcategory data (documented exclusion)"
)


def _resolve_raw_dir(raw_dir: Path | None = None) -> Path:
    """Return JMP raw xlsx directory, preferring explicit path then local then sibling repo."""
    if raw_dir is not None:
        return raw_dir
    repo_root = Path(__file__).resolve().parents[2]
    candidates = [
        repo_root / "data" / "original_data" / "jmp_country_files",
        repo_root.parent / "well-coverage" / "data" / "raw",
    ]
    for path in candidates:
        if path.is_dir() and any(path.glob("*.xlsx")):
            return path
    return candidates[0]


def survey_year(survey_id: str) -> int | None:
    match = re.search(r"_(\d{4})_", str(survey_id))
    return int(match.group(1)) if match else None


def _pct_value(rows: list[tuple], row_idx: int, col: int, offset: int) -> float | None:
    if row_idx >= len(rows):
        return None
    row = rows[row_idx]
    idx = col + offset
    if idx >= len(row):
        return None
    value = row[idx]
    if isinstance(value, (int, float)):
        return round(float(value), 4)
    return None


def _sum_pct(
    rows: list[tuple], row_indices: list[int], col: int, offset: int
) -> float | None:
    total = 0.0
    found = False
    for row_idx in row_indices:
        value = _pct_value(rows, row_idx, col, offset)
        if value is not None:
            total += value
            found = True
    return round(total, 4) if found else None


def other_improved_unclassified_pct(
    other_improved_pct: float | None, delivered_pct: float | None
) -> float | None:
    """Part of row 99 that cart + tanker do not explain (rows 102-103).

    None when row 99 itself is absent — "unknown", never a manufactured zero.
    Floored at 0 because rounding can make the children sum a hair past the
    parent. Purely diagnostic: it is never added to a model category.
    """
    if other_improved_pct is None:
        return None
    return round(max(0.0, other_improved_pct - (delivered_pct or 0.0)), 4)


def _extract_block_pcts(rows: list[tuple], col: int) -> dict[str, float | None]:
    out: dict[str, float | None] = {}
    for suffix, offset in (("urban", 3), ("rural", 4), ("total", 5)):
        out[f"piped_{suffix}_pct"] = _pct_value(rows, ROW_PIPED, col, offset)
        out[f"non_piped_{suffix}_pct"] = _pct_value(rows, ROW_NON_PIPED, col, offset)
        out[f"borehole_{suffix}_pct"] = _sum_pct(rows, BOREHOLE_ROWS, col, offset)
        out[f"packaged_{suffix}_pct"] = _sum_pct(rows, PACKAGED_ROWS, col, offset)
        out[f"delivered_{suffix}_pct"] = _sum_pct(rows, DELIVERED_ROWS, col, offset)
        out[f"other_unpiped_{suffix}_pct"] = _sum_pct(
            rows, OTHER_UNPIPED_ROWS, col, offset
        )
        other_improved = _pct_value(rows, ROW_OTHER_IMPROVED, col, offset)
        out[f"other_improved_{suffix}_pct"] = other_improved
        unclassified = other_improved_unclassified_pct(
            other_improved, out[f"delivered_{suffix}_pct"]
        )
        out[f"other_improved_unclassified_{suffix}_pct"] = unclassified
    return out


def _survey_has_subcategory_data(record: dict[str, Any]) -> bool:
    keys = (
        "packaged_urban_pct",
        "packaged_rural_pct",
        "delivered_urban_pct",
        "delivered_rural_pct",
        "borehole_urban_pct",
        "borehole_rural_pct",
        "borehole_total_pct",
        "other_unpiped_urban_pct",
        "other_unpiped_rural_pct",
    )
    return any(record.get(key) is not None for key in keys)


def _is_numeric_cell(value: Any) -> bool:
    """True for a number, or for a number stored as text (e.g. the '100' that
    admin-source ladder workbooks write into their survey blocks)."""
    if isinstance(value, bool):
        return False
    if isinstance(value, (int, float)):
        return True
    if isinstance(value, str):
        try:
            float(value.strip().replace("%", ""))
        except ValueError:
            return False
        return True
    return False


def _has_any_subcategory_cell(rows: list[tuple], blocks: list[tuple]) -> bool:
    """Whether ANY survey block holds a value in ANY household-subcategory row.

    Used to tell a *variant layout we could parse harder* apart from a workbook
    that simply has no subcategory data to give (ISR, FLK). Text is counted as
    data here on purpose: the point is to prove absence, so the check is
    deliberately generous about what it accepts as presence.
    """
    for col, _survey_id in blocks:
        for row_idx in SUBCATEGORY_ROWS:
            if row_idx >= len(rows):
                continue
            row = rows[row_idx]
            for offset in (3, 4, 5):
                idx = col + offset
                if idx < len(row) and _is_numeric_cell(row[idx]):
                    return True
    return False


def parse_country(path: Path) -> tuple[str, list[dict[str, Any]], str | None]:
    """Parse one JMP country workbook. Returns (iso3, survey records, error)."""
    iso3 = path.stem
    try:
        workbook = openpyxl.load_workbook(path, read_only=True, data_only=True)
    except Exception as exc:
        return iso3, [], f"load error: {exc}"

    if "Water Data" not in workbook.sheetnames:
        workbook.close()
        return iso3, [], "no Water Data sheet"

    rows = list(workbook["Water Data"].iter_rows(values_only=True))
    workbook.close()

    if len(rows) < 102:
        return iso3, [], "sheet too short"

    ids_row = rows[1]
    country = rows[0][3] if len(rows[0]) > 3 else iso3
    blocks = [
        (col, ids_row[col])
        for col in range(len(ids_row))
        if ids_row[col] and re.match(r"^[A-Z]{3}_\d{4}_", str(ids_row[col]))
    ]

    records: list[dict[str, Any]] = []
    for col, survey_id in blocks:
        year = survey_year(str(survey_id))
        pcts = _extract_block_pcts(rows, col)
        has_data = any(
            pcts.get(f"{key}_total_pct") is not None
            for key in ("packaged", "delivered", "borehole", "piped", "other_unpiped")
        )
        if not has_data:
            continue
        record = {
            "iso3": iso3,
            "country": country,
            "survey": survey_id,
            "year": year,
            "source_type": rows[2][col] if col < len(rows[2]) else None,
            **pcts,
        }
        records.append(record)

    if blocks and not records:
        # The workbook lists surveys but none of the fixed template rows hold a
        # number. Two very different situations hide behind that:
        #
        #  * ISR and FLK: admin-source "ladder" workbooks. Same 6-column block
        #    grid, but the blocks carry only harmonized ladder estimates (often
        #    stored as TEXT, e.g. "100") and every household subcategory row is
        #    empty. There is nothing to recover, so these are a documented
        #    exclusion rather than a parser gap — ISR's piped is ~100 (gap ~0)
        #    and FLK is ~3,800 people.
        #  * anything else: real subcategory numbers sit somewhere we did not
        #    look, i.e. a genuine template variant worth a parser.
        if not _has_any_subcategory_cell(rows, blocks):
            return iso3, [], VARIANT_LADDER_ONLY_ERROR
        return (
            iso3,
            [],
            f"{len(blocks)} survey blocks but no numeric data in the expected "
            "template rows (possible template variant)",
        )

    return iso3, records, None


def _unpiped_budget(
    piped_pct: float | None, non_piped_pct: float | None
) -> float | None:
    if piped_pct is not None:
        return max(0.0, 100.0 - piped_pct)
    if non_piped_pct is not None:
        return max(0.0, non_piped_pct)
    return None


def normalize_unpiped_subcategories(
    *,
    piped_pct: float | None,
    non_piped_pct: float | None,
    packaged_pct: float | None,
    delivered_pct: float | None,
    borehole_pct: float | None,
    other_unpiped_pct: float | None,
) -> tuple[float, float, float, float]:
    """Scale tracked + other unpiped shares to fit the unpiped budget."""
    budget = _unpiped_budget(piped_pct, non_piped_pct)
    packaged = packaged_pct or 0.0
    delivered = delivered_pct or 0.0
    borehole = borehole_pct or 0.0
    other = other_unpiped_pct or 0.0
    total = packaged + delivered + borehole + other

    if budget is None:
        return packaged, delivered, borehole, other
    if total <= budget or total == 0:
        return packaged, delivered, borehole, other

    scale = budget / total
    return (
        round(packaged * scale, 4),
        round(delivered * scale, 4),
        round(borehole * scale, 4),
        round(other * scale, 4),
    )


def select_survey_for_country(
    records: list[dict[str, Any]],
    piped_reference_year: int | None,
    max_year_gap: int = DEFAULT_MAX_YEAR_GAP,
) -> dict[str, Any] | None:
    """Pick the most recent survey within max_year_gap of the piped reference year."""
    if not records:
        return None

    candidates = sorted(records, key=lambda r: r.get("year") or 0, reverse=True)
    if piped_reference_year is None:
        for rec in candidates:
            if _survey_has_subcategory_data(rec):
                return rec
        return None

    for rec in candidates:
        year = rec.get("year")
        if year is None:
            continue
        if abs(year - piped_reference_year) > max_year_gap:
            continue
        if _survey_has_subcategory_data(rec):
            return rec
    return None


def resolve_urban_rural_with_total_fallback(
    record: dict[str, Any], category: str
) -> tuple[float | None, float | None]:
    """Return urban/rural percentages for ``category``, falling back to total.

    Some JMP surveys (household-frequency phone surveys and similar rapid
    instruments, e.g. MMR_2021_HFPS or UKR_2024_SWS) report country totals with
    no urban/rural split. Without a fallback the whole category ships as zero
    even though the survey measured it. The fallback only fires when BOTH
    strata are unknown, so measured splits are never overwritten.
    """
    urban = record.get(f"{category}_urban_pct")
    rural = record.get(f"{category}_rural_pct")
    if urban is None and rural is None:
        total = record.get(f"{category}_total_pct")
        if total is not None:
            return total, total
    return urban, rural


def resolve_borehole_urban_rural(
    record: dict[str, Any],
) -> tuple[float | None, float | None]:
    """Return urban/rural borehole percentages, falling back to total when needed."""
    return resolve_urban_rural_with_total_fallback(record, "borehole")


def apply_piped_reference_and_normalize(
    survey: dict[str, Any],
    urban_piped: float | None,
    rural_piped: float | None,
) -> dict[str, Any]:
    """Normalize unpiped subcategories against piped reference shares from merged data."""
    resolved = {
        category: resolve_urban_rural_with_total_fallback(survey, category)
        for category in ("packaged", "delivered", "borehole", "other_unpiped")
    }

    pkg_u, del_u, bh_u, other_u = normalize_unpiped_subcategories(
        piped_pct=urban_piped,
        non_piped_pct=survey.get("non_piped_urban_pct"),
        packaged_pct=resolved["packaged"][0],
        delivered_pct=resolved["delivered"][0],
        borehole_pct=resolved["borehole"][0],
        other_unpiped_pct=resolved["other_unpiped"][0],
    )
    pkg_r, del_r, bh_r, other_r = normalize_unpiped_subcategories(
        piped_pct=rural_piped,
        non_piped_pct=survey.get("non_piped_rural_pct"),
        packaged_pct=resolved["packaged"][1],
        delivered_pct=resolved["delivered"][1],
        borehole_pct=resolved["borehole"][1],
        other_unpiped_pct=resolved["other_unpiped"][1],
    )

    normalized = {
        "packaged": (pkg_u, pkg_r),
        "delivered": (del_u, del_r),
        "borehole": (bh_u, bh_r),
        "other_unpiped": (other_u, other_r),
    }
    out = dict(survey)
    for category, (value_u, value_r) in normalized.items():
        raw_u, raw_r = resolved[category]
        out[f"{category}_urban_pct"] = value_u if raw_u is not None else 0.0
        out[f"{category}_rural_pct"] = value_r if raw_r is not None else 0.0
    return out


def build_country_breakdown(
    all_records: list[dict[str, Any]],
    piped_reference: dict[str, dict[str, Any]] | None = None,
    max_year_gap: int = DEFAULT_MAX_YEAR_GAP,
) -> list[dict[str, Any]]:
    """Build per-country breakdown using year-filtered same-survey selection."""
    piped_reference = piped_reference or {}
    by_iso: dict[str, list[dict[str, Any]]] = {}
    for record in all_records:
        by_iso.setdefault(record["iso3"], []).append(record)

    breakdown_rows: list[dict[str, Any]] = []
    for iso3, records in sorted(by_iso.items()):
        ref = piped_reference.get(iso3, {})
        survey = select_survey_for_country(
            records,
            piped_reference_year=ref.get("piped_year"),
            max_year_gap=max_year_gap,
        )
        if survey is None:
            breakdown_rows.append({"iso3": iso3, "country": records[0].get("country")})
            continue

        normalized = apply_piped_reference_and_normalize(
            survey,
            urban_piped=ref.get("URBANPiped"),
            rural_piped=ref.get("RURALPiped"),
        )
        entry = {
            "iso3": iso3,
            "country": normalized.get("country"),
            "survey": normalized.get("survey"),
            "survey_year": normalized.get("year"),
            "piped_reference_year": ref.get("piped_year"),
        }
        entry.update(breakdown_to_model_columns(normalized))
        entry.update(diagnostic_columns(normalized))
        breakdown_rows.append(entry)

    return breakdown_rows


def breakdown_to_model_columns(row: dict[str, Any]) -> dict[str, float | None]:
    """Map parser output to merged_data column names."""
    packaged_urban, packaged_rural = resolve_urban_rural_with_total_fallback(
        row, "packaged"
    )
    delivered_urban, delivered_rural = resolve_urban_rural_with_total_fallback(
        row, "delivered"
    )
    borehole_urban, borehole_rural = resolve_urban_rural_with_total_fallback(
        row, "borehole"
    )
    other_urban, other_rural = resolve_urban_rural_with_total_fallback(
        row, "other_unpiped"
    )
    return {
        "URBANPackaged": packaged_urban,
        "RURALPackaged": packaged_rural,
        "URBANDelivered": delivered_urban,
        "RURALDelivered": delivered_rural,
        "URBANBorehole": borehole_urban,
        "RURALBorehole": borehole_rural,
        "URBANOtherUnpiped": other_urban,
        "RURALOtherUnpiped": other_rural,
    }


DIAGNOSTIC_COLUMNS = (
    "OtherImprovedUnclassified_URBAN",
    "OtherImprovedUnclassified_RURAL",
)


def diagnostic_columns(row: dict[str, Any]) -> dict[str, float | None]:
    """Diagnostic (non-model) columns for the per-country breakdown.

    ``OtherImprovedUnclassified_*`` is the part of JMP's "Other improved
    sources" row that cart + tanker do not explain (rows 102-103): 45.5pp rural
    / 61.3pp urban for the Philippines, where it is the FIES "community water
    system" category. It is deliberately NOT folded into any of the five model
    categories — its composition is ambiguous, so the mass belongs in the
    sampled Unallocated bucket, which is exactly where it already lands. The
    column exists so that fact is visible instead of invisible.

    None stays None (unknown), and the same total -> urban/rural fallback the
    model columns use applies, so total-only surveys report the diagnostic too.
    """
    urban, rural = resolve_urban_rural_with_total_fallback(
        row, "other_improved_unclassified"
    )
    return {
        "OtherImprovedUnclassified_URBAN": urban,
        "OtherImprovedUnclassified_RURAL": rural,
    }


def map_country_names_to_alpha3(
    country_names,
    manual_alpha3_mapping: dict[str, str] | None = None,
) -> dict[str, str | None]:
    """Map WHO/JMP country names to alpha3 codes, refusing silent collisions.

    ``pycountry.countries.search_fuzzy`` is a best-effort matcher: "Niger"
    resolves to Nigeria (NGA), so before the manual override was added, one of
    the two countries silently lost its row. Any two DISTINCT country names
    resolving to the same alpha3 now raises instead. The same name repeated
    across year rows is normal and is deduplicated before the check.
    """
    import pycountry

    manual_alpha3_mapping = manual_alpha3_mapping or {}

    mapping: dict[str, str | None] = {}
    for name in dict.fromkeys(country_names):
        if name in manual_alpha3_mapping:
            mapping[name] = manual_alpha3_mapping[name]
            continue
        cleaned = str(name).split("(")[0].strip()
        try:
            mapping[name] = pycountry.countries.search_fuzzy(cleaned)[0].alpha_3
        except LookupError:
            mapping[name] = None

    by_alpha3: dict[str, list[str]] = {}
    for name, alpha3 in mapping.items():
        if isinstance(alpha3, str) and alpha3:
            by_alpha3.setdefault(alpha3, []).append(str(name))
    collisions = {a: sorted(n) for a, n in by_alpha3.items() if len(n) > 1}
    if collisions:
        detail = "; ".join(f"{a} <- {names}" for a, names in sorted(collisions.items()))
        raise ValueError(
            "Ambiguous country name -> alpha3 mapping (add an explicit entry to "
            f"DEFAULT_MANUAL_ALPHA3_MAPPING): {detail}"
        )
    return mapping


def _latest_non_null(group, column: str) -> float | None:
    """Value of ``column`` from the latest year in ``group`` that has one."""
    import pandas as pd

    if column not in group.columns:
        return None
    series = group[column]
    for value in series:
        if pd.notna(value):
            return float(value)
    return None


def load_piped_reference_from_water_csv(
    water_csv_path: Path,
    manual_alpha3_mapping: dict[str, str] | None = None,
) -> dict[str, dict[str, Any]]:
    """Load piped year and urban/rural piped shares per alpha3 from the WHO CSV.

    Urban and rural are resolved INDEPENDENTLY: each takes the value from the
    most recent year in which that stratum has a piped number. Taking the
    latest year unconditionally silently dropped the reference whenever the
    newest row's cell was "-" (NaN) — e.g. USA rural and SLB urban — which in
    turn skipped normalisation for those strata entirely.

    ``piped_year`` is the most recent year carrying any piped value (urban,
    rural or total); it only drives survey selection, not the shares.
    """
    import pandas as pd

    df = pd.read_csv(water_csv_path, low_memory=False)
    df.replace({"<1": 0, ">99": 100, "-": pd.NA}, inplace=True)
    for col in ("URBANPiped", "RURALPiped", "TOTALPiped"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    alpha3_by_name = map_country_names_to_alpha3(
        df["Country"], manual_alpha3_mapping=manual_alpha3_mapping
    )
    df["alpha3"] = df["Country"].map(alpha3_by_name)
    df = df.sort_values(["Country", "Year"], ascending=[True, False])

    piped_cols = [
        col for col in ("URBANPiped", "RURALPiped", "TOTALPiped") if col in df.columns
    ]

    reference: dict[str, dict[str, Any]] = {}
    for country, group in df.groupby("Country", sort=False):
        alpha3 = alpha3_by_name.get(country)
        if not isinstance(alpha3, str) or not alpha3:
            continue
        group = group.sort_values("Year", ascending=False)

        piped_year = None
        if piped_cols:
            has_piped = group[piped_cols].notna().any(axis=1)
            years = group.loc[has_piped, "Year"].dropna()
            if not years.empty:
                piped_year = int(years.iloc[0])
        if piped_year is None:
            years = group["Year"].dropna()
            piped_year = int(years.iloc[0]) if not years.empty else None

        reference[alpha3] = {
            "piped_year": piped_year,
            "URBANPiped": _latest_non_null(group, "URBANPiped"),
            "RURALPiped": _latest_non_null(group, "RURALPiped"),
        }
    return reference


def parse_all_countries(
    raw_dir: Path | None = None,
    piped_reference: dict[str, dict[str, Any]] | None = None,
    max_year_gap: int = DEFAULT_MAX_YEAR_GAP,
) -> tuple[list[dict], list[tuple]]:
    """Parse every xlsx in raw_dir. Returns (country breakdown rows, errors)."""
    directory = _resolve_raw_dir(raw_dir)
    files = sorted(directory.glob("*.xlsx"))
    all_records: list[dict[str, Any]] = []
    errors: list[tuple[str, str]] = []

    for path in files:
        iso3, records, err = parse_country(path)
        if err:
            errors.append((iso3, err))
        else:
            all_records.extend(records)

    return (
        build_country_breakdown(
            all_records,
            piped_reference=piped_reference,
            max_year_gap=max_year_gap,
        ),
        errors,
    )
