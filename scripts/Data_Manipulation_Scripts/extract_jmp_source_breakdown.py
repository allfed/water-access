"""Extract JMP source-type breakdown for all countries and write CSV outputs.

Run from repo root:
  python scripts/Data_Manipulation_Scripts/extract_jmp_source_breakdown.py

Outputs:
  data/processed/semi-processed/jmp_source_breakdown_latest.csv
  data/processed/semi-processed/jmp_source_breakdown_errors.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from jmp_source_parser import (  # noqa: E402
    DEFAULT_MANUAL_ALPHA3_MAPPING,
    parse_all_countries,
    load_piped_reference_from_water_csv,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = REPO_ROOT / "data" / "processed" / "semi-processed"
WATER_CSV = REPO_ROOT / "data" / "original_data" / "WHO Household Water Data - 2023 Data.csv"


def main() -> int:
    piped_reference = {}
    if WATER_CSV.exists():
        piped_reference = load_piped_reference_from_water_csv(
            WATER_CSV, manual_alpha3_mapping=DEFAULT_MANUAL_ALPHA3_MAPPING
        )

    latest_rows, errors = parse_all_countries(piped_reference=piped_reference)

    model_rows = []
    for row in latest_rows:
        model_row = {
            "alpha3": row["iso3"],
            "Entity": row.get("country"),
            "URBANPackaged": row.get("URBANPackaged"),
            "RURALPackaged": row.get("RURALPackaged"),
            "URBANDelivered": row.get("URBANDelivered"),
            "RURALDelivered": row.get("RURALDelivered"),
            "URBANBorehole": row.get("URBANBorehole"),
            "RURALBorehole": row.get("RURALBorehole"),
            "URBANOtherUnpiped": row.get("URBANOtherUnpiped"),
            "RURALOtherUnpiped": row.get("RURALOtherUnpiped"),
            "survey": row.get("survey"),
            "survey_year": row.get("survey_year"),
            "piped_reference_year": row.get("piped_reference_year"),
        }
        model_rows.append(model_row)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    breakdown_path = OUTPUT_DIR / "jmp_source_breakdown_latest.csv"
    errors_path = OUTPUT_DIR / "jmp_source_breakdown_errors.csv"

    pd.DataFrame(model_rows).to_csv(breakdown_path, index=False)
    pd.DataFrame(errors, columns=["iso3", "error"]).to_csv(errors_path, index=False)

    df = pd.read_csv(breakdown_path)
    for col in (
        "URBANPackaged",
        "RURALPackaged",
        "URBANDelivered",
        "RURALDelivered",
        "URBANBorehole",
        "RURALBorehole",
        "URBANOtherUnpiped",
        "RURALOtherUnpiped",
    ):
        if col in df.columns:
            n = df[col].notna().sum()
            nz = (df[col].fillna(0) > 0).sum()
            print(f"{col}: {n} countries with data ({nz} > 0)")

    with_survey = df["survey"].notna().sum() if "survey" in df.columns else 0
    print(f"countries with selected survey: {with_survey}")
    print(f"wrote {len(model_rows)} countries to {breakdown_path}")
    print(f"{len(errors)} parse errors -> {errors_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
