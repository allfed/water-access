"""Tests for JMP source breakdown parser."""

import sys
from pathlib import Path

import pytest

SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts" / "Data_Manipulation_Scripts"
sys.path.insert(0, str(SCRIPT_DIR))

from jmp_source_parser import (  # noqa: E402
    _resolve_raw_dir,
    apply_piped_reference_and_normalize,
    breakdown_to_model_columns,
    normalize_unpiped_subcategories,
    parse_country,
    resolve_borehole_urban_rural,
    select_survey_for_country,
)


@pytest.fixture
def raw_dir():
    return _resolve_raw_dir()


def test_parse_indonesia_has_packaged_and_borehole(raw_dir):
    path = raw_dir / "IDN.xlsx"
    if not path.exists():
        pytest.skip("IDN JMP file not available")
    iso3, records, err = parse_country(path)
    assert err is None
    assert iso3 == "IDN"
    assert records
    latest = max(records, key=lambda r: r.get("year") or 0)
    has_source_data = any(
        latest.get(f"{indicator}_{suffix}_pct") is not None
        for indicator in ("packaged", "delivered", "borehole", "piped", "other_unpiped")
        for suffix in ("urban", "rural", "total")
    )
    assert has_source_data


def test_breakdown_to_model_columns_maps_names():
    row = {
        "packaged_urban_pct": 1.0,
        "packaged_rural_pct": 2.0,
        "delivered_urban_pct": 3.0,
        "delivered_rural_pct": 4.0,
        "borehole_urban_pct": 5.0,
        "borehole_rural_pct": 6.0,
        "other_unpiped_urban_pct": 7.0,
        "other_unpiped_rural_pct": 8.0,
    }
    mapped = breakdown_to_model_columns(row)
    assert mapped["URBANPackaged"] == 1.0
    assert mapped["RURALBorehole"] == 6.0
    assert mapped["URBANOtherUnpiped"] == 7.0


def test_borehole_total_fallback_when_urban_rural_missing():
    row = {
        "borehole_urban_pct": None,
        "borehole_rural_pct": None,
        "borehole_total_pct": 12.5,
    }
    urban, rural = resolve_borehole_urban_rural(row)
    assert urban == 12.5
    assert rural == 12.5
    mapped = breakdown_to_model_columns(row)
    assert mapped["URBANBorehole"] == 12.5
    assert mapped["RURALBorehole"] == 12.5


def test_borehole_urban_rural_preserved_when_present():
    row = {
        "borehole_urban_pct": 5.0,
        "borehole_rural_pct": 20.0,
        "borehole_total_pct": 12.0,
    }
    mapped = breakdown_to_model_columns(row)
    assert mapped["URBANBorehole"] == 5.0
    assert mapped["RURALBorehole"] == 20.0


def test_normalize_unpiped_subcategories_scales_to_budget():
    pkg, delivered, borehole, other = normalize_unpiped_subcategories(
        piped_pct=100.0,
        non_piped_pct=0.0,
        packaged_pct=80.0,
        delivered_pct=10.0,
        borehole_pct=5.0,
        other_unpiped_pct=5.0,
    )
    assert pkg == 0.0
    assert delivered == 0.0
    assert borehole == 0.0
    assert other == 0.0


def test_normalize_unpiped_subcategories_scales_proportionally():
    pkg, delivered, borehole, other = normalize_unpiped_subcategories(
        piped_pct=90.0,
        non_piped_pct=10.0,
        packaged_pct=8.0,
        delivered_pct=4.0,
        borehole_pct=4.0,
        other_unpiped_pct=4.0,
    )
    assert pytest.approx(pkg + delivered + borehole + other) == 10.0
    assert pytest.approx(pkg) == 4.0
    assert pytest.approx(other) == 2.0


def test_select_survey_for_country_respects_year_gap():
    records = [
        {"year": 2015, "survey": "OLD", "packaged_urban_pct": 80.0},
        {"year": 2022, "survey": "NEW", "packaged_urban_pct": 10.0},
    ]
    selected = select_survey_for_country(records, piped_reference_year=2022, max_year_gap=5)
    assert selected["survey"] == "NEW"
    none_within_gap = select_survey_for_country(
        records[:1], piped_reference_year=2022, max_year_gap=5
    )
    assert none_within_gap is None


def test_apply_piped_reference_and_normalize_preserves_other_unpiped():
    survey = {
        "packaged_urban_pct": 80.0,
        "packaged_rural_pct": 40.0,
        "delivered_urban_pct": None,
        "delivered_rural_pct": None,
        "borehole_urban_pct": None,
        "borehole_rural_pct": None,
        "other_unpiped_urban_pct": 5.0,
        "other_unpiped_rural_pct": 20.0,
        "non_piped_urban_pct": 10.0,
        "non_piped_rural_pct": 30.0,
    }
    normalized = apply_piped_reference_and_normalize(
        survey, urban_piped=95.0, rural_piped=70.0
    )
    mapped = breakdown_to_model_columns(normalized)
    assert mapped["URBANPackaged"] < 80.0
    assert mapped["URBANOtherUnpiped"] > 0.0
    assert (
        mapped["URBANPackaged"]
        + mapped["URBANOtherUnpiped"]
        <= 5.0 + 1e-6
    )