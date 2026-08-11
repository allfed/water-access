"""Tests for JMP source breakdown parser."""

import sys
from pathlib import Path

import pytest

SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts" / "Data_Manipulation_Scripts"
sys.path.insert(0, str(SCRIPT_DIR))

from jmp_source_parser import (  # noqa: E402
    DEFAULT_MANUAL_ALPHA3_MAPPING,
    _resolve_raw_dir,
    apply_piped_reference_and_normalize,
    breakdown_to_model_columns,
    load_piped_reference_from_water_csv,
    map_country_names_to_alpha3,
    normalize_unpiped_subcategories,
    parse_country,
    resolve_borehole_urban_rural,
    resolve_urban_rural_with_total_fallback,
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

class TestTotalOnlySurveyFallback:
    """Surveys reporting only country totals must not ship zeros.

    Before the fix only borehole had a total -> urban/rural fallback, so
    total-only instruments (MMR_2021_HFPS, UKR_2024_SWS) silently shipped
    packaged / delivered / other-unpiped as 0.
    """

    def test_fallback_applies_to_every_category(self):
        row = {
            "packaged_urban_pct": None,
            "packaged_rural_pct": None,
            "packaged_total_pct": 22.7,
            "delivered_urban_pct": None,
            "delivered_rural_pct": None,
            "delivered_total_pct": 6.0,
            "other_unpiped_urban_pct": None,
            "other_unpiped_rural_pct": None,
            "other_unpiped_total_pct": 19.4,
        }
        for category, expected in (
            ("packaged", 22.7),
            ("delivered", 6.0),
            ("other_unpiped", 19.4),
        ):
            urban, rural = resolve_urban_rural_with_total_fallback(row, category)
            assert urban == expected
            assert rural == expected

        mapped = breakdown_to_model_columns(row)
        assert mapped["URBANPackaged"] == 22.7
        assert mapped["RURALPackaged"] == 22.7
        assert mapped["URBANDelivered"] == 6.0
        assert mapped["URBANOtherUnpiped"] == 19.4

    def test_fallback_does_not_override_a_measured_split(self):
        row = {
            "packaged_urban_pct": 30.0,
            "packaged_rural_pct": None,
            "packaged_total_pct": 22.7,
        }
        urban, rural = resolve_urban_rural_with_total_fallback(row, "packaged")
        assert urban == 30.0
        assert rural is None

    def test_myanmar_like_total_only_survey_ships_packaged(self):
        survey = {
            "packaged_urban_pct": None,
            "packaged_rural_pct": None,
            "packaged_total_pct": 22.7,
            "delivered_urban_pct": None,
            "delivered_rural_pct": None,
            "delivered_total_pct": None,
            "borehole_urban_pct": None,
            "borehole_rural_pct": None,
            "borehole_total_pct": 30.0,
            "other_unpiped_urban_pct": None,
            "other_unpiped_rural_pct": None,
            "other_unpiped_total_pct": 19.4,
        }
        normalized = apply_piped_reference_and_normalize(
            survey, urban_piped=40.0, rural_piped=20.0
        )
        mapped = breakdown_to_model_columns(normalized)
        assert mapped["URBANPackaged"] > 0
        assert mapped["RURALPackaged"] > 0
        # 22.7 + 30 + 19.4 = 72.1 > urban budget of 60 -> scaled to fit
        urban_sum = (
            mapped["URBANPackaged"]
            + mapped["URBANDelivered"]
            + mapped["URBANBorehole"]
            + mapped["URBANOtherUnpiped"]
        )
        assert urban_sum == pytest.approx(60.0, abs=1e-3)

    def test_ukraine_like_total_over_100_fits_budget_after_normalize(self):
        # UKR_2024_SWS-like: raw totals sum past 100; must land inside the
        # unpiped budget once normalised.
        survey = {
            "packaged_urban_pct": None,
            "packaged_rural_pct": None,
            "packaged_total_pct": 29.0,
            "delivered_urban_pct": None,
            "delivered_rural_pct": None,
            "delivered_total_pct": 6.0,
            "borehole_urban_pct": None,
            "borehole_rural_pct": None,
            "borehole_total_pct": 60.0,
            "other_unpiped_urban_pct": None,
            "other_unpiped_rural_pct": None,
            "other_unpiped_total_pct": 24.0,
        }
        normalized = apply_piped_reference_and_normalize(
            survey, urban_piped=93.0, rural_piped=70.0
        )
        mapped = breakdown_to_model_columns(normalized)
        for stratum, piped in (("URBAN", 93.0), ("RURAL", 70.0)):
            total = sum(
                mapped[f"{stratum}{cat}"]
                for cat in ("Packaged", "Delivered", "Borehole", "OtherUnpiped")
            )
            assert total <= (100.0 - piped) + 1e-6
            assert mapped[f"{stratum}Packaged"] > 0

    def test_missing_category_still_ships_zero_not_none(self):
        survey = {
            "packaged_urban_pct": None,
            "packaged_rural_pct": None,
            "packaged_total_pct": None,
            "borehole_urban_pct": 5.0,
            "borehole_rural_pct": 10.0,
        }
        normalized = apply_piped_reference_and_normalize(
            survey, urban_piped=50.0, rural_piped=50.0
        )
        mapped = breakdown_to_model_columns(normalized)
        assert mapped["URBANPackaged"] == 0.0
        assert mapped["RURALDelivered"] == 0.0
        assert mapped["URBANBorehole"] == 5.0


class TestCountryNameMapping:
    def test_niger_maps_to_ner_not_nigeria(self):
        mapping = map_country_names_to_alpha3(
            ["Niger", "Nigeria"], DEFAULT_MANUAL_ALPHA3_MAPPING
        )
        assert mapping["Niger"] == "NER"
        assert mapping["Nigeria"] == "NGA"

    def test_collision_raises_instead_of_overwriting(self):
        with pytest.raises(ValueError) as excinfo:
            map_country_names_to_alpha3(["Niger", "Nigeria"], manual_alpha3_mapping={})
        message = str(excinfo.value)
        assert "NGA" in message
        assert "Niger" in message and "Nigeria" in message

    def test_repeated_names_are_not_a_collision(self):
        mapping = map_country_names_to_alpha3(["Kenya", "Kenya", "Kenya"])
        assert mapping == {"Kenya": "KEN"}

    def test_who_country_list_has_no_collisions(self):
        import pandas as pd

        water_csv = (
            Path(__file__).resolve().parents[1]
            / "data"
            / "original_data"
            / "WHO Household Water Data - 2023 Data.csv"
        )
        if not water_csv.exists():
            pytest.skip("WHO water CSV not available")
        names = pd.read_csv(water_csv, low_memory=False)["Country"]
        mapping = map_country_names_to_alpha3(names, DEFAULT_MANUAL_ALPHA3_MAPPING)
        assert mapping["Niger"] == "NER"


class TestPipedReferenceYearSelection:
    """Each stratum takes the latest year that actually has a piped value."""

    @staticmethod
    def _write_csv(tmp_path, rows):
        import pandas as pd

        path = tmp_path / "water.csv"
        pd.DataFrame(rows).to_csv(path, index=False)
        return path

    def test_stratum_falls_back_to_an_older_year(self, tmp_path):
        path = self._write_csv(
            tmp_path,
            [
                {
                    "Country": "Kenya",
                    "Year": 2015,
                    "URBANPiped": 60,
                    "RURALPiped": 20,
                    "TOTALPiped": 30,
                },
                {
                    "Country": "Kenya",
                    "Year": 2022,
                    "URBANPiped": 65,
                    "RURALPiped": "-",
                    "TOTALPiped": 35,
                },
            ],
        )
        ref = load_piped_reference_from_water_csv(path)
        assert ref["KEN"]["URBANPiped"] == 65.0
        assert ref["KEN"]["RURALPiped"] == 20.0
        assert ref["KEN"]["piped_year"] == 2022

    def test_reference_stays_none_when_no_year_has_the_stratum(self, tmp_path):
        path = self._write_csv(
            tmp_path,
            [
                {
                    "Country": "Kenya",
                    "Year": 2015,
                    "URBANPiped": 60,
                    "RURALPiped": "-",
                    "TOTALPiped": 30,
                },
                {
                    "Country": "Kenya",
                    "Year": 2022,
                    "URBANPiped": 65,
                    "RURALPiped": "-",
                    "TOTALPiped": 35,
                },
            ],
        )
        ref = load_piped_reference_from_water_csv(path)
        assert ref["KEN"]["RURALPiped"] is None

    def test_piped_year_is_the_latest_year_with_any_piped_value(self, tmp_path):
        path = self._write_csv(
            tmp_path,
            [
                {
                    "Country": "Kenya",
                    "Year": 2015,
                    "URBANPiped": 60,
                    "RURALPiped": 20,
                    "TOTALPiped": 30,
                },
                {
                    "Country": "Kenya",
                    "Year": 2022,
                    "URBANPiped": "-",
                    "RURALPiped": "-",
                    "TOTALPiped": "-",
                },
            ],
        )
        ref = load_piped_reference_from_water_csv(path)
        assert ref["KEN"]["piped_year"] == 2015
        assert ref["KEN"]["URBANPiped"] == 60.0

    def test_real_csv_recovers_usa_rural_slb_urban_and_niger(self):
        water_csv = (
            Path(__file__).resolve().parents[1]
            / "data"
            / "original_data"
            / "WHO Household Water Data - 2023 Data.csv"
        )
        if not water_csv.exists():
            pytest.skip("WHO water CSV not available")
        ref = load_piped_reference_from_water_csv(
            water_csv, DEFAULT_MANUAL_ALPHA3_MAPPING
        )
        assert ref["USA"]["RURALPiped"] is not None
        assert ref["SLB"]["URBANPiped"] is not None
        assert ref["NER"]["URBANPiped"] is not None
        assert ref["NGA"]["URBANPiped"] == 15.0
