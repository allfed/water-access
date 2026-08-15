"""Tests for JMP source breakdown parser."""

import sys
from pathlib import Path

import pytest

SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts" / "Data_Manipulation_Scripts"
sys.path.insert(0, str(SCRIPT_DIR))

from jmp_source_parser import (  # noqa: E402
    DEFAULT_MANUAL_ALPHA3_MAPPING,
    SUBCATEGORY_ROWS,
    VARIANT_SERVICE_LADDER_ONLY_ERROR,
    _has_any_subcategory_cell,
    _resolve_raw_dir,
    apply_piped_reference_and_normalize,
    breakdown_to_model_columns,
    diagnostic_columns,
    load_piped_reference_from_water_csv,
    map_country_names_to_alpha3,
    reconcile_categories_over_100_pct,
    other_improved_unclassified_pct,
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
        for indicator in ("packaged", "delivered", "borehole", "piped", "unpiped_unaffected")
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
        "unpiped_unaffected_urban_pct": 7.0,
        "unpiped_unaffected_rural_pct": 8.0,
    }
    mapped = breakdown_to_model_columns(row)
    assert mapped["URBANPackaged"] == 1.0
    assert mapped["RURALBorehole"] == 6.0
    assert mapped["URBANUnpipedUnaffected"] == 7.0


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


def test_reconcile_categories_over_100_pct_scales_to_budget():
    pkg, delivered, borehole, unaffected, unknown, _, _ = reconcile_categories_over_100_pct(
        piped_pct=90.0,
        packaged_pct=80.0,
        delivered_pct=10.0,
        borehole_pct=5.0,
        unpiped_unaffected_pct=5.0,
    )
    assert pkg == 8.0
    assert delivered == 1.0
    assert borehole == 0.5
    assert unaffected == 0.5
    assert unknown is None


def test_reconcile_categories_over_100_pct_scales_proportionally():
    pkg, delivered, borehole, unaffected, unknown, _, _ = reconcile_categories_over_100_pct(
        piped_pct=90.0,
        packaged_pct=8.0,
        delivered_pct=4.0,
        borehole_pct=4.0,
        unpiped_unaffected_pct=4.0,
    )
    known = [v for v in (pkg, delivered, borehole, unaffected, unknown) if v is not None]
    assert pytest.approx(sum(known)) == 10.0
    assert pytest.approx(pkg) == 4.0
    assert pytest.approx(unaffected) == 2.0


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


def test_apply_piped_reference_and_normalize_preserves_unpiped_unaffected():
    survey = {
        "packaged_urban_pct": 80.0,
        "packaged_rural_pct": 40.0,
        "delivered_urban_pct": None,
        "delivered_rural_pct": None,
        "borehole_urban_pct": None,
        "borehole_rural_pct": None,
        "unpiped_unaffected_urban_pct": 5.0,
        "unpiped_unaffected_rural_pct": 20.0,
    }
    normalized = apply_piped_reference_and_normalize(
        survey, urban_piped=95.0, rural_piped=70.0
    )
    mapped = breakdown_to_model_columns(normalized)
    assert mapped["URBANPackaged"] < 80.0
    assert mapped["URBANUnpipedUnaffected"] > 0.0
    assert (
        mapped["URBANPackaged"]
        + mapped["URBANUnpipedUnaffected"]
        <= 5.0 + 1e-6
    )


def test_apply_piped_reference_uses_country_piped_when_global_missing():
    """When WHO has no stratum piped, budget from country row 6 (100 - piped)."""
    survey = {
        "piped_urban_pct": 99.0,
        "piped_rural_pct": 97.0,
        "packaged_urban_pct": 50.0,
        "packaged_rural_pct": 20.0,
        "delivered_urban_pct": None,
        "delivered_rural_pct": None,
        "borehole_urban_pct": None,
        "borehole_rural_pct": None,
        "unpiped_unaffected_urban_pct": 5.0,
        "unpiped_unaffected_rural_pct": 10.0,
    }
    normalized = apply_piped_reference_and_normalize(
        survey, urban_piped=None, rural_piped=None
    )
    mapped = breakdown_to_model_columns(normalized)
    # Urban budget = 100 - 99 = 1% from country row 6 piped
    assert mapped["URBANPackaged"] == pytest.approx(50 / 55, abs=0.001)
    assert pytest.approx(mapped["URBANPackaged"] + mapped["URBANUnpipedUnaffected"]) == 1.0
    # Rural budget = 100 - 97 = 3%
    assert mapped["RURALPackaged"] == pytest.approx(2.0, abs=0.001)
    assert mapped["RURALUnpipedUnaffected"] == pytest.approx(1.0, abs=0.001)


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
            "unpiped_unaffected_urban_pct": None,
            "unpiped_unaffected_rural_pct": None,
            "unpiped_unaffected_total_pct": 19.4,
        }
        for category, expected in (
            ("packaged", 22.7),
            ("delivered", 6.0),
            ("unpiped_unaffected", 19.4),
        ):
            urban, rural = resolve_urban_rural_with_total_fallback(row, category)
            assert urban == expected
            assert rural == expected

        mapped = breakdown_to_model_columns(row)
        assert mapped["URBANPackaged"] == 22.7
        assert mapped["RURALPackaged"] == 22.7
        assert mapped["URBANDelivered"] == 6.0
        assert mapped["URBANUnpipedUnaffected"] == 19.4

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
            "unpiped_unaffected_urban_pct": None,
            "unpiped_unaffected_rural_pct": None,
            "unpiped_unaffected_total_pct": 19.4,
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
            + mapped["URBANUnpipedUnaffected"]
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
            "unpiped_unaffected_urban_pct": None,
            "unpiped_unaffected_rural_pct": None,
            "unpiped_unaffected_total_pct": 24.0,
        }
        normalized = apply_piped_reference_and_normalize(
            survey, urban_piped=93.0, rural_piped=70.0
        )
        mapped = breakdown_to_model_columns(normalized)
        for stratum, piped in (("URBAN", 93.0), ("RURAL", 70.0)):
            total = sum(
                mapped[f"{stratum}{cat}"]
                for cat in ("Packaged", "Delivered", "Borehole", "UnpipedUnaffected")
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


class TestOtherImprovedUnclassifiedDiagnostic:
    """Row 99 minus cart + tanker: the ambiguous remainder (rows 102-103).

    It is reported, never reallocated — the mass belongs in the sampled
    Unallocated bucket, so this must stay OUT of the five model categories.
    """

    def test_remainder_is_row99_minus_delivered(self):
        assert other_improved_unclassified_pct(61.56186, 0.28063) == 61.2812

    def test_none_row99_stays_unknown(self):
        assert other_improved_unclassified_pct(None, 0.5) is None

    def test_missing_delivered_counts_as_zero(self):
        assert other_improved_unclassified_pct(45.5, None) == 45.5

    def test_floored_at_zero(self):
        # rounding can push the children a hair past the parent
        assert other_improved_unclassified_pct(2.0, 2.0001) == 0.0

    def test_diagnostic_columns_use_the_total_fallback(self):
        row = {
            "other_improved_unclassified_urban_pct": None,
            "other_improved_unclassified_rural_pct": None,
            "other_improved_unclassified_total_pct": 12.5,
        }
        assert diagnostic_columns(row) == {
            "OtherImprovedUnclassified_URBAN": 12.5,
            "OtherImprovedUnclassified_RURAL": 12.5,
        }

    def test_diagnostic_columns_stay_none_when_unknown(self):
        assert diagnostic_columns({}) == {
            "OtherImprovedUnclassified_URBAN": None,
            "OtherImprovedUnclassified_RURAL": None,
        }

    def test_diagnostic_is_not_a_model_column(self):
        row = {
            "other_improved_unclassified_urban_pct": 61.2813,
            "other_improved_unclassified_rural_pct": 45.4635,
            "borehole_urban_pct": 1.0,
        }
        mapped = breakdown_to_model_columns(row)
        assert "OtherImprovedUnclassified_URBAN" not in mapped
        assert all(
            mapped[col] in (None, 1.0)
            for col in (
                "URBANPackaged",
                "URBANDelivered",
                "URBANBorehole",
                "URBANUnpipedUnaffected",
            )
        )

    def test_philippines_workbook_reports_the_remainder(self, raw_dir):
        path = raw_dir / "PHL.xlsx"
        if not path.exists():
            pytest.skip("PHL JMP file not available")
        _iso3, records, err = parse_country(path)
        assert err is None
        fies = [r for r in records if r["survey"] == "PHL_2023_FIES"]
        assert fies, "PHL_2023_FIES block not found"
        record = fies[0]
        # FIES "community water system": 61.3pp urban / 45.5pp rural sit in the
        # unextracted sub-'Other' rows of the row-99 block.
        assert record["other_improved_unclassified_urban_pct"] == pytest.approx(
            61.28, abs=0.05
        )
        assert record["other_improved_unclassified_rural_pct"] == pytest.approx(
            45.46, abs=0.05
        )
        # and none of it leaked into a model category: Delivered is still just
        # cart + tanker, UnpipedUnaffected still just wells/springs/rain/surface.
        mapped = breakdown_to_model_columns(record)
        assert mapped["URBANDelivered"] == pytest.approx(0.28, abs=0.05)
        assert mapped["URBANUnpipedUnaffected"] == pytest.approx(3.4008, abs=0.05)
        assert (
            mapped["URBANUnpipedUnaffected"]
            < record["other_improved_unclassified_urban_pct"]
        )


class TestVariantWorkbookClassification:
    """ISR/FLK ship admin-source service-ladder workbooks: survey blocks, no
    household subcategory data. They are a documented exclusion, not a parser
    gap, and must say so precisely instead of reading as a generic parse
    failure."""

    @pytest.mark.parametrize("iso3", ["ISR", "FLK"])
    def test_service_ladder_only_workbooks_are_classified(self, raw_dir, iso3):
        path = raw_dir / f"{iso3}.xlsx"
        if not path.exists():
            pytest.skip(f"{iso3} JMP file not available")
        parsed_iso3, records, err = parse_country(path)
        assert parsed_iso3 == iso3
        assert records == []
        assert err == VARIANT_SERVICE_LADDER_ONLY_ERROR

    @staticmethod
    def _rows_with(row_idx, col, offset, value):
        rows = [tuple([None] * 12) for _ in range(120)]
        row = list(rows[row_idx])
        row[col + offset] = value
        rows[row_idx] = tuple(row)
        return rows

    def test_empty_subcategory_rows_read_as_no_data(self):
        rows = [tuple([None] * 12) for _ in range(120)]
        assert not _has_any_subcategory_cell(rows, [(1, "ISR_2018_PWH")])

    def test_a_numeric_subcategory_cell_counts_as_data(self):
        rows = self._rows_with(SUBCATEGORY_ROWS[0], 1, 3, 12.5)
        assert _has_any_subcategory_cell(rows, [(1, "XXX_2020_CEN")])

    def test_text_numbers_count_as_data_too(self):
        # deliberately generous: the check exists to PROVE absence, so a value
        # we merely failed to parse must never be classified as "nothing there"
        rows = self._rows_with(SUBCATEGORY_ROWS[-1], 1, 4, "12.5")
        assert _has_any_subcategory_cell(rows, [(1, "XXX_2020_CEN")])

    def test_parent_row_99_alone_is_not_subcategory_data(self):
        # ISR_2018_PWH carries a harmonized service-ladder "100" in row 99 and
        # nothing else; that must not disqualify the service-ladder-only
        # classification.
        from jmp_source_parser import ROW_OTHER_IMPROVED

        assert ROW_OTHER_IMPROVED not in SUBCATEGORY_ROWS
        rows = self._rows_with(ROW_OTHER_IMPROVED, 1, 3, "100")
        assert not _has_any_subcategory_cell(rows, [(1, "ISR_2018_PWH")])

    def test_non_numeric_text_is_not_data(self):
        rows = self._rows_with(SUBCATEGORY_ROWS[0], 1, 3, "No")
        assert not _has_any_subcategory_cell(rows, [(1, "ISR_2011_DWQ")])

    def test_only_isr_and_flk_trip_the_classification(self):
        """Regression guard on the repo's own workbook set."""
        errors_csv = (
            Path(__file__).resolve().parents[1]
            / "data"
            / "processed"
            / "semi-processed"
            / "jmp_source_breakdown_errors.csv"
        )
        if not errors_csv.exists():
            pytest.skip("errors CSV not built")
        import pandas as pd

        errors = pd.read_csv(errors_csv)
        classified = sorted(
            errors.loc[errors["error"] == VARIANT_SERVICE_LADDER_ONLY_ERROR, "iso3"]
        )
        assert classified == ["FLK", "ISR"]


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
