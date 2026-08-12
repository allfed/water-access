"""Tests for the research-fill piped-budget reconciliation rule.

The rule (agreed on the 2026-08-06 Kevin/James call, carve-out added
2026-08-08): research-filled source-breakdown rows are fitted to the WHO/JMP
unpiped budget (100 - piped) exactly like the JMP parser does for survey rows,
EXCEPT when WHO piped is a rounded ">99" (>= 99.5) — there the measured
category values win and an effective piped share is emitted instead.
"""

import sys
from pathlib import Path

SCRIPT_DIR = (
    Path(__file__).resolve().parents[1] / "scripts" / "Data_Manipulation_Scripts"
)
sys.path.insert(0, str(SCRIPT_DIR))

import pytest  # noqa: E402

from build_unified_source_breakdown import (  # noqa: E402
    jmp_row_is_informative,
    reconcile_stratum_with_piped_budget,
)


def make_values(packaged=None, delivered=None, borehole=None, other=None):
    return {
        "Packaged": packaged,
        "Delivered": delivered,
        "Borehole": borehole,
        "OtherUnpiped": other,
    }


class TestDefaultBudgetRule:
    """piped < 99.5: James's proportional downscale, identical in spirit to
    normalize_unpiped_subcategories."""

    def test_scales_down_when_over_budget(self):
        values = make_values(packaged=10.0, borehole=10.0)
        adjusted, piped_eff, note = reconcile_stratum_with_piped_budget(values, 90.0)
        assert adjusted["Packaged"] == 5.0
        assert adjusted["Borehole"] == 5.0
        assert piped_eff is None
        assert note == "scaled_to_unpiped_budget(scale=0.5)"

    def test_no_change_when_within_budget(self):
        values = make_values(packaged=3.0, borehole=4.0)
        adjusted, piped_eff, note = reconcile_stratum_with_piped_budget(values, 90.0)
        assert adjusted == values
        assert piped_eff is None
        assert note is None

    def test_none_values_stay_none_when_scaling(self):
        values = make_values(borehole=20.0)  # Packaged/Delivered/Other unknown
        adjusted, _, note = reconcile_stratum_with_piped_budget(values, 90.0)
        assert adjusted["Borehole"] == 10.0
        assert adjusted["Packaged"] is None
        assert adjusted["Delivered"] is None
        assert adjusted["OtherUnpiped"] is None
        assert note is not None

    def test_year_drift_small_overshoot(self):
        # COG-rural-like: piped 11, categories sum 92.3 -> budget 89
        values = make_values(packaged=0.2, borehole=11.9, other=80.2)
        adjusted, piped_eff, note = reconcile_stratum_with_piped_budget(values, 11.0)
        total = sum(v for v in adjusted.values() if v is not None)
        assert abs(total - 89.0) < 0.01
        assert piped_eff is None
        assert "scaled_to_unpiped_budget" in note


class TestRoundedPipedCarveOut:
    """piped >= 99.5: measured categories win; effective piped is emitted."""

    def test_denmark_case_keeps_measured_boreholes(self):
        values = make_values(delivered=0.0, borehole=4.0, other=0.0)
        adjusted, piped_eff, note = reconcile_stratum_with_piped_budget(values, 100.0)
        assert adjusted["Borehole"] == 4.0
        assert piped_eff == 96.0
        assert note == "piped_capped"

    def test_anguilla_case_effective_piped_near_zero(self):
        values = make_values(packaged=60.7, delivered=0.0, borehole=1.0, other=35.9)
        adjusted, piped_eff, note = reconcile_stratum_with_piped_budget(values, 100.0)
        assert adjusted == values
        assert abs(piped_eff - 2.4) < 0.01
        assert note == "piped_capped"

    def test_categories_over_100_are_capped_proportionally(self):
        values = make_values(packaged=60.0, other=60.0)
        adjusted, piped_eff, note = reconcile_stratum_with_piped_budget(values, 100.0)
        assert adjusted["Packaged"] == 50.0
        assert adjusted["OtherUnpiped"] == 50.0
        assert piped_eff == 0.0
        assert "piped_capped(scale=" in note

    def test_threshold_is_99_5(self):
        values = make_values(borehole=4.0)
        _, piped_eff, note = reconcile_stratum_with_piped_budget(values, 99.5)
        assert piped_eff == 96.0 and note == "piped_capped"
        adjusted, piped_eff, note = reconcile_stratum_with_piped_budget(values, 99.4)
        # below threshold: budget rule applies, budget 0.6 < 4 -> scaled down
        assert piped_eff is None
        assert adjusted["Borehole"] == 0.6

    def test_evidence_based_zeros_keep_full_piped(self):
        values = make_values(packaged=0.0, delivered=0.0, borehole=0.0, other=0.0)
        adjusted, piped_eff, note = reconcile_stratum_with_piped_budget(values, 100.0)
        assert adjusted == values
        assert piped_eff == 100.0
        assert note == "piped_capped"


class TestNoReference:
    def test_no_piped_reference_leaves_row_untouched(self):
        values = make_values(packaged=60.7, other=35.9)
        adjusted, piped_eff, note = reconcile_stratum_with_piped_budget(values, None)
        assert adjusted == values
        assert piped_eff is None
        assert note is None

    def test_all_unknown_values_untouched(self):
        values = make_values()
        adjusted, piped_eff, note = reconcile_stratum_with_piped_budget(values, 100.0)
        assert adjusted == values
        assert piped_eff is None
        assert note is None


class TestValidator:
    """The validator must not stay silent when it cannot check a row."""

    @staticmethod
    def _row(alpha3, urban=None, rural=None, piped_effective=None):
        import pandas as pd

        urban = urban or {}
        rural = rural or {}
        row = {"alpha3": alpha3}
        for stratum, values in (("URBAN", urban), ("RURAL", rural)):
            for cat in ("Packaged", "Delivered", "Borehole", "OtherUnpiped"):
                row[f"{stratum}{cat}"] = values.get(cat, pd.NA)
            row[f"{stratum}PipedEffective"] = (piped_effective or {}).get(stratum, pd.NA)
        return row

    def test_missing_reference_is_reported_not_silently_passed(self):
        import pandas as pd

        from build_unified_source_breakdown import validate

        unified = pd.DataFrame([self._row("SLB", urban={"Borehole": 20.0})])
        problems = validate(unified, piped_reference={})
        urban_problems = problems[problems["field"] == "URBAN"]
        assert len(urban_problems) == 1
        assert "unverifiable" in urban_problems.iloc[0]["problem"]

    def test_present_reference_still_checks_the_sum(self):
        import pandas as pd

        from build_unified_source_breakdown import validate

        unified = pd.DataFrame([self._row("USA", rural={"OtherUnpiped": 43.7})])
        reference = {"USA": {"URBANPiped": 100.0, "RURALPiped": 95.0}}
        problems = validate(unified, piped_reference=reference)
        rural_problems = problems[problems["field"] == "RURAL"]
        assert len(rural_problems) == 1
        assert "piped (95.0) + unpiped categories > 100" in (
            rural_problems.iloc[0]["problem"]
        )

    def test_clean_row_produces_no_warnings(self):
        import pandas as pd

        from build_unified_source_breakdown import validate

        unified = pd.DataFrame(
            [
                self._row(
                    "KEN",
                    urban={"Borehole": 10.0, "OtherUnpiped": 5.0},
                    rural={"Borehole": 20.0, "OtherUnpiped": 10.0},
                )
            ]
        )
        reference = {"KEN": {"URBANPiped": 60.0, "RURALPiped": 30.0}}
        assert validate(unified, piped_reference=reference).empty

    def test_piped_effective_satisfies_the_reference_requirement(self):
        import pandas as pd

        from build_unified_source_breakdown import validate

        unified = pd.DataFrame(
            [
                self._row(
                    "DNK",
                    urban={"Borehole": 4.0},
                    piped_effective={"URBAN": 96.0},
                )
            ]
        )
        problems = validate(unified, piped_reference={})
        assert problems[problems["field"] == "URBAN"].empty

    def test_row99_remainder_over_tolerance_is_reported(self):
        import pandas as pd

        from build_unified_source_breakdown import validate

        row = self._row("PHL", urban={"Borehole": 1.0}, rural={"Borehole": 1.0})
        row["OtherImprovedUnclassified_URBAN"] = 61.2813
        row["OtherImprovedUnclassified_RURAL"] = 45.4635
        problems = validate(
            pd.DataFrame([row]),
            piped_reference={"PHL": {"URBANPiped": 60.0, "RURALPiped": 30.0}},
        )
        flagged = problems[
            problems["field"].str.contains("OtherImprovedUnclassified", na=False)
        ]
        assert sorted(flagged["field"]) == [
            "RURALOtherImprovedUnclassified",
            "URBANOtherImprovedUnclassified",
        ]
        assert flagged["value"].tolist() == [61.28, 45.46]
        assert all("unallocated by design" in p for p in flagged["problem"])

    def test_small_or_missing_remainder_is_not_reported(self):
        import pandas as pd

        from build_unified_source_breakdown import validate

        row = self._row("KEN", urban={"Borehole": 1.0})
        row["OtherImprovedUnclassified_URBAN"] = 2.0  # at the tolerance, not over
        row["OtherImprovedUnclassified_RURAL"] = pd.NA
        problems = validate(
            pd.DataFrame([row]),
            piped_reference={"KEN": {"URBANPiped": 60.0, "RURALPiped": 30.0}},
        )
        assert problems[
            problems["field"].str.contains("OtherImprovedUnclassified", na=False)
        ].empty

    def test_validator_tolerates_frames_without_the_diagnostic(self):
        import pandas as pd

        from build_unified_source_breakdown import validate

        unified = pd.DataFrame([self._row("KEN", urban={"Borehole": 10.0})])
        problems = validate(
            unified, piped_reference={"KEN": {"URBANPiped": 60.0, "RURALPiped": 30.0}}
        )
        assert problems.empty

    def test_coverage_gap_is_reported(self):
        import pandas as pd

        from build_unified_source_breakdown import validate

        unified = pd.DataFrame([self._row("KEN", urban={"Borehole": 10.0})])
        problems = validate(
            unified,
            piped_reference={"KEN": {"URBANPiped": 60.0, "RURALPiped": 30.0}},
            merged_alpha3=["KEN", "NIC", "TWN"],
        )
        coverage = problems[problems["field"] == "coverage"]
        assert sorted(coverage["alpha3"]) == ["NIC", "TWN"]


class TestBestSourceWins:
    """A JMP row only outranks a research fill when it measured something.

    ROU_2012_EUSILC / BIH_2015_HBS carry nothing but the OtherUnpiped catch-all;
    letting them win deleted Romania's 37% rural borehole share (census) — the
    wrong direction for the very review comment this branch answers.
    """

    @staticmethod
    def _row(**kwargs):
        import pandas as pd

        row = {
            col: kwargs.get(col, 0.0)
            for col in (
                "URBANPackaged",
                "RURALPackaged",
                "URBANDelivered",
                "RURALDelivered",
                "URBANBorehole",
                "RURALBorehole",
                "URBANOtherUnpiped",
                "RURALOtherUnpiped",
            )
        }
        row.update({k: v for k, v in kwargs.items() if k not in row})
        return pd.Series(row)

    def test_catch_all_only_row_is_not_informative(self):
        # ROU_2012_EUSILC as shipped: 10 urban / 65 rural OtherUnpiped, nothing else
        assert not jmp_row_is_informative(
            self._row(URBANOtherUnpiped=10.0, RURALOtherUnpiped=65.0)
        )

    def test_all_zero_row_is_not_informative(self):
        assert not jmp_row_is_informative(self._row())

    @pytest.mark.parametrize(
        "col", ["URBANPackaged", "RURALDelivered", "RURALBorehole"]
    )
    def test_any_positive_measured_category_is_informative(self, col):
        assert jmp_row_is_informative(self._row(**{col: 0.9}))

    def test_missing_values_are_not_informative(self):
        import pandas as pd

        assert not jmp_row_is_informative(
            pd.Series(
                {
                    "URBANPackaged": pd.NA,
                    "RURALBorehole": None,
                    "RURALOtherUnpiped": 65.0,
                }
            )
        )

    def test_plain_dict_row_works_too(self):
        assert jmp_row_is_informative({"URBANBorehole": 5.0})
        assert not jmp_row_is_informative({"URBANOtherUnpiped": 5.0})


class TestUnifiedCsvBestSourceOutcome:
    """Regression on the built artifact: ROU/BIH must keep their research fills."""

    @staticmethod
    def _unified():
        import pandas as pd

        path = (
            Path(__file__).resolve().parents[1]
            / "data"
            / "processed"
            / "semi-processed"
            / "jmp_source_breakdown_unified.csv"
        )
        if not path.exists():
            pytest.skip("unified breakdown not built")
        return pd.read_csv(path).set_index("alpha3")

    @pytest.mark.parametrize("alpha3", ["ROU", "BIH"])
    def test_demoted_countries_ship_research_boreholes(self, alpha3):
        row = self._unified().loc[alpha3]
        assert row["data_origin"] == "research"
        assert row["RURALBorehole"] > 0
        assert "demoted" in str(row["research_notes"])

    def test_an_informative_jmp_row_still_wins(self):
        unified = self._unified()
        # Kenya-style: a real household survey, research fill (if any) stays out
        assert unified.loc["KEN", "data_origin"] == "jmp"
