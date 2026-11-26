"""Tests for the research-fill piped-budget reconciliation rule.

Research rows trust their category values over the WHO piped headline
(trust_categories_over_piped=True): when piped + categories exceed 100,
piped is lowered to 100 - sum(categories). JMP survey rows use the default
scale-down rule (trust=False).
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
)
from jmp_source_parser import reconcile_categories_over_100_pct  # noqa: E402


class TestDefaultBudgetRule:
    """trust=False: proportional downscale to the unpiped budget."""

    def test_scales_down_when_over_budget(self):
        packaged, delivered, borehole, unaffected, unknown, piped_eff, note = (
            reconcile_categories_over_100_pct(
                piped_pct=90.0,
                packaged_pct=10.0,
                borehole_pct=10.0,
            )
        )
        assert packaged == 5.0
        assert borehole == 5.0
        assert piped_eff is None
        assert note == "scaled_to_unpiped_budget(scale=0.5)"

    def test_no_change_when_within_budget(self):
        packaged, delivered, borehole, unaffected, unknown, piped_eff, note = (
            reconcile_categories_over_100_pct(
                piped_pct=90.0,
                packaged_pct=3.0,
                borehole_pct=4.0,
            )
        )
        assert packaged == 3.0
        assert borehole == 4.0
        assert piped_eff is None
        assert note is None

    def test_unknown_categories_stay_none_when_scaling(self):
        packaged, delivered, borehole, unaffected, unknown, _, note = (
            reconcile_categories_over_100_pct(
                piped_pct=90.0,
                borehole_pct=20.0,
            )
        )
        assert borehole == 10.0
        assert packaged is None
        assert delivered is None
        assert unaffected is None
        assert unknown is None
        assert note is not None

    def test_year_drift_small_overshoot(self):
        # COG-rural-like: piped 11, categories sum 92.3 -> budget 89
        packaged, delivered, borehole, unaffected, unknown, piped_eff, note = (
            reconcile_categories_over_100_pct(
                piped_pct=11.0,
                packaged_pct=0.2,
                borehole_pct=11.9,
                unpiped_unaffected_pct=80.2,
            )
        )
        total = sum(
            v
            for v in (packaged, delivered, borehole, unaffected, unknown)
            if v is not None
        )
        assert abs(total - 89.0) < 0.01
        assert piped_eff is None
        assert "scaled_to_unpiped_budget" in note


class TestTrustCategoriesOverPiped:
    """trust=True: researched categories win; piped is lowered when they conflict."""

    def test_denmark_case_keeps_measured_boreholes(self):
        packaged, delivered, borehole, unaffected, unknown, piped_eff, note = (
            reconcile_categories_over_100_pct(
                piped_pct=100.0,
                delivered_pct=0.0,
                borehole_pct=4.0,
                unpiped_unaffected_pct=0.0,
                trust_categories_over_piped=True,
            )
        )
        assert borehole == 4.0
        assert piped_eff == 96.0
        assert note == "piped_lowered"

    def test_anguilla_case_effective_piped_near_zero(self):
        packaged, delivered, borehole, unaffected, unknown, piped_eff, note = (
            reconcile_categories_over_100_pct(
                piped_pct=100.0,
                packaged_pct=60.7,
                delivered_pct=0.0,
                borehole_pct=1.0,
                unpiped_unaffected_pct=35.9,
                trust_categories_over_piped=True,
            )
        )
        assert packaged == 60.7
        assert abs(piped_eff - 2.4) < 0.01
        assert note == "piped_lowered"

    def test_high_piped_with_zero_categories_unchanged(self):
        packaged, delivered, borehole, unaffected, unknown, piped_eff, note = (
            reconcile_categories_over_100_pct(
                piped_pct=100.0,
                packaged_pct=0.0,
                delivered_pct=0.0,
                borehole_pct=0.0,
                unpiped_unaffected_pct=0.0,
                trust_categories_over_piped=True,
            )
        )
        assert packaged == 0.0
        assert piped_eff is None
        assert note is None

    def test_rounded_piped_below_100_still_scales_without_trust(self):
        packaged, delivered, borehole, unaffected, unknown, piped_eff, note = (
            reconcile_categories_over_100_pct(
                piped_pct=99.4,
                borehole_pct=4.0,
            )
        )
        assert piped_eff is None
        assert borehole == 0.6


class TestNoReference:
    def test_no_piped_reference_leaves_row_untouched(self):
        packaged, delivered, borehole, unaffected, unknown, piped_eff, note = (
            reconcile_categories_over_100_pct(
                packaged_pct=60.7,
                unpiped_unaffected_pct=35.9,
            )
        )
        assert packaged == 60.7
        assert unaffected == 35.9
        assert unknown is None
        assert piped_eff is None
        assert note is None

    def test_all_unknown_values_untouched(self):
        packaged, delivered, borehole, unaffected, unknown, piped_eff, note = (
            reconcile_categories_over_100_pct(piped_pct=100.0)
        )
        assert packaged is None
        assert delivered is None
        assert borehole is None
        assert unaffected is None
        assert unknown is None
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
            for cat in (
                "Packaged",
                "Delivered",
                "Borehole",
                "UnpipedUnaffected",
                "UnpipedUnknown",
            ):
                row[f"{stratum}{cat}"] = values.get(cat, pd.NA)
            row[f"{stratum}PipedEffective"] = (piped_effective or {}).get(
                stratum, pd.NA
            )
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

        unified = pd.DataFrame([self._row("USA", rural={"UnpipedUnaffected": 43.7})])
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
                    urban={"Borehole": 10.0, "UnpipedUnaffected": 5.0},
                    rural={"Borehole": 20.0, "UnpipedUnaffected": 10.0},
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

    ROU_2012_EUSILC / BIH_2015_HBS carry nothing but the UnpipedUnaffected catch-all;
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
                "URBANUnpipedUnaffected",
                "RURALUnpipedUnaffected",
            )
        }
        row.update({k: v for k, v in kwargs.items() if k not in row})
        return pd.Series(row)

    def test_catch_all_only_row_is_not_informative(self):
        # ROU_2012_EUSILC as shipped: 10 urban / 65 rural UnpipedUnaffected, nothing else
        assert not jmp_row_is_informative(
            self._row(URBANUnpipedUnaffected=10.0, RURALUnpipedUnaffected=65.0)
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
                    "RURALUnpipedUnaffected": 65.0,
                }
            )
        )

    def test_plain_dict_row_works_too(self):
        assert jmp_row_is_informative({"URBANBorehole": 5.0})
        assert not jmp_row_is_informative({"URBANUnpipedUnaffected": 5.0})


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
