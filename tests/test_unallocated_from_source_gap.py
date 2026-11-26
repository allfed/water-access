"""Tests for the explicit Unallocated gap step.

Background: the six shipped source shares (piped + packaged + delivered +
borehole + unpiped-unaffected + unpiped-unknown) do not tile 100 for most
countries. ``apply_unallocated_from_source_gap`` records the leftover as
URBANUnallocated / RURALUnallocated.
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = REPO_ROOT / "scripts" / "Data_Manipulation_Scripts"
sys.path.insert(0, str(SCRIPT_DIR))

from imputation_script import (  # noqa: E402
    apply_unallocated_from_source_gap,
    reconcile_research_breakdown_with_piped,
    reconcile_research_rows_with_final_piped,
)


def source_row(alpha3, **kwargs):
    row = {"alpha3": alpha3}
    for stratum in ("URBAN", "RURAL"):
        for cat in (
            "Piped",
            "Packaged",
            "Delivered",
            "Borehole",
            "UnpipedUnaffected",
            "UnpipedUnknown",
        ):
            row[f"{stratum}{cat}"] = kwargs.get(f"{stratum}{cat}", 0.0)
    return row


class TestUnallocatedFromSourceGap:
    def test_gap_becomes_unallocated_without_changing_categories(self, tmp_path):
        df = pd.DataFrame(
            [
                source_row(
                    "KEN",
                    URBANPiped=60.0,
                    URBANBorehole=10.0,
                    URBANUnpipedUnaffected=5.0,
                    RURALPiped=30.0,
                    RURALBorehole=20.0,
                )
            ]
        )
        out = apply_unallocated_from_source_gap(df, report_path=tmp_path / "report.csv")
        assert out.at[0, "URBANUnpipedUnaffected"] == pytest.approx(5.0)
        assert out.at[0, "URBANUnallocated"] == pytest.approx(25.0)
        assert out.at[0, "RURALUnpipedUnaffected"] == pytest.approx(0.0)
        assert out.at[0, "RURALUnallocated"] == pytest.approx(50.0)

    def test_complete_row_is_untouched(self, tmp_path):
        df = pd.DataFrame(
            [
                source_row(
                    "DEU",
                    URBANPiped=99.0,
                    URBANBorehole=1.0,
                    RURALPiped=98.0,
                    RURALBorehole=2.0,
                )
            ]
        )
        out = apply_unallocated_from_source_gap(df, report_path=tmp_path / "report.csv")
        assert out.at[0, "URBANBorehole"] == 1.0
        assert out.at[0, "URBANUnpipedUnaffected"] == 0.0
        assert out.at[0, "URBANUnallocated"] == 0.0
        assert out.at[0, "RURALUnallocated"] == 0.0

    def test_unknown_country_still_gets_unallocated(self, tmp_path):
        df = pd.DataFrame([source_row("XXX", URBANPiped=40.0, RURALPiped=40.0)])
        out = apply_unallocated_from_source_gap(df, report_path=tmp_path / "report.csv")
        assert out.at[0, "URBANUnpipedUnaffected"] == 0.0
        assert out.at[0, "URBANUnallocated"] == pytest.approx(60.0)

    def test_report_columns_and_rows(self, tmp_path):
        df = pd.DataFrame(
            [
                source_row("KEN", URBANPiped=60.0, RURALPiped=30.0),
                source_row("DEU", URBANPiped=100.0, RURALPiped=100.0),
            ]
        )
        apply_unallocated_from_source_gap(df, report_path=tmp_path / "report.csv")
        report = pd.read_csv(tmp_path / "report.csv")
        assert len(report) == 4
        assert list(report.columns) == ["alpha3", "stratum", "gap", "unallocated"]

    def test_over_100_row_raises(self, tmp_path):
        df = pd.DataFrame(
            [
                source_row(
                    "KEN", URBANPiped=50.0, RURALPiped=95.0, RURALUnpipedUnaffected=43.7
                )
            ]
        )
        with pytest.raises(ValueError, match="sum past"):
            apply_unallocated_from_source_gap(df, report_path=tmp_path / "report.csv")

    def test_out_of_range_category_raises(self, tmp_path):
        df = pd.DataFrame([source_row("KEN", URBANPiped=-5.0, RURALPiped=10.0)])
        with pytest.raises(ValueError, match=r"outside \[0, 100\]"):
            apply_unallocated_from_source_gap(df, report_path=tmp_path / "report.csv")

    def test_no_nans_in_new_columns(self, tmp_path):
        df = pd.DataFrame(
            [
                source_row("KEN", URBANPiped=60.0, RURALPiped=30.0),
                source_row("XXX"),
            ]
        )
        out = apply_unallocated_from_source_gap(df, report_path=tmp_path / "report.csv")
        assert out["URBANUnallocated"].notna().all()
        assert out["RURALUnallocated"].notna().all()


class TestReconciliationCoversJmpRows:
    """Both reconciliation passes now sweep data_origin == 'jmp' as well."""

    @staticmethod
    def _frame(origin, **kwargs):
        row = source_row("AAA", **kwargs)
        row["data_origin"] = origin
        return pd.DataFrame([row])

    def test_prenormalised_jmp_row_is_unchanged(self):
        df = self._frame(
            "jmp",
            URBANPiped=60.0,
            URBANBorehole=25.0,
            URBANUnpipedUnaffected=10.0,
            RURALPiped=30.0,
            RURALBorehole=40.0,
            RURALUnpipedUnaffected=20.0,
        )
        before = df.copy()
        after = reconcile_research_breakdown_with_piped(df)
        pd.testing.assert_frame_equal(after, before)

    def test_over_budget_jmp_row_is_scaled_down(self):
        df = self._frame(
            "jmp",
            URBANPiped=50.0,
            URBANBorehole=60.0,
            URBANUnpipedUnaffected=40.0,
            RURALPiped=50.0,
        )
        after = reconcile_research_breakdown_with_piped(df)
        assert after.at[0, "URBANBorehole"] == pytest.approx(30.0)
        assert after.at[0, "URBANUnpipedUnaffected"] == pytest.approx(20.0)
        assert after.at[0, "URBANPiped"] == 50.0

    def test_usa_rural_like_row_is_fixed_by_the_final_pass(self):
        df = self._frame(
            "jmp", URBANPiped=100.0, RURALPiped=95.0, RURALUnpipedUnaffected=43.7
        )
        track = pd.DataFrame(
            {"alpha3": ["AAA"], "URBANPiped": [pd.NA], "RURALPiped": [pd.NA]}
        )
        after = reconcile_research_rows_with_final_piped(df, track)
        assert after.at[0, "RURALUnpipedUnaffected"] == pytest.approx(5.0)
        assert after.at[0, "RURALPiped"] == 95.0
        total = (
            after.at[0, "RURALPiped"]
            + after.at[0, "RURALPackaged"]
            + after.at[0, "RURALDelivered"]
            + after.at[0, "RURALBorehole"]
            + after.at[0, "RURALUnpipedUnaffected"]
        )
        assert total <= 100.5

    def test_jmp_row_with_interpolated_piped_takes_the_piped_lowering_branch(self):
        # Lebanon case: JMP survey categories are real data, the piped share was
        # neighbour-interpolated. Interpolation sits below survey data in the
        # hierarchy, so piped gives way and the categories are kept intact.
        df = self._frame("jmp", URBANPiped=80.0, URBANBorehole=60.0, RURALPiped=80.0)
        track = pd.DataFrame(
            {
                "alpha3": ["AAA"],
                "URBANPiped": ["neighbor interpolated"],
                "RURALPiped": ["neighbor interpolated"],
            }
        )
        after = reconcile_research_rows_with_final_piped(df, track)
        assert after.at[0, "URBANPiped"] == pytest.approx(40.0)
        assert after.at[0, "URBANBorehole"] == 60.0
        # RURAL had no categories to conflict with, so its piped is untouched
        assert after.at[0, "RURALPiped"] == 80.0

    def test_jmp_row_with_measured_piped_scales_categories_down(self):
        df = self._frame("jmp", URBANPiped=80.0, URBANBorehole=60.0, RURALPiped=80.0)
        track = pd.DataFrame(
            {"alpha3": ["AAA"], "URBANPiped": [pd.NA], "RURALPiped": [pd.NA]}
        )
        after = reconcile_research_rows_with_final_piped(df, track)
        assert after.at[0, "URBANPiped"] == 80.0
        assert after.at[0, "URBANBorehole"] == pytest.approx(20.0)

    def test_jmp_row_absent_from_track_scales_categories_down(self):
        df = self._frame("jmp", URBANPiped=80.0, URBANBorehole=60.0, RURALPiped=80.0)
        track = pd.DataFrame(
            {
                "alpha3": ["ZZZ"],
                "URBANPiped": ["neighbor interpolated"],
                "RURALPiped": [pd.NA],
            }
        )
        after = reconcile_research_rows_with_final_piped(df, track)
        assert after.at[0, "URBANPiped"] == 80.0
        assert after.at[0, "URBANBorehole"] == pytest.approx(20.0)

    def test_research_row_still_takes_the_piped_lowering_branch(self):
        df = self._frame(
            "research", URBANPiped=82.0, URBANBorehole=69.0, RURALPiped=82.0
        )
        track = pd.DataFrame(
            {"alpha3": ["AAA"], "URBANPiped": [82.0], "RURALPiped": [82.0]}
        )
        after = reconcile_research_rows_with_final_piped(df, track)
        assert after.at[0, "URBANPiped"] == pytest.approx(31.0)
        assert after.at[0, "URBANBorehole"] == 69.0
