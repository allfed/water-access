"""Tests for the explicit-Unallocated / JMP-ladder fill step and the
reconciliation pass now that it also sweeps JMP-origin rows.

Background: the five shipped source shares (piped + packaged + delivered +
borehole + other-unpiped) do not tile 100 for most countries. Downstream that
gap used to vanish into a silent residual, implicitly classifying ~348M people
as resilient. `apply_ladder_fill_and_unallocated` splits the gap into the part
JMP itself attributes to unimproved/surface sources (moved into OtherUnpiped)
and the part that is genuinely unknown (URBANUnallocated / RURALUnallocated).
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = REPO_ROOT / "scripts" / "Data_Manipulation_Scripts"
sys.path.insert(0, str(SCRIPT_DIR))

from imputation_script import (  # noqa: E402
    apply_ladder_fill_and_unallocated,
    load_who_ladder_shares,
    reconcile_research_breakdown_with_piped,
    reconcile_research_rows_with_final_piped,
)

WATER_CSV = REPO_ROOT / "data" / "original_data" / "WHO Household Water Data - 2023 Data.csv"


def write_water_csv(tmp_path, rows):
    path = tmp_path / "water.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def water_row(country, year, urban_unimproved, urban_surface, rural_unimproved, rural_surface):
    return {
        "Country": country,
        "Year": year,
        "URBANUnimproved": urban_unimproved,
        "URBANSurface water": urban_surface,
        "RURALUnimproved": rural_unimproved,
        "RURALSurface water": rural_surface,
    }


def source_row(alpha3, **kwargs):
    row = {"alpha3": alpha3}
    for stratum in ("URBAN", "RURAL"):
        for cat in ("Piped", "Packaged", "Delivered", "Borehole", "OtherUnpiped"):
            row[f"{stratum}{cat}"] = kwargs.get(f"{stratum}{cat}", 0.0)
    return row


class TestLoadWhoLadderShares:
    def test_latest_year_with_data_per_column(self, tmp_path):
        path = write_water_csv(
            tmp_path,
            [
                water_row("Kenya", 2015, 5, 2, 10, 8),
                water_row("Kenya", 2022, 3, "-", 9, 6),
            ],
        )
        shares = load_who_ladder_shares(path)
        assert shares["KEN"]["URBANUnimproved"] == 3.0
        # 2022 urban surface is blank -> falls back to the 2015 value
        assert shares["KEN"]["URBANSurface water"] == 2.0
        assert shares["KEN"]["RURALUnimproved"] == 9.0

    def test_blank_column_for_a_present_country_reads_as_zero(self, tmp_path):
        path = write_water_csv(tmp_path, [water_row("Kenya", 2022, "-", "-", 9, 6)])
        shares = load_who_ladder_shares(path)
        assert shares["KEN"]["URBANUnimproved"] == 0.0
        assert shares["KEN"]["URBANSurface water"] == 0.0

    def test_less_than_one_reads_as_a_half_percent(self, tmp_path):
        path = write_water_csv(tmp_path, [water_row("Kenya", 2022, "<1", 0, 9, 6)])
        shares = load_who_ladder_shares(path)
        assert shares["KEN"]["URBANUnimproved"] == 0.5

    def test_real_csv_loads(self):
        if not WATER_CSV.exists():
            pytest.skip("WHO water CSV not available")
        shares = load_who_ladder_shares(WATER_CSV)
        assert len(shares) > 200
        # Niger must be its own country, not overwritten by Nigeria.
        assert "NER" in shares and "NGA" in shares


class TestLadderFill:
    @pytest.fixture
    def water_csv(self, tmp_path):
        return write_water_csv(
            tmp_path,
            [
                # Kenya: urban ladder 8pp available, rural 20pp
                water_row("Kenya", 2022, 5, 3, 12, 8),
                # Germany: no unimproved/surface at all
                water_row("Germany", 2022, 0, 0, 0, 0),
            ],
        )

    def test_gap_is_split_into_fill_and_unallocated(self, water_csv, tmp_path):
        df = pd.DataFrame(
            [
                source_row(
                    "KEN",
                    URBANPiped=60.0,
                    URBANBorehole=10.0,
                    URBANOtherUnpiped=5.0,
                    RURALPiped=30.0,
                    RURALBorehole=20.0,
                )
            ]
        )
        out = apply_ladder_fill_and_unallocated(
            df, water_csv, report_path=tmp_path / "report.csv"
        )
        # urban gap 25, ladder 8 -> OtherUnpiped 5 + 8 = 13, unallocated 17
        assert out.at[0, "URBANOtherUnpiped"] == pytest.approx(13.0)
        assert out.at[0, "URBANUnallocated"] == pytest.approx(17.0)
        # rural gap 50, ladder 20 -> OtherUnpiped 20, unallocated 30
        assert out.at[0, "RURALOtherUnpiped"] == pytest.approx(20.0)
        assert out.at[0, "RURALUnallocated"] == pytest.approx(30.0)

    def test_fill_is_capped_by_the_gap(self, water_csv, tmp_path):
        df = pd.DataFrame(
            [source_row("KEN", URBANPiped=97.0, RURALPiped=97.0)]
        )
        out = apply_ladder_fill_and_unallocated(
            df, water_csv, report_path=tmp_path / "report.csv"
        )
        # urban gap is 3 but the ladder offers 8 -> fill capped at 3
        assert out.at[0, "URBANOtherUnpiped"] == pytest.approx(3.0)
        assert out.at[0, "URBANUnallocated"] == pytest.approx(0.0)

    def test_complete_row_is_untouched(self, water_csv, tmp_path):
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
        out = apply_ladder_fill_and_unallocated(
            df, water_csv, report_path=tmp_path / "report.csv"
        )
        assert out.at[0, "URBANBorehole"] == 1.0
        assert out.at[0, "URBANOtherUnpiped"] == 0.0
        assert out.at[0, "URBANUnallocated"] == 0.0
        assert out.at[0, "RURALUnallocated"] == 0.0

    def test_country_absent_from_who_gets_no_fill_but_still_gets_columns(
        self, water_csv, tmp_path
    ):
        df = pd.DataFrame([source_row("XXX", URBANPiped=40.0, RURALPiped=40.0)])
        out = apply_ladder_fill_and_unallocated(
            df, water_csv, report_path=tmp_path / "report.csv"
        )
        assert out.at[0, "URBANOtherUnpiped"] == 0.0
        assert out.at[0, "URBANUnallocated"] == pytest.approx(60.0)
        report = pd.read_csv(tmp_path / "report.csv")
        assert not report.loc[report["alpha3"] == "XXX", "who_ladder_row"].iloc[0]

    def test_report_columns_and_rows(self, water_csv, tmp_path):
        df = pd.DataFrame(
            [
                source_row("KEN", URBANPiped=60.0, RURALPiped=30.0),
                source_row("DEU", URBANPiped=100.0, RURALPiped=100.0),
            ]
        )
        apply_ladder_fill_and_unallocated(
            df, water_csv, report_path=tmp_path / "report.csv"
        )
        report = pd.read_csv(tmp_path / "report.csv")
        assert len(report) == 4
        for col in ("alpha3", "stratum", "gap", "fill", "unallocated"):
            assert col in report.columns

    def test_over_100_row_raises(self, water_csv, tmp_path):
        # USA-rural-like overshoot: piped 95 + other 43.7. The reconciliation
        # pass is supposed to have removed these; if one survives, the ladder
        # step must refuse to write rather than ship an impossible row.
        df = pd.DataFrame(
            [source_row("KEN", URBANPiped=50.0, RURALPiped=95.0, RURALOtherUnpiped=43.7)]
        )
        with pytest.raises(ValueError, match="sum past"):
            apply_ladder_fill_and_unallocated(
                df, water_csv, report_path=tmp_path / "report.csv"
            )

    def test_out_of_range_category_raises(self, water_csv, tmp_path):
        df = pd.DataFrame([source_row("KEN", URBANPiped=-5.0, RURALPiped=10.0)])
        with pytest.raises(ValueError, match=r"outside \[0, 100\]"):
            apply_ladder_fill_and_unallocated(
                df, water_csv, report_path=tmp_path / "report.csv"
            )

    def test_no_nans_in_new_columns_for_the_real_frame(self, water_csv, tmp_path):
        df = pd.DataFrame(
            [
                source_row("KEN", URBANPiped=60.0, RURALPiped=30.0),
                source_row("XXX"),
            ]
        )
        out = apply_ladder_fill_and_unallocated(
            df, water_csv, report_path=tmp_path / "report.csv"
        )
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
            URBANOtherUnpiped=10.0,
            RURALPiped=30.0,
            RURALBorehole=40.0,
            RURALOtherUnpiped=20.0,
        )
        before = df.copy()
        after = reconcile_research_breakdown_with_piped(df)
        pd.testing.assert_frame_equal(after, before)

    def test_over_budget_jmp_row_is_scaled_down(self):
        df = self._frame(
            "jmp",
            URBANPiped=50.0,
            URBANBorehole=60.0,
            URBANOtherUnpiped=40.0,
            RURALPiped=50.0,
        )
        after = reconcile_research_breakdown_with_piped(df)
        assert after.at[0, "URBANBorehole"] == pytest.approx(30.0)
        assert after.at[0, "URBANOtherUnpiped"] == pytest.approx(20.0)
        assert after.at[0, "URBANPiped"] == 50.0

    def test_usa_rural_like_row_is_fixed_by_the_final_pass(self):
        df = self._frame("jmp", URBANPiped=100.0, RURALPiped=95.0, RURALOtherUnpiped=43.7)
        track = pd.DataFrame({"alpha3": ["AAA"], "URBANPiped": [pd.NA], "RURALPiped": [pd.NA]})
        after = reconcile_research_rows_with_final_piped(df, track)
        assert after.at[0, "RURALOtherUnpiped"] == pytest.approx(5.0)
        assert after.at[0, "RURALPiped"] == 95.0
        total = (
            after.at[0, "RURALPiped"]
            + after.at[0, "RURALPackaged"]
            + after.at[0, "RURALDelivered"]
            + after.at[0, "RURALBorehole"]
            + after.at[0, "RURALOtherUnpiped"]
        )
        assert total <= 100.5

    def test_jmp_row_never_takes_the_piped_lowering_branch(self):
        # An imputed piped share on a JMP row must not be overwritten by the
        # survey categories (that branch is for research rows only).
        df = self._frame("jmp", URBANPiped=80.0, URBANBorehole=60.0, RURALPiped=80.0)
        track = pd.DataFrame({"alpha3": ["AAA"], "URBANPiped": [80.0], "RURALPiped": [80.0]})
        after = reconcile_research_rows_with_final_piped(df, track)
        assert after.at[0, "URBANPiped"] == 80.0
        assert after.at[0, "URBANBorehole"] == pytest.approx(20.0)

    def test_research_row_still_takes_the_piped_lowering_branch(self):
        df = self._frame("research", URBANPiped=82.0, URBANBorehole=69.0, RURALPiped=82.0)
        track = pd.DataFrame({"alpha3": ["AAA"], "URBANPiped": [82.0], "RURALPiped": [82.0]})
        after = reconcile_research_rows_with_final_piped(df, track)
        assert after.at[0, "URBANPiped"] == pytest.approx(31.0)
        assert after.at[0, "URBANBorehole"] == 69.0
