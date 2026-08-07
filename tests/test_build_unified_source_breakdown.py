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

from build_unified_source_breakdown import (  # noqa: E402
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
