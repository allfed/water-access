# flake8: noqa: E501
"""Chunk-planning behaviour that the cos-latitude population weighting touches.

The full chunked-vs-monolithic identity proof lives in
``scripts/validate_chunking.py`` (it needs the real grid). These are the cheap
guards around the coupling that weighting introduces: the planner's pre-pass has
to normalise with the *same* weight the run does, and it needs the latitude
columns to do it.
"""

import numpy as np
import pandas as pd
import pytest

import src.chunked_analysis as chunked
import src.gis_global_module as gis


def _zone_frame():
    """Two countries, four cells, spanning a wide latitude range."""
    return pd.DataFrame(
        {
            "ISOCODE": ["RUS", "RUS", "KEN", "KEN"],
            "shapeID": ["RUS-1", "RUS-2", "KEN-1", "KEN-2"],
            "pop_density": np.float32([100.0, 100.0, 100.0, 100.0]),
            "slope_1": np.float32([1.0, 2.0, 3.0, 4.0]),
            "left": np.float32([0.0, 0.0, 0.0, 0.0]),
            "right": np.float32([5 / 60, 5 / 60, 5 / 60, 5 / 60]),
            "top": np.float32([60.5, 50.5, 0.5, 1.5]),
            "bottom": np.float32([59.5, 49.5, -0.5, 0.5]),
        }
    )


def _country_table():
    return pd.DataFrame(
        {
            "alpha3": ["RUS", "KEN"],
            "Population": [1_000_000, 1_000_000],
            "Average Weight": [70.0, 65.0],
        }
    )


def _write(tmp_path, frame, name="zones.parquet"):
    path = tmp_path / name
    frame.to_parquet(path, index=False)
    return path


class TestPlanChunksAreaWeighting:
    def test_plan_records_the_flag(self, tmp_path, monkeypatch):
        country_file = tmp_path / "countries.csv"
        _country_table().to_csv(country_file, index=False)
        zone_file = _write(tmp_path, _zone_frame())

        default = chunked.plan_chunks(zone_file, country_file=country_file)
        off = chunked.plan_chunks(
            zone_file, country_file=country_file, area_weighted_population=False
        )
        on = chunked.plan_chunks(
            zone_file, country_file=country_file, area_weighted_population=True
        )
        # Area weighting is the model default, so a plan built without the
        # kwarg must be usable by a default run.
        assert default.area_weighted_population is True
        assert off.area_weighted_population is False
        assert on.area_weighted_population is True

    def test_density_totals_match_the_run_bit_for_bit(self, tmp_path):
        """The pre-pass divisor must equal the one the run derives itself.

        A last-bit difference here is exactly what would make a split country's
        chunked result diverge from the monolithic one.
        """
        country_file = tmp_path / "countries.csv"
        countries = _country_table()
        countries.to_csv(country_file, index=False)
        frame = _zone_frame()
        zone_file = _write(tmp_path, frame)

        plan = chunked.plan_chunks(
            zone_file, country_file=country_file, area_weighted_population=True
        )
        merged = gis.optimise_zone_dtypes(frame.copy()).merge(
            countries, left_on="ISOCODE", right_on="alpha3", how="inner"
        )
        weights = gis.population_split_weights(merged, area_weighted_population=True)
        expected = weights.groupby(merged["ISOCODE"], observed=True).sum()
        assert set(plan.density_totals) == {str(k) for k in expected.index}
        for isocode, total in zip(expected.index, expected.to_numpy()):
            planned = plan.density_totals[str(isocode)]
            # float32 in, float32 out: promoting the divisor would promote every
            # share and stop a chunked run matching a whole-world one.
            assert planned.dtype == np.float32
            assert planned == total


class TestChunkColumns:
    def test_latitude_columns_return_only_when_area_weighted(self, tmp_path):
        zone_file = _write(tmp_path, _zone_frame())
        off = chunked._chunk_columns(zone_file, area_weighted_population=False)
        on = chunked._chunk_columns(zone_file, area_weighted_population=True)
        default = chunked._chunk_columns(zone_file)
        assert set(off).isdisjoint(chunked.COORDINATE_COLUMNS)
        assert {"top", "bottom"}.issubset(on)
        # left/right stay dropped either way -- only the latitude bounds are
        # needed, and they are half the coordinate footprint.
        assert {"left", "right"}.isdisjoint(on)
        assert default == on


class TestPlanFlagMustMatchRun:
    def test_mismatched_plan_is_rejected(self, tmp_path, monkeypatch):
        country_file = tmp_path / "countries.csv"
        _country_table().to_csv(country_file, index=False)
        zone_file = _write(tmp_path, _zone_frame())
        monkeypatch.setattr(gis, "COUNTRY_DATA_FILE", country_file)

        plan = chunked.plan_chunks(
            zone_file, country_file=country_file, area_weighted_population=False
        )
        with pytest.raises(ValueError, match="rebuild the plan"):
            chunked.run_chunked_global_analysis(
                crr_adjustment=0,
                time_gathering_water=2.84,
                practical_limit_bicycle=40,
                practical_limit_buckets=20,
                met=4,
                watts=40,
                hill_polarity="downhill_uphill",
                urban_adjustment=1.3,
                rural_adjustment=1.4,
                zone_file=zone_file,
                plan=plan,
                area_weighted_population=True,
            )
