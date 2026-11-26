# flake8: noqa: E501
"""Chunk-planning behaviour that the cos-latitude population weighting touches.

The planner's pre-pass has to normalise with the *same* weight the run does,
and it needs the latitude columns to do it.
"""

import numpy as np
import pandas as pd

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
            "right": np.float32([30 / 3600, 30 / 3600, 30 / 3600, 30 / 3600]),
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
    def test_density_totals_match_the_run_bit_for_bit(self, tmp_path):
        """The pre-pass divisor must equal the one the run derives itself.

        A last-bit difference here is exactly what would make a split country's
        chunked result diverge from a whole-grid one.
        """
        country_file = tmp_path / "countries.csv"
        countries = _country_table()
        countries.to_csv(country_file, index=False)
        frame = _zone_frame()
        zone_file = _write(tmp_path, frame)

        plan = chunked.plan_chunks(zone_file, country_file=country_file)
        merged = gis.optimise_zone_dtypes(frame.copy()).merge(
            countries, left_on="ISOCODE", right_on="alpha3", how="inner"
        )
        weights = gis.population_split_weights(merged)
        expected = weights.groupby(merged["ISOCODE"], observed=True).sum()
        assert set(plan.density_totals) == {str(k) for k in expected.index}
        for isocode, total in zip(expected.index, expected.to_numpy()):
            planned = plan.density_totals[str(isocode)]
            # float32 in, float32 out: promoting the divisor would promote every
            # share and stop a chunked run matching a whole-grid one.
            assert planned.dtype == np.float32
            assert planned == total


class TestChunkColumns:
    def test_only_latitude_bounds_are_read(self, tmp_path):
        zone_file = _write(tmp_path, _zone_frame())
        columns = chunked._chunk_columns(zone_file)
        # Only the latitude bounds are needed, and they are half the
        # coordinate footprint.
        assert {"top", "bottom"}.issubset(columns)
        assert {"left", "right"}.isdisjoint(columns)
