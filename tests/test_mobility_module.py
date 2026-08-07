import warnings

import numpy as np
import pandas as pd
import pytest
from scipy.optimize import fsolve
from src.mobility_module import (
    harmonic_mean_velocity,
    linspace_creator,
    max_safe_load,
    mobility_models,
    mobility_models as MobilityModels,
    model_variables,
    model_options,
    HPV_variables,
    MET_values,
)


class TestLinspaceCreator:
    def test_returns_numpy_array(self):
        max_values = np.array([10, 20, 30])
        min_value = 5
        resolution = 5
        result = linspace_creator(max_values, min_value, resolution)
        assert isinstance(result, np.ndarray)

    def test_returns_correct_shape_for_res_1(self):
        max_values = np.array([10, 20, 30])
        min_value = 5
        resolution = 1
        result = linspace_creator(max_values, min_value, resolution)
        assert result.shape == (3,)

    def test_returns_correct_shape_for_res_0(self):
        max_values = np.array([10, 20, 30])
        min_value = 5
        resolution = 0
        result = linspace_creator(max_values, min_value, resolution)
        assert result.shape == (3, 1)

    def test_returns_correct_shape_for_res_greater_than_1(self):
        max_values = np.array([10, 20, 30])
        min_value = 5
        resolution = 5
        result = linspace_creator(max_values, min_value, resolution)
        assert result.shape == (3, 5)

    def test_returns_correct_values_for_res_1(self):
        max_values = np.array([10, 20, 30])
        min_value = 5
        resolution = 1
        result = linspace_creator(max_values, min_value, resolution)
        expected_result = np.array([10, 20, 30])
        assert np.array_equal(result, expected_result)

    def test_returns_correct_values_for_res_0(self):
        max_values = np.array([10, 20, 30])
        min_value = 5
        resolution = 0
        result = linspace_creator(max_values, min_value, resolution)
        expected_result = np.array([[5], [5], [5]])
        assert np.array_equal(result, expected_result)

    def test_returns_correct_values_for_res_greater_than_1(self):
        max_values = np.array([10, 20, 30])
        min_value = 5
        resolution = 5
        result = linspace_creator(max_values, min_value, resolution)
        expected_result = np.array(
            [
                [5.0, 6.25, 7.5, 8.75, 10.0],
                [5.0, 8.75, 12.5, 16.25, 20.0],
                [5.0, 11.25, 17.5, 23.75, 30.0],
            ]
        )
        assert np.array_equal(result, expected_result)

    def test_raises_error_for_negative_resolution(self):
        max_values = np.array([10, 20, 30])
        min_value = 5
        resolution = -1
        with pytest.raises(AssertionError):
            linspace_creator(max_values, min_value, resolution)


class TestMaxSafeLoad:
    def test_returns_numpy_array(self):
        m_HPV_only = np.array([50, 60, 70])
        LoadCapacity = np.array([100, 120, 150])
        F_max = 500
        s = 0.1
        g = 9.8
        result = max_safe_load(m_HPV_only, LoadCapacity, F_max, s, g)
        assert isinstance(result, np.ndarray)

    def test_returns_correct_shape(self):
        m_HPV_only = np.array([50, 60, 70])
        LoadCapacity = np.array([100, 120, 150])
        F_max = 500
        s = 0.1
        g = 9.8
        result = max_safe_load(m_HPV_only, LoadCapacity, F_max, s, g)
        assert result.shape == (3,)

    def test_returns_correct_values(self):
        m_HPV_only = np.array([50, 60, 70])
        LoadCapacity = np.array([100, 120, 150])
        F_max = 500
        s = 0.1
        g = 9.8
        result = max_safe_load(m_HPV_only, LoadCapacity, F_max, s, g)
        expected_result = np.array([100, 120, 150])
        assert np.allclose(result, expected_result)

    def test_handles_single_value(self):
        m_HPV_only = np.array([50])
        LoadCapacity = 100
        F_max = 500
        s = 0.1
        g = 9.8
        result = max_safe_load(m_HPV_only, LoadCapacity, F_max, s, g)
        expected_result = 461.055414879
        assert np.isclose(result, expected_result)

    def test_handles_zero_slope(self):
        m_HPV_only = np.array([50, 60, 70])
        LoadCapacity = np.array([100, 120, 150])
        F_max = 500
        s = 0
        g = 9.8
        result = max_safe_load(m_HPV_only, LoadCapacity, F_max, s, g)
        expected_result = np.array([100.0, 120.0, 150.0])
        assert np.allclose(result, expected_result)

    def test_handles_divide_by_zero(self):
        m_HPV_only = np.array([50, 60, 70])
        LoadCapacity = np.array([100, 120, 150])
        F_max = 0
        s = 0.1
        g = 9.8
        result = max_safe_load(m_HPV_only, LoadCapacity, F_max, s, g)
        expected_result = np.array([-50, -60, -70])
        assert np.allclose(result, expected_result)


def _build_walking_objects(met=4.5, human_mass=62, ulhillpo=0, lhillpo=1):
    """Construct the mv/mo/hpv/met objects the Lankford walking path needs.

    Mirrors gis_global_module.calculate_and_merge_walking_distance /
    run_walking_model: a single "Buckets" HPV, Lankford model selection, and
    country-specific MET budget. Default polarity is flat_uphill (loaded leg
    uphill) so slope has a clear, monotonic sign on the loaded velocity.
    """
    row = {
        "Name": "Buckets",
        "LoadLimit": 20,
        "PracticalLimit": 20,
        "AverageSpeedWithoutLoad": 1.2,
        "Drive": 0,
        "GroundContact": 0,
        "Pilot": 0,
        "Crr": 0,
        "Efficiency": 1,
        "Weight": 0.5,
    }
    param_df = pd.DataFrame([row])
    mv = model_variables(m1=human_mass)
    mo = model_options(ulhillpo=ulhillpo, lhillpo=lhillpo)
    mo.model_selection = 3  # Lankford model
    hpv = HPV_variables(param_df, mv)
    met_values = MET_values(
        mv,
        country_weight=human_mass,
        met=met,
        use_country_specific_weights=True,
    )
    return mv, mo, hpv, met_values


class TestHarmonicMeanVelocity:
    """Guards the zero-safe round-trip mean.

    The harmonic mean is correct for a round trip (equal distance per leg), but
    the naive 2/(1/a + 1/b) form divides by zero once the velocity floor makes
    a 0 m/s leg reachable. These pin both the equivalence on normal data and
    the zero handling.
    """

    def test_matches_naive_formula_on_positive_velocities(self):
        loaded = np.array([1.2, 0.8, 1.5, 2.0])
        unloaded = np.array([1.4, 1.1, 1.5, 0.5])
        naive = 2 / (1 / loaded + 1 / unloaded)
        assert np.allclose(harmonic_mean_velocity(loaded, unloaded), naive)

    def test_zero_leg_gives_zero_without_warning(self):
        loaded = np.array([0.0, 1.2, 0.0])
        unloaded = np.array([1.4, 0.0, 0.0])
        with warnings.catch_warnings():
            # An unreachable leg must not emit divide-by-zero warnings; on a
            # full run that would be millions of them.
            warnings.simplefilter("error", RuntimeWarning)
            result = harmonic_mean_velocity(loaded, unloaded)
        assert np.array_equal(result, np.zeros(3))

    def test_is_bounded_by_the_slower_leg(self):
        loaded = np.array([0.5, 1.0, 2.0])
        unloaded = np.array([2.0, 1.0, 0.5])
        result = harmonic_mean_velocity(loaded, unloaded)
        slower = np.minimum(loaded, unloaded)
        faster = np.maximum(loaded, unloaded)
        # Harmonic mean sits between the two legs, nearer the slower one.
        assert np.all(result >= slower - 1e-12)
        assert np.all(result <= faster + 1e-12)

    def test_preserves_shape_for_3d_input(self):
        loaded = np.full((2, 3, 4), 1.2)
        unloaded = np.full((2, 3, 4), 0.8)
        result = harmonic_mean_velocity(loaded, unloaded)
        assert result.shape == (2, 3, 4)
        assert np.allclose(result, 2 * 1.2 * 0.8 / (1.2 + 0.8))


class TestLankfordSolution:
    def test_loaded_velocity_non_increasing_with_slope(self):
        """Guards the coupled slope-units / cubic-sign / clipping fix (1.1-1.3):
        loaded walking velocity must be non-increasing as slope rises, and
        finite and non-negative at every point including the clipped steep end.
        """
        mv, mo, hpv, met_values = _build_walking_objects(
            met=4.5, human_mass=62, ulhillpo=0, lhillpo=1  # flat_uphill
        )
        slopes_deg = [0, 5, 10, 20]
        loaded_velocities = []
        for slope in slopes_deg:
            loaded_velocity, _, _ = MobilityModels.single_lankford_run(
                mv, mo, met_values, hpv, slope, load_attempted=15
            )
            loaded_velocities.append(loaded_velocity)

        # All finite and non-negative (including the clipped 20-degree case).
        assert all(np.isfinite(v) for v in loaded_velocities)
        assert all(v >= 0 for v in loaded_velocities)

        # Monotonically non-increasing with slope.
        for prev, curr in zip(loaded_velocities, loaded_velocities[1:]):
            assert curr <= prev + 1e-9

        # Pin the units/sign fix by MAGNITUDE, not just direction: with the
        # correct percentage-grade units, loaded speed at 20 deg falls to well
        # under half the flat value (measured ~0.20 vs ~1.12 m/s). The old
        # slope-blind code (degrees/45) produced only a ~4% drop, which a mere
        # "20deg < flat" check would still pass -- so require a real collapse.
        assert loaded_velocities[-1] < 0.5 * loaded_velocities[0]

    def test_velocities_non_negative_across_met_sweep(self):
        """Guards the velocity floor (1.5): across a MET sweep including a low
        value, both loaded and unloaded reachable velocities are finite and
        >= 0 (never NaN or negative).
        """
        for met in [2.0, 3.0, 4.5, 6.0]:
            mv, mo, hpv, met_values = _build_walking_objects(
                met=met, human_mass=62, ulhillpo=1, lhillpo=1  # uphill both legs
            )
            for slope in [0, 5, 10, 20]:
                (
                    loaded_velocity,
                    unloaded_velocity,
                    _,
                ) = MobilityModels.single_lankford_run(
                    mv, mo, met_values, hpv, slope, load_attempted=15
                )
                assert np.isfinite(loaded_velocity)
                assert np.isfinite(unloaded_velocity)
                assert loaded_velocity >= 0
                assert unloaded_velocity >= 0

    def test_solution_velocity_decreases_with_slope_direct(self):
        """Solve Lankford_solution directly via fsolve at several grades and
        confirm the root velocity decreases with slope (independent of the
        object-construction path).
        """
        mv, _, _, met_values = _build_walking_objects(met=4.5, human_mass=62)
        m_load = 62.0  # unloaded person only
        velocities = []
        for slope_deg in [0, 5, 10, 20]:
            s = (slope_deg / 360) * (2 * np.pi)
            v = fsolve(
                MobilityModels.Lankford_solution,
                1.0,
                args=(m_load, met_values, s),
                full_output=True,
            )
            velocities.append(v[0][0])

        assert all(np.isfinite(v) for v in velocities)
        for prev, curr in zip(velocities, velocities[1:]):
            assert curr <= prev + 1e-9
        assert velocities[-1] < velocities[0]


mobility_models = mobility_models()
sprott_model = mobility_models.sprott_model
bike_power_solution = mobility_models.bike_power_solution
sprott_solution = mobility_models.sprott_solution
