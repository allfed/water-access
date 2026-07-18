import numpy as np
import pandas as pd
import pytest
from scipy.optimize import fsolve
from src.mobility_module import (
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

        # Sanity: the fix actually makes speed slope-sensitive (not near-flat).
        assert loaded_velocities[-1] < loaded_velocities[0]

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
