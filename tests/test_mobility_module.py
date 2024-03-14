import numpy as np
import pytest
from src.mobility_module import linspace_creator
import pdb
import sys
from pathlib import Path

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
        expected_result = np.array([[ 5.,  6.25,  7.5 ,  8.75, 10.],
                                    [ 5.,  8.75, 12.5 , 16.25, 20.],
                                    [ 5., 11.25, 17.5 , 23.75, 30.]])
        assert np.array_equal(result, expected_result)

    def test_raises_error_for_negative_resolution(self):
        max_values = np.array([10, 20, 30])
        min_value = 5
        resolution = -1
        with pytest.raises(AssertionError):
            linspace_creator(max_values, min_value, resolution)
            

from src.mobility_module import linspace_creator, max_safe_load

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
        expected_result = np.array([100., 120., 150.])
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


from src.mobility_module import mobility_models
import src.mobility_module as mm
mobility_models = mobility_models()
sprott_model = mobility_models.sprott_model
bike_power_solution = mobility_models.bike_power_solution
sprott_solution = mobility_models.sprott_solution
from scipy.optimize import fsolve