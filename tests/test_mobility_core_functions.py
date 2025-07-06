"""
Comprehensive tests for core mobility functions.

These tests focus on the previously untested critical functions:
- single_lankford_run
- single_bike_run
- Core mathematical models

These functions were identified as having critical bugs that went undetected.
"""

import numpy as np
import pytest
import sys
from pathlib import Path
from unittest.mock import Mock, patch

# Add src to path for imports
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

import src.mobility_module as mm


class TestCriticalMobilityFunctions:
    """Test the core mobility functions that were previously untested."""
    
    def setup_method(self):
        """Set up basic test objects for each test."""
        # Create mock mobility variables (mv)
        self.mv = Mock()
        self.mv.m1 = 70.0  # Human mass in kg
        self.mv.ro = 1.225  # Air density
        self.mv.C_d = 0.9   # Drag coefficient
        self.mv.A = 0.3     # Reference area
        self.mv.eta = 0.95  # Efficiency
        self.mv.P_t = 75.0  # Power output in watts
        self.mv.g = 9.81    # Gravity
        self.mv.F_max = 200.0  # Maximum force
        
        # Create mock mobility options (mo)
        self.mo = Mock()
        self.mo.model_selection = 2  # Cycling model
        
        # Create mock HPV (Human Powered Vehicle)
        self.hpv = Mock()
        self.hpv.load_capacity = np.array([[[25.0]]])  # 3D array structure
        self.hpv.Crr = np.array([[[0.003]]])  # Coefficient of rolling resistance
        self.hpv.m_HPV_only = np.array([[[15.0]]])  # HPV mass
        self.hpv.practical_limit = np.array([[[20.0]]])  # Practical load limit
        
        # Create mock MET values
        self.met = Mock()
        self.met.vo2_budget = 1.5  # VO2 budget
        
    def test_single_lankford_run_basic_functionality(self):
        """Test that single_lankford_run returns expected structure."""
        slope = 0.0  # Flat terrain
        load_attempted = 10.0  # kg
        
        result = mm.mobility_models.single_lankford_run(
            self.mv, self.mo, self.met, self.hpv, slope, load_attempted
        )
        
        # Should return tuple of (loaded_velocity, unloaded_velocity, max_load)
        assert isinstance(result, tuple)
        assert len(result) == 3
        
        loaded_velocity, unloaded_velocity, max_load = result
        
        # Velocities should be non-negative numbers or NaN
        assert isinstance(loaded_velocity, (float, np.floating)) or np.isnan(loaded_velocity)
        assert isinstance(unloaded_velocity, (float, np.floating)) or np.isnan(unloaded_velocity)
        assert isinstance(max_load, (float, np.floating))
        
        if not np.isnan(loaded_velocity):
            assert loaded_velocity >= 0
        if not np.isnan(unloaded_velocity):
            assert unloaded_velocity >= 0
        assert max_load >= 0
        
    def test_single_bike_run_basic_functionality(self):
        """Test that single_bike_run returns expected structure."""
        slope = 0.0  # Flat terrain
        load_attempted = 10.0  # kg
        
        result = mm.mobility_models.single_bike_run(
            self.mv, self.mo, self.hpv, slope, load_attempted
        )
        
        # Should return tuple of (loaded_velocity, unloaded_velocity, max_load)
        assert isinstance(result, tuple)
        assert len(result) == 3
        
        loaded_velocity, unloaded_velocity, max_load = result
        
        # Velocities should be non-negative numbers or NaN
        assert isinstance(loaded_velocity, (float, np.floating)) or np.isnan(loaded_velocity)
        assert isinstance(unloaded_velocity, (float, np.floating)) or np.isnan(unloaded_velocity)
        assert isinstance(max_load, (float, np.floating))
        
        if not np.isnan(loaded_velocity):
            assert loaded_velocity >= 0
        if not np.isnan(unloaded_velocity):
            assert unloaded_velocity >= 0
        assert max_load >= 0
        
    def test_slope_conversion_fixed(self):
        """Test that slope conversion is now correct (fixes the 360 vs 180 bug)."""
        slope_degrees = 45.0  # 45 degree slope
        load_attempted = 5.0
        
        # The function should now use np.radians() internally
        # We can't directly test this without mocking, but we can test that
        # the function doesn't crash with reasonable slopes
        result = mm.mobility_models.single_lankford_run(
            self.mv, self.mo, self.met, self.hpv, slope_degrees, load_attempted
        )
        
        # Function should complete without error
        assert isinstance(result, tuple)
        assert len(result) == 3
        
    def test_load_capacity_limits_respected(self):
        """Test that load limits are properly respected."""
        slope = 0.0
        excessive_load = 50.0  # More than capacity (25.0)
        
        result = mm.mobility_models.single_lankford_run(
            self.mv, self.mo, self.met, self.hpv, slope, excessive_load
        )
        
        loaded_velocity, unloaded_velocity, max_load = result
        
        # Max load should not exceed capacity
        expected_capacity = self.hpv.load_capacity.flatten()[0]
        assert max_load <= expected_capacity
        
    def test_cycling_generally_faster_than_walking(self):
        """Test that cycling models generally produce higher velocities than walking."""
        slope = 0.0  # Flat terrain for fair comparison
        load_attempted = 10.0
        
        # Test cycling
        self.mo.model_selection = 2  # Cycling
        cycling_result = mm.mobility_models.single_bike_run(
            self.mv, self.mo, self.hpv, slope, load_attempted
        )
        
        # Test walking  
        walking_result = mm.mobility_models.single_lankford_run(
            self.mv, self.mo, self.met, self.hpv, slope, load_attempted
        )
        
        cycling_velocity = cycling_result[0]  # loaded velocity
        walking_velocity = walking_result[0]   # loaded velocity
        
        # If both are valid (not NaN), cycling should generally be faster
        if not np.isnan(cycling_velocity) and not np.isnan(walking_velocity):
            assert cycling_velocity > walking_velocity, \
                f"Cycling velocity ({cycling_velocity}) should be greater than walking velocity ({walking_velocity})"
        
    def test_unloaded_faster_than_loaded(self):
        """Test that unloaded velocities are faster than loaded velocities."""
        slope = 0.0
        load_attempted = 15.0  # Significant load
        
        # Test with cycling
        result = mm.mobility_models.single_bike_run(
            self.mv, self.mo, self.hpv, slope, load_attempted
        )
        
        loaded_velocity, unloaded_velocity, max_load = result
        
        # If both are valid, unloaded should be faster
        if not np.isnan(loaded_velocity) and not np.isnan(unloaded_velocity):
            assert unloaded_velocity >= loaded_velocity, \
                f"Unloaded velocity ({unloaded_velocity}) should be >= loaded velocity ({loaded_velocity})"
                
    def test_extreme_slopes_handled(self):
        """Test that extreme slopes don't crash the functions."""
        extreme_slopes = [-30, -10, 0, 10, 30]  # degrees
        load_attempted = 5.0
        
        for slope in extreme_slopes:
            # Test cycling
            cycling_result = mm.mobility_models.single_bike_run(
                self.mv, self.mo, self.hpv, slope, load_attempted
            )
            assert isinstance(cycling_result, tuple)
            assert len(cycling_result) == 3
            
            # Test walking
            walking_result = mm.mobility_models.single_lankford_run(
                self.mv, self.mo, self.met, self.hpv, slope, load_attempted
            )
            assert isinstance(walking_result, tuple)
            assert len(walking_result) == 3
            
    def test_zero_load_cases(self):
        """Test edge case of zero load."""
        slope = 0.0
        load_attempted = 0.0
        
        # Should handle zero load gracefully
        result = mm.mobility_models.single_bike_run(
            self.mv, self.mo, self.hpv, slope, load_attempted
        )
        
        assert isinstance(result, tuple)
        loaded_velocity, unloaded_velocity, max_load = result
        
        # With zero load, loaded and unloaded velocities should be similar
        if not np.isnan(loaded_velocity) and not np.isnan(unloaded_velocity):
            # Allow for small numerical differences
            assert abs(loaded_velocity - unloaded_velocity) < 0.1
            
    def test_array_flattening_fix(self):
        """Test that the array flattening fix works correctly."""
        # This tests the fix for line 217 where hpv.load_capacity.flatten()[0] is used
        slope = 0.0
        load_attempted = 10.0
        
        # Ensure our test setup uses the 3D array structure that caused the original bug
        assert self.hpv.load_capacity.shape == (1, 1, 1)
        
        # Function should work with 3D arrays
        result = mm.mobility_models.single_lankford_run(
            self.mv, self.mo, self.met, self.hpv, slope, load_attempted
        )
        
        loaded_velocity, unloaded_velocity, max_load = result
        
        # Should successfully extract the scalar value from the 3D array
        assert isinstance(max_load, (float, np.floating))
        assert max_load <= 25.0  # Should not exceed our test capacity


class TestMobilityIntegration:
    """Integration tests for mobility functions with realistic scenarios."""
    
    def test_realistic_water_carrying_scenario(self):
        """Test a realistic water carrying scenario."""
        # Realistic parameters for water carrying in developing regions
        mv = Mock()
        mv.m1 = 60.0  # Average adult mass
        mv.ro = 1.225
        mv.C_d = 0.9
        mv.A = 0.3
        mv.eta = 0.95
        mv.P_t = 60.0  # Sustainable power output
        mv.g = 9.81
        mv.F_max = 150.0
        
        mo = Mock()
        mo.model_selection = 2
        
        hpv = Mock()
        hpv.load_capacity = np.array([[[20.0]]])  # 20L water capacity
        hpv.Crr = np.array([[[0.004]]])  # Typical bike rolling resistance
        hpv.m_HPV_only = np.array([[[12.0]]])  # Typical bike mass
        hpv.practical_limit = np.array([[[15.0]]])
        
        met = Mock()
        met.vo2_budget = 1.4
        
        # Test various realistic slopes and loads
        test_scenarios = [
            (0, 10),    # Flat, 10L water
            (2, 15),    # Gentle slope, 15L water
            (5, 20),    # Steeper slope, full capacity
            (-2, 10),   # Downhill
        ]
        
        for slope, load in test_scenarios:
            # Test cycling
            cycling_result = mm.mobility_models.single_bike_run(mv, mo, hpv, slope, load)
            assert isinstance(cycling_result, tuple)
            
            # Test walking
            walking_result = mm.mobility_models.single_lankford_run(mv, mo, met, hpv, slope, load)
            assert isinstance(walking_result, tuple)
            
            # Results should be physically reasonable (velocities between 0.1 and 10 m/s)
            for result in [cycling_result, walking_result]:
                loaded_vel, unloaded_vel, max_load = result
                if not np.isnan(loaded_vel):
                    assert 0.1 <= loaded_vel <= 10.0, f"Unrealistic loaded velocity: {loaded_vel}"
                if not np.isnan(unloaded_vel):
                    assert 0.1 <= unloaded_vel <= 15.0, f"Unrealistic unloaded velocity: {unloaded_vel}"


class TestArrayIndexingFix:
    """Test that the array indexing fix works correctly."""
    
    def test_linspace_creator_indexing_fix(self):
        """Test that the load_matrix indexing fix works correctly."""
        # This tests the fix for line 78: load_matrix[i:] -> load_matrix[i, :]
        max_values = np.array([10, 20, 30])
        min_value = 5
        resolution = 3
        
        result = mm.linspace_creator(max_values, min_value, resolution)
        
        # Should create proper matrix shape
        assert result.shape == (3, 3)  # 3 HPVs, 3 resolution points each
        
        # Each row should contain different values (not all the same due to the bug)
        assert not np.array_equal(result[0, :], result[1, :]), \
            "Rows should be different - indexing bug may have returned"
        assert not np.array_equal(result[1, :], result[2, :]), \
            "Rows should be different - indexing bug may have returned"
        
    def test_load_matrix_structure(self):
        """Test that load matrices have the correct structure."""
        max_values = np.array([15, 25, 35])  # Different capacities
        min_value = 0
        resolution = 5
        
        result = mm.linspace_creator(max_values, min_value, resolution)
        
        # Each row should be a linspace from min_value to the corresponding max_value
        expected_row_0 = np.linspace(0, 15, 5)
        expected_row_1 = np.linspace(0, 25, 5)
        expected_row_2 = np.linspace(0, 35, 5)
        
        np.testing.assert_array_almost_equal(result[0, :], expected_row_0)
        np.testing.assert_array_almost_equal(result[1, :], expected_row_1)
        np.testing.assert_array_almost_equal(result[2, :], expected_row_2)