
# Generated by CodiumAI
from src.gis_monte_carlo import sample_normal
from src.gis_monte_carlo import sample_lognormal
from src.gis_monte_carlo import run_simulation
from src.gis_monte_carlo import process_mc_results

import numpy as np
import pandas as pd
from pathlib import Path
import pytest
import os

class TestSampleNormal:

    # The function returns a numpy array of size n.
    def test_returns_array_of_size_n(self):
        low = 0
        high = 10
        n = 5
        result = sample_normal(low, high, n)
        assert isinstance(result, np.ndarray)
        assert len(result) == n

    # The function returns an empty numpy array when n is 0.
    def test_returns_empty_array_when_n_is_0(self):
        low = 0
        high = 10
        n = 0
        result = sample_normal(low, high, n)
        assert isinstance(result, np.ndarray)
        assert len(result) == 0

class TestSampleLognormal:

    def test_returns_array_of_size_n(self):
        low = 1
        high = 10
        n = 5
        result = sample_lognormal(low, high, n)
        assert isinstance(result, np.ndarray)
        assert len(result) == n

    def test_returns_empty_array_when_n_is_0(self):
        low = 1
        high = 10
        n = 0
        result = sample_lognormal(low, high, n)
        assert isinstance(result, np.ndarray)
        assert len(result) == 0

    def test_raises_assertion_error_when_low_is_less_than_0(self):
        low = -1
        high = 10
        n = 5
        with pytest.raises(AssertionError):
            sample_lognormal(low, high, n)
            
class TestRunSimulation:

    def test_valid_input_returns_result(self):
        crr_adjustment = 1
        time_gathering_water = 2.5
        practical_limit_bicycle = 10
        practical_limit_buckets = 5
        met = 1.2
        result = run_simulation(crr_adjustment, time_gathering_water, practical_limit_bicycle, practical_limit_buckets, met, calculate_distance=False)
        assert isinstance(result, pd.DataFrame)

    def test_invalid_crr_adjustment_raises_assertion_error(self):
        crr_adjustment = "0.5"
        time_gathering_water = 2.5
        practical_limit_bicycle = 10
        practical_limit_buckets = 5
        met = 1.2
        with pytest.raises(AssertionError):
            run_simulation(crr_adjustment, time_gathering_water, practical_limit_bicycle, practical_limit_buckets, met)

    def test_invalid_time_gathering_water_raises_assertion_error(self):
        crr_adjustment = 1
        time_gathering_water = "invalid"
        practical_limit_bicycle = 10
        practical_limit_buckets = 5
        met = 1.2
        with pytest.raises(AssertionError):
            run_simulation(crr_adjustment, time_gathering_water, practical_limit_bicycle, practical_limit_buckets, met)

    def test_invalid_practical_limit_bicycle_raises_assertion_error(self):
        crr_adjustment = 1
        time_gathering_water = 2.5
        practical_limit_bicycle = "invalid"
        practical_limit_buckets = 5
        met = 1.2
        with pytest.raises(AssertionError):
            run_simulation(crr_adjustment, time_gathering_water, practical_limit_bicycle, practical_limit_buckets, met)

    def test_invalid_practical_limit_buckets_raises_assertion_error(self):
        crr_adjustment = 1
        time_gathering_water = 2.5
        practical_limit_bicycle = 10
        practical_limit_buckets = "invalid"
        met = 1.2
        with pytest.raises(AssertionError):
            run_simulation(crr_adjustment, time_gathering_water, practical_limit_bicycle, practical_limit_buckets, met)

    def test_invalid_met_raises_assertion_error(self):
        crr_adjustment = 1
        time_gathering_water = 2.5
        practical_limit_bicycle = 10
        practical_limit_buckets = 5
        met = "invalid"
        with pytest.raises(AssertionError):
            run_simulation(crr_adjustment, time_gathering_water, practical_limit_bicycle, practical_limit_buckets, met)



from src.gis_monte_carlo import process_mc_results

class TestProcessMCResults:

    def test_process_mc_results_saves_results_to_output_dir(self):
        # Arrange
        simulation_results = [pd.DataFrame({'percent_with_water': [0.5, 0.6, 0.7]}),
                              pd.DataFrame({'percent_with_water': [0.4, 0.3, 0.2]})]
        output_dir = 'test_results'

        # Act
        process_mc_results(simulation_results, plot=False, output_dir=output_dir)

        # Assert
        assert os.path.exists(os.path.join(output_dir, 'median_results.csv'))
        assert os.path.exists(os.path.join(output_dir, 'min_results.csv'))
        assert os.path.exists(os.path.join(output_dir, 'max_results.csv'))
        assert os.path.exists(os.path.join(output_dir, '95th_percentile_results.csv'))
        assert os.path.exists(os.path.join(output_dir, '5th_percentile_results.csv'))
        assert os.path.exists(os.path.join(output_dir, 'simulation_results.pkl'))

    def test_process_mc_results_plots_chloropleth_maps_when_plot_is_true(self, mocker):
        # Arrange
        simulation_results = [pd.DataFrame({'percent_with_water': [0.5, 0.6, 0.7]}),
                              pd.DataFrame({'percent_with_water': [0.4, 0.3, 0.2]})]
        mocker.patch('src.gis_monte_carlo.gis.plot_chloropleth')

        # Act
        process_mc_results(simulation_results, plot=True)

        # Assert
        assert src.gis_monte_carlo.gis.plot_chloropleth.call_count == 5

    def test_process_mc_results_does_not_plot_chloropleth_maps_when_plot_is_false(self, mocker):
        # Arrange
        simulation_results = [pd.DataFrame({'percent_with_water': [0.5, 0.6, 0.7]}),
                              pd.DataFrame({'percent_with_water': [0.4, 0.3, 0.2]})]
        mocker.patch('src.gis_monte_carlo.gis.plot_chloropleth')

        # Act
        process_mc_results(simulation_results, plot=False)

        # Assert
        assert src.gis_monte_carlo.gis.plot_chloropleth.call_count == 0

    def test_process_mc_results_raises_error_when_simulation_results_is_not_list(self):
        # Arrange
        simulation_results = 'invalid_results'

        # Act & Assert
        with pytest.raises(TypeError):
            process_mc_results(simulation_results, plot=False)

    def test_process_mc_results_raises_error_when_output_dir_is_not_string(self):
        # Arrange
        simulation_results = [pd.DataFrame({'percent_with_water': [0.5, 0.6, 0.7]})]
        output_dir = 123

        # Act & Assert
        with pytest.raises(TypeError):
            process_mc_results(simulation_results, plot=False, output_dir=output_dir)