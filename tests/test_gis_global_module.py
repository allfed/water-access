import numpy as np
import pandas as pd
import pytest

from src.gis_global_module import weighted_mean
from src.gis_global_module import weighted_median_series
from src.gis_global_module import weighted_median
from src.gis_global_module import weighted_median, run_weighted_median_on_grouped_df
from src.gis_global_module import load_data
from src.gis_global_module import manage_urban_rural
from src.gis_global_module import manage_slope
from src.gis_global_module import merge_and_adjust_population



class TestWeightedMean:
    def test_weighted_mean_returns_correct_result(self):
        var = np.array([1, 2, 3, 4, 5])
        wts = np.array([0.1, 0.2, 0.3, 0.2, 0.2])
        result = weighted_mean(var, wts)
        expected_result = 3.2
        assert np.isclose(result, expected_result)

    def test_weighted_mean_returns_zero_when_var_is_empty(self):
        var = np.array([])
        wts = np.array([0.1, 0.2, 0.3, 0.2, 0.2])
        result = weighted_mean(var, wts)
        expected_result = 0
        assert np.isclose(result, expected_result)

    def test_weighted_mean_returns_zero_when_wts_is_empty(self):
        var = np.array([1, 2, 3, 4, 5])
        wts = np.array([])
        result = weighted_mean(var, wts)
        expected_result = 0
        assert np.isclose(result, expected_result)

    def test_weighted_mean_returns_zero_when_both_var_and_wts_are_empty(self):
        var = np.array([])
        wts = np.array([])
        result = weighted_mean(var, wts)
        expected_result = 0
        assert np.isclose(result, expected_result)


class TestWeightedMedianSeries:
    def test_weighted_median_series_returns_correct_result(self):
        val = np.array([1, 2, 3, 4, 5])
        weight = np.array([0.1, 0.2, 0.3, 0.2, 0.2])
        result = weighted_median_series(val, weight)
        expected_result = 3.0
        assert np.isclose(result, expected_result)

    def test_weighted_median_series_returns_nan_when_val_is_empty(self):
        val = np.array([])
        weight = np.array([0.1, 0.2, 0.3, 0.2, 0.2])
        result = weighted_median_series(val, weight)
        expected_result = np.nan
        assert np.isnan(result)

    def test_weighted_median_series_returns_nan_when_weight_is_empty(self):
        val = np.array([1, 2, 3, 4, 5])
        weight = np.array([])
        result = weighted_median_series(val, weight)
        expected_result = np.nan
        assert np.isnan(result)

    def test_weighted_median_series_returns_nan_when_both_val_and_weight_are_empty(self):
        val = np.array([])
        weight = np.array([])
        result = weighted_median_series(val, weight)
        expected_result = np.nan
        assert np.isnan(result)

class TestWeightedMedian:
    def test_weighted_median_returns_correct_result(self):
        df = pd.DataFrame({'value': [1, 2, 3, 4, 5], 'weight': [0.1, 0.2, 0.3, 0.2, 0.2]})
        result = weighted_median(df, 'value', 'weight')
        expected_result = 3.0
        assert np.isclose(result, expected_result)

    def test_weighted_median_returns_error_when_df_is_empty(self):
        with pytest.raises(ValueError):
            df = pd.DataFrame({'value': [], 'weight': [0.1, 0.2, 0.3, 0.2, 0.2]})
            result = weighted_median(df, 'value', 'weight')

    def test_weighted_median_returns_error_when_weight_is_empty(self):
        with pytest.raises(ValueError):
            df = pd.DataFrame({'value': [1, 2, 3, 4, 5], 'weight': []})
            result = weighted_median(df, 'value', 'weight')

    def test_weighted_median_returns_error_when_both_df_and_weight_are_empty(self):
        with pytest.raises(IndexError):
            df = pd.DataFrame({'value': [], 'weight': []})
            result = weighted_median(df, 'value', 'weight')

class TestLoadData:
    def test_load_data_returns_dataframes(self):
        urb_data_file = "./data/GIS/GIS_data_zones.csv"
        country_data_file = "./data/processed/country_data_master_interpolated.csv"
        df_zones_input, df_input = load_data(urb_data_file, country_data_file)
        assert isinstance(df_zones_input, pd.DataFrame)
        assert isinstance(df_input, pd.DataFrame)

    def test_load_data_loads_correct_data(self):
        urb_data_file = "./data/GIS/GIS_data_zones.csv"
        country_data_file = "./data/processed/country_data_master_interpolated.csv"
        df_zones_input, df_input = load_data(urb_data_file, country_data_file)
        # Add assertions to check if the loaded data is correct
        # For example:
        assert len(df_zones_input) > 0
        assert len(df_input) > 0

class TestManageUrbanRural:
    def test_manage_urban_rural_converts_dtw_1_to_kilometers(self):
        df_zones_input = pd.DataFrame({'dtw_1': [1000, 2000, 3000], 'URBAN_1': [10, 20, 30]})
        expected_result = pd.DataFrame({'dtw_1': [1.0, 2.0, 3.0], 'URBAN_1': [10, 20, 30], 'urban_rural': [0, 1, 1]})
        result = manage_urban_rural(df_zones_input)
        print(result)
        print(expected_result)
        pd.testing.assert_frame_equal(result, expected_result)

    def test_manage_urban_rural_creates_urban_rural_column(self):
        df_zones_input = pd.DataFrame({'dtw_1': [1000, 2000, 3000], 'URBAN_1': [10, 20, 30]})
        result = manage_urban_rural(df_zones_input)
        assert 'urban_rural' in result.columns

    def test_manage_urban_rural_sets_urban_rural_to_1_for_urban_zones(self):
        df_zones_input = pd.DataFrame({'dtw_1': [1000, 2000, 3000], 'URBAN_1': [16, 20, 30]})
        result = manage_urban_rural(df_zones_input)
        assert all(result['urban_rural'] == 1)

    def test_manage_urban_rural_sets_urban_rural_to_0_for_rural_zones(self):
        df_zones_input = pd.DataFrame({'dtw_1': [1000, 2000, 3000], 'URBAN_1': [0, 10, 15]})
        result = manage_urban_rural(df_zones_input)
        assert all(result['urban_rural'] == 0)


class TestManageSlope:
    def test_manage_slope_modifies_dataframe(self):
        df_zones_input = pd.DataFrame({'slope_1': [1, 2, 3, 4, 5]})
        result = manage_slope(df_zones_input)
        assert 'slope_1' in result.columns

    def test_manage_slope_returns_same_dataframe(self):
        df_zones_input = pd.DataFrame({'slope_1': [1, 2, 3, 4, 5]})
        result = manage_slope(df_zones_input)
        pd.testing.assert_frame_equal(result, df_zones_input)


class TestMergeAndAdjustPopulation:
    def test_merge_and_adjust_population_returns_dataframe(self):
        df_zones_input = pd.DataFrame({'ISOCODE': ['USA', 'CAN'], 'pop_count_15_1': [1000, 2000]})
        df_input = pd.DataFrame({'alpha3': ['USA', 'CAN'], 'Population': [1000000, 2000000]})
        result = merge_and_adjust_population(df_zones_input, df_input)
        assert isinstance(result, pd.DataFrame)

    def test_merge_and_adjust_population_returns_expected_result(self):
        df_zones_input = pd.DataFrame({'ISOCODE': ['USA', 'CAN'], 'pop_count_15_1': [1000, 2000]})
        df_input = pd.DataFrame({'alpha3': ['USA', 'CAN'], 'Population': [1000000, 2000000]})
        result = merge_and_adjust_population(df_zones_input, df_input)
        expected_result = pd.DataFrame({
            'ISOCODE': ['USA', 'CAN'],
            'pop_count_15_1': [1000, 2000],
            'AdjPopFloat': [111.11111111111111, 222.22222222222223],
            'pop_density_perc': [0.0001111111111111111, 0.0001111111111111111],
            'pop_zone': [111.11111111111111, 222.22222222222223],
            'country_pop_raw': [111.11111111111111, 222.22222222222223],
            'country_pop_ratio': [333.3333333333333, 333.3333333333333],
            'any_pop': [1, 1]
        })
        pd.testing.assert_frame_equal(result, expected_result)

    def test_merge_and_adjust_population_handles_empty_dataframes(self):
        df_zones_input = pd.DataFrame()
        df_input = pd.DataFrame()
        result = merge_and_adjust_population(df_zones_input, df_input)
        assert len(result) == 0

import numpy as np
import pandas as pd
import pytest

from src.gis_global_module import road_analysis_old


class TestRoadAnalysisOld:
    def test_road_analysis_old_returns_dataframe(self):
        df_zones = pd.DataFrame({'grip_1_1': [0, 1, 0], 'grip_2_1': [1, 0, 0], 'grip_3_1': [0, 0, 1]})
        result = road_analysis_old(df_zones)
        assert isinstance(result, pd.DataFrame)

    def test_road_analysis_old_returns_expected_result(self):
        df_zones = pd.DataFrame({'grip_1_1': [0, 1, 0], 'grip_2_1': [1, 0, 0], 'grip_3_1': [0, 0, 1]})
        result = road_analysis_old(df_zones)
        expected_result = pd.DataFrame({
            'grip_1_1': [0, 1, 0],
            'grip_2_1': [1, 0, 0],
            'grip_3_1': [0, 0, 1],
            'dominant_road_type': ['Primary Roads', 'Highways', 'Tertiary Roads'],
            'Crr': [0.02, 0.01, 0.03]
        })
        pd.testing.assert_frame_equal(result, expected_result)

    def test_road_analysis_old_handles_all_zeros(self):
        df_zones = pd.DataFrame({'grip_1_1': [0, 0, 0], 'grip_2_1': [0, 0, 0], 'grip_3_1': [0, 0, 0]})
        result = road_analysis_old(df_zones)
        expected_result = pd.DataFrame({
            'grip_1_1': [0, 0, 0],
            'grip_2_1': [0, 0, 0],
            'grip_3_1': [0, 0, 0],
            'dominant_road_type': ['No Roads', 'No Roads', 'No Roads'],
            'Crr': [np.nan, np.nan, np.nan]
        })
        pd.testing.assert_frame_equal(result, expected_result)

    def test_road_analysis_old_handles_missing_values(self):
        df_zones = pd.DataFrame({'grip_1_1': [0, 1, np.nan], 'grip_2_1': [1, np.nan, 0], 'grip_3_1': [np.nan, 0, 1]})
        result = road_analysis_old(df_zones)
        expected_result = pd.DataFrame({
            'grip_1_1': [0, 1, np.nan],
            'grip_2_1': [1, np.nan, 0],
            'grip_3_1': [np.nan, 0, 1],
            'dominant_road_type': ['Primary Roads', np.nan, 'Tertiary Roads'],
            'Crr': [0.02, np.nan, 0.03]
        })
        pd.testing.assert_frame_equal(result, expected_result)