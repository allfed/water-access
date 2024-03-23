import numpy as np
import pandas as pd
import pytest
import pdb
from pathlib import Path
import sys

from src.gis_global_module import (
    weighted_mean,
    weighted_median_series,
    weighted_median,
    load_data,
    manage_urban_rural,
    manage_slope,
    merge_and_adjust_population,
    process_country_data,
    calculate_max_distances,
    run_walking_model,
    calculate_and_merge_walking_distance,
    calculate_population_water_access,
    calculate_water_rations,
    calculate_and_merge_bicycle_distance,
    aggregate_country_level_data,
    calculate_weighted_median,
    crr_add_uncertainty,
    road_analysis,
    load_hpv_parameters,
    extract_slope_crr,
    run_bicycle_model,
    process_and_save_results,
    clean_up_data,
)


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
            'alpha3': ['USA', 'CAN'],
            'Population': [1000000, 2000000],
            'AdjPopFloat': [111.11111111111111, 222.22222222222223],
            'pop_density_perc': [1.0, 1.0],
            'pop_zone': [1000000.0, 2000000.0],
            'country_pop_raw': [1000000.0, 2000000.0],
            'country_pop_ratio': [111.11111111111111, 222.22222222222223],
            'any_pop': [1, 1]
        })
        pd.testing.assert_frame_equal(result, expected_result)


    def test_merge_and_adjust_population_handles_empty_dataframes(self):
        df_zones_input = pd.DataFrame()
        df_input = pd.DataFrame()
        with pytest.raises(AssertionError):
            merge_and_adjust_population(df_zones_input, df_input)


class TestCrrAddUncertainty:
    def test_crr_add_uncertainty_returns_correct_result(self):
        road_type = "Primary Roads"
        adjustment = 1
        result = crr_add_uncertainty(road_type, adjustment)
        expected_result = "Secondary Roads"
        assert result == expected_result

    def test_crr_add_uncertainty_returns_highest_road_type_when_adjustment_exceeds_upper_bound(self):
        road_type = "Local Roads"
        adjustment = 2
        result = crr_add_uncertainty(road_type, adjustment)
        expected_result = "No Roads"
        assert result == expected_result

    def test_crr_add_uncertainty_returns_lowest_road_type_when_adjustment_exceeds_lower_bound(self):
        road_type = "Primary Roads"
        adjustment = -10
        result = crr_add_uncertainty(road_type, adjustment)
        expected_result = "Highways"
        assert result == expected_result

    def test_crr_add_uncertainty_returns_same_road_type_when_adjustment_is_zero(self):
        road_type = "Tertiary Roads"
        adjustment = 0
        result = crr_add_uncertainty(road_type, adjustment)
        expected_result = "Tertiary Roads"
        assert result == expected_result



class TestRoadAnalysis:
    def test_road_analysis_returns_dataframe(self):
        df_zones = pd.DataFrame({'grip_1_1': [0, 0, 0], 'grip_2_1': [0, 0, 0], 'grip_3_1': [0, 0, 0], 'grip_4_1': [0, 0, 0], 'grip_5_1': [0, 0, 0]})
        result = road_analysis(df_zones)
        assert isinstance(result, pd.DataFrame)

    def test_road_analysis_returns_expected_result(self):
        df_zones = pd.DataFrame({'grip_1_1': [0, 1, 0], 'grip_2_1': [0, 0, 1], 'grip_3_1': [1, 0, 0], 'grip_4_1': [0, 0, 0], 'grip_5_1': [0, 0, 0]})
        result = road_analysis(df_zones)
        expected_result = pd.DataFrame({
            'grip_1_1': [0, 1, 0],
            'grip_2_1': [0, 0, 1],
            'grip_3_1': [1, 0, 0],
            'grip_4_1': [0, 0, 0],
            'grip_5_1': [0, 0, 0],
            'dominant_road_type': ['Secondary Roads', 'Highways', 'Primary Roads'],
            'Crr': [0.004, 0.002, 0.003]
        })
        pd.testing.assert_frame_equal(result, expected_result)

    def test_road_analysis_handles_all_zeros(self):
        df_zones = pd.DataFrame({'grip_1_1': [0, 0, 0], 'grip_2_1': [0, 0, 0], 'grip_3_1': [0, 0, 0], 'grip_4_1': [0, 0, 0], 'grip_5_1': [0, 0, 0]})
        result = road_analysis(df_zones)
        assert 'dominant_road_type' in result.columns
        assert 'Crr' in result.columns

    def test_road_analysis_handles_crr_adjustment(self):
        df_zones = pd.DataFrame({'grip_1_1': [0, 1, 0], 'grip_2_1': [0, 0, 1], 'grip_3_1': [1, 0, 0], 'grip_4_1': [0, 0, 0], 'grip_5_1': [0, 0, 0]})
        result = road_analysis(df_zones, crr_adjustment=1)
        expected_result = pd.DataFrame({
            'grip_1_1': [0, 1, 0],
            'grip_2_1': [0, 0, 1],
            'grip_3_1': [1, 0, 0],
            'grip_4_1': [0, 0, 0],
            'grip_5_1': [0, 0, 0],
            'dominant_road_type': ['Secondary Roads', 'Highways', 'Primary Roads'],
            'Crr': [0.100, 0.003, 0.004]
        })
        pd.testing.assert_frame_equal(result, expected_result)



class TestLoadHPVParameters:
    def test_load_hpv_parameters_returns_dataframe(self):
        file_path_params = "./data/lookup tables/mobility-model-parameters.csv"
        hpv_name = "HPV1"
        result = load_hpv_parameters(file_path_params, hpv_name)
        assert isinstance(result, pd.DataFrame)




class TestExtractSlopeCrr:
    def test_extract_slope_crr_returns_tuple_of_series(self):
        df_zones = pd.DataFrame({'slope_1': [1, 2, 3, 4, 5], 'Crr': [0.1, 0.2, 0.3, 0.4, 0.5]})
        result = extract_slope_crr(df_zones)
        assert isinstance(result, tuple)
        assert isinstance(result[0], pd.Series)
        assert isinstance(result[1], pd.Series)

    def test_extract_slope_crr_returns_correct_series(self):
        df_zones = pd.DataFrame({'slope_1': [1, 2, 3, 4, 5], 'Crr': [0.1, 0.2, 0.3, 0.4, 0.5]})
        slope_zones, Crr_values = extract_slope_crr(df_zones)
        expected_slope_zones = pd.Series([1, 2, 3, 4, 5], name='slope_1')
        expected_Crr_values = pd.Series([0.1, 0.2, 0.3, 0.4, 0.5], name="Crr")
        pd.testing.assert_series_equal(slope_zones, expected_slope_zones)
        pd.testing.assert_series_equal(Crr_values, expected_Crr_values)



class TestRunBicycleModel:
    def test_run_bicycle_model_returns_numpy_array(self):

        project_root = Path().resolve().parent
        sys.path.append(str(project_root))
        import src.mobility_module as mm
        file_path_params = "./data/lookup tables/mobility-model-parameters.csv"
        param_df = load_hpv_parameters(file_path_params, "Bicycle")
        param_df["PracticalLimit"] = 40
        mo = mm.model_options()
        mo.model_selection = 2  # Cycling model
        mv = mm.model_variables()
        met = mm.MET_values(mv)
        hpv = mm.HPV_variables(param_df, mv)
        slope_zones = [0, 1, 2]
        Crr_values = [0.01, 0.02, 0.03]
        load_attempt = 1
        result = run_bicycle_model(mv, mo, hpv, slope_zones, Crr_values, load_attempt)
        assert isinstance(result, np.ndarray)

    def test_run_bicycle_model_adjusts_project_root_and_imports_mobility_module(self):
        project_root = Path().resolve().parent
        sys.path.append(str(project_root))
        import src.mobility_module as mm
        file_path_params = "./data/lookup tables/mobility-model-parameters.csv"
        param_df = load_hpv_parameters(file_path_params, "Bicycle")
        param_df["PracticalLimit"] = 40
        mo = mm.model_options()
        mo.model_selection = 2  # Cycling model
        mv = mm.model_variables()
        met = mm.MET_values(mv)
        hpv = mm.HPV_variables(param_df, mv)
        slope_zones = [0, 1, 2]
        Crr_values = [0.01, 0.02, 0.03]
        load_attempt = 1
        result = run_bicycle_model(mv, mo, hpv, slope_zones, Crr_values, load_attempt)
        assert str(project_root) in sys.path
        assert "src.mobility_module" in sys.modules

    def test_run_bicycle_model_calls_single_bike_run_for_each_slope_zone_and_Crr_value(self, monkeypatch):
        project_root = Path().resolve().parent
        sys.path.append(str(project_root))
        import src.mobility_module as mm
        file_path_params = "./data/lookup tables/mobility-model-parameters.csv"
        param_df = load_hpv_parameters(file_path_params, "Bicycle")
        param_df["PracticalLimit"] = 40
        mo = mm.model_options()
        mo.model_selection = 2  # Cycling model
        mv = mm.model_variables()
        hpv = mm.HPV_variables(param_df, mv)
        slope_zones = [0, 1, 2]
        Crr_values = [0.01, 0.02, 0.03]
        load_attempt = 1

        calls = []

        def mock_single_bike_run(mv, mo, hpv, slope, load_attempt):
            calls.append((slope, hpv.Crr))

        monkeypatch.setattr("src.mobility_module.mobility_models.single_bike_run", mock_single_bike_run)

        run_bicycle_model(mv, mo, hpv, slope_zones, Crr_values, load_attempt)

        expected_calls = [(0, 0.01), (1, 0.02), (2, 0.03)]
        assert calls == expected_calls


# Test case for saving results as CSV
def test_process_and_save_results_saves_csv(tmp_path):
    # Create a temporary directory for the CSV file
    export_file_location = tmp_path / "output/"
    export_file_location.mkdir()

    # Create a sample DataFrame and results array
    df_zones = pd.DataFrame({"fid": [1, 2, 3]})
    results = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
    velocity_type = "walk"

    # Call the function with save_csv=True
    updated_df = process_and_save_results(df_zones, results, export_file_location, velocity_type, save_csv=True)

    # Check if the CSV file is saved
    expected_csv_file = export_file_location / f"{velocity_type}_velocity_by_zone.csv"
    assert expected_csv_file.exists(), "CSV file was not saved"

    # Check if the DataFrame is updated with the new columns
    assert "loaded_velocity_walk" in updated_df.columns
    assert "unloaded_velocity_walk" in updated_df.columns
    assert "average_velocity_walk" in updated_df.columns
    assert "max_load_walk" in updated_df.columns

# Test case for not saving results as CSV
def test_process_and_save_results_does_not_save_csv(tmp_path):
    # Create a temporary directory for the CSV file
    export_file_location = tmp_path / "output/"
    export_file_location.mkdir()

    # Create a sample DataFrame and results array
    df_zones = pd.DataFrame({"fid": [1, 2, 3]})
    results = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
    velocity_type = "walk"

    # Call the function with save_csv=False
    updated_df = process_and_save_results(df_zones, results, export_file_location, velocity_type, save_csv=False)

    # Check if the CSV file is not saved
    expected_csv_file = export_file_location / "walk_velocity_by_zone.csv"
    assert not expected_csv_file.exists()

    # Check if the DataFrame is updated with the new columns
    assert "loaded_velocity_walk" in updated_df.columns
    assert "unloaded_velocity_walk" in updated_df.columns
    assert "average_velocity_walk" in updated_df.columns
    assert "max_load_walk" in updated_df.columns
    

class TestCalculateAndMergeBicycleDistance:
    def test_calculate_and_merge_bicycle_distance_calculates_distance_when_flag_is_true(self):
        df_zones = pd.DataFrame({'fid': [1, 2, 3], 'zone': ['A', 'B', 'C'], 'slope_1': [0, 1, 2], 'Crr': [0.01, 0.02, 0.03]})
        calculate_distance = True
        export_file_location = "./data/processed/"
        practical_limit_bicycle = 40

        result = calculate_and_merge_bicycle_distance(df_zones, calculate_distance, export_file_location, practical_limit_bicycle)

        assert 'loaded_velocity_bicycle' in result.columns

    def test_calculate_and_merge_bicycle_distance_merges_bicycle_distance_from_file(self):
        df_zones = pd.DataFrame({'fid': [1, 2, 3], 'zone': ['A', 'B', 'C']})
        calculate_distance = False
        export_file_location = "./data/processed/"
        practical_limit_bicycle = 40

        result = calculate_and_merge_bicycle_distance(df_zones, calculate_distance, export_file_location, practical_limit_bicycle)

        assert 'loaded_velocity_bicycle' in result.columns
        assert result['average_velocity_bicycle'].notnull().all()


class TestRunWalkingModel:
    def test_run_walking_model_returns_array(self):
        project_root = Path().resolve().parent
        sys.path.append(str(project_root))
        import src.mobility_module as mm

        file_path_params = "./data/lookup tables/mobility-model-parameters.csv"
        param_df = load_hpv_parameters(file_path_params, "Buckets")
        param_df["PracticalLimit"] = 20

        mo = mm.model_options()
        mo.model_selection = 3  # Lankford model
        mv = mm.model_variables()
        met = mm.MET_values(mv, met=3.3)
        hpv = mm.HPV_variables(param_df, mv)

        slope_zones = [0, 1, 2]

        result = run_walking_model(mv, mo, met, hpv, slope_zones, load_attempt=20)
        assert isinstance(result, np.ndarray)

    def test_run_walking_model_calls_single_walk_run_for_each_slope_zone(self, monkeypatch):
        project_root = Path().resolve().parent
        sys.path.append(str(project_root))
        import src.mobility_module as mm
        file_path_params = "./data/lookup tables/mobility-model-parameters.csv"
        param_df = load_hpv_parameters(file_path_params, "Walking")
        param_df["PracticalLimit"] = 20
        mo = mm.model_options()
        mo.model_selection = 3  # Lankford model
        mv = mm.model_variables()
        met = mm.MET_values(mv, met=3.3)
        hpv = mm.HPV_variables(param_df, mv)
        slope_zones = [0, 1, 2]
        load_attempt = 1

        calls = []

        def mock_single_lankford_run(mv, mo, met, hpv, slope, load_attempt):
            calls.append(slope)

        monkeypatch.setattr("src.mobility_module.mobility_models.single_lankford_run", mock_single_lankford_run)

        run_walking_model(mv, mo, met, hpv, slope_zones, load_attempt)

        expected_calls = [0, 1, 2]
        assert calls == expected_calls

class TestCalculateAndMergeWalkingDistance:
    def test_calculate_and_merge_walking_distance_calculates_distance_when_flag_is_true(self):
        df_zones = pd.DataFrame({'fid': [1, 2, 3], 'zone': ['A', 'B', 'C'], 'slope_1': [0, 1, 2], 'Crr': [0.01, 0.02, 0.03]})
        calculate_distance = True
        export_file_location = "./data/processed/"
        practical_limit_walking = 20

        result = calculate_and_merge_walking_distance(df_zones, calculate_distance, export_file_location, practical_limit_walking)

        assert 'loaded_velocity_walk' in result.columns

    def test_calculate_and_merge_walking_distance_merges_walking_distance_from_file(self):
        df_zones = pd.DataFrame({'fid': [1, 2, 3], 'zone': ['A', 'B', 'C']})
        calculate_distance = False
        export_file_location = "./data/processed/"
        practical_limit_walking = 20

        result = calculate_and_merge_walking_distance(df_zones, calculate_distance, export_file_location, practical_limit_walking)

        assert 'loaded_velocity_walk' in result.columns
        assert result['average_velocity_walk'].notnull().all()


class TestCalculateMaxDistances:
    def test_calculate_max_distances_returns_dataframe(self):
        df_zones = pd.DataFrame({
            "average_velocity_bicycle": [10, 15, 20],
            "average_velocity_walk": [5, 7, 10],
            "max_load_bicycle": [5, 10, 15]
        })
        time_gathering_water = 2
        result = calculate_max_distances(df_zones, time_gathering_water)
        assert isinstance(result, pd.DataFrame)

    def test_calculate_max_distances_returns_expected_result(self):
        df_zones = pd.DataFrame({
            "average_velocity_bicycle": [10, 15, 20],
            "average_velocity_walk": [5, 7, 10],
            "max_load_bicycle": [5, 10, 15]
        })
        time_gathering_water = 6
        result = calculate_max_distances(df_zones, time_gathering_water)
        expected_result = pd.DataFrame({
            "average_velocity_bicycle": [10, 15, 20],
            "average_velocity_walk": [5, 7, 10],
            "max_load_bicycle": [5, 10, 15],
            "max distance cycling": [30.0, 45.0, 60.0],
            "max distance walking": [15.0, 21.0, 30.0],
            "water_ration_kms": [150.0, 450.0, 900.0]
        })
        pd.testing.assert_frame_equal(result, expected_result)


class TestCalculatePopulationWaterAccess:
    def test_calculate_population_water_access_returns_dataframe(self):
        df_zones = pd.DataFrame({'pop_zone': [1000, 2000, 3000],
                                 'urban_rural': [0, 1, 1],
                                 'URBANPiped': [50, 60, 70],
                                 'RURALPiped': [30, 40, 50],
                                 'URBANNon-piped': [20, 30, 40],
                                 'RURALNon-piped': [10, 20, 30],
                                 'dtw_1': [100, 200, 300],
                                 'max distance cycling': [500, 600, 700],
                                 'max distance walking': [200, 300, 400],
                                 'PBO': [80, 90, 100]})
        result = calculate_population_water_access(df_zones)
        assert isinstance(result, pd.DataFrame)

    def test_calculate_population_water_access_returns_expected_result(self):
        df_zones = pd.DataFrame({'pop_zone': [1000, 2000, 3000],
                                 'urban_rural': [0, 1, 1],
                                 'URBANPiped': [50, 60, 70],
                                 'RURALPiped': [30, 40, 50],
                                 'URBANNon-piped': [20, 30, 40],
                                 'RURALNon-piped': [10, 20, 30],
                                 'dtw_1': [100, 200, 300],
                                 'max distance cycling': [500, 600, 700],
                                 'max distance walking': [200, 300, 400],
                                 'PBO': [80, 90, 100]})
        result = calculate_population_water_access(df_zones)
        expected_result = pd.DataFrame({'pop_zone': [1000, 2000, 3000],
                                         'urban_rural': [0, 1, 1],
                                         'URBANPiped': [50, 60, 70],
                                         'RURALPiped': [30, 40, 50],
                                         'URBANNon-piped': [20, 30, 40],
                                         'RURALNon-piped': [10, 20, 30],
                                         'dtw_1': [100, 200, 300],
                                         'max distance cycling': [500, 600, 700],
                                         'max distance walking': [200, 300, 400],
                                         'PBO': [80, 90, 100],
                                         'zone_pop_piped': [300.0, 1200.0, 2100.0],
                                         'zone_pop_unpiped': [100.0, 600.0, 1200.0],
                                         'zone_cycling_okay': [1, 1, 1],
                                         'zone_walking_okay': [1, 1, 1],
                                         'fraction_of_zone_with_cycling_access': [0.8, 0.9, 1.0],
                                         'fraction_of_zone_with_walking_access': [1, 1, 1],
                                         'population_piped_with_cycling_access': [240.0, 1080.0, 2100.0],
                                         'population_piped_with_walking_access': [300.0, 1200.0, 2100.0],
                                         'population_piped_with_access': [300.0, 1200.0, 2100.0],
                                         'zone_pop_with_water': [400.0, 1800.0, 3300.0],
                                         'zone_pop_without_water': [600.0, 200.0, -300.0]})
        pd.testing.assert_frame_equal(result, expected_result)

class TestCalculateWaterRations:
    def test_calculate_water_rations_returns_dataframe(self):
        df_zones = pd.DataFrame({
            "dtw_1": [1000, 2000, 3000],
            "water_ration_kms": [10, 20, 30],
            "pop_zone": [1000, 2000, 3000],
            "Average household size (number of members)": [2, 3, 4],
            "PBO": [0.5, 0.6, 0.7]
        })
        result = calculate_water_rations(df_zones)
        assert isinstance(result, pd.DataFrame)

    def test_calculate_water_rations_calculates_rations(self):
        df_zones = pd.DataFrame({
            "dtw_1": [1000, 2000, 3000],
            "water_ration_kms": [10, 20, 30],
            "pop_zone": [1000, 2000, 3000],
            "Average household size (number of members)": [2, 3, 4],
            "PBO": [0.5, 0.6, 0.7]
        })
        result = calculate_water_rations(df_zones)
        expected_result = pd.DataFrame({
            "dtw_1": [1000, 2000, 3000],
            "water_ration_kms": [10, 20, 30],
            "pop_zone": [1000, 2000, 3000],
            "Average household size (number of members)": [2, 3, 4],
            "PBO": [0.5, 0.6, 0.7],
            "water_rations_per_bike": [0.01, 0.01, 0.01],
            "bikes_in_zone": [250.0, 400.0, 525.0],
            "water_rations_achievable": [2.50, 4.0, 5.25]
        })
        pd.testing.assert_frame_equal(result, expected_result)

class TestAggregateCountryLevelData:
    def test_aggregate_country_level_data_returns_dataframe(self):
        df_zones = pd.DataFrame({
            "ISOCODE": ["USA", "CAN"],
            "Entity": ["United States", "Canada"],
            "country_pop_raw": [1000000, 2000000],
            "zone_pop_with_water": [500000, 1000000],
            "zone_pop_without_water": [500000, 1000000],
            "population_piped_with_access": [300000, 400000],
            "population_piped_with_cycling_access": [200000, 300000],
            "population_piped_with_walking_access": [100000, 200000],
            "Nat Piped": ["Yes", "No"],
            "region": ["North America", "North America"],
            "subregion": ["Northern America", "Northern America"]
        })
        
        result = aggregate_country_level_data(df_zones)
        
        assert isinstance(result, pd.DataFrame)
    
    def test_aggregate_country_level_data_returns_expected_result(self):
        df_zones = pd.DataFrame({
            "ISOCODE": ["USA", "USA", "CAN", "CAN"],
            "Entity": ["United States", "United States", "Canada", "Canada"],
            "country_pop_raw": [1000000, 1000000, 2000000, 2000000],
            "zone_pop_with_water": [250000, 250000, 500000, 500000],
            "zone_pop_without_water": [250000, 250000, 500000, 500000],
            "population_piped_with_access": [150000, 150000, 200000, 200000],
            "population_piped_with_cycling_access": [100000, 100000, 150000, 150000],
            "population_piped_with_walking_access": [50000, 50000, 100000, 100000],
            "Nat Piped": ["Yes", "Yes", "No", "No"],
            "region": ["North America", "North America", "North America", "North America"],
            "subregion": ["Northern America", "Northern America", "Northern America", "Northern America"]
        })
        
        expected_result = pd.DataFrame({
            "ISOCODE": ["CAN", "USA"],
            "Entity": ["Canada", "United States"],
            "country_pop_raw": [2000000, 1000000],
            "zone_pop_with_water": [1000000, 500000],
            "zone_pop_without_water": [1000000, 500000],
            "population_piped_with_access": [400000, 300000],
            "population_piped_with_cycling_access": [300000, 200000],
            "population_piped_with_walking_access": [200000, 100000],
            "Nat Piped": ["No", "Yes"],
            "region": ["North America", "North America"],
            "subregion": ["Northern America", "Northern America"]
        })
        
        result = aggregate_country_level_data(df_zones)
        pd.testing.assert_frame_equal(result, expected_result)


class TestCalculateWeightedMedian:
    def test_calculate_weighted_median_returns_dataframe(self):
        df_zones = pd.DataFrame({'ISOCODE': ['USA', 'CAN'], 'dtw_1': [1000, 2000], 'pop_zone': [1000000, 2000000]})
        result = calculate_weighted_median(df_zones)
        assert isinstance(result, pd.DataFrame)

    def test_calculate_weighted_median_returns_expected_result(self):
        df_zones = pd.DataFrame({'ISOCODE': ['USA', 'USA', 'CAN', 'CAN'], 'dtw_1': [1000, 2000, 3000, 4000], 'pop_zone': [1000000, 2000000, 3000000, 4000000]})
        result = calculate_weighted_median(df_zones)
        expected_result = pd.DataFrame({'ISOCODE': ['CAN', 'USA'], 'weighted_med': [4000, 2000]})
        expected_result.set_index('ISOCODE', inplace=True)
        pd.testing.assert_frame_equal(result, expected_result)


class TestCleanUpData:
    def test_clean_up_data_removes_nan_rows(self):
        df_countries = pd.DataFrame({"ISOCODE": ["USA", "CAN", "LBY"], "weighted_med": [1.0, None, 3.0]})
        cleaned_df, _, _ = clean_up_data(df_countries)
        assert len(cleaned_df) == 2
        assert cleaned_df["ISOCODE"].tolist() == ["USA", "LBY"]
        assert cleaned_df["weighted_med"].tolist() == [1.0, 3.0]

    def test_clean_up_data_removes_outliers(self):
        df_countries = pd.DataFrame({"ISOCODE": ["USA", "CAN", "LBY"], "weighted_med": [1.0, 10.0, 3.0]})
        cleaned_df, _, _ = clean_up_data(df_countries)
        assert len(cleaned_df) == 2
        assert cleaned_df["ISOCODE"].tolist() == ["USA", "LBY"]
        assert cleaned_df["weighted_med"].tolist() == [1.0, 3.0]

    def test_clean_up_data_removes_specific_countries(self):
        df_countries = pd.DataFrame({"ISOCODE": ["USA", "ATG", "GUM", "LBY"], "weighted_med": [1.0, 2.0, 3.0, 4.0]})
        cleaned_df, _, _ = clean_up_data(df_countries)
        assert len(cleaned_df) == 2
        assert cleaned_df["ISOCODE"].tolist() == ["USA", "LBY"]
        assert cleaned_df["weighted_med"].tolist() == [1.0, 4.0]
        

class TestProcessCountryData:
    def test_process_country_data_returns_dataframe(self):
        df_zones = pd.DataFrame({'ISOCODE': ['USA', 'CAN', 'LBY'], 'pop_zone': [1000, 2000, 1500], 'dtw_1': [10, 20, 15],
                    'Entity': ['USA', 'CAN', 'Libya'], 'Nat Piped': [1000, 2000, 1500],
                    'country_pop_raw': [1000, 2000, 1500],
                    'population_piped_with_access': [1000, 2000, 1500],
                    'population_piped_with_cycling_access': [1000, 2000, 1500],
                    'population_piped_with_walking_access': [1000, 2000, 1500],
                    'region': ['North America', 'North America', 'Africa'],
                    'subregion': ['Northern America', 'Northern America', 'Northern Africa'],
                    'zone_pop_with_water': [1000, 2000, 1500],
                    'zone_pop_without_water': [0, 0, 0]})
        result = process_country_data(df_zones)
        assert isinstance(result, pd.DataFrame)

    def test_process_country_data_returns_expected_result(self):
        df_zones = pd.DataFrame({'ISOCODE': ['USA', 'CAN', 'LBY'], 'pop_zone': [1000, 2000, 1500], 'dtw_1': [10, 20, 15],
                    'Entity': ['USA', 'CAN', 'Libya'], 'Nat Piped': [1000, 2000, 1500],
                    'country_pop_raw': [1000, 2000, 1500],
                    'population_piped_with_access': [1000, 2000, 1500],
                    'population_piped_with_cycling_access': [1000, 2000, 1500],
                    'population_piped_with_walking_access': [1000, 2000, 1500],
                    'region': ['North America', 'North America', 'Africa'],
                    'subregion': ['Northern America', 'Northern America', 'Northern Africa'],
                    'zone_pop_with_water': [1000, 2000, 1500],
                    'zone_pop_without_water': [0, 0, 0]})
        result = process_country_data(df_zones)
        expected_result = pd.DataFrame({
            'ISOCODE': ['LBY', 'USA'], 
            'Entity': ['Libya', 'USA'], 
            'country_pop_raw': [1500, 1000],
            'country_pop_with_water': [1500, 1000],
            'country_pop_without_water': [0, 0],
            'population_piped_with_access': [1500, 1000],
            'population_piped_with_cycling_access': [1500, 1000],
            'population_piped_with_walking_access': [1500, 1000],
            'Nat Piped': [1500, 1000],
            'region': ['Africa', 'North America'],
            'subregion': ['Northern Africa', 'Northern America'],
            'weighted_med': [15, 10],
            'percent_with_water': [100.0, 100.0],
            'percent_without_water': [0.0, 0.0]                   
        }, index=pd.Int64Index([1, 2], dtype='int64'))
        pd.testing.assert_frame_equal(result, expected_result)

    def test_process_country_data_raises_assertion_error_when_df_zones_is_empty(self):
        df_zones = pd.DataFrame()
        with pytest.raises(AssertionError):
            process_country_data(df_zones)

    def test_process_country_data_issues_warning_when_df_zones_contains_nan_values(self, capsys):
        df_zones = pd.DataFrame({'ISOCODE': ['USA', 'CAN', 'LBY'], 'pop_zone': [1000, 2000, 1500], 'dtw_1': [10, 20, 15],
                    'Entity': ['USA', 'CAN', 'Libya'], 'Nat Piped': [1000, 2000, 1500],
                    'country_pop_raw': [1000, 2000, 1500],
                    'population_piped_with_access': [1000, 2000, 1500],
                    'population_piped_with_cycling_access': [1000, 2000, 1500],
                    'population_piped_with_walking_access': [1000, 2000, 1500],
                    'region': ['North America', 'North America', 'Africa'],
                    'subregion': ['Northern America', 'Northern America', 'Northern Africa'],
                    'zone_pop_with_water': [1000, 2000, 1500],
                    'zone_pop_without_water': [0, 0, 0]})
        with pytest.raises(AssertionError):
            process_country_data(df_zones)
            captured = capsys.readouterr()
            assert "Input dataframe contains NaN values" in captured.err

    def test_process_country_data_raises_error_when_df_countries_is_empty(self):
        df_zones = pd.DataFrame()
        with pytest.raises(AssertionError):
            process_country_data(df_zones)

    def test_process_country_data_raises_index_error_when_df_countries_contains_nan_values(self):
        df_zones = pd.DataFrame({'ISOCODE': ['USA', 'CAN', 'LBY'], 'pop_zone': [1000, np.nan, 1500], 'dtw_1': [10, 20, 15],
                    'Entity': ['USA', 'CAN', 'Libya'], 'Nat Piped': [1000, 2000, 1500],
                    'country_pop_raw': [1000, 2000, 1500],
                    'population_piped_with_access': [1000, 2000, 1500],
                    'population_piped_with_cycling_access': [1000, 2000, 1500],
                    'population_piped_with_walking_access': [1000, 2000, 1500],
                    'region': ['North America', 'North America', 'Africa'],
                    'subregion': ['Northern America', 'Northern America', 'Northern Africa'],
                    'zone_pop_with_water': [1000, 2000, 1500],
                    'zone_pop_without_water': [0, 0, 0]})
        with pytest.raises(IndexError):
            process_country_data(df_zones)



    def test_process_country_data_renames_and_creates_percentage_columns(self):
        df_zones = pd.DataFrame({'ISOCODE': ['USA', 'CAN', 'LBY'], 'pop_zone': [1000, 2000, 1500], 'dtw_1': [10, 20, 15],
                    'Entity': ['USA', 'CAN', 'Libya'], 'Nat Piped': [1000, 2000, 1500],
                    'country_pop_raw': [1000, 2000, 1500],
                    'population_piped_with_access': [1000, 2000, 1500],
                    'population_piped_with_cycling_access': [1000, 2000, 1500],
                    'population_piped_with_walking_access': [1000, 2000, 1500],
                    'region': ['North America', 'North America', 'Africa'],
                    'subregion': ['Northern America', 'Northern America', 'Northern Africa'],
                    'zone_pop_with_water': [1000, 2000, 1500],
                    'zone_pop_without_water': [0, 0, 0]})
        result = process_country_data(df_zones)
        assert 'percent_with_water' in result.columns
        assert 'percent_without_water' in result.columns

    def test_process_country_data_prints_summary_of_removed_countries(self, capsys):
        df_zones = pd.DataFrame({'ISOCODE': ['USA', 'CAN', 'LBY'], 'pop_zone': [1000, 2000, 1500], 'dtw_1': [10, 20, 15],
                    'Entity': ['USA', 'CAN', 'Libya'], 'Nat Piped': [1000, 2000, 1500],
                    'country_pop_raw': [1000, 2000, 1500],
                    'population_piped_with_access': [1000, 2000, 1500],
                    'population_piped_with_cycling_access': [1000, 2000, 1500],
                    'population_piped_with_walking_access': [1000, 2000, 1500],
                    'region': ['North America', 'North America', 'Africa'],
                    'subregion': ['Northern America', 'Northern America', 'Northern Africa'],
                    'zone_pop_with_water': [1000, 2000, 1500],
                    'zone_pop_without_water': [0, 0, 0]})
        process_country_data(df_zones)
        captured = capsys.readouterr()
        assert "Countries removed from analysis due to being further than Libya's median:" in captured.out
        assert "Countries removed manually:" in captured.out