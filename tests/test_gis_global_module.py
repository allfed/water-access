import numpy as np
import pandas as pd
import numpy as np
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
    calculate_weighted_results,
    crr_add_uncertainty,
    road_analysis,
    load_hpv_parameters,
    extract_slope_crr,
    run_bicycle_model,
    process_and_save_results,
    clean_up_data,
    map_hill_polarity,
    adjust_euclidean,
    run_global_analysis,
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

    def test_weighted_median_series_returns_nan_when_both_val_and_weight_are_empty(
        self,
    ):
        val = np.array([])
        weight = np.array([])
        result = weighted_median_series(val, weight)
        expected_result = np.nan
        assert np.isnan(result)


class TestWeightedMedian:
    def test_weighted_median_returns_correct_result(self):
        df = pd.DataFrame(
            {"value": [1, 2, 3, 4, 5], "weight": [0.1, 0.2, 0.3, 0.2, 0.2]}
        )
        result = weighted_median(df, "value", "weight")
        expected_result = 3.0
        assert np.isclose(result, expected_result)

    def test_weighted_median_returns_error_when_df_is_empty(self):
        with pytest.raises(ValueError):
            df = pd.DataFrame({"value": [], "weight": [0.1, 0.2, 0.3, 0.2, 0.2]})
            result = weighted_median(df, "value", "weight")

    def test_weighted_median_returns_error_when_weight_is_empty(self):
        with pytest.raises(ValueError):
            df = pd.DataFrame({"value": [1, 2, 3, 4, 5], "weight": []})
            result = weighted_median(df, "value", "weight")

    def test_weighted_median_returns_error_when_both_df_and_weight_are_empty(self):
        with pytest.raises(IndexError):
            df = pd.DataFrame({"value": [], "weight": []})
            result = weighted_median(df, "value", "weight")


class TestLoadData:
    def test_load_data_returns_dataframes(self):
        urb_data_file = "./data/GIS/GIS_data_zones_sample.csv"
        country_data_file = "./data/processed/merged_data.csv"
        df_zones_input, df_input = load_data(urb_data_file, country_data_file)
        assert isinstance(df_zones_input, pd.DataFrame)
        assert isinstance(df_input, pd.DataFrame)

    def test_load_data_loads_correct_data(self):
        urb_data_file = "./data/GIS/GIS_data_zones_sample.csv"
        country_data_file = "./data/processed/merged_data.csv"
        df_zones_input, df_input = load_data(urb_data_file, country_data_file)
        # Add assertions to check if the loaded data is correct
        # For example:
        assert len(df_zones_input) > 0
        assert len(df_input) > 0


class TestManageUrbanRural:
    def test_manage_urban_rural_converts_dtw_1_to_kilometers(self):
        df_zones_input = pd.DataFrame(
            {"dtw_1": [1000, 2000, 3000], "GHS_SMOD": [10, 20, 30]}
        )
        expected_result = pd.DataFrame(
            {
                "dtw_1": [1.0, 2.0, 3.0],
                "GHS_SMOD": [10, 20, 30],
                "urban_rural": [0, 1, 1],
            }
        )
        result = manage_urban_rural(df_zones_input)
        print(result)
        print(expected_result)
        pd.testing.assert_frame_equal(result, expected_result)

    def test_manage_urban_rural_creates_urban_rural_column(self):
        df_zones_input = pd.DataFrame(
            {"dtw_1": [1000, 2000, 3000], "GHS_SMOD": [10, 20, 30]}
        )
        result = manage_urban_rural(df_zones_input)
        assert "urban_rural" in result.columns

    def test_manage_urban_rural_sets_urban_rural_to_1_for_urban_zones(self):
        df_zones_input = pd.DataFrame(
            {"dtw_1": [1000, 2000, 3000], "GHS_SMOD": [16, 20, 30]}
        )
        result = manage_urban_rural(df_zones_input)
        assert all(result["urban_rural"] == 1)

    def test_manage_urban_rural_sets_urban_rural_to_0_for_rural_zones(self):
        df_zones_input = pd.DataFrame(
            {"dtw_1": [1000, 2000, 3000], "GHS_SMOD": [0, 10, 15]}
        )
        result = manage_urban_rural(df_zones_input)
        assert all(result["urban_rural"] == 0)


class TestManageSlope:
    def test_manage_slope_modifies_dataframe(self):
        df_zones_input = pd.DataFrame({"slope_1": [1, 2, 3, 4, 5]})
        result = manage_slope(df_zones_input)
        assert "slope_1" in result.columns

    def test_manage_slope_returns_same_dataframe(self):
        df_zones_input = pd.DataFrame({"slope_1": [1, 2, 3, 4, 5]})
        result = manage_slope(df_zones_input)
        pd.testing.assert_frame_equal(result, df_zones_input)


class TestMergeAndAdjustPopulation:
    def test_merge_and_adjust_population_returns_dataframe(self):
        df_zones_input = pd.DataFrame(
            {"ISOCODE": ["USA", "CAN"], "pop_density": [1000, 2000]}
        )
        df_input = pd.DataFrame(
            {"alpha3": ["USA", "CAN"], "Population": [1000000, 2000000]}
        )
        result = merge_and_adjust_population(df_zones_input, df_input)
        assert isinstance(result, pd.DataFrame)

    def test_merge_and_adjust_population_returns_expected_result(self):
        df_zones_input = pd.DataFrame(
            {"ISOCODE": ["USA", "CAN"], "pop_density": [1000, 2000]}
        )
        df_input = pd.DataFrame(
            {"alpha3": ["USA", "CAN"], "Population": [1000000, 2000000]}
        )
        result = merge_and_adjust_population(df_zones_input, df_input)
        expected_result = pd.DataFrame(
            {
                "ISOCODE": ["USA", "CAN"],
                "pop_density": [1000, 2000],
                "alpha3": ["USA", "CAN"],
                "Population": [1000000, 2000000],
                "pop_density_perc": [1.0, 1.0],
                "pop_zone": [1000000.0, 2000000.0],
                "country_pop_raw": [1000000.0, 2000000.0],
                "any_pop": [1, 1],
            }
        )
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

    def test_crr_add_uncertainty_returns_highest_road_type_when_adjustment_exceeds_upper_bound(
        self,
    ):
        road_type = "Local Roads"
        adjustment = 2
        result = crr_add_uncertainty(road_type, adjustment)
        expected_result = "No Roads"
        assert result == expected_result

    def test_crr_add_uncertainty_returns_lowest_road_type_when_adjustment_exceeds_lower_bound(
        self,
    ):
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
        df_zones = pd.DataFrame(
            {
                "grip_1_1": [0, 0, 0],
                "grip_2_1": [0, 0, 0],
                "grip_3_1": [0, 0, 0],
                "grip_4_1": [0, 0, 0],
                "grip_5_1": [0, 0, 0],
            }
        )
        result = road_analysis(df_zones)
        assert isinstance(result, pd.DataFrame)

    def test_road_analysis_returns_expected_result(self):
        df_zones = pd.DataFrame(
            {
                "grip_1_1": [0, 1, 0],
                "grip_2_1": [0, 0, 1],
                "grip_3_1": [1, 0, 0],
                "grip_4_1": [0, 0, 0],
                "grip_5_1": [0, 0, 0],
            }
        )
        result = road_analysis(df_zones)
        expected_result = pd.DataFrame(
            {
                "grip_1_1": [0, 1, 0],
                "grip_2_1": [0, 0, 1],
                "grip_3_1": [1, 0, 0],
                "grip_4_1": [0, 0, 0],
                "grip_5_1": [0, 0, 0],
                "dominant_road_type": ["Secondary Roads", "Highways", "Primary Roads"],
                "Crr": [0.004, 0.002, 0.003],
            }
        )
        pd.testing.assert_frame_equal(result, expected_result)

    def test_road_analysis_handles_all_zeros(self):
        df_zones = pd.DataFrame(
            {
                "grip_1_1": [0, 0, 0],
                "grip_2_1": [0, 0, 0],
                "grip_3_1": [0, 0, 0],
                "grip_4_1": [0, 0, 0],
                "grip_5_1": [0, 0, 0],
            }
        )
        result = road_analysis(df_zones)
        assert "dominant_road_type" in result.columns
        assert "Crr" in result.columns

    def test_road_analysis_handles_crr_adjustment(self):
        df_zones = pd.DataFrame(
            {
                "grip_1_1": [0, 1, 0],
                "grip_2_1": [0, 0, 1],
                "grip_3_1": [1, 0, 0],
                "grip_4_1": [0, 0, 0],
                "grip_5_1": [0, 0, 0],
            }
        )
        result = road_analysis(df_zones, crr_adjustment=1)
        expected_result = pd.DataFrame(
            {
                "grip_1_1": [0, 1, 0],
                "grip_2_1": [0, 0, 1],
                "grip_3_1": [1, 0, 0],
                "grip_4_1": [0, 0, 0],
                "grip_5_1": [0, 0, 0],
                "dominant_road_type": ["Secondary Roads", "Highways", "Primary Roads"],
                "Crr": [0.006, 0.003, 0.004],
            }
        )
        pd.testing.assert_frame_equal(result, expected_result)


class TestLoadHPVParameters:
    def test_load_hpv_parameters_returns_dataframe(self):
        file_path_params = "./data/lookup tables/mobility-model-parameters.csv"
        hpv_name = "HPV1"
        result = load_hpv_parameters(file_path_params, hpv_name)
        assert isinstance(result, pd.DataFrame)


class TestExtractSlopeCrr:
    def test_extract_slope_crr_returns_tuple_of_series(self):
        df_zones = pd.DataFrame(
            {
                "slope_1": [1, 2, 3, 4, 5],
                "Crr": [0.1, 0.2, 0.3, 0.4, 0.5],
                "Average Weight": [60, 60, 60, 60, 60],
            }
        )
        result = extract_slope_crr(df_zones)
        assert isinstance(result, tuple)
        assert isinstance(result[0], pd.Series)
        assert isinstance(result[1], pd.Series)
        assert isinstance(result[2], pd.Series)

    def test_extract_slope_crr_returns_correct_series(self):
        df_zones = pd.DataFrame(
            {
                "slope_1": [1, 2, 3, 4, 5],
                "Crr": [0.1, 0.2, 0.3, 0.4, 0.5],
                "Average Weight": [60, 60, 60, 60, 60],
            }
        )
        slope_zones, Crr_values, country_average_weights = extract_slope_crr(df_zones)
        expected_slope_zones = pd.Series([1, 2, 3, 4, 5], name="slope_1")
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
        hpv = mm.HPV_variables(param_df, mv)
        slope_zones = [0, 1, 2]
        Crr_values = [0.01, 0.02, 0.03]
        country_average_weights = [60, 50, 70]
        load_attempt = 1
        result = run_bicycle_model(
            mv, mo, hpv, slope_zones, Crr_values, country_average_weights, load_attempt
        )
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
        hpv = mm.HPV_variables(param_df, mv)
        slope_zones = [0, 1, 2]
        Crr_values = [0.01, 0.02, 0.03]
        country_average_weights = [60, 50, 70]
        load_attempt = 1
        result = run_bicycle_model(
            mv, mo, hpv, slope_zones, Crr_values, country_average_weights, load_attempt
        )
        assert str(project_root) in sys.path
        assert "src.mobility_module" in sys.modules

    def test_run_bicycle_model_calls_single_bike_run_for_each_slope_zone_and_Crr_value(
        self, monkeypatch
    ):
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
        country_average_weights = [60, 50, 70]
        load_attempt = 1

        calls = []

        def mock_single_bike_run(mv, mo, hpv, slope, load_attempt):
            calls.append((slope, hpv.Crr))

        monkeypatch.setattr(
            "src.mobility_module.mobility_models.single_bike_run", mock_single_bike_run
        )

        run_bicycle_model(
            mv, mo, hpv, slope_zones, Crr_values, country_average_weights, load_attempt
        )

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
    updated_df = process_and_save_results(
        df_zones, results, export_file_location, velocity_type, save_csv=True
    )

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
    updated_df = process_and_save_results(
        df_zones, results, export_file_location, velocity_type, save_csv=False
    )

    # Check if the CSV file is not saved
    expected_csv_file = export_file_location / "walk_velocity_by_zone.csv"
    assert not expected_csv_file.exists()

    # Check if the DataFrame is updated with the new columns
    assert "loaded_velocity_walk" in updated_df.columns
    assert "unloaded_velocity_walk" in updated_df.columns
    assert "average_velocity_walk" in updated_df.columns
    assert "max_load_walk" in updated_df.columns


class TestCalculateAndMergeBicycleDistance:
    def test_calculate_and_merge_bicycle_distance_calculates_distance_when_flag_is_true(
        self,
    ):
        df_zones = pd.DataFrame(
            {
                "fid": [1, 2, 3],
                "zone": ["A", "B", "C"],
                "slope_1": [0, 1, 2],
                "Crr": [0.01, 0.02, 0.03],
                "Average Weight": [60, 50, 70],
            }
        )
        calculate_distance = True
        export_file_location = "./data/processed/"
        practical_limit_bicycle = 40

        result = calculate_and_merge_bicycle_distance(
            df_zones, calculate_distance, export_file_location, practical_limit_bicycle
        )

        assert "loaded_velocity_bicycle" in result.columns

    def test_calculate_and_merge_bicycle_distance_merges_bicycle_distance_from_file(
        self,
    ):
        df_zones = pd.DataFrame(
            {
                "fid": [1, 2, 3],
                "zone": ["A", "B", "C"],
                "slope_1": [0, 1, 2],
                "Crr": [0.01, 0.02, 0.03],
                "Average Weight": [60, 50, 70],
            }
        )
        calculate_distance = True
        export_file_location = "./data/processed/"
        practical_limit_bicycle = 40

        result = calculate_and_merge_bicycle_distance(
            df_zones, calculate_distance, export_file_location, practical_limit_bicycle
        )

        assert "loaded_velocity_bicycle" in result.columns
        assert result["average_velocity_bicycle"].notnull().all()


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
        met = 3.3
        hpv = mm.HPV_variables(param_df, mv)

        slope_zones = [0, 1, 2]
        country_average_weights = [60, 50, 70]

        result = run_walking_model(
            mv, mo, met, hpv, slope_zones, country_average_weights, load_attempt=20
        )
        assert isinstance(result, np.ndarray)

    def test_run_walking_model_calls_single_walk_run_for_each_slope_zone(
        self, monkeypatch
    ):
        project_root = Path().resolve().parent
        sys.path.append(str(project_root))
        import src.mobility_module as mm

        file_path_params = "./data/lookup tables/mobility-model-parameters.csv"
        param_df = load_hpv_parameters(file_path_params, "Walking")
        param_df["PracticalLimit"] = 20
        mo = mm.model_options()
        mo.model_selection = 3  # Lankford model
        mv = mm.model_variables()
        met = 3.3
        hpv = mm.HPV_variables(param_df, mv)
        slope_zones = [0, 1, 2]
        country_average_weights = [60, 50, 70]
        load_attempt = 1

        calls = []

        def mock_single_lankford_run(mv, mo, met, hpv, slope, load_attempt):
            calls.append(slope)

        monkeypatch.setattr(
            "src.mobility_module.mobility_models.single_lankford_run",
            mock_single_lankford_run,
        )

        run_walking_model(
            mv, mo, met, hpv, slope_zones, country_average_weights, load_attempt
        )

        expected_calls = [0, 1, 2]
        assert calls == expected_calls


class TestCalculateAndMergeWalkingDistance:
    def test_calculate_and_merge_walking_distance_calculates_distance_when_flag_is_true(
        self,
    ):
        df_zones = pd.DataFrame(
            {
                "fid": [1, 2, 3],
                "zone": ["A", "B", "C"],
                "slope_1": [0, 1, 2],
                "Crr": [0.01, 0.02, 0.03],
                "Average Weight": [60, 50, 70],
            }
        )
        calculate_distance = True
        export_file_location = "./data/processed/"
        practical_limit_walking = 20

        result = calculate_and_merge_walking_distance(
            df_zones, calculate_distance, export_file_location, practical_limit_walking
        )

        assert "loaded_velocity_walk" in result.columns

    def test_calculate_and_merge_walking_distance_merges_walking_distance_from_file(
        self,
    ):
        df_zones = pd.DataFrame({"fid": [1, 2, 3], "zone": ["A", "B", "C"]})
        calculate_distance = False
        export_file_location = "./data/processed/"
        practical_limit_walking = 20

        result = calculate_and_merge_walking_distance(
            df_zones, calculate_distance, export_file_location, practical_limit_walking
        )

        assert "loaded_velocity_walk" in result.columns
        assert result["average_velocity_walk"].notnull().all()


class TestCalculateMaxDistances:
    def test_calculate_max_distances_returns_dataframe(self):
        df_zones = pd.DataFrame(
            {
                "average_velocity_bicycle": [10, 15, 20],
                "average_velocity_walk": [5, 7, 10],
                "max_load_bicycle": [5, 10, 15],
            }
        )
        time_gathering_water = 2
        result = calculate_max_distances(df_zones, time_gathering_water)
        assert isinstance(result, pd.DataFrame)

    def test_calculate_max_distances_returns_expected_result(self):
        df_zones = pd.DataFrame(
            {
                "average_velocity_bicycle": [10, 15, 20],
                "average_velocity_walk": [5, 7, 10],
                "max_load_bicycle": [5, 10, 15],
            }
        )
        time_gathering_water = 6
        result = calculate_max_distances(df_zones, time_gathering_water)
        expected_result = pd.DataFrame(
            {
                "average_velocity_bicycle": [10, 15, 20],
                "average_velocity_walk": [5, 7, 10],
                "max_load_bicycle": [5, 10, 15],
                "max distance cycling": [30.0, 45.0, 60.0],
                "max distance walking": [15.0, 21.0, 30.0],
                "water_ration_kms": [150.0, 450.0, 900.0],
            }
        )
        pd.testing.assert_frame_equal(result, expected_result)


class TestCalculatePopulationWaterAccess:
    def test_calculate_population_water_access_returns_dataframe(self):
        df_zones = pd.DataFrame(
            {
                "pop_zone": [1000, 2000, 3000],
                "urban_rural": [0, 1, 1],
                "URBANPiped": [50, 60, 70],
                "RURALPiped": [30, 40, 50],
                "URBANNon-piped": [20, 30, 40],
                "RURALNon-piped": [10, 20, 30],
                "dtw_1": [100, 200, 300],
                "max distance cycling": [500, 600, 700],
                "max distance walking": [200, 300, 400],
                "PBO": [80, 90, 100],
            }
        )
        result = calculate_population_water_access(df_zones)
        assert isinstance(result, pd.DataFrame)

    def test_calculate_population_water_access_returns_expected_result(self):
        df_zones = pd.DataFrame(
            {
                "pop_zone": [1000, 2000, 3000],
                "urban_rural": [0, 1, 1],
                "URBANPiped": [50, 60, 70],
                "RURALPiped": [30, 40, 50],
                "URBANNon-piped": [20, 30, 40],
                "RURALNon-piped": [10, 20, 30],
                "dtw_1": [100, 200, 500],
                "max distance cycling": [500, 600, 700],
                "max distance walking": [200, 300, 400],
                "PBO": [80, 90, 100],
            }
        )
        result = calculate_population_water_access(df_zones)
        expected_result = pd.DataFrame(
            {
                "pop_zone": [1000, 2000, 3000],
                "urban_rural": [0, 1, 1],
                "URBANPiped": [50, 60, 70],
                "RURALPiped": [30, 40, 50],
                "URBANNon-piped": [20, 30, 40],
                "RURALNon-piped": [10, 20, 30],
                "dtw_1": [100, 200, 500],
                "max distance cycling": [500, 600, 700],
                "max distance walking": [200, 300, 400],
                "PBO": [80, 90, 100],
                "zone_pop_piped": [300.0, 1200.0, 2100.0],
                "zone_pop_unpiped": [700.0, 800.0, 900.0],
                "zone_cycling_okay": [1, 1, 1],
                "zone_walking_okay": [1, 1, 0],
                "fraction_of_zone_with_cycling_access": [0.8, 0.9, 1.0],
                "fraction_of_zone_with_walking_access": [1, 1, 0],
                "population_piped_with_cycling_access": [240.0, 1080.0, 2100.0],
                "population_piped_with_walking_access": [300.0, 1200.0, 0.0],
                "population_piped_with_access": [300.0, 1200.0, 2100.0],
                "population_piped_with_only_cycling_access": [0.0, 0.0, 2100.0],
                "zone_pop_with_water": [1000.0, 2000.0, 3000.0],
                "zone_pop_without_water": [0.0, 0.0, 0.0],
            }
        )
        print(result.columns)
        pd.testing.assert_frame_equal(result, expected_result)


class TestCalculateWaterRations:
    def test_calculate_water_rations_returns_dataframe(self):
        df_zones = pd.DataFrame(
            {
                "dtw_1": [1000, 2000, 3000],
                "water_ration_kms": [10, 20, 30],
                "pop_zone": [1000, 2000, 3000],
                "Household_Size": [2, 3, 4],
                "PBO": [50, 60, 70],
            }
        )
        result = calculate_water_rations(df_zones)
        assert isinstance(result, pd.DataFrame)

    def test_calculate_water_rations_calculates_rations(self):
        df_zones = pd.DataFrame(
            {
                "dtw_1": [1000, 2000, 3000],
                "water_ration_kms": [10, 20, 30],
                "pop_zone": [1000, 2000, 3000],
                "Household_Size": [2, 3, 4],
                "PBO": [50, 60, 70],
            }
        )
        result = calculate_water_rations(df_zones)
        expected_result = pd.DataFrame(
            {
                "dtw_1": [1000, 2000, 3000],
                "water_ration_kms": [10, 20, 30],
                "pop_zone": [1000, 2000, 3000],
                "Household_Size": [2, 3, 4],
                "PBO": [50, 60, 70],
                "water_rations_per_bike": [0.01, 0.01, 0.01],
                "bikes_in_zone": [250.0, 400.0, 525.0],
                "water_rations_achievable": [2.50, 4.0, 5.25],
            }
        )
        pd.testing.assert_frame_equal(result, expected_result)


class TestAggregateCountryLevelData:
    def test_aggregate_country_level_data_returns_dataframe(self):
        df_zones = pd.DataFrame(
            {
                "ISOCODE": ["USA", "CAN"],
                "Entity": ["United States", "Canada"],
                "country_pop_raw": [1000000, 2000000],
                "zone_pop_with_water": [500000, 1000000],
                "zone_pop_without_water": [500000, 1000000],
                "population_piped_with_access": [300000, 400000],
                "population_piped_with_cycling_access": [200000, 300000],
                "population_piped_with_walking_access": [100000, 200000],
                "population_piped_with_only_cycling_access": [150000, 200000],
                "NATPiped": ["Yes", "No"],
                "region": ["North America", "North America"],
                "subregion": ["Northern America", "Northern America"],
                "max distance cycling": [20, 30],
                "max distance walking": [5, 10],
            }
        )

        result = aggregate_country_level_data(df_zones)

        assert isinstance(result, pd.DataFrame)

    def test_aggregate_country_level_data_returns_expected_result(self):
        df_zones = pd.DataFrame(
            {
                "ISOCODE": ["USA", "USA", "CAN", "CAN"],
                "Entity": ["United States", "United States", "Canada", "Canada"],
                "country_pop_raw": [1000000, 1000000, 2000000, 2000000],
                "zone_pop_with_water": [250000, 250000, 500000, 500000],
                "zone_pop_without_water": [250000, 250000, 500000, 500000],
                "population_piped_with_access": [150000, 150000, 200000, 200000],
                "population_piped_with_cycling_access": [
                    100000,
                    100000,
                    150000,
                    150000,
                ],
                "population_piped_with_walking_access": [50000, 50000, 100000, 100000],
                "population_piped_with_only_cycling_access": [
                    50000,
                    50000,
                    75000,
                    75000,
                ],
                "NATPiped": ["Yes", "Yes", "No", "No"],
                "region": [
                    "North America",
                    "North America",
                    "North America",
                    "North America",
                ],
                "subregion": [
                    "Northern America",
                    "Northern America",
                    "Northern America",
                    "Northern America",
                ],
                "max distance cycling": [20, 20, 30, 30],
                "max distance walking": [5, 5, 10, 10],
            }
        )

        expected_result = pd.DataFrame(
            {
                "ISOCODE": ["CAN", "USA"],
                "Entity": ["Canada", "United States"],
                "country_pop_raw": [2000000, 1000000],
                "zone_pop_with_water": [1000000, 500000],
                "zone_pop_without_water": [1000000, 500000],
                "population_piped_with_access": [400000, 300000],
                "population_piped_with_cycling_access": [300000, 200000],
                "population_piped_with_walking_access": [200000, 100000],
                "population_piped_with_only_cycling_access": [150000, 100000],
                "NATPiped": ["No", "Yes"],
                "region": ["North America", "North America"],
                "subregion": ["Northern America", "Northern America"],
            }
        )

        result = aggregate_country_level_data(df_zones)
        pd.testing.assert_frame_equal(result, expected_result)


class TestCleanUpData:
    def test_clean_up_data_removes_nan_rows(self):
        df_countries = pd.DataFrame(
            {"ISOCODE": ["USA", "CAN", "LBY"], "weighted_med": [1.0, None, 3.0]}
        )
        cleaned_df, _, _ = clean_up_data(df_countries)
        assert len(cleaned_df) == 2
        assert cleaned_df["ISOCODE"].tolist() == ["USA", "LBY"]
        assert cleaned_df["weighted_med"].tolist() == [1.0, 3.0]

    def test_clean_up_data_removes_outliers(self):
        df_countries = pd.DataFrame(
            {"ISOCODE": ["USA", "CAN", "LBY"], "weighted_med": [1.0, 10.0, 3.0]}
        )
        cleaned_df, _, _ = clean_up_data(df_countries)
        assert len(cleaned_df) == 2
        assert cleaned_df["ISOCODE"].tolist() == ["USA", "LBY"]
        assert cleaned_df["weighted_med"].tolist() == [1.0, 3.0]

    def test_clean_up_data_removes_specific_countries(self):
        df_countries = pd.DataFrame(
            {
                "ISOCODE": ["USA", "ATG", "GUM", "LBY"],
                "weighted_med": [1.0, 2.0, 3.0, 4.0],
            }
        )
        cleaned_df, _, _ = clean_up_data(df_countries)
        assert len(cleaned_df) == 2
        assert cleaned_df["ISOCODE"].tolist() == ["USA", "LBY"]
        assert cleaned_df["weighted_med"].tolist() == [1.0, 4.0]


class TestProcessCountryData:
    def test_process_country_data_returns_dataframe(self):
        df_zones = pd.DataFrame(
            {
                "ISOCODE": ["USA", "CAN", "LBY"],
                "pop_zone": [1000, 2000, 1500],
                "dtw_1": [10, 20, 150],
                "Entity": ["USA", "CAN", "Libya"],
                "NATPiped": [1000, 2000, 1500],
                "country_pop_raw": [1000, 2000, 1500],
                "population_piped_with_access": [1000, 2000, 0],
                "population_piped_with_cycling_access": [1000, 2000, 0],
                "population_piped_with_walking_access": [1000, 2000, 0],
                "population_piped_with_only_cycling_access": [0, 0, 0],
                "region": ["North America", "North America", "Africa"],
                "subregion": [
                    "Northern America",
                    "Northern America",
                    "Northern Africa",
                ],
                "zone_pop_with_water": [1000, 2000, 0],
                "zone_pop_without_water": [0, 0, 1500],
                "max distance cycling": [10, 20, 30],
                "max distance walking": [5, 10, 15],
            }
        )
        result = process_country_data(df_zones)
        assert isinstance(result, pd.DataFrame)

    def test_process_country_data_returns_expected_result(self):
        df_zones = pd.DataFrame(
            {
                "ISOCODE": ["USA", "CAN", "LBY"],
                "pop_zone": [1000, 2000, 1500],
                "dtw_1": [10, 20, 150],
                "Entity": ["USA", "Canada", "Libya"],
                "NATPiped": [1000, 2000, 1500],
                "country_pop_raw": [1000, 2000, 1500],
                "population_piped_with_access": [1000, 2000, 0],
                "population_piped_with_cycling_access": [1000, 2000, 0],
                "population_piped_with_walking_access": [1000, 2000, 0],
                "population_piped_with_only_cycling_access": [0, 0, 0],
                "region": ["North America", "North America", "Africa"],
                "subregion": [
                    "Northern America",
                    "Northern America",
                    "Northern Africa",
                ],
                "zone_pop_with_water": [1000, 2000, 0],
                "zone_pop_without_water": [0, 0, 1500],
                "max distance cycling": [10, 20, 30],
                "max distance walking": [5, 10, 15],
            }
        )
        result = process_country_data(df_zones)
        expected_result = pd.DataFrame(
            {
                "ISOCODE": ["CAN", "LBY", "USA", "GLOBAL"],
                "Entity": ["Canada", "Libya", "USA", "Global"],
                "country_pop_raw": [2000, 1500, 1000, 4500],
                "country_pop_with_water": [2000, 0, 1000, 3000],
                "country_pop_without_water": [0, 1500, 0, 1500],
                "population_piped_with_access": [2000, 0, 1000, 3000],
                "population_piped_with_cycling_access": [2000, 0, 1000, 3000],
                "population_piped_with_walking_access": [2000, 0, 1000, 3000],
                "population_piped_with_only_cycling_access": [0, 0, 0, 0],
                "NATPiped": [2000.0, 1500.0, 1000.0, np.nan],
                "region": ["North America", "Africa", "North America", None],
                "subregion": [
                    "Northern America",
                    "Northern Africa",
                    "Northern America",
                    np.nan,
                ],
                "weighted_med": [20, 150, 10, 20],
                "weighted_med_cycling": [20, 30, 10, 20],
                "weighted_5th_cycling": [20, 30, 10, 10],
                "weighted_95th_cycling": [20, 30, 10, 30],
                "weighted_med_walking": [10, 15, 5, 10],
                "weighted_5th_walking": [10, 15, 5, 5],
                "weighted_95th_walking": [10, 15, 5, 15],
                "percent_with_water": [100.0, 0.0, 100.0, (2 / 3) * 100],
                "percent_without_water": [0.0, 100.0, 0.0, (1 / 3) * 100],
                "percent_piped_with_cycling_access": [100.0, 0.0, 100.0, (2 / 3) * 100],
                "percent_piped_with_walking_access": [100.0, 0.0, 100.0, (2 / 3) * 100],
                "proportion_piped_access_from_cycling": [0.0, np.nan, 0.0, 0.0],
                "percent_with_only_cycling_access": [0.0, 0.0, 0.0, 0.0],
            },
            index=pd.Int64Index([0, 1, 2, 3], dtype="int64"),
        )
        # find all differences between result and expected result
        diff = result.compare(expected_result)
        print(diff)

        pd.testing.assert_frame_equal(result, expected_result)

    def test_process_country_data_raises_assertion_error_when_df_zones_is_empty(self):
        df_zones = pd.DataFrame()
        with pytest.raises(AssertionError):
            process_country_data(df_zones)

    def test_process_country_data_issues_warning_when_df_zones_contains_nan_values(
        self, capsys
    ):
        df_zones = pd.DataFrame(
            {
                "ISOCODE": ["USA", "CAN", "LBY"],
                "pop_zone": [1000, 2000, 1500],
                "dtw_1": [10, 20, 15],
                "Entity": ["USA", "CAN", "Libya"],
                "NATPiped": [1000, 2000, 1500],
                "country_pop_raw": [1000, 2000, 1500],
                "population_piped_with_access": [1000, 2000, 1500],
                "population_piped_with_cycling_access": [1000, 2000, 1500],
                "population_piped_with_walking_access": [1000, 2000, 1500],
                "population_piped_with_only_cycling_access": [0, 0, 0],
                "region": ["North America", "North America", "Africa"],
                "subregion": [
                    "Northern America",
                    "Northern America",
                    "Northern Africa",
                ],
                "zone_pop_with_water": [1000, 2000, 1500],
                "zone_pop_without_water": [0, 0, 0],
                "max distance cycling": [10, 20, 30],
                "max distance walking": [5, 10, 15],
            }
        )
        with pytest.raises(AssertionError):
            process_country_data(df_zones)
            captured = capsys.readouterr()
            assert "Input dataframe contains NaN values" in captured.err

    def test_process_country_data_raises_error_when_df_countries_is_empty(self):
        df_zones = pd.DataFrame()
        with pytest.raises(AssertionError):
            process_country_data(df_zones)

    def test_process_country_data_raises_index_error_when_df_countries_contains_nan_values(
        self,
    ):
        df_zones = pd.DataFrame(
            {
                "ISOCODE": ["USA", "CAN", "LBY"],
                "pop_zone": [1000, np.nan, 1500],
                "dtw_1": [10, 20, 15],
                "Entity": ["USA", "CAN", "Libya"],
                "NATPiped": [1000, 2000, 1500],
                "country_pop_raw": [1000, 2000, 1500],
                "population_piped_with_access": [1000, 2000, 1500],
                "population_piped_with_cycling_access": [1000, 2000, 1500],
                "population_piped_with_walking_access": [1000, 2000, 1500],
                "population_piped_with_only_cycling_access": [0, 0, 0],
                "region": ["North America", "North America", "Africa"],
                "subregion": [
                    "Northern America",
                    "Northern America",
                    "Northern Africa",
                ],
                "zone_pop_with_water": [1000, 2000, 1500],
                "zone_pop_without_water": [0, 0, 0],
                "max distance cycling": [10, 20, 30],
                "max distance walking": [5, 10, 15],
            }
        )
        with pytest.raises(IndexError):
            process_country_data(df_zones)

    def test_process_country_data_renames_and_creates_percentage_columns(self):
        df_zones = pd.DataFrame(
            {
                "ISOCODE": ["USA", "CAN", "LBY"],
                "pop_zone": [1000, 2000, 1500],
                "dtw_1": [10, 20, 15],
                "Entity": ["USA", "CAN", "Libya"],
                "NATPiped": [1000, 2000, 1500],
                "country_pop_raw": [1000, 2000, 1500],
                "population_piped_with_access": [1000, 2000, 1500],
                "population_piped_with_cycling_access": [1000, 2000, 1500],
                "population_piped_with_walking_access": [1000, 2000, 1500],
                "population_piped_with_only_cycling_access": [0, 0, 0],
                "region": ["North America", "North America", "Africa"],
                "subregion": [
                    "Northern America",
                    "Northern America",
                    "Northern Africa",
                ],
                "zone_pop_with_water": [1000, 2000, 1500],
                "zone_pop_without_water": [0, 0, 0],
                "max distance cycling": [10, 20, 30],
                "max distance walking": [5, 10, 15],
            }
        )
        result = process_country_data(df_zones)
        assert "percent_with_water" in result.columns
        assert "percent_without_water" in result.columns

    def test_process_country_data_prints_summary_of_removed_countries(self, capsys):
        df_zones = pd.DataFrame(
            {
                "ISOCODE": ["USA", "CAN", "LBY"],
                "pop_zone": [1000, 2000, 1500],
                "dtw_1": [10, 20, 15],
                "Entity": ["USA", "CAN", "Libya"],
                "NATPiped": [1000, 2000, 1500],
                "country_pop_raw": [1000, 2000, 1500],
                "population_piped_with_access": [1000, 2000, 1500],
                "population_piped_with_cycling_access": [1000, 2000, 1500],
                "population_piped_with_walking_access": [1000, 2000, 1500],
                "population_piped_with_only_cycling_access": [0, 0, 0],
                "region": ["North America", "North America", "Africa"],
                "subregion": [
                    "Northern America",
                    "Northern America",
                    "Northern Africa",
                ],
                "zone_pop_with_water": [1000, 2000, 1500],
                "zone_pop_without_water": [0, 0, 0],
                "max distance cycling": [10, 20, 30],
                "max distance walking": [5, 10, 15],
            }
        )
        process_country_data(df_zones)
        captured = capsys.readouterr()
        assert (
            "Countries removed from analysis due to being further than Libya's median:"
            in captured.out
        )
        assert "Countries removed manually:" in captured.out


class TestMapHillPolarity:
    def test_map_hill_polarity_returns_correct_result(self):
        hill_polarity = "uphill_downhill"
        result = map_hill_polarity(hill_polarity)
        expected_result = (1, -1)
        assert result == expected_result

    def test_map_hill_polarity_returns_correct_result_for_downhill_uphill(self):
        hill_polarity = "downhill_uphill"
        result = map_hill_polarity(hill_polarity)
        expected_result = (-1, 1)
        assert result == expected_result

    def test_map_hill_polarity_returns_correct_result_for_uphill_flat(self):
        hill_polarity = "uphill_flat"
        result = map_hill_polarity(hill_polarity)
        expected_result = (1, 0)
        assert result == expected_result

    def test_map_hill_polarity_returns_correct_result_for_flat_uphill(self):
        hill_polarity = "flat_uphill"
        result = map_hill_polarity(hill_polarity)
        expected_result = (0, 1)
        assert result == expected_result

    def test_map_hill_polarity_returns_correct_result_for_downhill_flat(self):
        hill_polarity = "downhill_flat"
        result = map_hill_polarity(hill_polarity)
        expected_result = (-1, 0)
        assert result == expected_result

    def test_map_hill_polarity_returns_correct_result_for_flat_downhill(self):
        hill_polarity = "flat_downhill"
        result = map_hill_polarity(hill_polarity)
        expected_result = (0, -1)
        assert result == expected_result


class TestAdjustEuclidean:
    def test_adjust_euclidean_returns_correct_result(self):
        df_zones_input = pd.DataFrame(
            {"dtw_1": [1000, 2000, 3000], "urban_rural": [1, 0, 1]}
        )
        expected_result = pd.DataFrame(
            {"dtw_1": [4000, 6000, 12000], "urban_rural": [1, 0, 1]}
        )

        result = adjust_euclidean(
            df_zones_input, urban_adjustment=4, rural_adjustment=3
        )

        pd.testing.assert_frame_equal(result, expected_result)

    def test_adjust_euclidean_raises_error_when_urban_rural_is_non_binary(self):
        df_zones_input = pd.DataFrame(
            {"dtw_1": [1000, 2000, 3000], "urban_rural": [1, 2, 1]}
        )

        with pytest.raises(ValueError):
            adjust_euclidean(df_zones_input, urban_adjustment=4, rural_adjustment=3)


# Add test cases for run_global_analysis
class TestRunGlobalAnalysis:
    def test_run_global_analysis_null_run(self):
        # If time gathering water is 0, percentage of population without water access
        # should be equal to percentage piped
        df_countries, df_districts, df_zones = run_global_analysis(
            crr_adjustment=0,
            time_gathering_water=0,
            practical_limit_bicycle=40,
            practical_limit_buckets=20,
            met=4.5,
            watts=75,
            hill_polarity="flat_uphill",
            urban_adjustment=1.345,
            rural_adjustment=1.2,
            calculate_distance=True,
            plot=False,
            human_mass=62,  # gets overridden by country specific weight
            use_sample_data=True,
        )

        # drop global row
        df_countries = df_countries[df_countries["ISOCODE"] != "GLOBAL"]

        # calculate MAPE between percent_without_water and NATPiped
        # Extract the true and predicted values
        y_true = df_countries["NATPiped"]
        y_pred = df_countries["percent_without_water"]
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

        # As we are using national urban and rural piped percentages, we gain better accuracy for the zonal
        # results at the expense of the country level results. Country MAPE should average out at 19% globally
        # when using *national* piped data
        assert mape < 20.0

        df_countries["pop_unpiped"] = (
            df_countries["country_pop_with_water"]
            - df_countries["population_piped_with_access"]
        )
        y_calc = (
            (df_countries["country_pop_raw"] - df_countries["pop_unpiped"])
            / df_countries["country_pop_raw"]
            * 100
        )
        y_calc = np.array(y_calc)
        mae_calc = np.mean(np.abs(y_calc - y_pred))

        # This should be approx zero, as no areas modelled as piped should receive any water
        assert mae_calc < 0.01


class TestCalculateWeightedResults:
    def test_calculate_weighted_results_returns_dataframe(self):
        df_zones = pd.DataFrame(
            {
                "ISOCODE": ["USA", "USA", "CAN", "CAN", "CAN", "CAN"],
                "dtw_1": [1000, 2000, 3000, 4000, 5000, 6000],
                "pop_zone": [1000, 2000, 3000, 4000, 5000, 6000],
                "max distance cycling": [10, 20, 30, 40, 50, 60],
                "max distance walking": [5, 10, 15, 20, 25, 30],
            }
        )
        result = calculate_weighted_results(df_zones)
        assert isinstance(result, pd.DataFrame)

    def test_calculate_weighted_results_returns_expected_result(self):
        df_zones = pd.DataFrame(
            {
                "ISOCODE": ["USA", "USA", "CAN", "CAN", "CAN", "CAN"],
                "dtw_1": [1000, 2000, 3000, 4000, 5000, 6000],
                "pop_zone": [1000, 2000, 3000, 4000, 5000, 6000],
                "max distance cycling": [10, 20, 30, 40, 50, 60],
                "max distance walking": [5, 10, 15, 20, 25, 30],
            }
        )
        result = calculate_weighted_results(df_zones)
        expected_result = pd.DataFrame(
            {
                "ISOCODE": ["CAN", "USA"],
                "weighted_med": [5000, 2000],
                "weighted_med_cycling": [50, 20],
                "weighted_5th_cycling": [30, 10],
                "weighted_95th_cycling": [60, 20],
                "weighted_med_walking": [25, 10],
                "weighted_5th_walking": [15, 5],
                "weighted_95th_walking": [30, 10],
            }
        )
        pd.testing.assert_frame_equal(result, expected_result)

    def test_calculate_weighted_results_handles_empty_dataframe(self):
        df_zones = pd.DataFrame(
            {
                "ISOCODE": [],
                "dtw_1": [],
                "pop_zone": [],
                "max distance cycling": [],
                "max distance walking": [],
            }
        )
        with pytest.raises(ValueError):
            result = calculate_weighted_results(df_zones)

    def test_calculate_weighted_results_handles_single_country(self):
        df_zones = pd.DataFrame(
            {
                "ISOCODE": ["USA", "USA", "USA", "USA", "USA"],
                "dtw_1": [1000, 2000, 3000, 4000, 5000],
                "pop_zone": [1000, 2000, 3000, 4000, 5000],
                "max distance cycling": [10, 20, 30, 40, 50],
                "max distance walking": [5, 10, 15, 20, 25],
            }
        )
        result = calculate_weighted_results(df_zones)
        expected_result = pd.DataFrame(
            {
                "ISOCODE": ["USA"],
                "weighted_med": [4000],
                "weighted_med_cycling": [40],
                "weighted_5th_cycling": [10],
                "weighted_95th_cycling": [50],
                "weighted_med_walking": [20],
                "weighted_5th_walking": [5],
                "weighted_95th_walking": [25],
            }
        )
        pd.testing.assert_frame_equal(result, expected_result)


if __name__ == "__main__":
    TestRunGlobalAnalysis().test_run_global_analysis_null_run()
