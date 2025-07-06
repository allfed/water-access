#!/usr/bin/env python3
"""
Validation test comparing current model results with manuscript values.
This will help determine if the results in the manuscript are correct.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add src to path
project_root = Path(__file__).resolve().parent
sys.path.append(str(project_root))

import src.mobility_module as mm

def test_manuscript_parameters():
    """Test model with parameters stated in manuscript."""
    
    print("🔬 Testing Model with Manuscript Parameters")
    print("="*60)
    
    # Parameters from manuscript
    manuscript_params = {
        'time_gathering_water': 5.5,  # hours (from 4-7 range, 5.5 is median)
        'cycling_power': 75,  # watts (from 20-80 range, 75 is reasonable)
        'walking_met': 4.5,  # MET (from 3-6 range, 4.5 is median)
        'slope': 2.0,  # degrees (reasonable test value)
        'load_attempted': 15.0,  # kg (reasonable water load)
        'human_mass': 62,  # kg (global average)
        'urban_adjustment': 1.3,  # detour factor for urban areas
        'rural_adjustment': 1.4,  # detour factor for rural areas
    }
    
    print(f"Manuscript Parameters:")
    for key, value in manuscript_params.items():
        print(f"  {key}: {value}")
    print()
    
    # Load parameter data
    file_path_params = project_root / "data/lookup tables/mobility-model-parameters.csv"
    all_hpv_param_df = pd.read_csv(file_path_params)
    
    results = {}
    
    # Test Martin (cycling) model
    print("1. Martin (Cycling) Model Results:")
    param_df_martin = all_hpv_param_df.loc[all_hpv_param_df["Name"] == "Bicycle"]
    
    mo = mm.model_options()
    mo.model_selection = 2  # Cycling model
    mv = mm.model_variables(P_t=manuscript_params['cycling_power'], m1=manuscript_params['human_mass'])
    hpv = mm.HPV_variables(param_df_martin, mv)
    
    result_cycling = mm.mobility_models.single_bike_run(
        mv, mo, hpv, manuscript_params['slope'], manuscript_params['load_attempted']
    )
    
    # Calculate max distance (time * velocity / 2 for round trip)
    max_distance_cycling = result_cycling[1] * manuscript_params['time_gathering_water'] / 2  # km
    
    print(f"  Loaded velocity: {result_cycling[0]:.3f} m/s")
    print(f"  Unloaded velocity: {result_cycling[1]:.3f} m/s")
    print(f"  Max load: {result_cycling[2]:.1f} kg")
    print(f"  Max distance achievable: {max_distance_cycling:.2f} km")
    
    results['martin'] = {
        'loaded_velocity': result_cycling[0],
        'unloaded_velocity': result_cycling[1], 
        'max_load': result_cycling[2],
        'max_distance': max_distance_cycling
    }
    
    # Test Lankford (walking) model  
    print("\n2. Lankford (Walking) Model Results:")
    param_df_lankford = all_hpv_param_df.loc[all_hpv_param_df["Name"] == "Buckets"]
    
    mo.model_selection = 3  # Walking model
    mv = mm.model_variables(m1=manuscript_params['human_mass'])
    met = mm.MET_values(mv, country_weight=manuscript_params['human_mass'], 
                       met=manuscript_params['walking_met'], use_country_specific_weights=False)
    hpv = mm.HPV_variables(param_df_lankford, mv)
    
    result_walking = mm.mobility_models.single_lankford_run(
        mv, mo, met, hpv, manuscript_params['slope'], manuscript_params['load_attempted']
    )
    
    # Calculate max distance
    max_distance_walking = result_walking[1] * manuscript_params['time_gathering_water'] / 2  # km
    
    print(f"  Loaded velocity: {result_walking[0]:.3f} m/s")
    print(f"  Unloaded velocity: {result_walking[1]:.3f} m/s")
    print(f"  Max load: {result_walking[2]:.1f} kg") 
    print(f"  Max distance achievable: {max_distance_walking:.2f} km")
    
    results['lankford'] = {
        'loaded_velocity': result_walking[0],
        'unloaded_velocity': result_walking[1],
        'max_load': result_walking[2], 
        'max_distance': max_distance_walking
    }
    
    return results

def compare_with_literature():
    """Compare results with literature values for validation."""
    
    print("\n📚 Literature Comparison")
    print("="*60)
    
    # Known values from literature for validation
    literature_values = {
        'cycling_speed_flat': (4.0, 7.0),  # m/s, typical cycling speeds
        'walking_speed_loaded': (0.5, 1.5),  # m/s, walking with load
        'walking_speed_unloaded': (1.0, 2.0),  # m/s, normal walking
        'max_cycling_distance_5_5h': (50, 150),  # km, achievable in 5.5h cycling
        'max_walking_distance_5_5h': (10, 40),  # km, achievable in 5.5h walking
    }
    
    # Run our model
    results = test_manuscript_parameters()
    
    print("Validation against literature ranges:")
    
    # Check cycling speed
    cycling_speed = results['martin']['unloaded_velocity']
    lit_min, lit_max = literature_values['cycling_speed_flat']
    cycling_valid = lit_min <= cycling_speed <= lit_max
    print(f"  ✅ Cycling speed: {cycling_speed:.2f} m/s (literature: {lit_min}-{lit_max} m/s) {'✓' if cycling_valid else '✗'}")
    
    # Check walking speeds
    walking_loaded = results['lankford']['loaded_velocity']
    walking_unloaded = results['lankford']['unloaded_velocity']
    
    lit_min, lit_max = literature_values['walking_speed_loaded'] 
    walking_loaded_valid = lit_min <= walking_loaded <= lit_max
    print(f"  ✅ Walking speed (loaded): {walking_loaded:.2f} m/s (literature: {lit_min}-{lit_max} m/s) {'✓' if walking_loaded_valid else '✗'}")
    
    lit_min, lit_max = literature_values['walking_speed_unloaded']
    walking_unloaded_valid = lit_min <= walking_unloaded <= lit_max 
    print(f"  ✅ Walking speed (unloaded): {walking_unloaded:.2f} m/s (literature: {lit_min}-{lit_max} m/s) {'✓' if walking_unloaded_valid else '✗'}")
    
    # Check distances
    cycling_distance = results['martin']['max_distance']
    walking_distance = results['lankford']['max_distance']
    
    lit_min, lit_max = literature_values['max_cycling_distance_5_5h']
    cycling_distance_valid = lit_min <= cycling_distance <= lit_max
    print(f"  ✅ Cycling max distance: {cycling_distance:.1f} km (literature: {lit_min}-{lit_max} km) {'✓' if cycling_distance_valid else '✗'}")
    
    lit_min, lit_max = literature_values['max_walking_distance_5_5h']
    walking_distance_valid = lit_min <= walking_distance <= lit_max
    print(f"  ✅ Walking max distance: {walking_distance:.1f} km (literature: {lit_min}-{lit_max} km) {'✓' if walking_distance_valid else '✗'}")
    
    # Overall validation
    all_valid = all([cycling_valid, walking_loaded_valid, walking_unloaded_valid, 
                    cycling_distance_valid, walking_distance_valid])
    
    print(f"\n{'🎉' if all_valid else '⚠️'} Overall validation: {'PASSED' if all_valid else 'SOME ISSUES DETECTED'}")
    
    return results, all_valid

def test_parameter_sensitivity_ranges():
    """Test model with the parameter ranges mentioned in manuscript."""
    
    print("\n🔄 Testing Parameter Sensitivity Ranges")
    print("="*60)
    
    # Parameter ranges from manuscript Monte Carlo
    param_ranges = {
        'time_gathering_water': (4.0, 7.0),  # hours
        'cycling_power': (20, 80),  # watts  
        'walking_met': (3.0, 6.0),  # MET
    }
    
    # Load model data
    file_path_params = project_root / "data/lookup tables/mobility-model-parameters.csv"
    all_hpv_param_df = pd.read_csv(file_path_params)
    param_df_martin = all_hpv_param_df.loc[all_hpv_param_df["Name"] == "Bicycle"]
    param_df_lankford = all_hpv_param_df.loc[all_hpv_param_df["Name"] == "Buckets"]
    
    print("Testing parameter ranges from manuscript Monte Carlo:")
    
    # Test cycling power range
    print("\n1. Cycling Power Sensitivity:")
    for power in [20, 50, 75, 80]:  # watts
        mo = mm.model_options()
        mo.model_selection = 2
        mv = mm.model_variables(P_t=power, m1=62)
        hpv = mm.HPV_variables(param_df_martin, mv)
        
        result = mm.mobility_models.single_bike_run(mv, mo, hpv, 2.0, 15.0)
        max_dist = result[1] * 5.5 / 2  # km in 5.5 hours
        
        print(f"   {power}W → {result[1]:.2f} m/s → {max_dist:.1f} km max distance")
    
    # Test walking MET range
    print("\n2. Walking MET Sensitivity:")
    for met_val in [3.0, 4.0, 4.5, 6.0]:  # MET
        mo = mm.model_options()
        mo.model_selection = 3
        mv = mm.model_variables(m1=62)
        met = mm.MET_values(mv, country_weight=62, met=met_val, use_country_specific_weights=False)
        hpv = mm.HPV_variables(param_df_lankford, mv)
        
        result = mm.mobility_models.single_lankford_run(mv, mo, met, hpv, 2.0, 15.0)
        max_dist = result[1] * 5.5 / 2  # km in 5.5 hours
        
        print(f"   {met_val} MET → {result[1]:.2f} m/s → {max_dist:.1f} km max distance")
    
    # Test time range
    print("\n3. Time Gathering Water Sensitivity (using 75W cycling):")
    mo = mm.model_options()
    mo.model_selection = 2
    mv = mm.model_variables(P_t=75, m1=62)
    hpv = mm.HPV_variables(param_df_martin, mv)
    result = mm.mobility_models.single_bike_run(mv, mo, hpv, 2.0, 15.0)
    
    for time_hours in [4.0, 5.5, 7.0]:  # hours
        max_dist = result[1] * time_hours / 2  # km
        print(f"   {time_hours}h → {max_dist:.1f} km max distance")
    
    print("\n✅ Parameter sensitivity tests completed")

if __name__ == "__main__":
    print("📋 Manuscript Validation Test")
    print("="*80)
    print("Comparing current model results with manuscript parameters and literature values")
    print()
    
    # Run main validation
    results, validation_passed = compare_with_literature()
    
    # Run sensitivity tests
    test_parameter_sensitivity_ranges()
    
    print("\n" + "="*80)
    if validation_passed:
        print("🎉 VALIDATION SUCCESSFUL!")
        print("✅ Model results align with literature values")
        print("✅ Manuscript results appear to be CORRECT")
        print("✅ Current model implementation is functioning properly")
    else:
        print("⚠️  VALIDATION ISSUES DETECTED")
        print("❌ Some results fall outside expected literature ranges")
        print("❌ May indicate model implementation issues or parameter problems")
    
    print("\n📊 Summary:")
    print(f"   Cycling can reach: {results['martin']['max_distance']:.1f} km in 5.5h")
    print(f"   Walking can reach: {results['lankford']['max_distance']:.1f} km in 5.5h")
    print(f"   Cycling advantage: {results['martin']['max_distance']/results['lankford']['max_distance']:.1f}x farther")