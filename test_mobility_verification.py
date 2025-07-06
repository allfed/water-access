#!/usr/bin/env python3
"""
Quick verification that mobility functions work with real objects.
This tests the same functions that sensitivity analysis uses.
"""

import sys
from pathlib import Path

# Add src to path
project_root = Path(__file__).resolve().parent
sys.path.append(str(project_root))

import src.mobility_module as mm
import pandas as pd

def test_mobility_functions():
    """Test mobility functions with real objects like sensitivity analysis does."""
    
    print("Testing mobility functions with real objects...")
    
    # Load real parameter data (like sensitivity analysis does)
    file_path_params = project_root / "data/lookup tables/mobility-model-parameters.csv"
    all_hpv_param_df = pd.read_csv(file_path_params)
    
    # Test Lankford (walking) model
    print("\n1. Testing Lankford (walking) model:")
    param_df_lankford = all_hpv_param_df.loc[all_hpv_param_df["Name"] == "Buckets"]
    
    # Initialize real model components
    mo = mm.model_options()
    mo.model_selection = 3  # Lankford model
    
    mv = mm.model_variables()
    mv.m1 = 62  # Human mass
    
    met = mm.MET_values(mv, country_weight=60, met=3.5, use_country_specific_weights=False)
    hpv = mm.HPV_variables(param_df_lankford, mv)
    
    # Test single_lankford_run
    try:
        slope = 0.0  # Flat terrain
        load_attempted = 15.0  # kg
        
        result = mm.mobility_models.single_lankford_run(
            mv, mo, met, hpv, slope, load_attempted
        )
        
        loaded_velocity, unloaded_velocity, max_load = result
        print(f"  ✅ Lankford run successful:")
        print(f"     Loaded velocity: {loaded_velocity:.3f} m/s")
        print(f"     Unloaded velocity: {unloaded_velocity:.3f} m/s") 
        print(f"     Max load: {max_load:.1f} kg")
        
    except Exception as e:
        print(f"  ❌ Lankford run failed: {e}")
        return False
    
    # Test Martin (cycling) model
    print("\n2. Testing Martin (cycling) model:")
    param_df_martin = all_hpv_param_df.loc[all_hpv_param_df["Name"] == "Bicycle"]
    
    mo.model_selection = 2  # Cycling model
    hpv = mm.HPV_variables(param_df_martin, mv)
    
    # Test single_bike_run
    try:
        result = mm.mobility_models.single_bike_run(
            mv, mo, hpv, slope, load_attempted
        )
        
        loaded_velocity, unloaded_velocity, max_load = result
        print(f"  ✅ Martin run successful:")
        print(f"     Loaded velocity: {loaded_velocity:.3f} m/s")
        print(f"     Unloaded velocity: {unloaded_velocity:.3f} m/s")
        print(f"     Max load: {max_load:.1f} kg")
        
    except Exception as e:
        print(f"  ❌ Martin run failed: {e}")
        return False
    
    print("\n✅ All mobility function tests passed!")
    return True

def test_parameter_sensitivity():
    """Test that changing parameters actually affects results."""
    
    print("\n3. Testing parameter sensitivity:")
    
    # Load parameter data
    file_path_params = project_root / "data/lookup tables/mobility-model-parameters.csv"
    all_hpv_param_df = pd.read_csv(file_path_params)
    param_df = all_hpv_param_df.loc[all_hpv_param_df["Name"] == "Bicycle"]
    
    # Test with default parameters
    mo = mm.model_options()
    mo.model_selection = 2
    mv = mm.model_variables(P_t=75)  # Default power
    hpv = mm.HPV_variables(param_df, mv)
    
    result1 = mm.mobility_models.single_bike_run(mv, mo, hpv, 0.0, 15.0)
    velocity1 = result1[1]  # unloaded velocity
    
    # Test with higher power
    mv2 = mm.model_variables(P_t=150)  # Double power
    hpv2 = mm.HPV_variables(param_df, mv2)
    
    result2 = mm.mobility_models.single_bike_run(mv2, mo, hpv2, 0.0, 15.0)
    velocity2 = result2[1]  # unloaded velocity
    
    print(f"  Default power (75W): {velocity1:.3f} m/s")
    print(f"  Higher power (150W): {velocity2:.3f} m/s")
    
    if velocity2 > velocity1:
        print(f"  ✅ Parameter sensitivity working: {velocity2/velocity1:.2f}x speed increase")
        return True
    else:
        print(f"  ❌ Parameter sensitivity not working: no speed increase")
        return False

if __name__ == "__main__":
    success1 = test_mobility_functions()
    success2 = test_parameter_sensitivity()
    
    if success1 and success2:
        print("\n🎉 All verification tests passed! Mobility functions work correctly.")
    else:
        print("\n⚠️  Some verification tests failed.")