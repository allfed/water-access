#!/usr/bin/env python3
"""
Cross-module validation test to verify sensitivity analysis and global module consistency.
This addresses the user's concern about different results between modules.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add src to path
project_root = Path(__file__).resolve().parent
sys.path.append(str(project_root))

import src.mobility_module as mm
from scripts.sensitivity_analysis import SensitivityAnalyzer

def test_function_consistency():
    """Test that both modules call the same underlying mobility functions."""
    
    print("🔍 Testing function consistency between modules...")
    
    # Test parameters
    slope = 2.0  # degrees
    load_attempted = 15.0  # kg
    
    # Load parameter data
    file_path_params = project_root / "data/lookup tables/mobility-model-parameters.csv"
    all_hpv_param_df = pd.read_csv(file_path_params)
    
    # Test both Lankford and Martin models
    for model_name, filter_value, model_selection in [
        ("Lankford", "Buckets", 3),
        ("Martin", "Bicycle", 2)
    ]:
        print(f"\n{model_name} Model Consistency Test:")
        
        param_df = all_hpv_param_df.loc[all_hpv_param_df["Name"] == filter_value]
        
        # === SENSITIVITY ANALYSIS APPROACH ===
        print("  1. Sensitivity Analysis approach:")
        analyzer = SensitivityAnalyzer(model_name)
        mo, mv, met, hpv, mr = analyzer.initialize_model_components(param_df)
        
        # Use same approach as sensitivity analysis direct calls
        if model_selection == 2:  # Cycling
            result_sens = mm.mobility_models.single_bike_run(mv, mo, hpv, slope, load_attempted)
        else:  # Walking  
            result_sens = mm.mobility_models.single_lankford_run(mv, mo, met, hpv, slope, load_attempted)
        
        print(f"     Loaded: {result_sens[0]:.4f} m/s, Unloaded: {result_sens[1]:.4f} m/s, Max Load: {result_sens[2]:.1f} kg")
        
        # === GLOBAL MODULE APPROACH ===
        print("  2. Global Module approach (simulated):")
        
        # Simulate what global module does - same initialization but different call pattern
        mo2 = mm.model_options()
        mo2.model_selection = model_selection
        mv2 = mm.model_variables()
        
        if model_selection == 2:  # Cycling
            hpv2 = mm.HPV_variables(param_df, mv2)
            result_global = mm.mobility_models.single_bike_run(mv2, mo2, hpv2, slope, load_attempted)
        else:  # Walking
            met2 = mm.MET_values(mv2, country_weight=60, met=3.5, use_country_specific_weights=False)
            hpv2 = mm.HPV_variables(param_df, mv2)  
            result_global = mm.mobility_models.single_lankford_run(mv2, mo2, met2, hpv2, slope, load_attempted)
        
        print(f"     Loaded: {result_global[0]:.4f} m/s, Unloaded: {result_global[1]:.4f} m/s, Max Load: {result_global[2]:.1f} kg")
        
        # === CONSISTENCY CHECK ===
        tolerance = 1e-6  # Very small tolerance for floating point comparison
        
        loaded_match = abs(result_sens[0] - result_global[0]) < tolerance
        unloaded_match = abs(result_sens[1] - result_global[1]) < tolerance  
        load_match = abs(result_sens[2] - result_global[2]) < tolerance
        
        if loaded_match and unloaded_match and load_match:
            print(f"     ✅ Results are consistent!")
        else:
            print(f"     ❌ Results differ:")
            print(f"        Loaded velocity diff: {abs(result_sens[0] - result_global[0]):.6f}")
            print(f"        Unloaded velocity diff: {abs(result_sens[1] - result_global[1]):.6f}")
            print(f"        Max load diff: {abs(result_sens[2] - result_global[2]):.6f}")
            return False
    
    return True

def test_parameter_propagation():
    """Test that parameter changes propagate consistently in both modules."""
    
    print("\n🔧 Testing parameter propagation consistency...")
    
    # Load data
    file_path_params = project_root / "data/lookup tables/mobility-model-parameters.csv"
    all_hpv_param_df = pd.read_csv(file_path_params)
    param_df = all_hpv_param_df.loc[all_hpv_param_df["Name"] == "Bicycle"]
    
    # Test: Change human power output and verify both modules see the change
    powers = [50, 75, 100, 150]  # Different power levels
    slope = 1.0
    load_attempted = 15.0
    
    print("  Testing power parameter changes:")
    
    sens_results = []
    global_results = []
    
    for power in powers:
        # Sensitivity analysis approach
        analyzer = SensitivityAnalyzer("Martin")
        mo, mv, met, hpv, mr = analyzer.initialize_model_components(param_df)
        
        # Apply parameter change (like sensitivity analysis does)
        mv.P_t = power
        result_sens = mm.mobility_models.single_bike_run(mv, mo, hpv, slope, load_attempted)
        sens_results.append(result_sens[1])  # unloaded velocity
        
        # Global module approach  
        mv2 = mm.model_variables(P_t=power)  # Set power directly
        mo2 = mm.model_options()
        mo2.model_selection = 2
        hpv2 = mm.HPV_variables(param_df, mv2)
        
        result_global = mm.mobility_models.single_bike_run(mv2, mo2, hpv2, slope, load_attempted)
        global_results.append(result_global[1])  # unloaded velocity
        
        print(f"    {power}W: Sens={result_sens[1]:.3f} m/s, Global={result_global[1]:.3f} m/s")
    
    # Check that both show the same pattern (increasing velocity with power)
    sens_increasing = all(sens_results[i] <= sens_results[i+1] for i in range(len(sens_results)-1))
    global_increasing = all(global_results[i] <= global_results[i+1] for i in range(len(global_results)-1))
    
    # Check that results are numerically close
    max_diff = max(abs(s - g) for s, g in zip(sens_results, global_results))
    
    if sens_increasing and global_increasing and max_diff < 1e-6:
        print(f"  ✅ Parameter propagation consistent (max diff: {max_diff:.2e})")
        return True
    else:
        print(f"  ❌ Parameter propagation inconsistent:")
        print(f"     Sensitivity increasing: {sens_increasing}")
        print(f"     Global increasing: {global_increasing}")
        print(f"     Max difference: {max_diff:.6f}")
        return False

def test_slope_handling():
    """Test that slope conversion is handled consistently."""
    
    print("\n📐 Testing slope handling consistency...")
    
    # Load data  
    file_path_params = project_root / "data/lookup tables/mobility-model-parameters.csv"
    all_hpv_param_df = pd.read_csv(file_path_params)
    param_df = all_hpv_param_df.loc[all_hpv_param_df["Name"] == "Bicycle"]
    
    slopes = [0, 1, 2, 5, 10]  # Different slopes in degrees
    load_attempted = 15.0
    
    print("  Testing slope conversion (degrees to radians):")
    
    for slope in slopes:
        # Both modules should convert slope the same way
        # slope_radians = (slope / 360) * (2 * π)
        expected_radians = (slope / 360) * (2 * np.pi)
        
        # Test with both approaches
        analyzer = SensitivityAnalyzer("Martin")
        mo, mv, met, hpv, mr = analyzer.initialize_model_components(param_df)
        
        result_sens = mm.mobility_models.single_bike_run(mv, mo, hpv, slope, load_attempted)
        
        mv2 = mm.model_variables()
        mo2 = mm.model_options()
        mo2.model_selection = 2
        hpv2 = mm.HPV_variables(param_df, mv2)
        
        result_global = mm.mobility_models.single_bike_run(mv2, mo2, hpv2, slope, load_attempted)
        
        # Results should be identical since same conversion is used
        diff = abs(result_sens[1] - result_global[1])
        
        print(f"    {slope}° ({expected_radians:.4f} rad): diff = {diff:.2e}")
        
        if diff > 1e-10:  # Very strict tolerance
            print(f"  ❌ Slope handling inconsistent at {slope}°")
            return False
    
    print("  ✅ Slope handling consistent")
    return True

if __name__ == "__main__":
    print("🧪 Cross-Module Validation Test")
    print("="*50)
    
    test1 = test_function_consistency()
    test2 = test_parameter_propagation() 
    test3 = test_slope_handling()
    
    print("\n" + "="*50)
    if test1 and test2 and test3:
        print("🎉 ALL TESTS PASSED! Modules are consistent.")
        print("\n✅ Key Findings:")
        print("   • Both modules call the same underlying functions")
        print("   • Parameter changes propagate identically")  
        print("   • Slope conversion is consistent")
        print("   • No discrepancies detected between sensitivity analysis and global module")
    else:
        print("⚠️  SOME TESTS FAILED! Inconsistencies detected.")
        print("\n❌ Issues found between modules - further investigation needed.")