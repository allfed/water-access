#!/usr/bin/env python3
"""
Test script for the refactored sensitivity analysis.

This script tests that our refactored sensitivity analysis produces reasonable results
and can identify potential issues compared to the original implementation.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Setup path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

# Import our refactored sensitivity analyzer
from scripts.sensitivity_analysis import SensitivityAnalyzer

def test_basic_functionality():
    """Test basic functionality of the SensitivityAnalyzer."""
    print("🧪 Testing basic functionality...")
    
    try:
        # Test Lankford model
        analyzer_lankford = SensitivityAnalyzer(model_type="Lankford")
        param_df, sens_df = analyzer_lankford.load_data()
        
        print(f"✅ Loaded {len(param_df)} parameter records for Lankford model")
        print(f"✅ Loaded {len(sens_df)} sensitivity variables")
        
        # Test model component initialization
        mo, mv, met, hpv, mr = analyzer_lankford.initialize_model_components(param_df)
        print(f"✅ Initialized model components (model selection: {mo.model_selection})")
        
        # Test direct model calls
        velocities = analyzer_lankford.run_direct_model_calls(mv, mo, met, hpv)
        print(f"✅ Direct model calls completed for {len(velocities)} slopes")
        
        # Print sample results
        print("\n📊 Sample results from direct model calls:")
        for i, v in enumerate(velocities[:2]):  # Show first 2 results
            print(f"  Slope {v['slope']}°: loaded_vel={v['loaded_velocity']:.2f} m/s, "
                  f"unloaded_vel={v['unloaded_velocity']:.2f} m/s, "
                  f"max_load={v['max_load']:.1f} kg")
        
        return True
        
    except Exception as e:
        print(f"❌ Basic functionality test failed: {e}")
        return False

def test_parameter_application():
    """Test that parameter changes are applied correctly."""
    print("\n🧪 Testing parameter application...")
    
    try:
        analyzer = SensitivityAnalyzer(model_type="Lankford")
        param_df, sens_df = analyzer.load_data()
        mo, mv, met, hpv, mr = analyzer.initialize_model_components(param_df)
        
        # Test a simple parameter change
        original_met = met.MET_of_sustainable_excercise
        
        # Apply a MET budget change
        hpv_new, mv_new, mo_new, met_new = analyzer.apply_sensitivity_parameter(
            "MET budget", 5.0, hpv, mv, mo, met
        )
        
        if met_new.MET_of_sustainable_excercise != original_met:
            print(f"✅ Parameter application works: MET changed from {original_met} to {met_new.MET_of_sustainable_excercise}")
            return True
        else:
            print(f"❌ Parameter application failed: MET unchanged ({original_met})")
            return False
            
    except Exception as e:
        print(f"❌ Parameter application test failed: {e}")
        return False

def test_both_models():
    """Test both Lankford and Martin models."""
    print("\n🧪 Testing both models...")
    
    results = {}
    
    for model_type in ["Lankford", "Martin"]:
        try:
            analyzer = SensitivityAnalyzer(model_type=model_type)
            param_df, sens_df = analyzer.load_data()
            mo, mv, met, hpv, mr = analyzer.initialize_model_components(param_df)
            
            velocities = analyzer.run_direct_model_calls(mv, mo, met, hpv)
            
            # Calculate average metrics
            avg_loaded_vel = np.mean([v['loaded_velocity'] for v in velocities if not np.isnan(v['loaded_velocity'])])
            avg_unloaded_vel = np.mean([v['unloaded_velocity'] for v in velocities if not np.isnan(v['unloaded_velocity'])])
            
            results[model_type] = {
                'avg_loaded_velocity': avg_loaded_vel,
                'avg_unloaded_velocity': avg_unloaded_vel,
                'model_selection': mo.model_selection
            }
            
            print(f"✅ {model_type} model (selection {mo.model_selection}): "
                  f"avg_loaded={avg_loaded_vel:.2f} m/s, avg_unloaded={avg_unloaded_vel:.2f} m/s")
            
        except Exception as e:
            print(f"❌ {model_type} model test failed: {e}")
            results[model_type] = None
    
    # Compare results
    if results["Lankford"] and results["Martin"]:
        print("\n📊 Model comparison:")
        print(f"  Lankford (walking) vs Martin (cycling) loaded velocity ratio: "
              f"{results['Martin']['avg_loaded_velocity'] / results['Lankford']['avg_loaded_velocity']:.2f}")
        print("  (Cycling should generally be faster than walking)")
    
    return all(results.values())

def test_single_sensitivity_run():
    """Test a single sensitivity parameter run."""
    print("\n🧪 Testing single sensitivity parameter run...")
    
    try:
        analyzer = SensitivityAnalyzer(model_type="Lankford")
        param_df, sens_df = analyzer.load_data()
        
        # Test with the first sensitivity parameter
        first_param = sens_df.iloc[0]
        print(f"Testing parameter: {first_param['Short Name']}")
        
        df_result = analyzer.run_single_sensitivity(first_param, param_df)
        
        print(f"✅ Single sensitivity run completed: {len(df_result)} results")
        print(f"✅ Result columns: {list(df_result.columns)}")
        
        # Check for reasonable values
        if not df_result['Velocity Kgs'].isna().all():
            velocity_kg_range = (df_result['Velocity Kgs'].min(), df_result['Velocity Kgs'].max())
            print(f"✅ Velocity Kgs range: {velocity_kg_range[0]:.2f} to {velocity_kg_range[1]:.2f}")
            return True
        else:
            print("❌ All Velocity Kgs values are NaN")
            return False
            
    except Exception as e:
        print(f"❌ Single sensitivity run test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🚀 Starting sensitivity analysis refactor tests...\n")
    
    tests = [
        ("Basic Functionality", test_basic_functionality),
        ("Parameter Application", test_parameter_application),
        ("Both Models", test_both_models),
        ("Single Sensitivity Run", test_single_sensitivity_run)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{'='*50}")
        result = test_func()
        results.append((test_name, result))
    
    print(f"\n{'='*50}")
    print("📋 TEST SUMMARY:")
    all_passed = True
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {status}: {test_name}")
        if not result:
            all_passed = False
    
    if all_passed:
        print("\n🎉 All tests passed! The refactored sensitivity analysis appears to be working correctly.")
    else:
        print("\n⚠️  Some tests failed. Please review the errors above.")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 