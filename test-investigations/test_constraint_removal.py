#!/usr/bin/env python3
"""
Test script to run the global model with and without the practical_limit_buckets constraint
to see if removing it fixes the 3.6 km vs 13.36 km discrepancy.
"""

import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import src.gis_global_module as gis
import src.mobility_module as mm

def test_constraint_removal():
    """Test the global model with and without the practical limit constraint."""
    
    print("=== TESTING PRACTICAL LIMIT CONSTRAINT REMOVAL ===")
    print("Using simplified test with run_global_analysis function")
    
    # Test parameters
    test_params = {
        'crr_adjustment': 0,
        'time_gathering_water': 5.5,
        'met': 4.5,
        'watts': 75,
        'hill_polarity': 0,
        'urban_adjustment': 0,
        'rural_adjustment': 0,
        'calculate_distance': True,
        'plot': False,
        'human_mass': 60,
        'use_sample_data': True  # Use sample data for testing
    }
    
    # Test 1: Original implementation with practical_limit_buckets = 20
    print("\n1. TESTING WITH ORIGINAL CONSTRAINT (practical_limit_buckets=20):")
    
    try:
        df_original = gis.run_global_analysis(
            practical_limit_bicycle=40,
            practical_limit_buckets=20,  # Original constraint
            **test_params
        )
        
        if 'max distance walking' in df_original.columns:
            walking_distances = df_original['max distance walking'].dropna()
            print(f"  Walking distances: {walking_distances.values}")
            print(f"  Mean: {walking_distances.mean():.2f} km")
            print(f"  Median: {walking_distances.median():.2f} km")
        else:
            print("  No walking distance column found")
            print(f"  Available columns: {df_original.columns.tolist()}")
        
    except Exception as e:
        print(f"  Error with original constraint: {e}")
    
    # Test 2: Disable constraint by setting it very high
    print("\n2. TESTING WITH DISABLED CONSTRAINT (practical_limit_buckets=1000):")
    
    try:
        df_disabled = gis.run_global_analysis(
            practical_limit_bicycle=40,
            practical_limit_buckets=1000,  # Effectively disabled
            **test_params
        )
        
        if 'max distance walking' in df_disabled.columns:
            walking_distances = df_disabled['max distance walking'].dropna()
            print(f"  Walking distances: {walking_distances.values}")
            print(f"  Mean: {walking_distances.mean():.2f} km")
            print(f"  Median: {walking_distances.median():.2f} km")
        else:
            print("  No walking distance column found")
            print(f"  Available columns: {df_disabled.columns.tolist()}")
        
    except Exception as e:
        print(f"  Error with disabled constraint: {e}")
    
    # Test 3: Use exactly 15L water capacity (matching sensitivity analysis)
    print("\n3. TESTING WITH EXACT SENSITIVITY ANALYSIS MATCH (practical_limit_buckets=15):")
    
    try:
        df_matched = gis.run_global_analysis(
            practical_limit_bicycle=40,
            practical_limit_buckets=15,  # Match sensitivity analysis
            **test_params
        )
        
        if 'max distance walking' in df_matched.columns:
            walking_distances = df_matched['max distance walking'].dropna()
            print(f"  Walking distances: {walking_distances.values}")
            print(f"  Mean: {walking_distances.mean():.2f} km")
            print(f"  Median: {walking_distances.median():.2f} km")
        else:
            print("  No walking distance column found")
            print(f"  Available columns: {df_matched.columns.tolist()}")
        
    except Exception as e:
        print(f"  Error with matched constraint: {e}")

def test_direct_mobility_calls():
    """Test the mobility model directly with different practical limits."""
    
    print("\n=== TESTING DIRECT MOBILITY MODEL CALLS ===")
    
    # Load parameters
    file_path_params = Path(__file__).parent / "data/lookup tables/mobility-model-parameters.csv"
    all_hpv_param_df = pd.read_csv(file_path_params)
    param_df = all_hpv_param_df.loc[all_hpv_param_df["Name"] == "Buckets"]
    
    print("Original Buckets parameters:")
    print(param_df[['Name', 'LoadLimit', 'PracticalLimit', 'AverageSpeedWithoutLoad']])
    
    # Test with different practical limits
    for practical_limit in [15, 20, 1000]:
        print(f"\\nTesting with practical_limit = {practical_limit}:")
        
        try:
            # Create modified parameters
            test_param_df = param_df.copy()
            test_param_df["PracticalLimit"] = practical_limit
            
            # Initialize model components
            mo = mm.model_options()
            mo.model_selection = 3  # Lankford walking model
            mv = mm.model_variables(m1=60)  # 60kg human
            met = mm.MET_values(mv, country_weight=60, met=4.5, use_country_specific_weights=False)
            hpv = mm.HPV_variables(test_param_df, mv)
            
            # Test with a moderate slope (2 degrees)
            slope = 2.0
            load_attempt = 15  # Try to carry 15L
            
            result = mm.mobility_models.single_lankford_run(
                mv, mo, met, hpv, slope, load_attempt
            )
            
            if isinstance(result, tuple) and len(result) == 3:
                loaded_velocity, unloaded_velocity, max_load = result
                avg_velocity = (loaded_velocity + unloaded_velocity) / 2
                
                # Calculate distance (like global model does)
                distance = avg_velocity * mv.t_hours * 3600 / 2 / 1000  # One-way distance in km
                
                print(f"  Loaded velocity: {loaded_velocity:.3f} m/s")
                print(f"  Unloaded velocity: {unloaded_velocity:.3f} m/s")
                print(f"  Average velocity: {avg_velocity:.3f} m/s")
                print(f"  Max load: {max_load:.1f} kg")
                print(f"  One-way distance: {distance:.2f} km")
                
            else:
                print(f"  Unexpected result: {result}")
                
        except Exception as e:
            print(f"  Error: {e}")

def main():
    """Main test function."""
    
    print("TESTING PRACTICAL LIMIT CONSTRAINT REMOVAL")
    print("=" * 60)
    
    test_constraint_removal()
    test_direct_mobility_calls()
    
    print("\\n" + "=" * 60)
    print("TEST COMPLETE")
    print("If removing the constraint increases distances significantly,")
    print("then the constraint was indeed the cause of the 3.6 km issue.")

if __name__ == "__main__":
    main()