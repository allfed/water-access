#!/usr/bin/env python3
"""
Quick test to see impact of flat terrain vs slopes on achievable distances.
"""

import sys
from pathlib import Path

# Add src to path
project_root = Path(__file__).resolve().parent
sys.path.append(str(project_root))

import src.mobility_module as mm
import pandas as pd

def test_terrain_impact():
    """Test the impact of terrain on achievable distances."""
    
    print("🏔️ Testing Terrain Impact on Achievable Distances")
    print("="*60)
    
    # Load parameter data
    file_path_params = project_root / "data/lookup tables/mobility-model-parameters.csv"
    all_hpv_param_df = pd.read_csv(file_path_params)
    param_df_martin = all_hpv_param_df.loc[all_hpv_param_df["Name"] == "Bicycle"]
    param_df_lankford = all_hpv_param_df.loc[all_hpv_param_df["Name"] == "Buckets"]
    
    # Test parameters
    test_slopes = [0.0, 1.0, 2.0, 5.0]  # degrees
    time_hours = 5.5
    load_kg = 15.0
    
    print("Testing different slopes with 15kg load:\n")
    
    # Test cycling
    print("CYCLING (Martin Model):")
    print("Slope | Loaded v | Unloaded v | Max Distance")
    print("------|----------|------------|-------------")
    
    for slope in test_slopes:
        mo = mm.model_options()
        mo.model_selection = 2
        mv = mm.model_variables(P_t=75, m1=62)
        hpv = mm.HPV_variables(param_df_martin, mv)
        
        result = mm.mobility_models.single_bike_run(mv, mo, hpv, slope, load_kg)
        max_dist = result[1] * time_hours / 2  # km in 5.5 hours round trip
        
        print(f"{slope:4.0f}° | {result[0]:7.2f} | {result[1]:9.2f} | {max_dist:10.1f} km")
    
    print("\nWALKING (Lankford Model):")
    print("Slope | Loaded v | Unloaded v | Max Distance") 
    print("------|----------|------------|-------------")
    
    for slope in test_slopes:
        mo = mm.model_options()
        mo.model_selection = 3
        mv = mm.model_variables(m1=62)
        met = mm.MET_values(mv, country_weight=62, met=4.5, use_country_specific_weights=False)
        hpv = mm.HPV_variables(param_df_lankford, mv)
        
        result = mm.mobility_models.single_lankford_run(mv, mo, met, hpv, slope, load_kg)
        max_dist = result[1] * time_hours / 2  # km in 5.5 hours round trip
        
        print(f"{slope:4.0f}° | {result[0]:7.2f} | {result[1]:9.2f} | {max_dist:10.1f} km")
    
    # Test without load
    print("\n" + "="*60)
    print("Testing WITHOUT load (0kg):\n")
    
    print("CYCLING (No Load):")
    print("Slope | Unloaded v | Max Distance")
    print("------|------------|-------------")
    
    for slope in test_slopes:
        mo = mm.model_options()
        mo.model_selection = 2
        mv = mm.model_variables(P_t=75, m1=62)
        hpv = mm.HPV_variables(param_df_martin, mv)
        
        result = mm.mobility_models.single_bike_run(mv, mo, hpv, slope, 0.0)
        max_dist = result[1] * time_hours / 2  # km in 5.5 hours round trip
        
        print(f"{slope:4.0f}° | {result[1]:9.2f} | {max_dist:10.1f} km")
    
    print("\nWALKING (No Load):")
    print("Slope | Unloaded v | Max Distance")
    print("------|------------|-------------")
    
    for slope in test_slopes:
        mo = mm.model_options()
        mo.model_selection = 3
        mv = mm.model_variables(m1=62)
        met = mm.MET_values(mv, country_weight=62, met=4.5, use_country_specific_weights=False)
        hpv = mm.HPV_variables(param_df_lankford, mv)
        
        result = mm.mobility_models.single_lankford_run(mv, mo, met, hpv, slope, 0.0)
        max_dist = result[1] * time_hours / 2  # km in 5.5 hours round trip
        
        print(f"{slope:4.0f}° | {result[1]:9.2f} | {max_dist:10.1f} km")

if __name__ == "__main__":
    test_terrain_impact()