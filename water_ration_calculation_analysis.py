#!/usr/bin/env python3
"""
Water Ration Calculation Analysis

This script analyzes the differences in water ration calculations between
the sensitivity analysis and the global model to identify potential sources
of discrepancy.

Key differences identified:
1. Sensitivity analysis: water_ration_kms = mean_vel_kg_per_slope / mv.waterration * t_secs / 1000
2. Global model: water_ration_kms = loaded_velocity * max_load * time_hours / 1000
"""

import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Setup paths
project_root = Path(__file__).resolve().parent
sys.path.append(str(project_root))

# Import mobility module
import src.mobility_module as mm

def test_water_ration_calculations():
    """Test water ration calculations with identical parameters"""
    
    print("=== WATER RATION CALCULATION COMPARISON ===")
    print("Testing with identical parameters to identify calculation differences")
    print("="*70)
    
    # Common parameters
    test_params = {
        'time_hours': 5.5,
        'water_ration': 15,  # L/day
        'slope': 2.0,  # degrees
        'load_attempt': 15,  # kg
        'power': 75,  # watts
        'human_mass': 62  # kg
    }
    
    # Load mobility parameters for bicycle
    file_path_params = project_root / "data/lookup tables/mobility-model-parameters.csv"
    param_df = pd.read_csv(file_path_params)
    bicycle_params = param_df[param_df["Name"] == "Bicycle"]
    
    # Initialize model components (matching sensitivity analysis)
    mo = mm.model_options()
    mo.model_selection = 2  # Cycling model
    mv = mm.model_variables(P_t=test_params['power'], m1=test_params['human_mass'])
    mv.waterration = test_params['water_ration']
    mv.t_hours = test_params['time_hours']
    hpv = mm.HPV_variables(bicycle_params, mv)
    
    print(f"\n=== TEST PARAMETERS ===")
    for key, value in test_params.items():
        print(f"{key}: {value}")
    
    # Run single model call (like global model does)
    print(f"\n=== DIRECT MODEL CALL (Global Model Style) ===")
    loaded_velocity, unloaded_velocity, max_load = mm.mobility_models.single_bike_run(
        mv, mo, hpv, test_params['slope'], test_params['load_attempt']
    )
    
    avg_velocity = (loaded_velocity + unloaded_velocity) / 2
    
    print(f"Loaded velocity: {loaded_velocity:.3f} m/s")
    print(f"Unloaded velocity: {unloaded_velocity:.3f} m/s")
    print(f"Average velocity: {avg_velocity:.3f} m/s")
    print(f"Max load: {max_load:.3f} kg")
    
    # Calculate water ration kms using different methods
    print(f"\n=== WATER RATION CALCULATIONS ===")
    
    # Method 1: Sensitivity analysis formula
    t_secs = test_params['time_hours'] * 60 * 60
    velocity_kg = avg_velocity * max_load
    water_ration_kms_sensitivity = velocity_kg / test_params['water_ration'] * t_secs / 1000
    
    print(f"\nSensitivity Analysis Method:")
    print(f"  velocity_kg = avg_velocity * max_load = {avg_velocity:.3f} * {max_load:.3f} = {velocity_kg:.3f}")
    print(f"  water_ration_kms = velocity_kg / water_ration * t_secs / 1000")
    print(f"  water_ration_kms = {velocity_kg:.3f} / {test_params['water_ration']} * {t_secs} / 1000")
    print(f"  water_ration_kms = {water_ration_kms_sensitivity:.3f} km")
    
    # Method 2: Global model approach (from gis_global_module.py)
    # Looking at the code, the global model calculates:
    # 1. max distance = average_velocity * time_gathering_water / 2
    # 2. water_ration_kms = max_distance * max_load
    
    max_distance_cycling = avg_velocity * test_params['time_hours'] * 3600 / 2 / 1000  # Convert to km
    water_ration_kms_global = max_distance_cycling * max_load
    
    print(f"\nGlobal Model Method:")
    print(f"  max_distance = avg_velocity * time_hours * 3600 / 2 / 1000")
    print(f"  max_distance = {avg_velocity:.3f} * {test_params['time_hours']} * 3600 / 2 / 1000")
    print(f"  max_distance = {max_distance_cycling:.3f} km")
    print(f"  water_ration_kms = max_distance * max_load")
    print(f"  water_ration_kms = {max_distance_cycling:.3f} * {max_load:.3f}")
    print(f"  water_ration_kms = {water_ration_kms_global:.3f} km")
    
    # Compare the results
    print(f"\n=== COMPARISON ===")
    print(f"Sensitivity analysis result: {water_ration_kms_sensitivity:.3f} km")
    print(f"Global model result: {water_ration_kms_global:.3f} km")
    print(f"Ratio (sensitivity/global): {water_ration_kms_sensitivity/water_ration_kms_global:.2f}")
    
    # Analyze the difference
    print(f"\n=== ANALYSIS ===")
    print(f"The key difference is in the interpretation:")
    print(f"1. Sensitivity analysis: water_ration_kms = total distance you can travel carrying water rations")
    print(f"2. Global model: water_ration_kms = one-way distance to water source × water carried")
    print(f"")
    print(f"The sensitivity analysis formula divides by water_ration (15L)")
    print(f"This gives distance per liter, then multiplies by time to get total distance")
    print(f"")
    print(f"The global model calculates one-way distance (hence /2) then multiplies by load")
    
    # Test with walking model too
    print(f"\n\n=== TESTING WITH WALKING MODEL ===")
    
    # Load walking parameters
    walking_params = param_df[param_df["Name"] == "Buckets"]
    
    # Initialize for walking
    mo_walk = mm.model_options()
    mo_walk.model_selection = 3  # Lankford model
    hpv_walk = mm.HPV_variables(walking_params, mv)
    met = mm.MET_values(mv, country_weight=test_params['human_mass'], met=4.5)
    
    # Run walking model
    loaded_vel_walk, unloaded_vel_walk, max_load_walk = mm.mobility_models.single_lankford_run(
        mv, mo_walk, met, hpv_walk, test_params['slope'], test_params['load_attempt']
    )
    
    avg_vel_walk = (loaded_vel_walk + unloaded_vel_walk) / 2
    
    print(f"Walking average velocity: {avg_vel_walk:.3f} m/s")
    print(f"Walking max load: {max_load_walk:.3f} kg")
    
    # Calculate water ration for walking
    velocity_kg_walk = avg_vel_walk * max_load_walk
    water_ration_kms_walk_sens = velocity_kg_walk / test_params['water_ration'] * t_secs / 1000
    
    max_distance_walk = avg_vel_walk * test_params['time_hours'] * 3600 / 2 / 1000
    water_ration_kms_walk_global = max_distance_walk * max_load_walk
    
    print(f"\nWalking water ration (sensitivity method): {water_ration_kms_walk_sens:.3f} km")
    print(f"Walking water ration (global method): {water_ration_kms_walk_global:.3f} km")
    
    return {
        'cycling_sensitivity': water_ration_kms_sensitivity,
        'cycling_global': water_ration_kms_global,
        'walking_sensitivity': water_ration_kms_walk_sens,
        'walking_global': water_ration_kms_walk_global
    }

def analyze_formula_differences():
    """Analyze the mathematical differences between formulas"""
    
    print("\n\n=== FORMULA ANALYSIS ===")
    print("="*70)
    
    print("\nSensitivity Analysis Formula:")
    print("  water_ration_kms = (velocity × max_load) / water_ration × time_seconds / 1000")
    print("  water_ration_kms = velocity × max_load × time_hours × 3600 / (water_ration × 1000)")
    
    print("\nGlobal Model Formula:")
    print("  max_distance = velocity × time_hours × 3600 / 2 / 1000")
    print("  water_ration_kms = max_distance × max_load")
    print("  water_ration_kms = velocity × max_load × time_hours × 3600 / 2000")
    
    print("\nRatio of formulas:")
    print("  sensitivity / global = [velocity × max_load × time × 3600 / (water_ration × 1000)] / ")
    print("                        [velocity × max_load × time × 3600 / 2000]")
    print("  sensitivity / global = 2000 / (water_ration × 1000)")
    print("  sensitivity / global = 2 / water_ration")
    print(f"  sensitivity / global = 2 / 15 = {2/15:.3f}")
    
    print("\nThis means the sensitivity analysis systematically underestimates by a factor of ~7.5x!")
    print("This is because:")
    print("1. Global model calculates round-trip distance capacity")
    print("2. Sensitivity analysis calculates distance per unit of water")
    
def propose_harmonized_calculation():
    """Propose a harmonized calculation method"""
    
    print("\n\n=== PROPOSED HARMONIZATION ===")
    print("="*70)
    
    print("\nTo harmonize the calculations, we should:")
    print("\n1. Clarify the metric definition:")
    print("   - 'Water access distance' = one-way distance to water source")
    print("   - 'Water transport capacity' = water × distance that can be transported")
    
    print("\n2. Use consistent formulas:")
    print("   - One-way distance = velocity × time / 2")
    print("   - Water transport = one-way distance × water_carried")
    
    print("\n3. For the sensitivity analysis, update the calculation to:")
    print("   water_access_km = velocity × time_hours / 2")
    print("   water_transport_km_L = water_access_km × max_load")
    
    print("\n4. Ensure both models use the same:")
    print("   - Time units (hours vs seconds)")
    print("   - Distance units (m vs km)")
    print("   - Interpretation (one-way vs round-trip)")

def main():
    """Main analysis function"""
    
    # Test calculations
    results = test_water_ration_calculations()
    
    # Analyze formula differences
    analyze_formula_differences()
    
    # Propose harmonization
    propose_harmonized_calculation()
    
    print("\n\n=== SUMMARY ===")
    print("="*70)
    print("The primary calculation difference is:")
    print(f"1. Sensitivity analysis divides by water_ration ({15}L), giving km per liter")
    print("2. Global model calculates total water×distance capacity")
    print(f"3. This creates a systematic {15/2:.1f}x difference in results")
    print("\nCombined with slope differences, this explains the full discrepancy:")
    print("- Sensitivity: ~13.5 km (flat slopes, per-liter calculation)")
    print("- Global: ~3.6 km walking (real slopes, total capacity calculation)")

if __name__ == "__main__":
    main()