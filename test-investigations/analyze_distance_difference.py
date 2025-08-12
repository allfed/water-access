#!/usr/bin/env python3
"""
Analyze the distance calculation difference between sensitivity analysis and global model.
"""

import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import mobility_module as mm

def analyze_formulas():
    """Analyze the distance calculation formulas used in both models."""
    
    print("=== FORMULA ANALYSIS ===")
    print("\n1. GLOBAL MODEL FORMULA:")
    print("   File: src/gis_global_module.py")
    print("   Formula: max_distance = average_velocity * time_gathering_water / 2")
    print("   - time_gathering_water = 5.5 hours (total time available)")
    print("   - Division by 2: converts total time to one-way travel time")
    print("   - Result: one-way distance to water source")
    
    print("\n2. SENSITIVITY ANALYSIS FORMULA:")
    print("   File: scripts/sensitivity_analysis_refactored.py")
    print("   Formula: one_way_distance_km = avg_velocity * mv.t_hours * 3600 / 2 / 1000")
    print("   - mv.t_hours = 5.5 hours (total time available)")
    print("   - * 3600: convert hours to seconds")
    print("   - / 2: convert to one-way travel time")
    print("   - / 1000: convert meters to kilometers")
    print("   - avg_velocity in m/s")
    
    print("\n3. UNIT ANALYSIS:")
    print("   Global: velocity(km/h) * time(h) / 2 = distance(km)")
    print("   Sensitivity: velocity(m/s) * time(h) * 3600 / 2 / 1000 = distance(km)")
    print("   Sensitivity simplified: velocity(m/s) * time(h) * 1.8 = distance(km)")
    
    return True

def test_identical_parameters():
    """Test both models with identical parameters to isolate differences."""
    
    print("\n=== IDENTICAL PARAMETER TEST ===")
    
    # Load parameter data (walking uses "Backpack" for Lankford model)
    param_df = pd.read_csv('data/lookup tables/mobility-model-parameters.csv')
    param_df = param_df.loc[param_df["Name"] == "Backpack"]
    
    if len(param_df) == 0:
        print("ERROR: No parameter data found")
        return None, None, None
    
    print(f"Using parameter data: {param_df['Name'].iloc[0]}")
    
    # Initialize components like sensitivity analysis does
    mo = mm.model_options()
    mv = mm.model_variables()
    met = mm.MET_values(mv, country_weight=60, met=4.5, use_country_specific_weights=False)
    hpv = mm.HPV_variables(param_df, mv)
    
    mo.model_selection = 3  # Walking (Lankford model)
    
    print(f"Model parameters:")
    print(f"  mv.t_hours: {mv.t_hours}")
    print(f"  mv.m1 (human mass): {mv.m1}")
    print(f"  met.MET_of_sustainable_excercise: {met.MET_of_sustainable_excercise}")
    
    # Test single slope like global model might use
    test_slopes = [0, 1, 2, 3, 4]
    velocities = []
    
    for slope in test_slopes:
        try:
            loaded_velocity, unloaded_velocity, max_load = mm.mobility_models.single_lankford_run(
                mv, mo, met, hpv, slope, 15  # 15L water capacity
            )
            avg_velocity = (loaded_velocity + unloaded_velocity) / 2
            velocities.append({
                'slope': slope,
                'avg_velocity_ms': avg_velocity,
                'loaded_velocity': loaded_velocity,
                'unloaded_velocity': unloaded_velocity
            })
            print(f"  Slope {slope}°: loaded={loaded_velocity:.3f}, unloaded={unloaded_velocity:.3f}, avg={avg_velocity:.3f} m/s")
        except Exception as e:
            print(f"  Slope {slope}°: ERROR - {e}")
    
    if velocities:
        # Calculate like sensitivity analysis does
        overall_avg_velocity = np.mean([v['avg_velocity_ms'] for v in velocities])
        
        print(f"\nOverall average velocity: {overall_avg_velocity:.3f} m/s")
        
        # CURRENT SENSITIVITY ANALYSIS FORMULA:
        sens_distance = overall_avg_velocity * mv.t_hours * 3600 / 2 / 1000
        print(f"Current sensitivity formula: {sens_distance:.3f} km")
        
        # GLOBAL MODEL EQUIVALENT (assume velocity in km/h):
        overall_avg_velocity_kmh = overall_avg_velocity * 3.6  # Convert m/s to km/h
        global_distance = overall_avg_velocity_kmh * mv.t_hours / 2
        print(f"Global model equivalent: {global_distance:.3f} km")
        
        # CORRECTED SENSITIVITY ANALYSIS:
        # If we use velocity in km/h like global model
        corrected_distance = overall_avg_velocity_kmh * mv.t_hours / 2
        print(f"Corrected sensitivity formula: {corrected_distance:.3f} km")
        
        print(f"\nDifference analysis:")
        print(f"  Current sensitivity: {sens_distance:.3f} km")
        print(f"  Global equivalent: {global_distance:.3f} km")
        print(f"  Ratio: {sens_distance / global_distance:.2f}x")
        
        return overall_avg_velocity, sens_distance, global_distance
    
    return None, None, None

def analyze_slope_impact():
    """Analyze how slope distribution affects the results."""
    
    print("\n=== SLOPE IMPACT ANALYSIS ===")
    
    # Load our previous slope analysis
    print("From previous analysis:")
    print("  Real-world slopes: mean=4.0°, 36% > 3°")
    print("  Sensitivity analysis slopes: [0,1,2,3,4,5]° (simple average)")
    print("  Population-weighted real-world mean: 2.67°")
    
    # Test different slope scenarios
    param_df = pd.read_csv('data/lookup tables/mobility-model-parameters.csv')
    param_df = param_df.loc[param_df["Name"] == "Backpack"]
    
    mo = mm.model_options()
    mv = mm.model_variables()
    met = mm.MET_values(mv, country_weight=60, met=4.5, use_country_specific_weights=False)
    hpv = mm.HPV_variables(param_df, mv)
    mo.model_selection = 3
    
    scenarios = {
        'Current sensitivity': [0, 1, 2, 3, 4, 5],
        'Real-world weighted': [2.67],  # Population-weighted mean
        'Real-world mean': [4.0],       # Simple mean
        'Flat terrain': [0],            # Best case
        'Steep terrain': [5]            # Worst case in sensitivity range
    }
    
    results = {}
    
    for scenario_name, slopes in scenarios.items():
        velocities = []
        for slope in slopes:
            try:
                loaded_velocity, unloaded_velocity, max_load = mm.mobility_models.single_lankford_run(
                    mv, mo, met, hpv, slope, 15
                )
                avg_velocity = (loaded_velocity + unloaded_velocity) / 2
                velocities.append(avg_velocity)
            except Exception as e:
                print(f"Error for {scenario_name} slope {slope}°: {e}")
                velocities.append(np.nan)
        
        if velocities and not all(np.isnan(velocities)):
            avg_velocity_ms = np.nanmean(velocities)
            avg_velocity_kmh = avg_velocity_ms * 3.6
            distance = avg_velocity_kmh * mv.t_hours / 2
            results[scenario_name] = {
                'avg_velocity_ms': avg_velocity_ms,
                'avg_velocity_kmh': avg_velocity_kmh,
                'distance_km': distance,
                'slopes': slopes
            }
            print(f"{scenario_name:20}: {avg_velocity_ms:.3f} m/s → {distance:.3f} km")
    
    return results

def main():
    """Main analysis function."""
    
    print("DISTANCE CALCULATION DIFFERENCE ANALYSIS")
    print("=" * 50)
    
    # Step 1: Analyze formulas
    analyze_formulas()
    
    # Step 2: Test with identical parameters
    avg_vel, sens_dist, global_dist = test_identical_parameters()
    
    # Step 3: Analyze slope impact
    slope_results = analyze_slope_impact()
    
    # Step 4: Summary
    print("\n=== SUMMARY ===")
    if sens_dist and global_dist:
        print(f"Formula difference: {sens_dist/global_dist:.2f}x")
        print("This suggests the formulas are equivalent when using the same velocity units.")
        
    print("\nSlope impact on distance:")
    if slope_results:
        baseline = slope_results.get('Current sensitivity', {}).get('distance_km', 0)
        for scenario, data in slope_results.items():
            if baseline > 0:
                ratio = data['distance_km'] / baseline
                print(f"  {scenario}: {data['distance_km']:.3f} km ({ratio:.2f}x current)")
    
    print("\nKey findings:")
    print("- Formula equivalence needs verification")
    print("- Slope distribution has significant impact")
    print("- Real-world slopes would reduce distances further")

if __name__ == "__main__":
    main()