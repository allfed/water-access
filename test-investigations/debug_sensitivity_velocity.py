#!/usr/bin/env python3
"""
Debug the velocity calculation in sensitivity analysis to understand the 8.55 km result.
"""

import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import mobility_module as mm

def debug_sensitivity_calculation():
    """Debug the exact calculation in sensitivity analysis."""
    
    print("=== DEBUGGING SENSITIVITY ANALYSIS CALCULATION ===")
    
    # Load data exactly like sensitivity analysis does
    project_root = Path(__file__).resolve().parent
    file_path_params = project_root / "data/lookup tables/mobility-model-parameters.csv"
    
    all_hpv_param_df = pd.read_csv(file_path_params)
    param_df = all_hpv_param_df.loc[all_hpv_param_df["Name"] == "Backpack"]
    
    print(f"Parameter data loaded: {len(param_df)} rows")
    
    # Initialize components
    mo = mm.model_options()
    mv = mm.model_variables()
    met = mm.MET_values(mv, country_weight=60, met=3.5, use_country_specific_weights=False)
    hpv = mm.HPV_variables(param_df, mv)
    
    mo.model_selection = 3  # Walking
    
    # Override load capacity like sensitivity analysis
    fixed_water_capacity = 15.0
    hpv.load_limit = np.array([[[fixed_water_capacity]]])
    hpv.practical_limit = np.array([[[fixed_water_capacity]]])
    
    print(f"Model setup:")
    print(f"  mv.t_hours: {mv.t_hours}")
    print(f"  met.MET_of_sustainable_excercise: {met.MET_of_sustainable_excercise}")
    print(f"  Fixed water capacity: {fixed_water_capacity}")
    
    # Test slopes like sensitivity analysis
    test_slopes = [0, 1, 2, 3, 4, 5]
    velocities = []
    
    print(f"\nTesting slopes: {test_slopes}")
    
    for slope in test_slopes:
        try:
            loaded_velocity, unloaded_velocity, max_load = mm.mobility_models.single_lankford_run(
                mv, mo, met, hpv, slope, fixed_water_capacity
            )
            
            # Ensure scalar values like sensitivity analysis
            loaded_velocity = float(loaded_velocity) if not np.isnan(loaded_velocity) else np.nan
            unloaded_velocity = float(unloaded_velocity) if not np.isnan(unloaded_velocity) else np.nan
            max_load = float(max_load) if not np.isnan(max_load) else np.nan
            
            # Calculate average velocity
            avg_velocity = (loaded_velocity + unloaded_velocity) / 2
            
            velocities.append({
                'slope': slope,
                'loaded_velocity': loaded_velocity,
                'unloaded_velocity': unloaded_velocity,
                'avg_velocity': avg_velocity,
                'max_load': max_load
            })
            
            print(f"  Slope {slope}°: loaded={loaded_velocity:.3f}, unloaded={unloaded_velocity:.3f}, avg={avg_velocity:.3f} m/s")
            
        except Exception as e:
            print(f"  Slope {slope}°: ERROR - {e}")
            velocities.append({
                'slope': slope,
                'loaded_velocity': np.nan,
                'unloaded_velocity': np.nan,
                'avg_velocity': np.nan,
                'max_load': np.nan
            })
    
    # Calculate metrics like sensitivity analysis
    valid_velocities = [v for v in velocities if not np.isnan(v['avg_velocity'])]
    
    if not valid_velocities:
        print("ERROR: No valid velocities calculated")
        return
    
    print(f"\nValid velocities: {len(valid_velocities)}")
    
    # Calculate average velocity across slopes
    avg_velocity = np.mean([v['avg_velocity'] for v in valid_velocities])
    
    print(f"Average velocity across slopes: {avg_velocity:.3f} m/s")
    
    # Calculate one-way distance (matching sensitivity analysis formula)
    one_way_distance_km = avg_velocity * mv.t_hours * 3600 / 2 / 1000
    
    print(f"\nCalculation breakdown:")
    print(f"  avg_velocity: {avg_velocity:.3f} m/s")
    print(f"  mv.t_hours: {mv.t_hours}")
    print(f"  * 3600: {avg_velocity * mv.t_hours * 3600:.3f} m")
    print(f"  / 2: {avg_velocity * mv.t_hours * 3600 / 2:.3f} m")
    print(f"  / 1000: {avg_velocity * mv.t_hours * 3600 / 2 / 1000:.3f} km")
    
    print(f"\nFinal result: {one_way_distance_km:.3f} km")
    
    # Compare with manual calculation
    manual_calc = avg_velocity * 3.6 * mv.t_hours / 2  # Convert to km/h first
    print(f"Manual calculation (convert to km/h first): {manual_calc:.3f} km")
    
    return avg_velocity, one_way_distance_km

if __name__ == "__main__":
    debug_sensitivity_calculation()