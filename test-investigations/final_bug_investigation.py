#!/usr/bin/env python3
"""
Final investigation to find the exact bug causing 3.6x reduction.
We know time should be 5.5h but results imply 1.56h.
"""

import pandas as pd
import numpy as np

print("=== FINAL BUG INVESTIGATION ===")
print()

# Known facts
print("KNOWN FACTS:")
print("- Theoretical velocity: 1.343 m/s")
print("- Raw data velocity: 1.316 m/s") 
print("- Expected time: 5.5 hours")
print("- Expected distance: ~13 km")
print("- Actual distance: 3.71 km")
print("- Implied time: 1.56 hours")
print("- Reduction factor: 3.53x")
print()

# Check if velocities are in different units somewhere
print("HYPOTHESIS 1: Velocity unit mismatch")
print("- What if velocities are in km/h somewhere?")
velocity_ms = 1.316
velocity_kmh = velocity_ms * 3.6
print(f"- 1.316 m/s = {velocity_kmh:.2f} km/h")
print("- But this would make distances larger, not smaller")
print()

# Check the actual country results more carefully
print("HYPOTHESIS 2: Check actual calculation in results")

results_file = 'results/country_median_results.csv'
try:
    df = pd.read_csv(results_file)
    
    # Look for velocity-related columns
    print("\nColumns in results file:")
    vel_cols = [col for col in df.columns if 'vel' in col.lower() or 'walk' in col.lower()]
    print(f"Velocity/walking columns: {vel_cols}")
    
    # Check a specific country
    if 'Entity' in df.columns and 'weighted_med_walking' in df.columns:
        # Find USA as example
        usa_data = df[df['Entity'].str.contains('United States', na=False)]
        if not usa_data.empty:
            print(f"\nUSA walking distance: {usa_data['weighted_med_walking'].iloc[0]:.2f} km")
            
            # Calculate implied parameters
            distance = usa_data['weighted_med_walking'].iloc[0]
            
            # If velocity = 1.316 m/s, what time was used?
            implied_time = distance * 2 * 1000 / (1.316 * 3600)
            print(f"Implied time (if v=1.316 m/s): {implied_time:.2f} hours")
            
            # If time = 5.5 hours, what velocity was used?
            implied_velocity = distance * 2 * 1000 / (5.5 * 3600)
            print(f"Implied velocity (if t=5.5h): {implied_velocity:.3f} m/s")
            
except Exception as e:
    print(f"Error reading results: {e}")

# Check the velocity data file for clues
print("\n\nHYPOTHESIS 3: Population weighting reduces effective velocity")

velocity_file = 'data/processed/walk_velocity_by_zone.csv'
try:
    df_vel = pd.read_csv(velocity_file)
    
    if 'average_velocity_walk' in df_vel.columns:
        # Check if there's zone-level distance calculation
        if 'max_distance_walk' in df_vel.columns:
            print("Found max_distance_walk in velocity file!")
            distances = df_vel['max_distance_walk'].dropna()
            print(f"Distance range: {distances.min():.2f} - {distances.max():.2f} km")
            print(f"Mean distance: {distances.mean():.2f} km")
            
        # Check for population column
        pop_cols = [col for col in df_vel.columns if 'pop' in col.lower()]
        if pop_cols:
            print(f"\nPopulation columns: {pop_cols}")
            
            # Calculate weighted average
            pop_col = pop_cols[0]
            data = df_vel[['average_velocity_walk', pop_col]].dropna()
            
            if len(data) > 0:
                total_pop = data[pop_col].sum()
                weighted_vel = (data['average_velocity_walk'] * data[pop_col]).sum() / total_pop
                unweighted_vel = data['average_velocity_walk'].mean()
                
                print(f"\nWeighted velocity: {weighted_vel:.3f} m/s")
                print(f"Unweighted velocity: {unweighted_vel:.3f} m/s")
                print(f"Ratio: {weighted_vel/unweighted_vel:.3f}x")
                
                # Calculate distances with both
                time_h = 5.5
                weighted_dist = weighted_vel * time_h * 3600 / 2 / 1000
                unweighted_dist = unweighted_vel * time_h * 3600 / 2 / 1000
                
                print(f"\nWith t=5.5h:")
                print(f"Weighted distance: {weighted_dist:.2f} km")
                print(f"Unweighted distance: {unweighted_dist:.2f} km")
                
except Exception as e:
    print(f"Error reading velocity file: {e}")

# The real smoking gun
print("\n\n🎯 THE SMOKING GUN:")
print("Looking at the exact calculation path...")
print()

# We found that process_zones_for_water_access has default=16
# But calculate_max_distances expects hours
# So let's trace the exact flow

print("CALL FLOW:")
print("1. run_global_analysis(time_gathering_water=5.5)")
print("   - Doc says this is MINUTES (wrong!)")
print("   - But everyone uses it as HOURS")
print()
print("2. process_zones_for_water_access(time_gathering_water=5.5)")
print("   - Has default=16 (suspicious!)")
print("   - Passes value directly to calculate_max_distances")
print()
print("3. calculate_max_distances(time_gathering_water=5.5)")
print("   - Uses as HOURS in velocity * time / 2")
print("   - Should give ~13 km")
print()
print("So where does 3.71 km come from?")
print()

# Final hypothesis
print("FINAL HYPOTHESIS: The weighted_med_walking is NOT max_distance_walking!")
print("- max_distance_walking = theoretical maximum distance")
print("- weighted_med_walking = actual median distance people travel")
print("- These could be different due to:")
print("  1. People don't always go the maximum distance")
print("  2. Water sources are closer than maximum")
print("  3. Some other constraint or calculation")

print("\n" + "="*60)
print("NEXT STEP: Check if weighted_med_walking calculation is different")
print("from max_distance_walking calculation!")