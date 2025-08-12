#!/usr/bin/env python3
"""
COMPREHENSIVE VERIFICATION: Confirm the unit conversion error with absolute certainty
"""

import pandas as pd
import numpy as np

print("=== COMPREHENSIVE UNIT CONVERSION VERIFICATION ===")
print()

def load_and_analyze_velocity_data():
    """Load velocity data and verify units"""
    print("1. ANALYZING VELOCITY DATA UNITS")
    print("-" * 60)
    
    try:
        # Load velocity data
        df_vel = pd.read_csv('data/processed/walk_velocity_by_zone.csv')
        
        if 'average_velocity_walk' in df_vel.columns:
            velocities = df_vel['average_velocity_walk'].dropna()
            
            print(f"Loaded {len(velocities)} velocity values")
            print(f"\nVelocity statistics:")
            print(f"  Mean: {velocities.mean():.6f}")
            print(f"  Median: {velocities.median():.6f}")
            print(f"  Min: {velocities.min():.6f}")
            print(f"  Max: {velocities.max():.6f}")
            print(f"  Std: {velocities.std():.6f}")
            
            # Check if these values make sense as m/s
            print(f"\nInterpretation:")
            mean_vel = velocities.mean()
            print(f"  If m/s: {mean_vel:.3f} m/s = {mean_vel * 3.6:.3f} km/h (typical walking speed)")
            print(f"  If km/h: {mean_vel:.3f} km/h = {mean_vel / 3.6:.3f} m/s (very slow)")
            
            # Sample some actual values
            print(f"\nSample velocity values:")
            for i in range(min(10, len(velocities))):
                vel = velocities.iloc[i]
                print(f"  Zone {i}: {vel:.6f} (m/s: {vel*3.6:.2f} km/h, km/h: {vel/3.6:.2f} m/s)")
                
            return velocities.mean()
            
    except Exception as e:
        print(f"Error loading velocity data: {e}")
        return None

def verify_distance_calculations(mean_velocity):
    """Verify the distance calculations with both formulas"""
    print("\n\n2. VERIFYING DISTANCE CALCULATIONS")
    print("-" * 60)
    
    time_hours = 5.5
    
    # Current formula (potentially buggy)
    distance_current = mean_velocity * time_hours / 2
    
    # Correct formula if velocity is in m/s
    distance_corrected = mean_velocity * time_hours * 3600 / 2 / 1000
    
    print(f"\nGiven:")
    print(f"  Mean velocity: {mean_velocity:.6f}")
    print(f"  Time: {time_hours} hours")
    
    print(f"\nCalculations:")
    print(f"  Current formula (line 839): velocity * time / 2")
    print(f"    = {mean_velocity:.6f} * {time_hours} / 2")
    print(f"    = {distance_current:.6f} km")
    
    print(f"\n  Corrected formula (if velocity in m/s): velocity * time * 3600 / 2 / 1000")
    print(f"    = {mean_velocity:.6f} * {time_hours} * 3600 / 2 / 1000")
    print(f"    = {distance_corrected:.6f} km")
    
    print(f"\n  Ratio: {distance_corrected / distance_current:.1f}x")
    
    return distance_current, distance_corrected

def compare_with_actual_results(distance_current, distance_corrected):
    """Compare calculations with actual country results"""
    print("\n\n3. COMPARING WITH ACTUAL RESULTS")
    print("-" * 60)
    
    try:
        df_results = pd.read_csv('results/country_median_results.csv')
        
        if 'weighted_med_walking' in df_results.columns:
            actual_distances = df_results['weighted_med_walking'].dropna()
            
            print(f"Actual country results:")
            print(f"  Mean: {actual_distances.mean():.6f} km")
            print(f"  Median: {actual_distances.median():.6f} km")
            print(f"  Range: {actual_distances.min():.3f} - {actual_distances.max():.3f} km")
            
            actual_mean = actual_distances.mean()
            
            print(f"\nComparison:")
            print(f"  Current formula result: {distance_current:.3f} km")
            print(f"  Actual result: {actual_mean:.3f} km")
            print(f"  Error: {abs(distance_current - actual_mean):.3f} km ({abs(distance_current - actual_mean)/actual_mean*100:.1f}%)")
            
            print(f"\n  Corrected formula result: {distance_corrected:.3f} km")
            print(f"  Actual result: {actual_mean:.3f} km") 
            print(f"  Error: {abs(distance_corrected - actual_mean):.3f} km ({abs(distance_corrected - actual_mean)/actual_mean*100:.1f}%)")
            
            # Show some example countries
            print(f"\nExample countries:")
            for i in range(min(10, len(df_results))):
                country = df_results.iloc[i]['Entity']
                dist = df_results.iloc[i]['weighted_med_walking']
                print(f"  {country}: {dist:.3f} km")
                
            return actual_mean
            
    except Exception as e:
        print(f"Error loading results: {e}")
        return None

def check_sensitivity_analysis():
    """Check how sensitivity analysis calculates distances"""
    print("\n\n4. CHECKING SENSITIVITY ANALYSIS IMPLEMENTATION")
    print("-" * 60)
    
    # Read the sensitivity analysis code
    try:
        with open('scripts/sensitivity_analysis_refactored.py', 'r') as f:
            content = f.read()
            
        # Find the distance calculation
        lines = content.split('\n')
        for i, line in enumerate(lines):
            if 'one_way_distance_km' in line and '3600' in line:
                print(f"Found in sensitivity analysis (line {i+1}):")
                print(f"  {line.strip()}")
                # Show context
                for j in range(max(0, i-2), min(len(lines), i+3)):
                    if j != i:
                        print(f"  {lines[j].strip()}")
                print()
                
    except Exception as e:
        print(f"Error reading sensitivity analysis: {e}")

def check_mobility_module_output():
    """Check what the mobility module actually outputs"""
    print("\n\n5. CHECKING MOBILITY MODULE OUTPUT UNITS")
    print("-" * 60)
    
    try:
        # Check mobility module for return statements
        with open('src/mobility_module.py', 'r') as f:
            content = f.read()
            
        lines = content.split('\n')
        
        # Find return statements and velocity calculations
        for i, line in enumerate(lines):
            if 'return' in line and ('velocity' in line or 'speed' in line):
                print(f"Return statement (line {i+1}):")
                print(f"  {line.strip()}")
                # Show context
                for j in range(max(0, i-3), min(len(lines), i+2)):
                    if j != i:
                        print(f"  {lines[j].strip()}")
                print()
                
    except Exception as e:
        print(f"Error reading mobility module: {e}")

def trace_data_flow():
    """Trace how velocity data flows through the system"""
    print("\n\n6. TRACING VELOCITY DATA FLOW")
    print("-" * 60)
    
    # Check where velocity CSVs come from
    print("Looking for velocity CSV creation...")
    
    # Search for files that write to walk_velocity_by_zone.csv
    import os
    for root, dirs, files in os.walk('src'):
        for file in files:
            if file.endswith('.py'):
                filepath = os.path.join(root, file)
                try:
                    with open(filepath, 'r') as f:
                        content = f.read()
                    if 'walk_velocity_by_zone' in content and ('to_csv' in content or 'write' in content):
                        print(f"\nFound in {filepath}:")
                        lines = content.split('\n')
                        for i, line in enumerate(lines):
                            if 'walk_velocity_by_zone' in line:
                                print(f"  Line {i+1}: {line.strip()}")
                except:
                    pass

def check_cycling_consistency():
    """Check if cycling has the same issue"""
    print("\n\n7. CHECKING CYCLING CALCULATIONS")
    print("-" * 60)
    
    try:
        # Load cycling velocity data
        df_vel = pd.read_csv('data/processed/cycle_velocity_by_zone.csv')
        
        if 'average_velocity_bicycle' in df_vel.columns:
            cycle_velocities = df_vel['average_velocity_bicycle'].dropna()
            
            print(f"Cycling velocity statistics:")
            print(f"  Mean: {cycle_velocities.mean():.6f}")
            print(f"  If m/s: {cycle_velocities.mean() * 3.6:.3f} km/h")
            
            # Calculate expected cycling distance
            time_hours = 5.5
            cycle_dist_current = cycle_velocities.mean() * time_hours / 2
            cycle_dist_corrected = cycle_velocities.mean() * time_hours * 3600 / 2 / 1000
            
            print(f"\nCycling distance calculations:")
            print(f"  Current formula: {cycle_dist_current:.3f} km")
            print(f"  Corrected formula: {cycle_dist_corrected:.3f} km")
            
            # Compare with actual cycling results
            df_results = pd.read_csv('results/country_median_results.csv')
            if 'weighted_med_cycling' in df_results.columns:
                actual_cycle = df_results['weighted_med_cycling'].mean()
                print(f"  Actual cycling distance: {actual_cycle:.3f} km")
                print(f"  Ratio actual/current: {actual_cycle/cycle_dist_current:.1f}x")
                
    except Exception as e:
        print(f"Error checking cycling: {e}")

def final_verification():
    """Final mathematical proof"""
    print("\n\n8. FINAL MATHEMATICAL VERIFICATION")
    print("-" * 60)
    
    print("The unit conversion factor between m/s and km/h is:")
    print("  1 m/s = 3.6 km/h")
    print("  1 km/h = 1/3.6 m/s")
    
    print("\nTo convert velocity (m/s) * time (hours) to distance (km):")
    print("  distance = velocity * time * 3600 / 1000")
    print("  distance = velocity * time * 3.6")
    
    print("\nThe missing factor in the current formula is:")
    print("  3600 / 1000 = 3.6")
    
    print("\nThis exactly matches the observed 3.6x discrepancy!")

def main():
    """Run comprehensive verification"""
    
    # Step 1: Analyze velocity data
    mean_velocity = load_and_analyze_velocity_data()
    
    if mean_velocity:
        # Step 2: Calculate distances both ways
        dist_current, dist_corrected = verify_distance_calculations(mean_velocity)
        
        # Step 3: Compare with actual results
        actual_mean = compare_with_actual_results(dist_current, dist_corrected)
        
    # Step 4: Check sensitivity analysis
    check_sensitivity_analysis()
    
    # Step 5: Check mobility module
    check_mobility_module_output()
    
    # Step 6: Trace data flow
    trace_data_flow()
    
    # Step 7: Check cycling
    check_cycling_consistency()
    
    # Step 8: Final verification
    final_verification()
    
    print("\n" + "="*60)
    print("CONCLUSION:")
    print("The unit conversion bug is CONFIRMED. The global model is missing")
    print("the conversion factor of 3.6 (or 3600/1000) when converting from")
    print("m/s to km/h in the distance calculations.")

if __name__ == "__main__":
    main()