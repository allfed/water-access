#!/usr/bin/env python3
"""
CRITICAL VERIFICATION: Double-check the unit conversion hypothesis
"""

import pandas as pd
import numpy as np

print("=== CRITICAL UNIT VERIFICATION ===")
print()

def verify_units_hypothesis():
    """Meticulously verify the unit conversion hypothesis"""
    
    print("1. CHECKING THE EXACT CALCULATION IN LINE 839")
    print()
    
    # Load the actual velocity data
    try:
        df_vel = pd.read_csv('data/processed/walk_velocity_by_zone.csv')
        
        if 'average_velocity_walk' in df_vel.columns:
            velocity_mean = df_vel['average_velocity_walk'].mean()
            print(f"Mean velocity from data: {velocity_mean:.6f}")
            
            # The exact calculation from line 839
            time_gathering_water = 5.5  # hours
            
            # Current (potentially buggy) calculation
            max_distance_current = velocity_mean * time_gathering_water / 2
            print(f"Line 839 calculation: {velocity_mean:.6f} * {time_gathering_water} / 2 = {max_distance_current:.6f}")
            
            # If velocity is in m/s, correct calculation should be:
            max_distance_correct = velocity_mean * time_gathering_water * 3600 / 2 / 1000
            print(f"Correct m/s calculation: {velocity_mean:.6f} * {time_gathering_water} * 3600 / 2 / 1000 = {max_distance_correct:.6f} km")
            
            # Check what the observed country results are
            try:
                df_results = pd.read_csv('results/country_median_results.csv')
                if 'weighted_med_walking' in df_results.columns:
                    observed_mean = df_results['weighted_med_walking'].mean()
                    print(f"Observed country mean: {observed_mean:.6f} km")
                    
                    print()
                    print("2. MATHEMATICAL VERIFICATION")
                    print()
                    
                    error_current = abs(max_distance_current - observed_mean)
                    error_correct = abs(max_distance_correct - observed_mean)
                    
                    print(f"Error if line 839 is correct: {error_current:.6f} km")
                    print(f"Error if m/s conversion needed: {error_correct:.6f} km")
                    
                    if error_current < error_correct:
                        print("🎯 LINE 839 IS ACTUALLY CORRECT!")
                        print("The hypothesis was WRONG!")
                        
                        # Calculate what units would make this work
                        implied_velocity_units = observed_mean * 2 / time_gathering_water
                        print(f"Implied velocity units from results: {implied_velocity_units:.6f}")
                        print(f"If this is km/h: {implied_velocity_units * 1000 / 3600:.6f} m/s")
                        
                    else:
                        print("✅ UNIT CONVERSION BUG CONFIRMED!")
                        print(f"The factor needed: {max_distance_correct / max_distance_current:.1f}x")
                        
            except Exception as e:
                print(f"Error loading results: {e}")
                
    except Exception as e:
        print(f"Error loading velocity data: {e}")

def check_mobility_module_units():
    """Check what units the mobility module actually returns"""
    
    print()
    print("3. CHECKING MOBILITY MODULE UNITS")
    print()
    
    # Look for any documentation or comments about units
    try:
        with open('src/mobility_module.py', 'r') as f:
            content = f.read()
        
        lines = content.split('\n')
        
        unit_mentions = []
        for i, line in enumerate(lines):
            if any(unit in line.lower() for unit in ['m/s', 'km/h', 'velocity', 'speed', 'unit']):
                unit_mentions.append((i+1, line.strip()))
        
        print("Lines mentioning units/velocity in mobility_module.py:")
        for line_num, line in unit_mentions[:10]:  # Show first 10
            print(f"  {line_num}: {line}")
            
        print(f"\nTotal lines mentioning units: {len(unit_mentions)}")
        
    except Exception as e:
        print(f"Error reading mobility module: {e}")

def test_with_sample_data():
    """Test the calculation with actual sample data"""
    
    print()
    print("4. TESTING WITH SAMPLE DATA")
    print()
    
    try:
        df_vel = pd.read_csv('data/processed/walk_velocity_by_zone.csv')
        
        # Take first few rows as sample
        sample_velocities = df_vel['average_velocity_walk'].head(5)
        time_hours = 5.5
        
        print("Sample velocities from data:")
        for i, vel in enumerate(sample_velocities):
            
            # Line 839 calculation
            dist_current = vel * time_hours / 2
            
            # Corrected calculation  
            dist_corrected = vel * time_hours * 3600 / 2 / 1000
            
            print(f"  Zone {i+1}: {vel:.6f} -> {dist_current:.6f} (current) vs {dist_corrected:.6f} km (corrected)")
        
        # Load a country result for comparison
        try:
            df_results = pd.read_csv('results/country_median_results.csv')
            first_country_dist = df_results['weighted_med_walking'].iloc[0]
            print(f"\nFirst country result: {first_country_dist:.6f} km")
            print("Which calculation matches better?")
            
        except Exception as e:
            print(f"Error loading results: {e}")
            
    except Exception as e:
        print(f"Error in sample test: {e}")

def check_other_distance_calculations():
    """Check if cycling has the same issue"""
    
    print()
    print("5. CHECKING CYCLING CALCULATIONS")
    print()
    
    try:
        # Check if cycling uses the same pattern
        with open('src/gis_global_module.py', 'r') as f:
            content = f.read()
        
        lines = content.split('\n')
        
        for i, line in enumerate(lines):
            if 'max distance cycling' in line and '=' in line:
                print(f"Line {i+1}: {line.strip()}")
                # Show context
                for j in range(max(0, i-1), min(len(lines), i+3)):
                    if j != i:
                        print(f"     {j+1}: {lines[j].strip()}")
                break
        
        # If cycling uses same calculation, the bug affects both
        print("\nIf cycling uses the same calculation pattern, it has the same unit issue.")
        
    except Exception as e:
        print(f"Error checking cycling: {e}")

def main():
    """Run complete verification"""
    
    print("DOUBLE-CHECKING THE UNIT CONVERSION HYPOTHESIS")
    print("=" * 60)
    
    verify_units_hypothesis()
    check_mobility_module_units()
    test_with_sample_data()
    check_other_distance_calculations()
    
    print()
    print("=" * 60)
    print("FINAL VERIFICATION RESULT:")
    print("Check the numbers above to confirm if the unit hypothesis is correct!")

if __name__ == "__main__":
    main()