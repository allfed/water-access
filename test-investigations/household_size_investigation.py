#!/usr/bin/env python3
"""
Investigate if household size calculations are causing the 3.6x discrepancy.
"""

import pandas as pd
import numpy as np

print("=== HOUSEHOLD SIZE INVESTIGATION ===")
print()

def check_household_size_factor():
    """Check if 3.6 matches the population-weighted average household size."""
    
    print("1. CHECKING IF 3.6 MATCHES HOUSEHOLD SIZES")
    print()
    
    # Load household size data
    try:
        df = pd.read_csv('data/processed/merged_data.csv')
        
        if 'Household_Size' in df.columns and 'Population' in df.columns:
            # Remove missing values
            household_data = df[['Household_Size', 'Population', 'Entity']].dropna()
            
            print(f"Countries with household size data: {len(household_data)}")
            
            # Calculate population-weighted average household size
            total_pop = household_data['Population'].sum()
            weighted_household_size = (household_data['Household_Size'] * household_data['Population']).sum() / total_pop
            
            print(f"Population-weighted average household size: {weighted_household_size:.2f}")
            print(f"Simple average household size: {household_data['Household_Size'].mean():.2f}")
            print(f"Median household size: {household_data['Household_Size'].median():.2f}")
            
            # Check how close this is to 3.6
            difference = abs(weighted_household_size - 3.6)
            print(f"Difference from 3.6: {difference:.2f}")
            
            if difference < 0.2:
                print("🎯 CLOSE MATCH! Household size could explain the 3.6x factor")
            else:
                print("❌ Not a close match to 3.6")
            
            # Show distribution
            print(f"\nHousehold size distribution:")
            print(f"  Range: {household_data['Household_Size'].min():.1f} - {household_data['Household_Size'].max():.1f}")
            print(f"  Std dev: {household_data['Household_Size'].std():.2f}")
            
            # Show some examples
            print(f"\nExamples:")
            for i in range(min(10, len(household_data))):
                row = household_data.iloc[i]
                print(f"  {row['Entity']}: {row['Household_Size']:.1f} people/household")
            
            return weighted_household_size
            
    except Exception as e:
        print(f"Error loading household data: {e}")
        return None

def search_for_household_water_calculations():
    """Search for any calculations that might multiply water needs by household size."""
    
    print("\n2. SEARCHING FOR HOUSEHOLD WATER CALCULATIONS")
    print()
    
    # Search in the main GIS module for household size usage
    gis_file = 'src/gis_global_module.py'
    
    try:
        with open(gis_file, 'r') as f:
            content = f.read()
        
        lines = content.split('\n')
        
        household_lines = []
        for i, line in enumerate(lines):
            if 'Household_Size' in line or 'household' in line.lower():
                household_lines.append((i+1, line.strip()))
        
        print(f"Found {len(household_lines)} lines mentioning household:")
        for line_num, line in household_lines:
            print(f"  Line {line_num}: {line}")
            
    except Exception as e:
        print(f"Error reading GIS file: {e}")

def check_for_per_household_water_requirement():
    """Check if there's any calculation of water requirement per household."""
    
    print("\n3. CHECKING FOR PER-HOUSEHOLD WATER CALCULATIONS")
    print()
    
    # Look for any multiplication of 15L by household size
    search_terms = [
        "15.*household",
        "household.*15", 
        "water.*household",
        "household.*water"
    ]
    
    files_to_search = [
        'src/gis_global_module.py',
        'src/mobility_module.py',
        'scripts/run_monte_carlo.py'
    ]
    
    for file_path in files_to_search:
        try:
            with open(file_path, 'r') as f:
                content = f.read()
            
            print(f"\nSearching {file_path}:")
            
            # Look for any mention of household in water-related context
            lines = content.split('\n')
            found_relevant = False
            
            for i, line in enumerate(lines):
                if ('household' in line.lower() and 
                    any(term in line.lower() for term in ['water', '15', 'ration'])):
                    print(f"  Line {i+1}: {line.strip()}")
                    found_relevant = True
            
            if not found_relevant:
                print("  No household-water calculations found")
                
        except Exception as e:
            print(f"  Error reading {file_path}: {e}")

def theoretical_calculation_test():
    """Test if water requirement per household could explain the discrepancy."""
    
    print("\n4. THEORETICAL CALCULATION TEST")
    print()
    
    # Scenario 1: Per person water requirement (current understanding)
    water_per_person = 15  # L/day
    velocity = 1.316  # m/s
    time_hours = 5.5
    
    distance_per_person = velocity * time_hours * 3600 / 2 / 1000
    print(f"Scenario 1 - Per person (15L/person):")
    print(f"  Distance: {distance_per_person:.2f} km")
    
    # Scenario 2: Per household water requirement 
    avg_household_size = 3.6  # hypothesis
    water_per_household = water_per_person * avg_household_size  # 54L/household
    
    # If someone calculates distance for household, then divides by household size
    distance_per_household = velocity * time_hours * 3600 / 2 / 1000  # Same calculation
    distance_per_person_from_household = distance_per_household / avg_household_size
    
    print(f"\nScenario 2 - Per household calculation error:")
    print(f"  Water per household: {water_per_household:.1f} L")
    print(f"  Distance per household: {distance_per_household:.2f} km")
    print(f"  Distance per person (divided): {distance_per_person_from_household:.2f} km")
    print(f"  Reduction factor: {distance_per_person / distance_per_person_from_household:.1f}x")
    
    # Check if this matches observed results
    observed_distance = 3.71
    print(f"\nComparison with observed results:")
    print(f"  Observed: {observed_distance:.2f} km")
    print(f"  Scenario 2 prediction: {distance_per_person_from_household:.2f} km")
    print(f"  Error: {abs(observed_distance - distance_per_person_from_household):.2f} km")
    
    if abs(observed_distance - distance_per_person_from_household) < 0.5:
        print("🎯 EXCELLENT MATCH! This could be the issue!")
    else:
        print("❌ Not a good match")

def check_weighted_median_calculation():
    """Check if the weighted median calculation involves household size."""
    
    print("\n5. CHECKING WEIGHTED MEDIAN CALCULATION")
    print()
    
    # Look at the weighted percentile function
    gis_file = 'src/gis_global_module.py'
    
    try:
        with open(gis_file, 'r') as f:
            content = f.read()
        
        # Find the weighted_percentile function
        lines = content.split('\n')
        
        for i, line in enumerate(lines):
            if 'def weighted_percentile' in line:
                print("Found weighted_percentile function:")
                
                # Print the function
                j = i
                while j < len(lines) and (j == i or lines[j].startswith(' ') or lines[j].strip() == ''):
                    print(f"  {j+1}: {lines[j]}")
                    j += 1
                    if j - i > 15:  # Limit output
                        break
                break
        
        # Check how it's called for walking distances
        for i, line in enumerate(lines):
            if 'weighted_percentile' in line and 'walking' in line:
                print(f"\nWeighted percentile call for walking:")
                print(f"  Line {i+1}: {line.strip()}")
                
                # Show context
                for k in range(max(0, i-2), min(len(lines), i+3)):
                    if k != i:
                        print(f"    {k+1}: {lines[k].strip()}")
        
    except Exception as e:
        print(f"Error: {e}")

def main():
    """Run the household size investigation."""
    
    print("INVESTIGATING THE HOUSEHOLD SIZE HYPOTHESIS")
    print("=" * 60)
    
    # Check if 3.6 matches household sizes
    avg_household_size = check_household_size_factor()
    
    # Search for household calculations
    search_for_household_water_calculations()
    
    # Check for per-household water calculations
    check_for_per_household_water_requirement()
    
    # Test theoretical calculation
    theoretical_calculation_test()
    
    # Check weighted median calculation
    check_weighted_median_calculation()
    
    print("\n" + "=" * 60)
    print("CONCLUSION:")
    
    if avg_household_size and abs(avg_household_size - 3.6) < 0.2:
        print("🎯 HOUSEHOLD SIZE HYPOTHESIS IS VIABLE!")
        print("The population-weighted household size is close to 3.6")
        print("This could explain the 3.6x reduction if there's a calculation error")
    else:
        print("❌ HOUSEHOLD SIZE HYPOTHESIS UNLIKELY")
        print("The household size doesn't match the 3.6x factor")

if __name__ == "__main__":
    main()