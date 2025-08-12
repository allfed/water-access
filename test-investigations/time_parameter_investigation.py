#!/usr/bin/env python3
"""
Investigation into the time parameter discrepancy.
The global model appears to use ~1.56 hours instead of 5.5 hours.
"""

import pandas as pd
import numpy as np
import sys
import os
from pathlib import Path
import re

def search_time_gathering_water():
    """Search for all occurrences of time_gathering_water in the codebase."""
    
    print("=== SEARCHING FOR TIME_GATHERING_WATER USAGE ===")
    
    files_to_search = [
        'src/gis_global_module.py',
        'src/gis_monte_carlo.py',
        'scripts/run_monte_carlo.py',
        'scripts/null_case.ipynb'
    ]
    
    findings = []
    
    for file_path in files_to_search:
        if os.path.exists(file_path):
            print(f"\nSearching {file_path}:")
            
            with open(file_path, 'r') as f:
                content = f.read()
            
            lines = content.split('\n')
            
            # Find all occurrences
            for i, line in enumerate(lines):
                if 'time_gathering_water' in line:
                    print(f"  Line {i+1}: {line.strip()}")
                    
                    # Check for any division or multiplication
                    if '/' in line or '*' in line:
                        print(f"    🔍 MATH OPERATION DETECTED")
                        findings.append({
                            'file': file_path,
                            'line': i+1,
                            'content': line.strip(),
                            'has_math': True
                        })
                    else:
                        findings.append({
                            'file': file_path,
                            'line': i+1,
                            'content': line.strip(),
                            'has_math': False
                        })
                    
                    # Get context
                    for j in range(max(0, i-2), min(len(lines), i+3)):
                        if j != i:
                            print(f"    Context {j+1}: {lines[j].strip()}")
    
    return findings

def check_distance_calculation():
    """Check the exact distance calculation formula."""
    
    print("\n=== CHECKING DISTANCE CALCULATION ===")
    
    gis_file = 'src/gis_global_module.py'
    
    if os.path.exists(gis_file):
        with open(gis_file, 'r') as f:
            content = f.read()
        
        lines = content.split('\n')
        
        # Find the calculate_max_distances function
        for i, line in enumerate(lines):
            if 'def calculate_max_distances' in line:
                print(f"\nFound function at line {i+1}")
                
                # Print the function
                j = i
                while j < len(lines) and (j == i or not lines[j].startswith('def ')):
                    print(f"{j+1}: {lines[j]}")
                    j += 1
                    if j - i > 30:  # Limit output
                        break
                
                break

def analyze_time_values():
    """Analyze what time values are actually being used."""
    
    print("\n=== ANALYZING TIME VALUES ===")
    
    # Check default values in function definitions
    search_patterns = [
        r'time_gathering_water\s*=\s*(\d+\.?\d*)',
        r'time_gathering_water:\s*float\s*=\s*(\d+\.?\d*)',
        r'time_gathering_water\s*=\s*(\d+)',
    ]
    
    files = ['src/gis_global_module.py', 'src/gis_monte_carlo.py', 'scripts/run_monte_carlo.py']
    
    for file_path in files:
        if os.path.exists(file_path):
            print(f"\nChecking {file_path}:")
            
            with open(file_path, 'r') as f:
                content = f.read()
            
            for pattern in search_patterns:
                matches = re.findall(pattern, content)
                if matches:
                    print(f"  Time values found: {matches}")
                    
                    # Convert to float and check
                    for match in matches:
                        time_val = float(match)
                        print(f"    - {time_val} hours")
                        
                        if abs(time_val - 1.56) < 0.1:
                            print(f"      🚨 FOUND SUSPICIOUS VALUE CLOSE TO 1.56!")
                        elif abs(time_val - 16) < 1:
                            print(f"      🔍 Found 16 - might be total daily hours?")

def check_water_access_calculations():
    """Check if there's a specific water access time calculation."""
    
    print("\n=== CHECKING WATER ACCESS CALCULATIONS ===")
    
    # Look for any calculations that might result in 1.56 hours
    possible_calculations = [
        "16 / 10.3 = 1.55",  # Some kind of division?
        "5.5 / 3.5 = 1.57",  # Division by some factor?
        "5.5 * 0.284 = 1.56",  # Multiplication by some factor?
    ]
    
    print("Possible calculations that could yield ~1.56 hours:")
    for calc in possible_calculations:
        print(f"  - {calc}")
    
    # Check if there's a daily limit being applied
    print("\nChecking if daily walking limit affects time:")
    
    # From our earlier findings
    daily_limit = 20  # km
    round_trip_distance = 3.71 * 2  # km
    trips_per_day = daily_limit / round_trip_distance
    
    print(f"  - Daily limit: {daily_limit} km")
    print(f"  - Round trip distance: {round_trip_distance:.2f} km")
    print(f"  - Max trips per day: {trips_per_day:.2f}")
    print(f"  - Time per trip if 5.5h total: {5.5/trips_per_day:.2f} hours")
    
    # Check if there's a "per trip" time being used
    effective_time = 1.56
    total_daily_time = effective_time * trips_per_day
    print(f"  - If 1.56h per trip: {total_daily_time:.2f} hours total daily")

def trace_velocity_to_distance():
    """Trace how velocity becomes distance in the pipeline."""
    
    print("\n=== TRACING VELOCITY TO DISTANCE ===")
    
    # We know:
    # - Raw velocity: 1.316 m/s
    # - Expected: velocity * 5.5 * 3600 / 2 / 1000 = 13.02 km
    # - Actual: 3.71 km
    
    raw_velocity = 1.316  # m/s
    
    print(f"Starting with raw velocity: {raw_velocity:.3f} m/s")
    
    # Test different time values
    test_times = [1.0, 1.5, 1.56, 2.0, 2.75, 3.0, 5.5, 16.0]
    
    print("\nTesting different time values:")
    for time_h in test_times:
        distance = raw_velocity * time_h * 3600 / 2 / 1000
        print(f"  Time = {time_h:4.2f}h → Distance = {distance:5.2f} km", end="")
        
        if abs(distance - 3.71) < 0.1:
            print(" 🎯 MATCHES ACTUAL RESULT!")
        else:
            print()

def check_process_zones_function():
    """Check the process_zones_for_water_access function which might use different time."""
    
    print("\n=== CHECKING PROCESS_ZONES_FOR_WATER_ACCESS ===")
    
    gis_file = 'src/gis_global_module.py'
    
    if os.path.exists(gis_file):
        with open(gis_file, 'r') as f:
            content = f.read()
        
        lines = content.split('\n')
        
        # Find the process_zones_for_water_access function
        for i, line in enumerate(lines):
            if 'def process_zones_for_water_access' in line:
                print(f"\nFound function at line {i+1}")
                
                # Check the default parameter
                if 'time_gathering_water' in line:
                    print(f"  🔍 Function signature: {line.strip()}")
                    
                    # Extract default value
                    match = re.search(r'time_gathering_water\s*=\s*(\d+)', line)
                    if match:
                        default_time = int(match.group(1))
                        print(f"  🚨 DEFAULT TIME: {default_time} (hours or minutes?)")
                        
                        if default_time == 16:
                            print(f"  🔍 This might be DAILY hours, not per-trip hours!")
                
                # Print the function body
                j = i + 1
                while j < len(lines) and (lines[j].startswith(' ') or lines[j].strip() == ''):
                    if 'calculate_max_distances' in lines[j]:
                        print(f"  Line {j+1}: {lines[j]}")
                        print(f"    🔍 CALLS calculate_max_distances")
                    j += 1
                    if j - i > 20:
                        break

def find_the_bug():
    """Try to pinpoint the exact bug location."""
    
    print("\n=== FINDING THE BUG ===")
    
    # Based on our investigation, let's check specific hypotheses
    
    print("HYPOTHESIS 1: time_gathering_water=16 is being used as daily hours")
    print("  - 16 hours daily ÷ multiple trips = fewer hours per trip")
    print("  - This would explain the reduction")
    
    print("\nHYPOTHESIS 2: Division by 2 is applied twice")
    print("  - Once in velocity calculation (loaded + unloaded) / 2")
    print("  - Again in distance calculation (round trip) / 2")
    print("  - But this would give 2x error, not 3.5x")
    
    print("\nHYPOTHESIS 3: Water access specific time")
    print("  - process_zones_for_water_access might use different time")
    print("  - Default parameter might be 16 instead of 5.5")
    
    print("\nMOST LIKELY: Check process_zones_for_water_access(time_gathering_water=16)")

def main():
    """Run the time parameter investigation."""
    
    print("🕐 TIME PARAMETER INVESTIGATION")
    print("=" * 60)
    print("Finding why 1.56 hours is used instead of 5.5 hours")
    print()
    
    # Search for time_gathering_water usage
    findings = search_time_gathering_water()
    
    # Check distance calculation
    check_distance_calculation()
    
    # Analyze time values
    analyze_time_values()
    
    # Check water access calculations
    check_water_access_calculations()
    
    # Trace velocity to distance
    trace_velocity_to_distance()
    
    # Check process_zones function
    check_process_zones_function()
    
    # Try to find the bug
    find_the_bug()
    
    print("\n" + "=" * 60)
    print("INVESTIGATION SUMMARY:")
    print("The most likely cause is that time_gathering_water=16 (daily hours)")
    print("is being used somewhere instead of 5.5 (hours per trip).")
    print("Check process_zones_for_water_access function!")

if __name__ == "__main__":
    main()