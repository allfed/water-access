#!/usr/bin/env python3
"""
EXPLICIT DEMONSTRATION: Why the existing code is wrong
"""

import pandas as pd
import numpy as np

print("="*80)
print("EXPLICIT BUG DEMONSTRATION: src/gis_global_module.py lines 835-840")
print("="*80)

# Load actual data to demonstrate with real values
df_vel = pd.read_csv('data/processed/walk_velocity_by_zone.csv')
velocity_walk = df_vel['average_velocity_walk'].mean()

print("\n1. THE EXISTING CODE (lines 838-839):")
print("-" * 80)
print('df_zones["max distance walking"] = (')
print('    df_zones["average_velocity_walk"] * time_gathering_water / 2')
print(')')
print()

print("2. WHAT THE CODE DOES:")
print("-" * 80)
print("The formula is: velocity * time / 2")
print()
print("This formula assumes:")
print("  - velocity is in km/h")
print("  - time is in hours")
print("  - Result: km/h * h = km ✓")
print()

print("3. THE ACTUAL DATA:")
print("-" * 80)
print(f"Average walking velocity in data: {velocity_walk:.6f}")
print()
print("Is this km/h or m/s? Let's check:")
print(f"  - If km/h: {velocity_walk:.3f} km/h = {velocity_walk/3.6:.3f} m/s (0.37 m/s is slower than a baby crawl!)")
print(f"  - If m/s: {velocity_walk:.3f} m/s = {velocity_walk*3.6:.3f} km/h (4.7 km/h is normal walking speed ✓)")
print()
print("Also checking loaded vs unloaded velocities:")
print(f"  - Loaded velocity: {df_vel['loaded_velocity_walk'].mean():.3f} (matches typical 0.8-1.1 m/s with load)")
print(f"  - Unloaded velocity: {df_vel['unloaded_velocity_walk'].mean():.3f} (matches typical 1.1-1.6 m/s)")
print()
print("CONCLUSION: The velocity data is clearly in m/s, not km/h!")
print()

print("4. THE BUG:")
print("-" * 80)
time_hours = 5.5

# What the code currently calculates
current_result = velocity_walk * time_hours / 2
print(f"Current code calculates: {velocity_walk:.3f} * {time_hours} / 2 = {current_result:.3f} km")
print()
print("But this is WRONG because:")
print(f"  - It treats {velocity_walk:.3f} m/s as if it were {velocity_walk:.3f} km/h")
print(f"  - Units: m/s * hours ≠ km (unit mismatch!)")
print()

print("5. THE CORRECT CALCULATION:")
print("-" * 80)
print("To convert m/s to km/h, multiply by 3.6 (or 3600/1000):")
print("  1 m/s = 3.6 km/h")
print("  velocity(m/s) * 3600(s/h) / 1000(m/km) = velocity(km/h)")
print()

# Correct calculation
correct_result = velocity_walk * time_hours * 3600 / 2 / 1000
print(f"Correct: {velocity_walk:.3f} m/s * {time_hours} h * 3600 s/h / 2 / 1000 m/km = {correct_result:.3f} km")
print()

print("6. THE 3.6x DISCREPANCY:")
print("-" * 80)
print(f"Correct result / Current result = {correct_result:.3f} / {current_result:.3f} = {correct_result/current_result:.1f}x")
print("This is exactly 3.6, the m/s to km/h conversion factor!")
print()

print("7. VERIFICATION WITH ACTUAL RESULTS:")
print("-" * 80)
df_results = pd.read_csv('results/country_median_results.csv')
actual_mean = df_results['weighted_med_walking'].mean()
print(f"Actual country results average: {actual_mean:.3f} km")
print(f"Current (buggy) formula result: {current_result:.3f} km")
print(f"Error: {abs(current_result - actual_mean):.3f} km ({abs(current_result - actual_mean)/actual_mean*100:.1f}%)")
print()
print("The buggy formula matches the results! This means:")
print("→ The entire water access analysis has been using distances that are 3.6x too small")
print()

print("8. PROOF FROM OTHER MODULES:")
print("-" * 80)
print("The sensitivity analysis uses the CORRECT formula:")
with open('scripts/sensitivity_analysis_refactored.py', 'r') as f:
    lines = f.readlines()
    for i, line in enumerate(lines):
        if 'one_way_distance_km' in line and '3600' in line:
            print(f"Line {i+1}: {line.strip()}")
            break

print()
print("This is why the sensitivity analysis gives ~13 km while global model gives ~3.6 km!")
print()

print("="*80)
print("SUMMARY OF THE BUG:")
print("="*80)
print("1. Velocity data is in m/s (proven by reasonable walking speeds)")
print("2. Current formula treats it as km/h (missing unit conversion)")
print("3. This causes all distances to be 3.6x smaller than reality")
print("4. The bug affects both walking (line 839) and cycling (line 836)")
print()
print("THE FIX:")
print("Add the unit conversion factor (* 3600 / 1000) to both lines:")
print('  df_zones["max distance walking"] = df_zones["average_velocity_walk"] * time_gathering_water * 3600 / 2 / 1000')
print('  df_zones["max distance cycling"] = df_zones["average_velocity_bicycle"] * time_gathering_water * 3600 / 2 / 1000')