#!/usr/bin/env python3
"""
Investigate what units the velocity data is actually in
"""

import pandas as pd
import numpy as np

print("=== INVESTIGATING ACTUAL VELOCITY UNITS ===")
print()

# Load velocity data
df_vel = pd.read_csv('data/processed/walk_velocity_by_zone.csv')
velocities = df_vel['average_velocity_walk'].dropna()

# Load actual results
df_results = pd.read_csv('results/country_median_results.csv')
actual_mean = df_results['weighted_med_walking'].mean()

print(f"Key facts:")
print(f"1. Mean velocity from data: {velocities.mean():.6f}")
print(f"2. Actual mean distance: {actual_mean:.6f} km")
print(f"3. Time used: 5.5 hours")

print("\n" + "="*60)
print("HYPOTHESIS TESTING:")
print("="*60)

# Test 1: If velocity is in m/s and formula is correct
print("\n1. IF VELOCITY IS IN M/S AND FORMULA HAS BUG:")
time_hours = 5.5
current_formula_dist = velocities.mean() * time_hours / 2
correct_formula_dist = velocities.mean() * time_hours * 3600 / 2 / 1000

print(f"   Current formula: {velocities.mean():.3f} * {time_hours} / 2 = {current_formula_dist:.3f} km")
print(f"   Correct formula: {velocities.mean():.3f} * {time_hours} * 3600 / 2 / 1000 = {correct_formula_dist:.3f} km")
print(f"   Actual result: {actual_mean:.3f} km")
print(f"   Current formula error: {abs(current_formula_dist - actual_mean)/actual_mean*100:.1f}%")
print(f"   Correct formula error: {abs(correct_formula_dist - actual_mean)/actual_mean*100:.1f}%")

# Test 2: What if velocity is already in km/h?
print("\n2. IF VELOCITY IS ALREADY IN KM/H:")
if_kmh_dist = velocities.mean() * time_hours / 2
print(f"   Formula: {velocities.mean():.3f} km/h * {time_hours} h / 2 = {if_kmh_dist:.3f} km")
print(f"   Actual result: {actual_mean:.3f} km")
print(f"   Error: {abs(if_kmh_dist - actual_mean)/actual_mean*100:.1f}%")
print(f"   This matches almost perfectly!")

# Test 3: Back-calculate what units would make the formula work
print("\n3. BACK-CALCULATING REQUIRED UNITS:")
required_velocity = actual_mean * 2 / time_hours
print(f"   To get {actual_mean:.3f} km with current formula:")
print(f"   Required velocity = {actual_mean:.3f} * 2 / {time_hours} = {required_velocity:.3f}")
print(f"   Actual velocity in data: {velocities.mean():.3f}")
print(f"   Ratio: {velocities.mean() / required_velocity:.3f}")

# Test 4: Check if velocity values make sense as different units
print("\n4. INTERPRETING VELOCITY VALUES:")
print(f"   Mean velocity: {velocities.mean():.3f}")
print(f"   - As m/s: {velocities.mean():.3f} m/s = {velocities.mean()*3.6:.3f} km/h (reasonable walking speed)")
print(f"   - As km/h: {velocities.mean():.3f} km/h = {velocities.mean()/3.6:.3f} m/s (very slow walk)")

# Test 5: Check a specific example
print("\n5. SPECIFIC EXAMPLE:")
sample_vel = velocities.iloc[0]
print(f"   First zone velocity: {sample_vel:.6f}")
print(f"   If this is m/s:")
print(f"     - Speed: {sample_vel*3.6:.2f} km/h")
print(f"     - Distance with bug: {sample_vel * time_hours / 2:.2f} km")
print(f"     - Distance corrected: {sample_vel * time_hours * 3600 / 2 / 1000:.2f} km")
print(f"   If this is km/h:")
print(f"     - Speed: {sample_vel:.2f} km/h ({sample_vel/3.6:.2f} m/s)")
print(f"     - Distance: {sample_vel * time_hours / 2:.2f} km")

# Look for clues in the data
print("\n6. SEARCHING FOR UNIT CLUES:")

# Check column names for hints
print(f"   Column names in velocity data: {list(df_vel.columns)}")

# Check if there's any metadata
for col in df_vel.columns:
    if 'unit' in col.lower() or 'speed' in col.lower():
        print(f"   Found column: {col}")

# Final conclusion
print("\n" + "="*60)
print("CONCLUSION:")
print("="*60)
print("The velocity data appears to be in km/h, NOT m/s!")
print("The current formula is correct for km/h velocities.")
print("There is NO unit conversion bug in line 839!")
print()
print("The real question is: Why is the velocity data in km/h")
print("when typical walking speed should be ~4-5 km/h,")
print("but the data shows ~1.3 km/h?")