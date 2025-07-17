#!/usr/bin/env python3
"""
Demonstrate the water_ration_kms calculation difference
"""

# Example values
velocity = 3.0  # m/s
max_load = 20.0  # kg
time_hours = 5.5
water_ration = 15.0  # L

print("=== WATER_RATION_KMS CALCULATION COMPARISON ===")
print(f"\nTest values:")
print(f"  Velocity: {velocity} m/s")
print(f"  Max load: {max_load} kg")
print(f"  Time: {time_hours} hours")
print(f"  Water ration: {water_ration} L")

print("\n--- SENSITIVITY ANALYSIS CALCULATION ---")
# From sensitivity_analysis.py line 319:
# water_ration_kms = mean_vel_kg_per_slope / mv.waterration * t_secs / 1000

mean_vel_kg = velocity * max_load  # velocity × load
print(f"Step 1: mean_vel_kg = velocity × max_load = {velocity} × {max_load} = {mean_vel_kg}")

t_secs = time_hours * 60 * 60
print(f"Step 2: t_secs = {time_hours} × 60 × 60 = {t_secs}")

water_ration_kms_sens = mean_vel_kg / water_ration * t_secs / 1000
print(f"Step 3: water_ration_kms = mean_vel_kg / water_ration × t_secs / 1000")
print(f"        water_ration_kms = {mean_vel_kg} / {water_ration} × {t_secs} / 1000")
print(f"        water_ration_kms = {water_ration_kms_sens:.2f}")

print(f"\nInterpretation: {water_ration_kms_sens:.2f} km per liter of water")
print(f"(This is distance you can travel per liter)")

print("\n--- GLOBAL MODEL CALCULATION ---")
# From gis_global_module.py lines 835-843:
# max distance cycling = average_velocity_bicycle * time_gathering_water / 2
# water_ration_kms = max distance cycling * max_load_bicycle

max_distance = velocity * time_hours * 3600 / 2 / 1000  # Convert to km
print(f"Step 1: max_distance = velocity × time × 3600 / 2 / 1000")
print(f"        max_distance = {velocity} × {time_hours} × 3600 / 2 / 1000")
print(f"        max_distance = {max_distance:.2f} km (one-way)")

water_ration_kms_global = max_distance * max_load
print(f"Step 2: water_ration_kms = max_distance × max_load")
print(f"        water_ration_kms = {max_distance:.2f} × {max_load}")
print(f"        water_ration_kms = {water_ration_kms_global:.2f}")

print(f"\nInterpretation: {water_ration_kms_global:.2f} L×km of water transport capacity")
print(f"(This is total liters × kilometers that can be transported)")

print("\n--- THE KEY DIFFERENCE ---")
print(f"Sensitivity: {water_ration_kms_sens:.2f} (divides by {water_ration}L)")
print(f"Global: {water_ration_kms_global:.2f} (doesn't divide by water ration)")
print(f"Ratio: {water_ration_kms_global / water_ration_kms_sens:.2f}x")

print("\n--- WHAT THEY ACTUALLY MEASURE ---")
print("Sensitivity: km per liter (efficiency metric)")
print("Global: total L×km capacity (absolute metric)")
print(f"\nTo convert sensitivity to same units as global:")
print(f"{water_ration_kms_sens:.2f} km/L × {water_ration} L = {water_ration_kms_sens * water_ration:.2f} km")
print(f"But this is total distance, not distance × load!")

print("\n--- THE REAL COMPARISON ---")
print("If we want to compare apples to apples:")
print(f"Sensitivity approach: {mean_vel_kg} kg⋅m/s × {t_secs}s / 1000 = {mean_vel_kg * t_secs / 1000:.2f} kg⋅km")
print(f"Global approach: {water_ration_kms_global:.2f} L⋅km")
print(f"These should be similar (since 1L ≈ 1kg for water)")

print("\nThe 7.5x difference comes from sensitivity dividing by 15L!")