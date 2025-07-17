#!/usr/bin/env python3
"""
Revised analysis of the calculation difference
"""

# Test values
velocity = 3.0  # m/s
max_load = 20.0  # kg  
time_hours = 5.5
water_ration = 15.0  # L

print("=== REVISED ANALYSIS ===")

print("\n1. WHAT EACH MODEL CALCULATES (without the division by 15):")

# Sensitivity (if we remove the division by 15)
vel_kg = velocity * max_load
total_kg_km_sens = vel_kg * time_hours * 3600 / 1000
print(f"Sensitivity (full time): {vel_kg} kg⋅m/s × {time_hours}h × 3600s/h / 1000 = {total_kg_km_sens:.0f} kg⋅km")

# Global 
one_way_distance = velocity * time_hours * 3600 / 2 / 1000
total_kg_km_global = one_way_distance * max_load
print(f"Global (one-way): {one_way_distance:.1f} km × {max_load} kg = {total_kg_km_global:.0f} kg⋅km")

print(f"\nRatio: {total_kg_km_sens / total_kg_km_global:.1f}x")
print("This 2x difference is because global uses time/2 (one-way distance)")

print("\n2. THE ACTUAL FORMULAS:")
print(f"Sensitivity: {total_kg_km_sens:.0f} / {water_ration} = {total_kg_km_sens/water_ration:.1f} km")
print(f"Global: {total_kg_km_global:.0f} kg⋅km (no division)")

print("\n3. WHAT'S REALLY HAPPENING:")
print("The variable name 'water_ration_kms' is misleading!")
print("- Sensitivity: measures km achievable per liter of water ration")
print("- Global: measures total water transport capacity (L×km)")
print("\nThese are fundamentally different metrics!")

print("\n4. TO COMPARE PROPERLY:")
print("We need to decide what we're measuring:")
print("a) One-way distance to water source?")
print("b) Total water transport capacity?")
print("c) Distance achievable per unit of water?")

# Check the original notebook calculation
print("\n5. ORIGINAL NOTEBOOK LOGIC:")
print("From the manuscript, we expect water ration distance of ~13.5 km")
print("This likely means: distance you can travel to get your daily water ration")
print(f"With {water_ration}L needed and {max_load}kg capacity:")
print(f"  One-way distance possible: {one_way_distance:.1f} km")
print(f"  But need {water_ration/max_load:.2f} trips for daily ration")
print(f"  Effective range: {one_way_distance * max_load / water_ration:.1f} km")