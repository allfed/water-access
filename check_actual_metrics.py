#!/usr/bin/env python3
"""
Check what metrics are actually being compared
"""

print("=== WHAT'S ACTUALLY BEING COMPARED ===")

print("\nGLOBAL MODEL:")
print("- Calculates 'max distance cycling/walking' = velocity × time / 2")
print("- This is ONE-WAY distance to water source")
print("- Reports: 3.6 km (walking), 11.8 km (cycling)")
print("- These are the 'weighted_med_walking' and 'weighted_med_cycling' values")

print("\nSENSITIVITY ANALYSIS:")
print("- Calculates 'water_ration_kms' = velocity × load / water_ration × time")
print("- This is a different metric entirely!")
print("- Reports: ~13.5 km")

print("\nTHE REAL ISSUE:")
print("We're comparing apples to oranges!")
print("- Global: One-way distance to water source (km)")
print("- Sensitivity: Some kind of water transport efficiency metric")

print("\nTO COMPARE PROPERLY:")
print("We need to calculate the same metric in both models")

# Let's calculate what sensitivity analysis would give for simple distance
velocity = 3.0  # m/s (reasonable cycling speed)
time_hours = 5.5

one_way_distance = velocity * time_hours * 3600 / 2 / 1000
print(f"\nIf sensitivity calculated one-way distance:")
print(f"Distance = {velocity} m/s × {time_hours} h × 3600 / 2 / 1000 = {one_way_distance:.1f} km")

# With different slopes reducing performance
print(f"\nWith realistic slope reduction (30% slower):")
print(f"Distance = {one_way_distance * 0.7:.1f} km")

print("\nThis is much closer to the global model results!")