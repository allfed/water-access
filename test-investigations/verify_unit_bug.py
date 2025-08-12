#!/usr/bin/env python3
"""
Verify the unit conversion bug between minutes and hours.
"""

print("=== UNIT CONVERSION BUG VERIFICATION ===")
print()

# The bug:
# - Documentation says time_gathering_water is in MINUTES
# - But calculate_max_distances uses it as HOURS
# - Default value of 5.5 would mean 5.5 minutes, not hours!

print("1. Documentation says: time_gathering_water in MINUTES")
print("2. calculate_max_distances uses it directly in velocity * time calculation")
print("3. Velocity is in m/s, so time should be in seconds or hours")
print()

# Let's calculate what happens
velocity = 1.316  # m/s

print("If time_gathering_water = 5.5:")
print()

# Scenario 1: Treated as minutes (BUG)
time_minutes = 5.5
time_hours_from_minutes = time_minutes / 60
distance_bug = velocity * time_hours_from_minutes * 3600 / 2 / 1000
print(f"  Scenario 1 - As MINUTES (5.5 min = {time_hours_from_minutes:.3f} hours):")
print(f"    Distance = {distance_bug:.2f} km")

# Scenario 2: Treated as hours (INTENDED)
time_hours = 5.5
distance_correct = velocity * time_hours * 3600 / 2 / 1000
print(f"  Scenario 2 - As HOURS (5.5 hours):")
print(f"    Distance = {distance_correct:.2f} km")

print()
print("Actual global model result: 3.71 km")
print()

# Check if minutes conversion explains it
print("To get 3.71 km, we need time in hours:")
required_time_hours = 3.71 * 2 * 1000 / (velocity * 3600)
required_time_minutes = required_time_hours * 60
print(f"  Required time: {required_time_hours:.3f} hours = {required_time_minutes:.1f} minutes")

print()
print("WAIT! Let's check if the code converts minutes to hours somewhere...")

# Maybe the default 16 in process_zones_for_water_access is the issue
print()
print("process_zones_for_water_access has default time_gathering_water=16")
print("If 16 is meant to be HOURS but docs say MINUTES...")

# What if someone passes 330 thinking it's minutes (5.5 hours)?
time_330_minutes = 330
time_330_as_hours = 330  # Bug: treating minutes value as hours!
distance_330_bug = velocity * time_330_as_hours * 3600 / 2 / 1000
print(f"\nIf someone passes 330 (minutes) but it's used as hours:")
print(f"  Distance = {distance_330_bug:.0f} km (way too high!)")

# The real issue might be mixed units
print("\n🔍 MIXED UNITS HYPOTHESIS:")
print("- run_global_analysis expects MINUTES (per docs)")
print("- But passes to calculate_max_distances which expects HOURS")
print("- NO CONVERSION happens between them!")

# Let's trace the actual values
print("\n🎯 TRACING THE BUG:")
print("1. run_global_analysis(time_gathering_water=5.5) # Meant as hours")
print("2. Docs say it's minutes, but no conversion happens")
print("3. calculate_max_distances gets 5.5 and uses as hours")
print("4. Result: Correct by accident if you ignore the docs!")
print()
print("But if someone reads the docs and passes 330 minutes...")
print("They would get wildly wrong results!")

# The 16 default might be a clue
print("\n🔍 THE 16 HOUR DEFAULT:")
print("process_zones_for_water_access(time_gathering_water=16)")
print("- 16 hours is unrealistic for a single water gathering trip")
print("- 16 hours might be total daily time available")
print("- But it's used as per-trip time in calculations!")

# Check what 16 hours gives
distance_16h = velocity * 16 * 3600 / 2 / 1000
print(f"\nWith 16 hours: distance = {distance_16h:.1f} km")
print("That's way too far for walking!")

print("\n" + "="*60)
print("CONCLUSION: There's a UNIT MISMATCH between documentation and code!")
print("- Docs say MINUTES but code uses HOURS")
print("- This causes confusion and potential bugs")
print("- The default of 16 in process_zones_for_water_access is suspicious")