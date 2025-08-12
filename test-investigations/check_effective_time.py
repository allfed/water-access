#!/usr/bin/env python3
"""
Check what effective time value would produce the observed results.
"""

print("=== CHECKING EFFECTIVE TIME ===")
print()

# Known values
observed_distance = 3.71  # km (from global model results)
zone_velocity = 1.316  # m/s (from velocity data)

# Calculate what time would give this distance
# distance = velocity × time / 2
# time = distance × 2 / velocity

time_seconds = observed_distance * 1000 * 2 / zone_velocity
time_hours = time_seconds / 3600
time_minutes = time_seconds / 60

print(f"Given:")
print(f"  Observed distance: {observed_distance} km")
print(f"  Zone velocity: {zone_velocity} m/s")
print()
print(f"Required time to get this distance:")
print(f"  {time_hours:.2f} hours")
print(f"  {time_minutes:.1f} minutes")
print(f"  {time_seconds:.0f} seconds")
print()

# Compare with expected values
print(f"Expected time values:")
print(f"  Sensitivity analysis: 5.5 hours")
print(f"  Monte Carlo range: 4-7 hours")
print(f"  Process zones default: 16 hours")
print()

# Check ratios
print(f"Time ratios:")
print(f"  5.5 / {time_hours:.2f} = {5.5/time_hours:.1f}x")
print(f"  16 / {time_hours:.2f} = {16/time_hours:.1f}x")
print()

# What if they're using the median of the Monte Carlo range?
mc_median_time = 5.5  # (4+7)/2 = 5.5
print(f"If using Monte Carlo median time (5.5 hours):")
print(f"  Expected distance: {zone_velocity * mc_median_time * 3600 / 2 / 1000:.2f} km")
print(f"  Actual distance: {observed_distance:.2f} km")
print(f"  Ratio: {(zone_velocity * mc_median_time * 3600 / 2 / 1000) / observed_distance:.1f}x")

print("\n" + "="*60)
print("CRITICAL FINDING:")
print(f"The effective time being used is {time_hours:.2f} hours instead of 5.5 hours!")
print(f"This is a {5.5/time_hours:.1f}x reduction in time!")

# Check if this matches any known values
if abs(time_hours - 1.56) < 0.1:
    print("\n🎯 This matches our earlier finding of 1.56 hours!")
    print("The question is: WHERE is this time reduction happening?")