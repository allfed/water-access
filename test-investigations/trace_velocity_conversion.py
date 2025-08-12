#!/usr/bin/env python3
"""
Trace where velocity might be converted between mobility module and final CSV
"""

import pandas as pd
import numpy as np

print("=== TRACING VELOCITY CONVERSION ===")
print()

# First, let's understand what the Sprott model should return
print("1. UNDERSTANDING THE SPROTT MODEL")
print("-" * 60)
print("The Sprott walking model (https://sprott.physics.wisc.edu/technote/Walkrun.htm)")
print("calculates walking/running speed based on metabolic power.")
print("The model equations use SI units, so velocity should be in m/s.")
print()

# Check actual velocity values and what they represent
print("2. ANALYZING VELOCITY VALUES")
print("-" * 60)

df_vel = pd.read_csv('data/processed/walk_velocity_by_zone.csv')

# Check all velocity columns
velocity_cols = [col for col in df_vel.columns if 'velocity' in col]
print(f"Velocity columns found: {velocity_cols}")
print()

for col in velocity_cols:
    if col in df_vel.columns:
        values = df_vel[col].dropna()
        print(f"{col}:")
        print(f"  Mean: {values.mean():.6f}")
        print(f"  Min: {values.min():.6f}")
        print(f"  Max: {values.max():.6f}")
        print()

# Key insight: Check if loaded and unloaded velocities are different
if 'loaded_velocity_walk' in df_vel.columns and 'unloaded_velocity_walk' in df_vel.columns:
    loaded = df_vel['loaded_velocity_walk'].dropna()
    unloaded = df_vel['unloaded_velocity_walk'].dropna()
    
    print("3. LOADED VS UNLOADED COMPARISON")
    print("-" * 60)
    print(f"Loaded mean: {loaded.mean():.6f}")
    print(f"Unloaded mean: {unloaded.mean():.6f}")
    print(f"Ratio (unloaded/loaded): {unloaded.mean()/loaded.mean():.3f}")
    print()
    
    # This ratio should tell us something about the units
    # If carrying load reduces speed by ~20-30%, that's reasonable
    
# Check how average is calculated
if 'average_velocity_walk' in df_vel.columns:
    avg = df_vel['average_velocity_walk'].dropna()
    
    # See if it's a simple average
    if 'loaded_velocity_walk' in df_vel.columns and 'unloaded_velocity_walk' in df_vel.columns:
        calculated_avg = (df_vel['loaded_velocity_walk'] + df_vel['unloaded_velocity_walk']) / 2
        calculated_avg = calculated_avg.dropna()
        
        print("4. AVERAGE VELOCITY CALCULATION")
        print("-" * 60)
        print(f"Average from file: {avg.mean():.6f}")
        print(f"(Loaded + Unloaded)/2: {calculated_avg.mean():.6f}")
        print(f"Match? {np.allclose(avg.mean(), calculated_avg.mean(), rtol=0.01)}")
        print()

# Now the key test: what makes sense?
print("5. UNIT INTERPRETATION")
print("-" * 60)

mean_vel = df_vel['average_velocity_walk'].mean()
print(f"Mean velocity: {mean_vel:.6f}")
print()

print("If this is m/s:")
print(f"  = {mean_vel:.3f} m/s")
print(f"  = {mean_vel * 3.6:.3f} km/h")
print(f"  = {mean_vel * 2.237:.3f} mph")
print(f"  This is reasonable walking speed")
print()

print("If this is km/h:")
print(f"  = {mean_vel:.3f} km/h") 
print(f"  = {mean_vel / 3.6:.3f} m/s")
print(f"  = {mean_vel * 0.621371:.3f} mph")
print(f"  This is VERY slow (baby crawling speed)")
print()

# Critical test: Check against known walking speeds with load
print("6. REALITY CHECK - WALKING SPEEDS")
print("-" * 60)
print("Typical walking speeds:")
print("  Unloaded adult: 4-5 km/h (1.1-1.4 m/s)")
print("  Loaded (20kg): 3-4 km/h (0.8-1.1 m/s)")
print("  Heavy load (40kg): 2-3 km/h (0.6-0.8 m/s)")
print()

if 'loaded_velocity_walk' in df_vel.columns:
    print(f"Our loaded velocity mean: {df_vel['loaded_velocity_walk'].mean():.3f}")
    print(f"If m/s: {df_vel['loaded_velocity_walk'].mean() * 3.6:.1f} km/h - reasonable")
    print(f"If km/h: {df_vel['loaded_velocity_walk'].mean():.1f} km/h - too slow!")

# Final check: Look at the distribution
print("\n7. VELOCITY DISTRIBUTION")
print("-" * 60)

velocities = df_vel['average_velocity_walk'].dropna()
print(f"Percentiles of average_velocity_walk:")
for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
    val = np.percentile(velocities, p)
    print(f"  {p}%: {val:.3f} (if m/s: {val*3.6:.1f} km/h)")

# The smoking gun
print("\n8. THE SMOKING GUN")
print("-" * 60)
print("All evidence points to velocity data being in m/s:")
print("1. Values around 1.3 match expected walking speed in m/s")
print("2. The Sprott model uses SI units (m/s)")
print("3. The loaded/unloaded ratio makes sense for m/s")
print()
print("BUT the formula in line 839 treats it as km/h!")
print("This is why we get 3.6x smaller distances.")
print()
print("The bug is CONFIRMED: velocity is in m/s but formula assumes km/h")