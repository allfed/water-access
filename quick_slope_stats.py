#!/usr/bin/env python3
"""
Quick Slope Statistics from GIS Data
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Load sample GIS data
project_root = Path(__file__).resolve().parent
gis_file = project_root / "data/GIS/GIS_data_zones_sample.csv"

print("Loading GIS sample data...")
df = pd.read_csv(gis_file)
print(f"Loaded {len(df)} rows")

# Extract slope data
slopes = df['slope_1'].dropna()
print(f"\nSlope data points: {len(slopes)}")

# Check units
print(f"Raw slope range: {slopes.min():.4f} to {slopes.max():.4f}")

# Slopes appear to be in reasonable range (likely degrees already)
if slopes.max() < 2:
    print("Converting from radians to degrees")
    slopes = slopes * 180 / np.pi

# Basic statistics
print(f"\nSlope Statistics (degrees):")
print(f"  Mean: {slopes.mean():.2f}")
print(f"  Median: {slopes.median():.2f}")
print(f"  Std Dev: {slopes.std():.2f}")
print(f"  Min: {slopes.min():.2f}")
print(f"  Max: {slopes.max():.2f}")

# Percentiles
print(f"\nPercentiles:")
for p in [10, 25, 50, 75, 90, 95]:
    print(f"  {p}th: {np.percentile(slopes, p):.2f}°")

# Compare with hardcoded slopes
print(f"\nComparison with sensitivity analysis:")
print(f"  Hardcoded slopes: [0, 1, 2, 3]")
print(f"  % of real slopes > 3°: {(slopes > 3).mean() * 100:.1f}%")
print(f"  % of real slopes > 5°: {(slopes > 5).mean() * 100:.1f}%")

# Quick performance estimate
# Assuming velocity decreases roughly with slope
# Using simplified relationship: velocity reduction ≈ 1 / (1 + slope/10)
hardcoded_slopes = np.array([0, 1, 2, 3])
hardcoded_perf = np.mean(1 / (1 + hardcoded_slopes/10))
real_perf = np.mean(1 / (1 + slopes/10))

print(f"\nSimplified Performance Estimate:")
print(f"  Hardcoded slope performance: {hardcoded_perf:.3f}")
print(f"  Real slope performance: {real_perf:.3f}")
print(f"  Performance ratio: {hardcoded_perf/real_perf:.2f}x")

# Weight by population if available
if 'pop_density' in df.columns:
    # Calculate weighted average slope
    df_clean = df.dropna(subset=['slope_1', 'pop_density'])
    if len(df_clean) > 0:
        weighted_mean = np.average(df_clean['slope_1'], weights=df_clean['pop_density'])
        print(f"\nPopulation-weighted mean slope: {weighted_mean:.2f}°")