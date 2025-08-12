# Unit Conversion Bug Report: Velocity Unit Mismatch in Distance Calculations

## Executive Summary

A critical unit conversion error has been identified in the global model's distance calculations. The model uses velocities in **meters per second (m/s)** but treats them as if they were in **kilometers per hour (km/h)** when calculating distances, resulting in distances that are 3.6x smaller than they should be.

## The Bug

### Location
File: `/src/gis_global_module.py`, lines 835-840

```python
# Max distance achievable (not round trip, just the distance from home to water source)
df_zones["max distance cycling"] = (
    df_zones["average_velocity_bicycle"] * time_gathering_water / 2
)
df_zones["max distance walking"] = (
    df_zones["average_velocity_walk"] * time_gathering_water / 2
)
```

### Root Cause
1. **Velocity data is in m/s**: The velocity CSV files (`walk_velocity_by_zone.csv`, `bicycle_velocity_by_zone.csv`) contain velocities in meters per second (e.g., ~1.3 m/s for walking)
2. **Formula assumes km/h**: The formula `velocity * time / 2` only works correctly if velocity is in km/h
3. **Missing conversion**: No conversion factor (3.6) is applied to convert m/s to km/h

### Impact
- All distances are **underestimated by a factor of 3.6**
- Walking distances: ~3.7 km instead of ~13.3 km
- Cycling distances: ~11.8 km instead of ~42.5 km

## Evidence

### 1. Velocity Data Units
From `/data/processed/walk_velocity_by_zone.csv`:
```csv
fid,loaded_velocity_walk,unloaded_velocity_walk,average_velocity_walk,max_load_walk
1,0.9857,1.5928,1.2893,20.0
2,0.9834,1.5891,1.2862,20.0
```
These values (~1.3 m/s) are clearly in meters per second.

### 2. Sensitivity Analysis (Correct Implementation)
File: `/scripts/sensitivity_analysis_refactored.py`, line 172:
```python
one_way_distance_km = avg_velocity * mv.t_hours * 3600 / 2 / 1000
```
This correctly converts:
- `* 3600`: hours to seconds
- `/ 1000`: meters to kilometers
- `/ 2`: total time to one-way time

### 3. Numerical Verification
- Velocity: 1.316 m/s (average from data)
- Time: 5.5 hours
- Expected: 1.316 × 5.5 × 3600 / 2 / 1000 = 13.03 km
- Actual global result: 3.71 km
- Ratio: 13.03 / 3.71 = 3.51 ≈ 3.6

## Other Occurrences

The correct formula is used in several other files:
- `/src/water_access_metrics.py` (line 41)
- `/scripts/sensitivity_analysis_refactored.py` (line 172)
- Multiple test and analysis scripts

## Fix

### Option 1: Convert velocities to km/h
```python
df_zones["max distance cycling"] = (
    df_zones["average_velocity_bicycle"] * 3.6 * time_gathering_water / 2
)
df_zones["max distance walking"] = (
    df_zones["average_velocity_walk"] * 3.6 * time_gathering_water / 2
)
```

### Option 2: Use full conversion formula
```python
df_zones["max distance cycling"] = (
    df_zones["average_velocity_bicycle"] * time_gathering_water * 3600 / 2 / 1000
)
df_zones["max distance walking"] = (
    df_zones["average_velocity_walk"] * time_gathering_water * 3600 / 2 / 1000
)
```

## Recommendation

Use **Option 2** for consistency with the sensitivity analysis and other modules. This makes the unit conversion explicit and matches the pattern used throughout the codebase.

## Impact on Results

Fixing this bug will:
1. Increase all distance calculations by 3.6x
2. Significantly change water access conclusions
3. Require re-evaluation of all results and manuscript conclusions
4. Better align global model with sensitivity analysis results

## Verification

After applying the fix, expected results should be:
- Walking: ~13 km (instead of ~3.7 km)
- Cycling: ~42 km (instead of ~11.8 km)

These align with theoretical calculations and sensitivity analysis results.