#!/usr/bin/env python3
"""
Compare Monte Carlo approach vs Sensitivity Analysis approach
to understand why they produce different walking distances.
"""

import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def analyze_monte_carlo_parameters():
    """Analyze the Monte Carlo parameters to understand the differences."""
    
    print("=== ANALYZING MONTE CARLO PARAMETERS ===")
    
    # Read the Monte Carlo script to understand parameter ranges
    mc_script = 'scripts/run_monte_carlo.py'
    if os.path.exists(mc_script):
        print(f"Reading {mc_script}:")
        
        with open(mc_script, 'r') as f:
            content = f.read()
            
        # Look for parameter definitions
        lines = content.split('\n')
        in_parameter_section = False
        
        for i, line in enumerate(lines):
            if 'WALKING' in line and 'CYCLING' in line and 'PARAMETERS' in line:
                in_parameter_section = True
                print(f"  Found parameter section at line {i+1}")
                continue
                
            if in_parameter_section:
                if line.strip().startswith('#') or 'walking' in line.lower() or 'met' in line.lower():
                    print(f"  {line.strip()}")
                
                # Look for specific parameter ranges
                if 'np.random' in line or 'uniform' in line or 'normal' in line:
                    print(f"  --> {line.strip()}")
                
                # Stop if we hit a new section
                if line.strip() and not line.startswith('#') and not any(keyword in line.lower() for keyword in ['met', 'walking', 'practical', 'random', 'uniform', 'normal']):
                    break
                    
    else:
        print(f"{mc_script} not found")

def analyze_real_world_data():
    """Analyze the real-world data used by Monte Carlo."""
    
    print("\n=== ANALYZING REAL-WORLD DATA ===")
    
    # Check the GIS data that might be used
    gis_files = [
        'data/GIS/GIS_data_zones_sample.csv',
        'data/GIS/updated_GIS_output.csv',
        'data/processed/merged_data.csv'
    ]
    
    for file_path in gis_files:
        if os.path.exists(file_path):
            print(f"\nAnalyzing {file_path}:")
            
            try:
                df = pd.read_csv(file_path)
                print(f"  Shape: {df.shape}")
                print(f"  Columns: {list(df.columns)}")
                
                # Look for slope-related columns
                slope_cols = [col for col in df.columns if 'slope' in col.lower()]
                if slope_cols:
                    print(f"  Slope columns: {slope_cols}")
                    for col in slope_cols:
                        values = df[col].dropna()
                        if len(values) > 0:
                            print(f"    {col}: min={values.min():.2f}, max={values.max():.2f}, mean={values.mean():.2f}, median={values.median():.2f}")
                
                # Look for CRR-related columns
                crr_cols = [col for col in df.columns if 'crr' in col.lower()]
                if crr_cols:
                    print(f"  CRR columns: {crr_cols}")
                    for col in crr_cols:
                        values = df[col].dropna()
                        if len(values) > 0:
                            print(f"    {col}: min={values.min():.4f}, max={values.max():.4f}, mean={values.mean():.4f}, median={values.median():.4f}")
                
                # Look for velocity-related columns
                velocity_cols = [col for col in df.columns if 'velocity' in col.lower() or 'speed' in col.lower()]
                if velocity_cols:
                    print(f"  Velocity columns: {velocity_cols}")
                    for col in velocity_cols:
                        values = df[col].dropna()
                        if len(values) > 0:
                            print(f"    {col}: min={values.min():.2f}, max={values.max():.2f}, mean={values.mean():.2f}, median={values.median():.2f}")
                
                # Look for distance-related columns
                distance_cols = [col for col in df.columns if 'distance' in col.lower() and 'walk' in col.lower()]
                if distance_cols:
                    print(f"  Walking distance columns: {distance_cols}")
                    for col in distance_cols:
                        values = df[col].dropna()
                        if len(values) > 0:
                            print(f"    {col}: min={values.min():.2f}, max={values.max():.2f}, mean={values.mean():.2f}, median={values.median():.2f}")
                            
                            # Check for values around 3.6 km
                            close_to_3_6 = values[(values >= 3.0) & (values <= 4.0)]
                            if len(close_to_3_6) > 0:
                                print(f"    *** Values close to 3.6 km: {len(close_to_3_6)} zones, mean={close_to_3_6.mean():.2f}")
                
            except Exception as e:
                print(f"  Error reading {file_path}: {e}")
        else:
            print(f"\n{file_path} not found")

def analyze_monte_carlo_results():
    """Analyze the Monte Carlo results in detail."""
    
    print("\n=== ANALYZING MONTE CARLO RESULTS IN DETAIL ===")
    
    # Look at the country median results more carefully
    results_file = 'results/country_median_results.csv'
    if os.path.exists(results_file):
        print(f"\nAnalyzing {results_file}:")
        
        df = pd.read_csv(results_file)
        
        # Focus on the weighted_med_walking column
        walking_distances = df['weighted_med_walking'].dropna()
        print(f"  Walking distances (weighted_med_walking):")
        print(f"    Count: {len(walking_distances)}")
        print(f"    Min: {walking_distances.min():.2f} km")
        print(f"    Max: {walking_distances.max():.2f} km")
        print(f"    Mean: {walking_distances.mean():.2f} km")
        print(f"    Median: {walking_distances.median():.2f} km")
        print(f"    Std: {walking_distances.std():.2f} km")
        
        # Show some examples of countries with different distances
        print(f"\n  Examples of countries with different walking distances:")
        sample_countries = df[['Entity', 'weighted_med_walking']].dropna().head(10)
        for _, row in sample_countries.iterrows():
            print(f"    {row['Entity']}: {row['weighted_med_walking']:.2f} km")
        
        # Check if there are any constraints or filters
        print(f"\n  Checking for constraints:")
        
        # Check percent_with_water
        if 'percent_with_water' in df.columns:
            water_access = df['percent_with_water'].dropna()
            print(f"    Water access: min={water_access.min():.1f}%, max={water_access.max():.1f}%, mean={water_access.mean():.1f}%")
        
        # Check percent_piped_with_walking_access
        if 'percent_piped_with_walking_access' in df.columns:
            walking_access = df['percent_piped_with_walking_access'].dropna()
            print(f"    Walking access: min={walking_access.min():.1f}%, max={walking_access.max():.1f}%, mean={walking_access.mean():.1f}%")
        
    else:
        print(f"{results_file} not found")

def compare_key_differences():
    """Compare the key differences between Monte Carlo and Sensitivity Analysis."""
    
    print("\n=== COMPARING KEY DIFFERENCES ===")
    
    print("Monte Carlo Approach:")
    print("  - Uses real-world terrain data (slopes, CRR)")
    print("  - Population-weighted results")
    print("  - Statistical distribution (median of countries)")
    print("  - Variable parameters across simulations")
    print("  - Reports: ~3.6 km (population-weighted median)")
    
    print("\nSensitivity Analysis Approach:")
    print("  - Uses fixed parameters (slopes [0-5°], CRR=0.003)")
    print("  - Single scenario calculation")
    print("  - Average across test slopes")
    print("  - Fixed MET=4.5, time=5.5h")
    print("  - Reports: ~13.36 km (single scenario)")
    
    print("\nKey Differences:")
    print("  1. Terrain: Real-world vs test slopes")
    print("  2. Population weighting: Yes vs No")
    print("  3. Statistical approach: Median vs Mean")
    print("  4. Parameter variation: Monte Carlo vs Fixed")
    print("  5. Scale: Global vs Single scenario")

def investigate_terrain_impact():
    """Investigate how real-world terrain impacts the results."""
    
    print("\n=== INVESTIGATING TERRAIN IMPACT ===")
    
    # Try to understand the terrain differences
    print("From previous analysis:")
    print("  - Real-world slopes: mean=4.0°, 36% > 3°")
    print("  - Sensitivity analysis: [0,1,2,3,4,5]°")
    print("  - Population-weighted real-world mean: 2.67°")
    
    # Estimate the impact
    print("\nEstimated impact of terrain differences:")
    print("  - Sensitivity analysis avg slope: 2.5°")
    print("  - Real-world avg slope: 4.0°")
    print("  - Higher slopes -> slower walking speeds")
    print("  - Expected impact: 20-30% reduction in speeds")
    
    # Check if this explains the difference
    estimated_reduction = 13.36 * 0.7  # 30% reduction
    print(f"\nIf terrain reduces speeds by 30%:")
    print(f"  - Sensitivity result: 13.36 km")
    print(f"  - With terrain correction: {estimated_reduction:.2f} km")
    print(f"  - Still higher than Monte Carlo: {estimated_reduction:.2f} vs 3.6 km")
    print(f"  - Remaining difference: {estimated_reduction / 3.6:.1f}x")

def main():
    """Main comparison function."""
    
    print("MONTE CARLO vs SENSITIVITY ANALYSIS COMPARISON")
    print("=" * 60)
    
    analyze_monte_carlo_parameters()
    analyze_real_world_data()
    analyze_monte_carlo_results()
    compare_key_differences()
    investigate_terrain_impact()
    
    print("\n" + "=" * 60)
    print("INVESTIGATION COMPLETE")
    print("Next step: Examine specific terrain/slope data to quantify the impact")

if __name__ == "__main__":
    main()