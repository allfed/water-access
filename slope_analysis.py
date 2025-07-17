#!/usr/bin/env python3
"""
Slope Analysis Script

This script analyzes the slope distributions in the GIS data to understand
why there's a discrepancy between sensitivity analysis and global model results.

The sensitivity analysis uses hardcoded slopes [0, 1, 2, 3] degrees while
the global model uses real-world slope data from GIS.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
import warnings

# Setup paths
project_root = Path(__file__).resolve().parent
sys.path.append(str(project_root))

# Import mobility module
import src.mobility_module as mm

# Set up plotting
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_gis_data():
    """Load GIS data to analyze slope distributions"""
    
    # Try to load full GIS data first
    try:
        gis_data_path = project_root / "data" / "GIS" / "updated_GIS_output_cleaned.csv"
        print(f"Loading GIS data from: {gis_data_path}")
        df = pd.read_csv(gis_data_path)
        print(f"Loaded {len(df)} rows from full GIS dataset")
        return df
    except FileNotFoundError:
        print("Full GIS data not found, trying sample data...")
        
    # Fallback to sample data
    try:
        sample_path = project_root / "data" / "GIS" / "GIS_data_zones_sample.csv"
        df = pd.read_csv(sample_path)
        print(f"Loaded {len(df)} rows from sample GIS dataset")
        warnings.warn("Using sample data - results may not be representative of full dataset")
        return df
    except FileNotFoundError:
        print("Error: Could not find GIS data files")
        return None

def analyze_slope_distribution(df):
    """Analyze the slope distribution in the GIS data"""
    
    if df is None:
        print("No data available for analysis")
        return None
    
    print("\n=== SLOPE DISTRIBUTION ANALYSIS ===")
    
    # Check if slope column exists
    if 'slope_1' not in df.columns:
        print("Error: 'slope_1' column not found in GIS data")
        print(f"Available columns: {list(df.columns)}")
        return None
    
    # Remove NaN values
    slope_data = df['slope_1'].dropna()
    
    print(f"Total slope data points: {len(slope_data)}")
    print(f"Slope statistics:")
    print(f"  Mean: {slope_data.mean():.2f}")
    print(f"  Median: {slope_data.median():.2f}")
    print(f"  Std Dev: {slope_data.std():.2f}")
    print(f"  Min: {slope_data.min():.2f}")
    print(f"  Max: {slope_data.max():.2f}")
    
    # Percentiles
    percentiles = [5, 10, 25, 50, 75, 90, 95, 99]
    print(f"\nSlope percentiles:")
    for p in percentiles:
        print(f"  {p}th percentile: {np.percentile(slope_data, p):.2f}")
    
    # Compare with sensitivity analysis hardcoded slopes
    hardcoded_slopes = [0, 1, 2, 3]
    print(f"\n=== COMPARISON WITH SENSITIVITY ANALYSIS ===")
    print(f"Hardcoded slopes in sensitivity analysis: {hardcoded_slopes}")
    
    # What percentage of real data falls within each hardcoded slope range?
    for i, slope in enumerate(hardcoded_slopes):
        if i == 0:
            # Count slopes <= 0.5 degrees
            count = (slope_data <= 0.5).sum()
            pct = (count / len(slope_data)) * 100
            print(f"Slopes <= 0.5°: {count} ({pct:.1f}%)")
        else:
            # Count slopes in range [prev_slope+0.5, slope+0.5]
            prev_slope = hardcoded_slopes[i-1]
            lower = prev_slope + 0.5
            upper = slope + 0.5
            count = ((slope_data > lower) & (slope_data <= upper)).sum()
            pct = (count / len(slope_data)) * 100
            print(f"Slopes {lower:.1f}° - {upper:.1f}°: {count} ({pct:.1f}%)")
    
    # Count slopes > 3.5 degrees
    count = (slope_data > 3.5).sum()
    pct = (count / len(slope_data)) * 100
    print(f"Slopes > 3.5°: {count} ({pct:.1f}%)")
    
    return slope_data

def create_slope_visualization(slope_data):
    """Create visualizations of slope distributions"""
    
    if slope_data is None:
        return
    
    print("\n=== CREATING SLOPE VISUALIZATIONS ===")
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Real-World Slope Distribution Analysis', fontsize=16, fontweight='bold')
    
    # 1. Histogram of slopes
    axes[0, 0].hist(slope_data, bins=100, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0, 0].set_title('Distribution of Slopes in GIS Data')
    axes[0, 0].set_xlabel('Slope (degrees)')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Add vertical lines for hardcoded slopes
    hardcoded_slopes = [0, 1, 2, 3]
    for slope in hardcoded_slopes:
        axes[0, 0].axvline(slope, color='red', linestyle='--', alpha=0.7, 
                          label=f'Hardcoded: {slope}°' if slope == 0 else '')
    if hardcoded_slopes:
        axes[0, 0].legend()
    
    # 2. Cumulative distribution
    sorted_slopes = np.sort(slope_data)
    cumulative = np.arange(1, len(sorted_slopes) + 1) / len(sorted_slopes)
    axes[0, 1].plot(sorted_slopes, cumulative, color='green', linewidth=2)
    axes[0, 1].set_title('Cumulative Distribution of Slopes')
    axes[0, 1].set_xlabel('Slope (degrees)')
    axes[0, 1].set_ylabel('Cumulative Probability')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Add vertical lines for hardcoded slopes
    for slope in hardcoded_slopes:
        axes[0, 1].axvline(slope, color='red', linestyle='--', alpha=0.7)
    
    # 3. Box plot comparison
    data_to_plot = [slope_data, hardcoded_slopes]
    labels = ['Real GIS Data', 'Hardcoded Slopes']
    
    # Create box plot
    bp = axes[1, 0].boxplot(data_to_plot, labels=labels, patch_artist=True)
    bp['boxes'][0].set_facecolor('lightblue')
    bp['boxes'][1].set_facecolor('lightcoral')
    axes[1, 0].set_title('Slope Distribution Comparison')
    axes[1, 0].set_ylabel('Slope (degrees)')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Zoomed histogram (0-10 degrees)
    slope_subset = slope_data[slope_data <= 10]
    axes[1, 1].hist(slope_subset, bins=50, alpha=0.7, color='lightgreen', edgecolor='black')
    axes[1, 1].set_title('Slope Distribution (0-10 degrees)')
    axes[1, 1].set_xlabel('Slope (degrees)')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Add vertical lines for hardcoded slopes
    for slope in hardcoded_slopes:
        axes[1, 1].axvline(slope, color='red', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    # Save the figure
    output_path = project_root / "results" / "slope_analysis_comparison.png"
    output_path.parent.mkdir(exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved slope analysis visualization to: {output_path}")
    
    plt.show()

def calculate_mobility_impact(slope_data):
    """Calculate the mobility impact of using real slopes vs hardcoded slopes"""
    
    if slope_data is None:
        return
    
    print("\n=== MOBILITY IMPACT ANALYSIS ===")
    
    # Load mobility parameters
    file_path_params = project_root / "data/lookup tables/mobility-model-parameters.csv"
    param_df = pd.read_csv(file_path_params)
    
    # Test with bicycle parameters
    bicycle_params = param_df[param_df["Name"] == "Bicycle"]
    if len(bicycle_params) == 0:
        print("Error: Bicycle parameters not found")
        return
    
    # Initialize mobility model components
    mo = mm.model_options()
    mo.model_selection = 2  # Cycling model
    mv = mm.model_variables(P_t=75)  # 75W power
    hpv = mm.HPV_variables(bicycle_params, mv)
    
    # Test slopes
    hardcoded_slopes = [0, 1, 2, 3]
    
    # Sample some real slopes for comparison
    np.random.seed(42)  # For reproducibility
    real_slopes_sample = np.random.choice(slope_data, size=100, replace=False)
    
    print(f"Testing mobility with {len(real_slopes_sample)} real slope samples...")
    
    # Calculate velocities for hardcoded slopes
    hardcoded_results = []
    for slope in hardcoded_slopes:
        try:
            loaded_vel, unloaded_vel, max_load = mm.mobility_models.single_bike_run(
                mv, mo, hpv, slope, 15  # 15kg load attempt
            )
            avg_vel = (loaded_vel + unloaded_vel) / 2
            hardcoded_results.append({
                'slope': slope,
                'avg_velocity': avg_vel,
                'max_load': max_load,
                'velocity_kg': avg_vel * max_load
            })
        except Exception as e:
            print(f"Error calculating for slope {slope}: {e}")
            hardcoded_results.append({
                'slope': slope,
                'avg_velocity': np.nan,
                'max_load': np.nan,
                'velocity_kg': np.nan
            })
    
    # Calculate velocities for real slopes (sample)
    real_results = []
    for slope in real_slopes_sample:
        try:
            loaded_vel, unloaded_vel, max_load = mm.mobility_models.single_bike_run(
                mv, mo, hpv, slope, 15  # 15kg load attempt
            )
            avg_vel = (loaded_vel + unloaded_vel) / 2
            real_results.append({
                'slope': slope,
                'avg_velocity': avg_vel,
                'max_load': max_load,
                'velocity_kg': avg_vel * max_load
            })
        except Exception as e:
            # Skip problematic slopes
            continue
    
    # Convert to DataFrames for analysis
    hardcoded_df = pd.DataFrame(hardcoded_results)
    real_df = pd.DataFrame(real_results)
    
    # Calculate summary statistics
    print(f"\n=== HARDCODED SLOPES RESULTS ===")
    print(f"Mean velocity: {hardcoded_df['avg_velocity'].mean():.3f} m/s")
    print(f"Mean max load: {hardcoded_df['max_load'].mean():.3f} kg")
    print(f"Mean velocity*kg: {hardcoded_df['velocity_kg'].mean():.3f}")
    
    print(f"\n=== REAL SLOPES RESULTS ===")
    print(f"Mean velocity: {real_df['avg_velocity'].mean():.3f} m/s")
    print(f"Mean max load: {real_df['max_load'].mean():.3f} kg")
    print(f"Mean velocity*kg: {real_df['velocity_kg'].mean():.3f}")
    
    # Calculate water ration kilometers
    t_hours = 5.5  # Default time
    water_ration = 15  # L
    
    hardcoded_water_ration_km = (hardcoded_df['velocity_kg'].mean() * t_hours * 3600) / (water_ration * 1000)
    real_water_ration_km = (real_df['velocity_kg'].mean() * t_hours * 3600) / (water_ration * 1000)
    
    print(f"\n=== WATER RATION KILOMETERS ===")
    print(f"Hardcoded slopes: {hardcoded_water_ration_km:.3f} km")
    print(f"Real slopes: {real_water_ration_km:.3f} km")
    print(f"Ratio (hardcoded/real): {hardcoded_water_ration_km/real_water_ration_km:.2f}")
    
    # Save results
    results_summary = {
        'hardcoded_slopes': hardcoded_slopes,
        'hardcoded_mean_velocity': hardcoded_df['avg_velocity'].mean(),
        'hardcoded_mean_velocity_kg': hardcoded_df['velocity_kg'].mean(),
        'hardcoded_water_ration_km': hardcoded_water_ration_km,
        'real_slopes_mean': real_df['slope'].mean(),
        'real_slopes_std': real_df['slope'].std(),
        'real_mean_velocity': real_df['avg_velocity'].mean(),
        'real_mean_velocity_kg': real_df['velocity_kg'].mean(),
        'real_water_ration_km': real_water_ration_km,
        'performance_ratio': hardcoded_water_ration_km/real_water_ration_km
    }
    
    return results_summary

def main():
    """Main analysis function"""
    
    print("=== SLOPE ANALYSIS INVESTIGATION ===")
    print("Investigating slope distribution differences between sensitivity analysis and global model")
    print("="*70)
    
    # Load GIS data
    df = load_gis_data()
    
    if df is None:
        print("Cannot proceed without GIS data")
        return
    
    # Analyze slope distribution
    slope_data = analyze_slope_distribution(df)
    
    if slope_data is None:
        print("Cannot proceed without slope data")
        return
    
    # Create visualizations
    create_slope_visualization(slope_data)
    
    # Calculate mobility impact
    impact_results = calculate_mobility_impact(slope_data)
    
    if impact_results:
        print(f"\n=== SUMMARY ===")
        print(f"The sensitivity analysis uses hardcoded slopes {impact_results['hardcoded_slopes']}")
        print(f"Real-world slopes have mean: {impact_results['real_slopes_mean']:.2f}° (±{impact_results['real_slopes_std']:.2f}°)")
        print(f"")
        print(f"Performance impact:")
        print(f"  Hardcoded slopes water ration: {impact_results['hardcoded_water_ration_km']:.2f} km")
        print(f"  Real slopes water ration: {impact_results['real_water_ration_km']:.2f} km")
        print(f"  Performance ratio: {impact_results['performance_ratio']:.2f}x")
        print(f"")
        print(f"This explains {'part of' if impact_results['performance_ratio'] < 3 else 'most of'} the discrepancy!")

if __name__ == "__main__":
    main()