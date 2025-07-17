#!/usr/bin/env python3
"""
GIS Slope Analysis

Analyzes real-world slope distributions from GIS data to quantify
the impact on water access performance.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys

# Setup paths
project_root = Path(__file__).resolve().parent
sys.path.append(str(project_root))

# Import mobility module
import src.mobility_module as mm

# Set plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def analyze_gis_slopes():
    """Analyze slope distribution from GIS data"""
    
    print("=== GIS SLOPE ANALYSIS ===")
    print("Analyzing real-world slope distributions")
    print("="*70)
    
    # Try different GIS files in order of preference
    gis_files = [
        "data/GIS/GIS_data_zones_sample.csv",  # Start with sample for speed
        "data/GIS/GIS_pre_processed_for_analysis.csv",
        "data/GIS/updated_GIS_output_cleaned.csv"
    ]
    
    df = None
    for file_path in gis_files:
        full_path = project_root / file_path
        if full_path.exists():
            print(f"\nLoading: {file_path}")
            try:
                # Read with low_memory=False to avoid dtype warnings
                df = pd.read_csv(full_path, low_memory=False)
                print(f"Loaded {len(df)} rows")
                break
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
                continue
    
    if df is None:
        print("Error: Could not load any GIS data files")
        return None
    
    # Extract slope data
    if 'slope_1' not in df.columns:
        print(f"Error: 'slope_1' column not found. Available columns: {list(df.columns[:10])}")
        return None
    
    # Remove NaN values and filter reasonable slopes
    slope_data = df['slope_1'].dropna()
    
    # Convert slope units if needed (assuming slopes are in reasonable range)
    # GIS slopes might be in percent grade or radians - check range
    print(f"\nRaw slope data range: {slope_data.min():.4f} to {slope_data.max():.4f}")
    
    # If slopes are very small (likely radians), convert to degrees
    if slope_data.max() < 2:  # Likely radians
        print("Converting slopes from radians to degrees")
        slope_data = slope_data * 180 / np.pi
    elif slope_data.max() > 100:  # Likely percent grade
        print("Converting slopes from percent grade to degrees")
        slope_data = np.arctan(slope_data / 100) * 180 / np.pi
    
    # Filter out unrealistic slopes (>45 degrees is very rare)
    original_count = len(slope_data)
    slope_data = slope_data[slope_data <= 45]
    filtered_count = len(slope_data)
    if filtered_count < original_count:
        print(f"Filtered out {original_count - filtered_count} extreme slopes (>45°)")
    
    return slope_data

def compare_slope_distributions(real_slopes):
    """Compare real slopes with sensitivity analysis hardcoded slopes"""
    
    print("\n=== SLOPE DISTRIBUTION COMPARISON ===")
    
    # Hardcoded slopes from sensitivity analysis
    hardcoded_slopes = [0, 1, 2, 3]
    
    # Statistics
    print(f"\nHardcoded slopes: {hardcoded_slopes}")
    print(f"Real slope statistics:")
    print(f"  Count: {len(real_slopes)}")
    print(f"  Mean: {real_slopes.mean():.2f}°")
    print(f"  Median: {real_slopes.median():.2f}°")
    print(f"  Std Dev: {real_slopes.std():.2f}°")
    print(f"  Min: {real_slopes.min():.2f}°")
    print(f"  Max: {real_slopes.max():.2f}°")
    
    # Percentiles
    percentiles = [5, 10, 25, 50, 75, 90, 95]
    print(f"\nReal slope percentiles:")
    for p in percentiles:
        print(f"  {p:>2}th percentile: {np.percentile(real_slopes, p):>6.2f}°")
    
    # Distribution comparison
    print(f"\nSlope distribution comparison:")
    for slope in hardcoded_slopes:
        if slope == 0:
            count = (real_slopes <= 0.5).sum()
        else:
            count = ((real_slopes > slope - 0.5) & (real_slopes <= slope + 0.5)).sum()
        pct = (count / len(real_slopes)) * 100
        print(f"  Slopes near {slope}°: {pct:.1f}%")
    
    # How much of real data is steeper than hardcoded max?
    steep_count = (real_slopes > 3).sum()
    steep_pct = (steep_count / len(real_slopes)) * 100
    print(f"\nSlopes > 3°: {steep_pct:.1f}% of real data")
    
    return {
        'mean': real_slopes.mean(),
        'median': real_slopes.median(),
        'std': real_slopes.std(),
        'pct_steep': steep_pct
    }

def calculate_performance_impact(real_slopes):
    """Calculate the performance impact of real slopes vs hardcoded slopes"""
    
    print("\n=== PERFORMANCE IMPACT ANALYSIS ===")
    
    # Load bicycle parameters
    file_path_params = project_root / "data/lookup tables/mobility-model-parameters.csv"
    param_df = pd.read_csv(file_path_params)
    bicycle_params = param_df[param_df["Name"] == "Bicycle"]
    
    # Initialize model
    mo = mm.model_options()
    mo.model_selection = 2  # Cycling
    mv = mm.model_variables(P_t=75)
    hpv = mm.HPV_variables(bicycle_params, mv)
    
    # Test with hardcoded slopes
    hardcoded_slopes = [0, 1, 2, 3]
    hardcoded_results = []
    
    print("\nCalculating performance for hardcoded slopes...")
    for slope in hardcoded_slopes:
        try:
            loaded_vel, unloaded_vel, max_load = mm.mobility_models.single_bike_run(
                mv, mo, hpv, slope, 15
            )
            avg_vel = (loaded_vel + unloaded_vel) / 2
            hardcoded_results.append({
                'slope': slope,
                'velocity': avg_vel,
                'max_load': max_load,
                'velocity_kg': avg_vel * max_load
            })
        except:
            continue
    
    # Sample real slopes for performance testing
    n_samples = min(200, len(real_slopes))
    real_sample = real_slopes.sample(n=n_samples, random_state=42)
    real_results = []
    
    print(f"\nCalculating performance for {n_samples} real slope samples...")
    for i, slope in enumerate(real_sample):
        if i % 50 == 0:
            print(f"  Progress: {i}/{n_samples}")
        try:
            loaded_vel, unloaded_vel, max_load = mm.mobility_models.single_bike_run(
                mv, mo, hpv, slope, 15
            )
            avg_vel = (loaded_vel + unloaded_vel) / 2
            real_results.append({
                'slope': slope,
                'velocity': avg_vel,
                'max_load': max_load,
                'velocity_kg': avg_vel * max_load
            })
        except:
            continue
    
    # Calculate averages
    hardcoded_df = pd.DataFrame(hardcoded_results)
    real_df = pd.DataFrame(real_results)
    
    hardcoded_avg_vel_kg = hardcoded_df['velocity_kg'].mean()
    real_avg_vel_kg = real_df['velocity_kg'].mean()
    
    # Calculate water ration distance
    t_hours = 5.5
    water_ration = 15
    
    # Using sensitivity analysis formula for consistency
    hardcoded_water_km = hardcoded_avg_vel_kg / water_ration * t_hours * 3600 / 1000
    real_water_km = real_avg_vel_kg / water_ration * t_hours * 3600 / 1000
    
    print(f"\n=== RESULTS ===")
    print(f"Hardcoded slopes (0-3°):")
    print(f"  Mean velocity: {hardcoded_df['velocity'].mean():.2f} m/s")
    print(f"  Mean velocity×kg: {hardcoded_avg_vel_kg:.1f}")
    print(f"  Water ration distance: {hardcoded_water_km:.1f} km")
    
    print(f"\nReal slopes (mean {real_slopes.mean():.1f}°):")
    print(f"  Mean velocity: {real_df['velocity'].mean():.2f} m/s") 
    print(f"  Mean velocity×kg: {real_avg_vel_kg:.1f}")
    print(f"  Water ration distance: {real_water_km:.1f} km")
    
    performance_ratio = hardcoded_water_km / real_water_km
    print(f"\nPerformance ratio (hardcoded/real): {performance_ratio:.2f}x")
    
    return {
        'hardcoded_water_km': hardcoded_water_km,
        'real_water_km': real_water_km,
        'performance_ratio': performance_ratio,
        'hardcoded_results': hardcoded_df,
        'real_results': real_df
    }

def create_visualization(real_slopes, performance_results):
    """Create visualization of slope analysis"""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('GIS Slope Analysis: Impact on Water Access', fontsize=16, fontweight='bold')
    
    # 1. Slope distribution histogram
    axes[0, 0].hist(real_slopes, bins=50, alpha=0.7, color='skyblue', edgecolor='black', density=True)
    axes[0, 0].axvline(real_slopes.mean(), color='red', linestyle='--', label=f'Mean: {real_slopes.mean():.1f}°')
    
    # Add hardcoded slopes
    for slope in [0, 1, 2, 3]:
        axes[0, 0].axvline(slope, color='green', linestyle=':', alpha=0.7)
    axes[0, 0].axvline(1.5, color='green', linestyle=':', alpha=0.7, label='Hardcoded slopes')
    
    axes[0, 0].set_xlabel('Slope (degrees)')
    axes[0, 0].set_ylabel('Density')
    axes[0, 0].set_title('Real-World Slope Distribution')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Cumulative distribution
    sorted_slopes = np.sort(real_slopes)
    cumulative = np.arange(1, len(sorted_slopes) + 1) / len(sorted_slopes)
    axes[0, 1].plot(sorted_slopes, cumulative * 100, 'b-', linewidth=2)
    
    # Mark percentiles
    for p in [25, 50, 75]:
        val = np.percentile(real_slopes, p)
        axes[0, 1].axvline(val, color='gray', linestyle='--', alpha=0.5)
        axes[0, 1].text(val, p + 5, f'{p}%', ha='center')
    
    axes[0, 1].set_xlabel('Slope (degrees)')
    axes[0, 1].set_ylabel('Cumulative Percentage (%)')
    axes[0, 1].set_title('Cumulative Slope Distribution')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_xlim(0, 20)
    
    # 3. Performance comparison
    categories = ['Hardcoded\nSlopes\n(0-3°)', 'Real GIS\nSlopes']
    water_distances = [
        performance_results['hardcoded_water_km'],
        performance_results['real_water_km']
    ]
    
    bars = axes[1, 0].bar(categories, water_distances, color=['green', 'skyblue'], alpha=0.7)
    axes[1, 0].set_ylabel('Water Ration Distance (km)')
    axes[1, 0].set_title('Performance Impact of Slope Distributions')
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, val in zip(bars, water_distances):
        height = bar.get_height()
        axes[1, 0].text(bar.get_x() + bar.get_width()/2., height + 1,
                       f'{val:.1f} km', ha='center', va='bottom')
    
    # 4. Velocity vs slope scatter
    real_df = performance_results['real_results']
    axes[1, 1].scatter(real_df['slope'], real_df['velocity'], alpha=0.5, s=30, color='skyblue')
    
    # Add hardcoded points
    hardcoded_df = performance_results['hardcoded_results']
    axes[1, 1].scatter(hardcoded_df['slope'], hardcoded_df['velocity'], 
                      color='green', s=100, marker='s', label='Hardcoded slopes')
    
    axes[1, 1].set_xlabel('Slope (degrees)')
    axes[1, 1].set_ylabel('Average Velocity (m/s)')
    axes[1, 1].set_title('Velocity vs Slope')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_xlim(-1, 20)
    
    plt.tight_layout()
    
    # Save figure
    output_path = project_root / "results" / "gis_slope_analysis.png"
    output_path.parent.mkdir(exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved visualization to: {output_path}")
    
    plt.show()

def main():
    """Main analysis function"""
    
    # Analyze GIS slopes
    real_slopes = analyze_gis_slopes()
    if real_slopes is None:
        return
    
    # Compare distributions
    slope_stats = compare_slope_distributions(real_slopes)
    
    # Calculate performance impact
    performance_results = calculate_performance_impact(real_slopes)
    
    # Create visualization
    create_visualization(real_slopes, performance_results)
    
    # Summary
    print("\n" + "="*70)
    print("SLOPE ANALYSIS SUMMARY")
    print("="*70)
    print(f"\nReal-world slopes are significantly steeper than test slopes:")
    print(f"  - Test slopes: 0-3° (flat terrain)")
    print(f"  - Real slopes: mean {slope_stats['mean']:.1f}°, {slope_stats['pct_steep']:.0f}% > 3°")
    print(f"\nPerformance impact:")
    print(f"  - Hardcoded slopes: {performance_results['hardcoded_water_km']:.1f} km")
    print(f"  - Real slopes: {performance_results['real_water_km']:.1f} km")
    print(f"  - Performance reduction: {performance_results['performance_ratio']:.1f}x")
    print(f"\nThis {performance_results['performance_ratio']:.1f}x slope effect, combined with")
    print(f"the 7.5x formula difference, explains the discrepancy between")
    print(f"sensitivity analysis and global model results.")

if __name__ == "__main__":
    main()