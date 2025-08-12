#!/usr/bin/env python3
"""
Investigate what constraints might be limiting the Monte Carlo results
to a narrow range around 3.6 km.
"""

import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def investigate_time_constraint():
    """Check if there's a time constraint limiting distances."""
    
    print("=== INVESTIGATING TIME CONSTRAINT ===")
    
    # The Monte Carlo uses time_gathering_water parameter
    # If this is too short, it would limit distances
    print("Time constraint hypothesis:")
    print("  - Formula: distance = velocity × time / 2")
    print("  - If time is constrained, distances would be capped")
    
    # Check what time values are being used
    print("\nTime parameters:")
    print("  - Sensitivity analysis: mv.t_hours = 5.5 hours")
    print("  - Global model: time_gathering_water = 5.5 hours")
    print("  - But Monte Carlo might use different time constraints")
    
    # Calculate what velocities would be needed for different distances
    time_hours = 5.5
    distances = [3.6, 9.35, 13.36]
    
    print(f"\nRequired velocities for different distances (time={time_hours}h):")
    for dist in distances:
        # distance = velocity × time / 2  ->  velocity = distance × 2 / time
        required_velocity = dist * 2 / time_hours
        print(f"  {dist} km → {required_velocity:.2f} km/h = {required_velocity/3.6:.2f} m/s")
    
    print("\nComparison with sensitivity analysis:")
    print("  - Sensitivity analysis velocity: 1.35 m/s = 4.86 km/h")
    print("  - Required for 3.6 km: 1.31 km/h = 0.36 m/s")
    print("  - This suggests velocity is NOT the constraint")

def investigate_met_constraint():
    """Check if MET values are constraining walking performance."""
    
    print("\n=== INVESTIGATING MET CONSTRAINT ===")
    
    print("MET constraint hypothesis:")
    print("  - Lower MET values → slower walking speeds")
    print("  - Monte Carlo might use different MET distributions")
    
    # Check what MET values the Monte Carlo uses
    print("\nMET values:")
    print("  - Sensitivity analysis: fixed MET = 4.5")
    print("  - Monte Carlo: likely uses a distribution")
    
    # Check if the Monte Carlo parameters file mentions MET
    mc_script = 'scripts/run_monte_carlo.py'
    if os.path.exists(mc_script):
        with open(mc_script, 'r') as f:
            content = f.read()
            
        lines = content.split('\n')
        for i, line in enumerate(lines):
            if 'met' in line.lower() or 'MET' in line:
                print(f"  Line {i+1}: {line.strip()}")
    
    print("\nIf Monte Carlo uses lower MET values:")
    print("  - MET 3.0: ~20% slower than MET 4.5")
    print("  - MET 2.5: ~30% slower than MET 4.5")
    print("  - This could partially explain the difference")

def investigate_practical_limits():
    """Check if there are practical limits constraining distances."""
    
    print("\n=== INVESTIGATING PRACTICAL LIMITS ===")
    
    print("Practical limits hypothesis:")
    print("  - Monte Carlo might have practical walking limits")
    print("  - E.g., maximum reasonable walking distance per day")
    
    # Check the sensitivity analysis variables for practical limits
    sens_vars_file = 'data/lookup tables/Sensitivity Analysis Variables.csv'
    if os.path.exists(sens_vars_file):
        df = pd.read_csv(sens_vars_file)
        
        # Look for practical limit variables
        practical_vars = df[df['Short Name'].str.contains('Practical', case=False, na=False)]
        if not practical_vars.empty:
            print("\nPractical limit variables found:")
            for _, row in practical_vars.iterrows():
                print(f"  {row['Short Name']}: {row['Default Value']} (min={row['Expected Min']}, max={row['Expected Max']})")
        else:
            print("\nNo practical limit variables found in sensitivity analysis")
    
    # Check if the global model has practical limits
    global_file = 'src/gis_global_module.py'
    if os.path.exists(global_file):
        with open(global_file, 'r') as f:
            content = f.read()
            
        lines = content.split('\n')
        for i, line in enumerate(lines):
            if 'practical' in line.lower() or 'limit' in line.lower():
                if 'walking' in line.lower() or 'distance' in line.lower():
                    print(f"  Line {i+1}: {line.strip()}")

def investigate_load_constraints():
    """Check if load carrying constraints affect distances."""
    
    print("\n=== INVESTIGATING LOAD CONSTRAINTS ===")
    
    print("Load constraint hypothesis:")
    print("  - Walking with heavy loads → slower speeds")
    print("  - Different load assumptions between models")
    
    print("\nLoad assumptions:")
    print("  - Sensitivity analysis: fixed 15L water capacity")
    print("  - Monte Carlo: might use different load assumptions")
    
    # Check if the Monte Carlo accounts for load weight
    print("\nLoad weight impact:")
    print("  - 15L water = 15 kg additional weight")
    print("  - Heavy loads significantly reduce walking speed")
    print("  - This could explain the lower Monte Carlo distances")

def investigate_population_weighting():
    """Check if population weighting affects the results."""
    
    print("\n=== INVESTIGATING POPULATION WEIGHTING ===")
    
    print("Population weighting hypothesis:")
    print("  - Monte Carlo uses population-weighted results")
    print("  - Areas with poor water access have more population")
    print("  - This could bias results toward shorter distances")
    
    # Check the Monte Carlo results to see if there's population bias
    results_file = 'results/country_median_results.csv'
    if os.path.exists(results_file):
        df = pd.read_csv(results_file)
        
        # Check correlation between population and walking distance
        if 'country_pop_raw' in df.columns and 'weighted_med_walking' in df.columns:
            pop_col = 'country_pop_raw'
            walk_col = 'weighted_med_walking'
            
            # Remove missing values
            data = df[[pop_col, walk_col]].dropna()
            
            if len(data) > 0:
                correlation = data[pop_col].corr(data[walk_col])
                print(f"\nCorrelation between population and walking distance: {correlation:.3f}")
                
                # Show examples of high/low population countries
                high_pop = data.nlargest(5, pop_col)
                low_pop = data.nsmallest(5, pop_col)
                
                print("\nHighest population countries:")
                for _, row in high_pop.iterrows():
                    print(f"  Population: {row[pop_col]:,.0f}, Walking: {row[walk_col]:.2f} km")
                
                print("\nLowest population countries:")
                for _, row in low_pop.iterrows():
                    print(f"  Population: {row[pop_col]:,.0f}, Walking: {row[walk_col]:.2f} km")

def investigate_velocity_constraints():
    """Check if there are velocity constraints in the models."""
    
    print("\n=== INVESTIGATING VELOCITY CONSTRAINTS ===")
    
    print("Velocity constraint hypothesis:")
    print("  - Monte Carlo might have maximum/minimum velocity limits")
    print("  - Real-world walking speeds have natural limits")
    
    # Check if there are velocity calculations in the processed data
    velocity_file = 'data/processed/walk_velocity_by_zone.csv'
    if os.path.exists(velocity_file):
        print(f"\nAnalyzing {velocity_file}:")
        
        df = pd.read_csv(velocity_file)
        
        velocity_cols = [col for col in df.columns if 'velocity' in col.lower() or 'speed' in col.lower()]
        if velocity_cols:
            for col in velocity_cols:
                values = df[col].dropna()
                if len(values) > 0:
                    print(f"  {col}:")
                    print(f"    Min: {values.min():.3f} m/s")
                    print(f"    Max: {values.max():.3f} m/s")
                    print(f"    Mean: {values.mean():.3f} m/s")
                    print(f"    Median: {values.median():.3f} m/s")
                    
                    # Check if velocities are capped at certain values
                    if values.max() < 2.0:
                        print(f"    *** Velocities appear to be capped at {values.max():.3f} m/s")
    else:
        print(f"\n{velocity_file} not found")

def main():
    """Main investigation function."""
    
    print("INVESTIGATING MONTE CARLO CONSTRAINTS")
    print("=" * 50)
    
    investigate_time_constraint()
    investigate_met_constraint()
    investigate_practical_limits()
    investigate_load_constraints()
    investigate_population_weighting()
    investigate_velocity_constraints()
    
    print("\n" + "=" * 50)
    print("CONSTRAINT INVESTIGATION COMPLETE")
    print("Most likely constraints:")
    print("  1. Load weight (15kg water reduces walking speed)")
    print("  2. Real-world terrain (steeper slopes)")
    print("  3. Population weighting (bias toward poor access areas)")
    print("  4. Lower MET values in Monte Carlo")

if __name__ == "__main__":
    main()