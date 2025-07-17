#!/usr/bin/env python3
"""
Parameter Comparison Script

This script compares the default parameters used in sensitivity analysis
versus the global model and Monte Carlo simulations to identify differences
that could contribute to the performance discrepancy.
"""

import pandas as pd
from pathlib import Path
import sys

# Setup paths
project_root = Path(__file__).resolve().parent
sys.path.append(str(project_root))

def load_sensitivity_defaults():
    """Load default parameters from sensitivity analysis"""
    
    # From Sensitivity Analysis Variables.csv
    sens_vars_path = project_root / "data/lookup tables/Sensitivity Analysis Variables.csv"
    sens_df = pd.read_csv(sens_vars_path)
    
    # Extract defaults
    defaults = {}
    for _, row in sens_df.iterrows():
        defaults[row['Short Name']] = {
            'default': row['Default Value'],
            'min': row['Expected Min'],
            'max': row['Expected Max'],
            'units': row['Units']
        }
    
    return defaults

def get_global_model_defaults():
    """Extract default parameters from global model code"""
    
    # From gis_global_module.py main function defaults
    global_defaults = {
        'crr_adjustment': 0,
        'time_gathering_water': 5.5,  # hours
        'practical_limit_bicycle': 40,  # kg
        'practical_limit_buckets': 20,  # kg
        'met': 4.5,
        'watts': 75,
        'hill_polarity': 'downhill_uphill',
        'urban_adjustment': 1.3,
        'rural_adjustment': 1.4,
        'human_mass': 62  # kg (gets overridden by country specific weight)
    }
    
    return global_defaults

def get_monte_carlo_ranges():
    """Extract parameter ranges from Monte Carlo simulation"""
    
    # From run_monte_carlo.py
    mc_ranges = {
        'crr_adjustment': {'min': -1, 'max': 1},
        'time_gathering_water': {'min': 4, 'max': 7},
        'practical_limit_bicycle': {'min': 30, 'max': 45},
        'practical_limit_buckets': {'min': 15, 'max': 25},
        'met': {'min': 3, 'max': 6},
        'watts': {'min': 20, 'max': 80},
        'hill_polarity': ['uphill_downhill', 'uphill_flat', 'flat_uphill', 'downhill_uphill'],
        'urban_adjustment': {'min': 1.2, 'max': 1.5},
        'rural_adjustment': {'pareto_params': {
            'shape': 0.20007812499999994,
            'scale': 0.19953125000000005,
            'loc': 1.0
        }}
    }
    
    return mc_ranges

def compare_parameters():
    """Compare parameters across different models"""
    
    print("=== PARAMETER COMPARISON ANALYSIS ===")
    print("Comparing default parameters across models")
    print("="*70)
    
    # Load parameters
    sens_defaults = load_sensitivity_defaults()
    global_defaults = get_global_model_defaults()
    mc_ranges = get_monte_carlo_ranges()
    
    # Key parameters to compare
    print("\n=== KEY PARAMETER DIFFERENCES ===")
    
    # 1. Time gathering water
    print("\n1. Time Gathering Water (T_hours):")
    print(f"   Sensitivity default: {sens_defaults['T_hours']['default']} hours")
    print(f"   Global model default: {global_defaults['time_gathering_water']} hours")
    print(f"   Monte Carlo range: {mc_ranges['time_gathering_water']['min']}-{mc_ranges['time_gathering_water']['max']} hours")
    print(f"   → SAME default (5.5 hours)")
    
    # 2. MET budget
    print("\n2. MET Budget (walking energy):")
    print(f"   Sensitivity default: {sens_defaults['MET budget']['default']}")
    print(f"   Global model default: {global_defaults['met']}")
    print(f"   Monte Carlo range: {mc_ranges['met']['min']}-{mc_ranges['met']['max']}")
    print(f"   → DIFFERENT: Sensitivity uses 3.5, Global uses 4.5 (29% higher)")
    
    # 3. Practical limits
    print("\n3. Practical Limit Cycling:")
    print(f"   Sensitivity default: {sens_defaults['Practical Limit Cycling']['default']} kg")
    print(f"   Global model default: {global_defaults['practical_limit_bicycle']} kg")
    print(f"   Monte Carlo range: {mc_ranges['practical_limit_bicycle']['min']}-{mc_ranges['practical_limit_bicycle']['max']} kg")
    print(f"   → DIFFERENT: Sensitivity uses 37.5 kg, Global uses 40 kg (7% higher)")
    
    print("\n4. Practical Limit Walking:")
    print(f"   Sensitivity default: {sens_defaults['Practical Limit Walking']['default']} kg")
    print(f"   Global model default: {global_defaults['practical_limit_buckets']} kg")
    print(f"   Monte Carlo range: {mc_ranges['practical_limit_buckets']['min']}-{mc_ranges['practical_limit_buckets']['max']} kg")
    print(f"   → SAME default (20 kg)")
    
    # 5. Human Power Output
    print("\n5. Human Power Output (cycling):")
    print(f"   Sensitivity default: {sens_defaults['Human Power Output']['default']} watts")
    print(f"   Global model default: {global_defaults['watts']} watts")
    print(f"   Monte Carlo range: {mc_ranges['watts']['min']}-{mc_ranges['watts']['max']} watts")
    print(f"   → SAME default (75 watts)")
    
    # 6. Water ration
    print("\n6. Water Ration:")
    print(f"   Sensitivity default: {sens_defaults['Water Ration']['default']} L/day")
    print(f"   Global model: Fixed at 15 L (mentioned in comments)")
    print(f"   → SAME default (15 L)")
    
    # 7. Hill polarity
    print("\n7. Hill Polarity:")
    print(f"   Sensitivity: Not specified (likely flat_flat or uphill_downhill)")
    print(f"   Global model default: {global_defaults['hill_polarity']}")
    print(f"   Monte Carlo options: {mc_ranges['hill_polarity']}")
    print(f"   → POTENTIALLY DIFFERENT: Global uses downhill_uphill")
    
    # 8. Terrain adjustments
    print("\n8. Euclidean Distance Adjustments:")
    print(f"   Sensitivity: Not used")
    print(f"   Global urban adjustment: {global_defaults['urban_adjustment']}")
    print(f"   Global rural adjustment: {global_defaults['rural_adjustment']}")
    print(f"   → DIFFERENT: Global model adjusts distances by 1.3x (urban) and 1.4x (rural)")
    
    # Calculate impact
    print("\n\n=== PARAMETER IMPACT ANALYSIS ===")
    
    # MET impact on walking
    met_ratio = global_defaults['met'] / sens_defaults['MET budget']['default']
    print(f"\n1. MET Impact:")
    print(f"   Global/Sensitivity ratio: {met_ratio:.2f}")
    print(f"   Higher MET = more energy = faster walking")
    print(f"   Expected impact: ~{(met_ratio-1)*100:.0f}% faster walking in global model")
    
    # Practical limit impact
    pl_ratio = global_defaults['practical_limit_bicycle'] / sens_defaults['Practical Limit Cycling']['default']
    print(f"\n2. Practical Limit Impact:")
    print(f"   Global/Sensitivity ratio: {pl_ratio:.2f}")
    print(f"   Higher limit = more water carried")
    print(f"   Expected impact: ~{(pl_ratio-1)*100:.0f}% more water capacity in global model")
    
    # Euclidean adjustment impact
    avg_adjustment = (global_defaults['urban_adjustment'] + global_defaults['rural_adjustment']) / 2
    print(f"\n3. Euclidean Adjustment Impact:")
    print(f"   Average adjustment: {avg_adjustment:.2f}x")
    print(f"   This INCREASES distances to water by {(avg_adjustment-1)*100:.0f}%")
    print(f"   Expected impact: Reduces water access performance by ~{(1-1/avg_adjustment)*100:.0f}%")
    
    # Combined impact estimate
    print("\n\n=== COMBINED PARAMETER IMPACT ===")
    print("Parameter differences between models:")
    print("1. MET: +29% performance (helps global model)")
    print("2. Practical limit: +7% capacity (helps global model)")
    print("3. Euclidean adjustment: -28% performance (hurts global model)")
    print("4. Hill polarity: Variable impact depending on terrain")
    
    print("\nNet parameter impact: Roughly neutral to slightly negative")
    print("This suggests parameters alone don't explain the discrepancy")
    
    return {
        'met_ratio': met_ratio,
        'practical_limit_ratio': pl_ratio,
        'euclidean_adjustment': avg_adjustment
    }

def main():
    """Main comparison function"""
    
    # Compare parameters
    impacts = compare_parameters()
    
    print("\n\n=== SUMMARY ===")
    print("="*70)
    print("Key findings:")
    print("1. Most default parameters are similar between models")
    print("2. Main differences:")
    print("   - MET: 3.5 vs 4.5 (sensitivity vs global)")
    print("   - Practical limit cycling: 37.5 vs 40 kg")
    print("   - Euclidean adjustments: Not used vs 1.3-1.4x")
    print("3. These parameter differences have offsetting effects")
    print("4. The main discrepancy sources remain:")
    print("   - Water ration calculation formulas (7.5x difference)")
    print("   - Slope distributions (awaiting GIS data analysis)")

if __name__ == "__main__":
    main()