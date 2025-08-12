#!/usr/bin/env python3
"""
Investigation script to trace the source of the 3.6 km walking distance
reported by the global model.
"""

import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def check_monte_carlo_results():
    """Check if there are existing Monte Carlo results to examine."""
    
    print("=== CHECKING FOR EXISTING MONTE CARLO RESULTS ===")
    
    # Check common result locations
    result_paths = [
        'results/',
        'data/processed/',
        'outputs/',
        '.'
    ]
    
    for path in result_paths:
        if os.path.exists(path):
            print(f"\nChecking {path}:")
            files = os.listdir(path)
            csv_files = [f for f in files if f.endswith('.csv')]
            
            if csv_files:
                print(f"  Found CSV files: {csv_files}")
                
                # Look for files that might contain Monte Carlo results
                for file in csv_files:
                    if any(keyword in file.lower() for keyword in ['monte', 'carlo', 'results', 'global', 'simulation']):
                        print(f"  -> Potentially relevant: {file}")
                        
                        # Quick peek at the file
                        try:
                            df = pd.read_csv(os.path.join(path, file))
                            print(f"     Columns: {list(df.columns)}")
                            print(f"     Shape: {df.shape}")
                            
                            # Look for walking distance columns
                            walking_cols = [col for col in df.columns if 'walk' in col.lower() or 'distance' in col.lower()]
                            if walking_cols:
                                print(f"     Walking-related columns: {walking_cols}")
                                
                                # Check for 3.6 km values
                                for col in walking_cols:
                                    if col in df.columns:
                                        values = df[col].dropna()
                                        if len(values) > 0:
                                            print(f"     {col}: min={values.min():.2f}, max={values.max():.2f}, mean={values.mean():.2f}, median={values.median():.2f}")
                                            
                                            # Check if any values are close to 3.6
                                            close_to_3_6 = values[(values >= 3.0) & (values <= 4.0)]
                                            if len(close_to_3_6) > 0:
                                                print(f"     *** Values close to 3.6: {close_to_3_6.values}")
                                            
                        except Exception as e:
                            print(f"     Error reading {file}: {e}")
            else:
                print(f"  No CSV files found")
        else:
            print(f"\nPath {path} does not exist")

def examine_monte_carlo_code():
    """Examine the Monte Carlo code to understand how results are processed."""
    
    print("\n=== EXAMINING MONTE CARLO CODE ===")
    
    # Look for the Monte Carlo processing code
    mc_files = [
        'src/monte_carlo.py',
        'scripts/run_monte_carlo.py',
        'src/monte_carlo_module.py'
    ]
    
    for file_path in mc_files:
        if os.path.exists(file_path):
            print(f"\nFound {file_path}")
            
            # Look for result processing functions
            with open(file_path, 'r') as f:
                content = f.read()
                
                # Look for functions that might calculate summary statistics
                if 'def process' in content:
                    print("  Contains result processing functions")
                    
                # Look for median/mean calculations
                if 'median' in content or 'mean' in content:
                    print("  Contains median/mean calculations")
                    
                # Look for walking distance references
                if 'walk' in content:
                    print("  Contains walking distance references")
                    
                # Extract lines containing '3.6' or similar
                lines = content.split('\n')
                for i, line in enumerate(lines):
                    if '3.6' in line or 'walking' in line.lower():
                        print(f"  Line {i+1}: {line.strip()}")
        else:
            print(f"\n{file_path} not found")

def check_references_to_3_6():
    """Check all references to 3.6 in the codebase."""
    
    print("\n=== CHECKING ALL REFERENCES TO 3.6 ===")
    
    # Search through analysis files for the 3.6 value
    analysis_files = [
        'CORRECTED_FINAL_SUMMARY.py',
        'final_analysis_summary.py',
        'discrepancy_analysis_summary.py',
        'corrected_analysis.py'
    ]
    
    for file_path in analysis_files:
        if os.path.exists(file_path):
            print(f"\nChecking {file_path}:")
            with open(file_path, 'r') as f:
                content = f.read()
                
                lines = content.split('\n')
                for i, line in enumerate(lines):
                    if '3.6' in line:
                        print(f"  Line {i+1}: {line.strip()}")
                        # Also show context
                        if i > 0:
                            print(f"    Context: {lines[i-1].strip()}")
                        if i < len(lines) - 1:
                            print(f"    Context: {lines[i+1].strip()}")
                        print()

def investigate_global_model_output():
    """Try to understand what the global model actually outputs."""
    
    print("\n=== INVESTIGATING GLOBAL MODEL OUTPUT ===")
    
    # Look at the global model to understand what it calculates
    global_file = 'src/gis_global_module.py'
    if os.path.exists(global_file):
        print(f"Examining {global_file}:")
        
        with open(global_file, 'r') as f:
            content = f.read()
            
            # Look for max distance calculation
            lines = content.split('\n')
            for i, line in enumerate(lines):
                if 'max distance walking' in line or 'max_distance_walking' in line:
                    print(f"  Line {i+1}: {line.strip()}")
                    # Show context
                    for j in range(max(0, i-2), min(len(lines), i+3)):
                        if j != i:
                            print(f"    {j+1}: {lines[j].strip()}")
                    print()
                    
                # Look for result processing
                if 'median' in line and 'walking' in line:
                    print(f"  Line {i+1}: {line.strip()}")
                    
                # Look for weighted calculations
                if 'weighted' in line and 'walking' in line:
                    print(f"  Line {i+1}: {line.strip()}")

def main():
    """Main investigation function."""
    
    print("INVESTIGATING THE SOURCE OF 3.6 KM WALKING DISTANCE")
    print("=" * 60)
    
    check_monte_carlo_results()
    examine_monte_carlo_code()
    check_references_to_3_6()
    investigate_global_model_output()
    
    print("\n" + "=" * 60)
    print("INVESTIGATION COMPLETE")
    print("Next steps: Run this script to identify where 3.6 km comes from")

if __name__ == "__main__":
    main()