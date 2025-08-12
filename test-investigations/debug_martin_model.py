#!/usr/bin/env python3
"""
Debug script to isolate the Martin model array issue.
"""

import sys
import os
import pandas as pd
import numpy as np

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import mobility_module as mm

def test_martin_model():
    print("Testing Martin model directly...")
    
    # Load parameter data
    param_df = pd.read_csv('data/lookup tables/mobility-model-parameters.csv')
    param_df = param_df.loc[param_df["Name"] == "Bicycle"]
    
    # Initialize components
    mo = mm.model_options()
    mv = mm.model_variables()
    hpv = mm.HPV_variables(param_df, mv)
    
    mo.model_selection = 2  # Martin cycling model
    
    print(f"hpv.Crr shape: {hpv.Crr.shape}")
    print(f"hpv.Crr value: {hpv.Crr}")
    print(f"hpv.m_HPV_only shape: {hpv.m_HPV_only.shape}")
    print(f"hpv.m_HPV_only value: {hpv.m_HPV_only}")
    print(f"hpv.load_capacity shape: {hpv.load_capacity.shape}")
    print(f"hpv.load_capacity value: {hpv.load_capacity}")
    
    try:
        result = mm.mobility_models.single_bike_run(mv, mo, hpv, 0, 15)
        print(f"Success! Result: {result}")
        print(f"Result type: {type(result)}")
        print(f"Result shapes: {[type(r) for r in result]}")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_martin_model()