#!/usr/bin/env python3
"""
Proposed Fixes for Model Harmonization

This script demonstrates how to fix the calculation discrepancies
between sensitivity analysis and global model.
"""

import numpy as np
from pathlib import Path
import sys

# Setup paths
project_root = Path(__file__).resolve().parent
sys.path.append(str(project_root))

def current_sensitivity_calculation(velocity_kg, water_ration, t_hours):
    """Current sensitivity analysis calculation (problematic)"""
    t_secs = t_hours * 3600
    water_ration_kms = velocity_kg / water_ration * t_secs / 1000
    return water_ration_kms

def current_global_calculation(avg_velocity, max_load, t_hours):
    """Current global model calculation"""
    # One-way distance to water source
    max_distance = avg_velocity * t_hours * 3600 / 2 / 1000  # km
    # Total water transport capacity
    water_ration_kms = max_distance * max_load
    return water_ration_kms

def proposed_harmonized_calculation(avg_velocity, max_load, t_hours, water_ration):
    """Proposed harmonized calculation for both models"""
    
    # Calculate one-way distance to water source (km)
    one_way_distance_km = avg_velocity * t_hours * 3600 / 2 / 1000
    
    # Calculate different metrics
    metrics = {
        'one_way_distance_km': one_way_distance_km,
        'round_trip_distance_km': one_way_distance_km * 2,
        'water_transport_capacity_L_km': one_way_distance_km * max_load,
        'daily_water_collection_trips': water_ration / max_load,
        'time_per_trip_hours': one_way_distance_km * 2 / (avg_velocity * 3.6),  # Round trip time
        'water_access_score': one_way_distance_km * max_load / water_ration
    }
    
    return metrics

def demonstrate_fixes():
    """Demonstrate the proposed fixes"""
    
    print("=== PROPOSED FIXES FOR MODEL HARMONIZATION ===")
    print("="*70)
    
    # Test parameters
    test_velocity = 3.0  # m/s
    test_max_load = 20.0  # kg
    test_time_hours = 5.5
    test_water_ration = 15.0  # L
    
    velocity_kg = test_velocity * test_max_load
    
    print(f"\nTest Parameters:")
    print(f"  Average velocity: {test_velocity} m/s")
    print(f"  Max load: {test_max_load} kg")
    print(f"  Time available: {test_time_hours} hours")
    print(f"  Water ration: {test_water_ration} L/day")
    
    # Current calculations
    print(f"\n=== CURRENT CALCULATIONS ===")
    
    sens_result = current_sensitivity_calculation(velocity_kg, test_water_ration, test_time_hours)
    global_result = current_global_calculation(test_velocity, test_max_load, test_time_hours)
    
    print(f"\nSensitivity analysis result: {sens_result:.2f} km")
    print(f"Global model result: {global_result:.2f} km")
    print(f"Ratio (global/sensitivity): {global_result/sens_result:.2f}x")
    print(f"→ These are measuring different things!")
    
    # Proposed harmonized calculation
    print(f"\n=== PROPOSED HARMONIZED CALCULATION ===")
    
    metrics = proposed_harmonized_calculation(test_velocity, test_max_load, test_time_hours, test_water_ration)
    
    print(f"\nClear Metric Definitions:")
    print(f"  One-way distance to water: {metrics['one_way_distance_km']:.2f} km")
    print(f"  Round-trip distance: {metrics['round_trip_distance_km']:.2f} km")
    print(f"  Water transport capacity: {metrics['water_transport_capacity_L_km']:.1f} L⋅km")
    print(f"  Daily trips needed: {metrics['daily_water_collection_trips']:.1f} trips")
    print(f"  Time per trip: {metrics['time_per_trip_hours']:.1f} hours")
    print(f"  Water access score: {metrics['water_access_score']:.2f}")
    
    # Code changes needed
    print(f"\n=== REQUIRED CODE CHANGES ===")
    
    print(f"\n1. In sensitivity_analysis.py:")
    print(f"   Replace:")
    print(f"     water_ration_kms = mean_vel_kg_per_slope / mv.waterration * t_secs / 1000")
    print(f"   With:")
    print(f"     one_way_distance_km = mean_avg_velocity * mv.t_hours / 2")
    print(f"     water_transport_capacity = one_way_distance_km * mean_max_load")
    
    print(f"\n2. In gis_global_module.py:")
    print(f"   Keep current calculation but clarify naming:")
    print(f"     one_way_distance_km = average_velocity * time_gathering_water / 2")
    print(f"     water_transport_capacity = one_way_distance_km * max_load")
    
    print(f"\n3. Add consistent metric reporting:")
    print(f"   - Always specify if distance is one-way or round-trip")
    print(f"   - Use consistent units (km, not m)")
    print(f"   - Report multiple metrics for clarity")
    
    # Additional recommendations
    print(f"\n=== ADDITIONAL RECOMMENDATIONS ===")
    
    print(f"\n1. Create shared calculation module:")
    print(f"   src/water_access_metrics.py")
    print(f"   - Standardized calculation functions")
    print(f"   - Clear documentation")
    print(f"   - Unit tests")
    
    print(f"\n2. Update variable names:")
    print(f"   - 'water_ration_kms' → 'water_transport_capacity_L_km'")
    print(f"   - 'max_distance' → 'one_way_distance_km'")
    print(f"   - Add units to all variable names")
    
    print(f"\n3. Add validation:")
    print(f"   - Check that velocities are reasonable (0-10 m/s)")
    print(f"   - Check that loads don't exceed limits")
    print(f"   - Verify time constraints are met")

def create_unified_metric_module():
    """Create a unified metric calculation module"""
    
    module_content = '''"""
Water Access Metrics Module

Provides standardized calculations for water access metrics across all models.
Ensures consistency between sensitivity analysis and global models.
"""

import numpy as np

class WaterAccessMetrics:
    """Calculate standardized water access metrics"""
    
    @staticmethod
    def calculate_metrics(velocity_m_s, max_load_kg, time_hours, 
                         water_ration_L=15, verbose=False):
        """
        Calculate comprehensive water access metrics.
        
        Parameters:
        -----------
        velocity_m_s : float
            Average velocity in meters per second
        max_load_kg : float
            Maximum water load in kilograms (≈ liters)
        time_hours : float
            Time available for water collection in hours
        water_ration_L : float
            Daily water requirement in liters (default: 15L)
        verbose : bool
            Print detailed calculations
            
        Returns:
        --------
        dict : Dictionary containing all water access metrics
        """
        
        # Convert velocity to km/h for easier interpretation
        velocity_km_h = velocity_m_s * 3.6
        
        # Calculate one-way distance (assuming round trip in available time)
        one_way_distance_km = velocity_m_s * time_hours * 3600 / 2 / 1000
        
        # Calculate round-trip distance
        round_trip_distance_km = one_way_distance_km * 2
        
        # Water transport capacity (how much water × distance can be moved)
        water_transport_L_km = one_way_distance_km * max_load_kg
        
        # Number of trips needed per day
        trips_per_day = water_ration_L / max_load_kg if max_load_kg > 0 else np.inf
        
        # Time per round trip
        time_per_trip_hours = round_trip_distance_km / velocity_km_h if velocity_km_h > 0 else np.inf
        
        # Total time needed for daily water
        total_time_daily_hours = time_per_trip_hours * trips_per_day
        
        # Water access feasibility (can daily needs be met in available time?)
        is_feasible = total_time_daily_hours <= time_hours
        
        # Water access score (higher is better)
        # Represents efficiency: water collected per unit time
        water_access_score = (max_load_kg / time_per_trip_hours 
                             if time_per_trip_hours > 0 else 0)
        
        metrics = {
            'velocity_m_s': velocity_m_s,
            'velocity_km_h': velocity_km_h,
            'max_load_kg': max_load_kg,
            'one_way_distance_km': one_way_distance_km,
            'round_trip_distance_km': round_trip_distance_km,
            'water_transport_L_km': water_transport_L_km,
            'trips_per_day': trips_per_day,
            'time_per_trip_hours': time_per_trip_hours,
            'total_time_daily_hours': total_time_daily_hours,
            'is_feasible': is_feasible,
            'water_access_score': water_access_score
        }
        
        if verbose:
            print(f"\\nWater Access Metrics Calculation:")
            print(f"  Velocity: {velocity_m_s:.2f} m/s ({velocity_km_h:.1f} km/h)")
            print(f"  Max load: {max_load_kg:.1f} kg")
            print(f"  Available time: {time_hours:.1f} hours")
            print(f"  Water ration: {water_ration_L:.1f} L/day")
            print(f"\\nResults:")
            print(f"  One-way distance: {one_way_distance_km:.2f} km")
            print(f"  Round-trip distance: {round_trip_distance_km:.2f} km")
            print(f"  Water transport capacity: {water_transport_L_km:.1f} L⋅km")
            print(f"  Trips needed per day: {trips_per_day:.1f}")
            print(f"  Time per trip: {time_per_trip_hours:.2f} hours")
            print(f"  Total daily time: {total_time_daily_hours:.2f} hours")
            print(f"  Feasible: {'Yes' if is_feasible else 'No'}")
            print(f"  Water access score: {water_access_score:.2f}")
        
        return metrics
    
    @staticmethod
    def compare_scenarios(scenario1, scenario2, names=('Scenario 1', 'Scenario 2')):
        """Compare two water access scenarios"""
        
        print(f"\\n=== Comparing {names[0]} vs {names[1]} ===")
        
        key_metrics = [
            ('one_way_distance_km', 'One-way distance', 'km'),
            ('water_transport_L_km', 'Water transport', 'L⋅km'),
            ('time_per_trip_hours', 'Time per trip', 'hours'),
            ('water_access_score', 'Access score', '')
        ]
        
        for metric, label, unit in key_metrics:
            val1 = scenario1[metric]
            val2 = scenario2[metric]
            ratio = val2 / val1 if val1 > 0 else np.inf
            
            print(f"{label}:")
            print(f"  {names[0]}: {val1:.2f} {unit}")
            print(f"  {names[1]}: {val2:.2f} {unit}")
            print(f"  Ratio: {ratio:.2f}x")
'''
    
    # Save the module
    module_path = project_root / "src" / "water_access_metrics.py"
    
    print(f"\n=== CREATING UNIFIED METRICS MODULE ===")
    print(f"Saving to: {module_path}")
    
    with open(module_path, 'w') as f:
        f.write(module_content)
    
    print("✓ Module created successfully!")
    print("\nUsage example:")
    print("  from src.water_access_metrics import WaterAccessMetrics")
    print("  metrics = WaterAccessMetrics.calculate_metrics(")
    print("      velocity_m_s=3.0, max_load_kg=20, time_hours=5.5, verbose=True)")

def main():
    """Main execution"""
    
    # Demonstrate fixes
    demonstrate_fixes()
    
    # Create unified module
    create_unified_metric_module()
    
    print("\n" + "="*70)
    print("IMPLEMENTATION COMPLETE")
    print("="*70)
    print("\nNext steps:")
    print("1. Update sensitivity_analysis.py to use new calculations")
    print("2. Update gis_global_module.py variable names for clarity")
    print("3. Run validation tests with both models")
    print("4. Document the harmonized metrics in README")

if __name__ == "__main__":
    main()