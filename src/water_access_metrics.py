"""
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
            print(f"\nWater Access Metrics Calculation:")
            print(f"  Velocity: {velocity_m_s:.2f} m/s ({velocity_km_h:.1f} km/h)")
            print(f"  Max load: {max_load_kg:.1f} kg")
            print(f"  Available time: {time_hours:.1f} hours")
            print(f"  Water ration: {water_ration_L:.1f} L/day")
            print(f"\nResults:")
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
        
        print(f"\n=== Comparing {names[0]} vs {names[1]} ===")
        
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
