#!/usr/bin/env python3
"""
Final attempt to solve the mystery by examining the actual velocity and distance data.
"""

import pandas as pd
import numpy as np

print("=== SOLVING THE 3.6x MYSTERY ===")
print()

# Check the actual walking velocity data
velocity_file = 'data/processed/walk_velocity_by_zone.csv'
try:
    df_vel = pd.read_csv(velocity_file)
    print(f"Loaded velocity data: {len(df_vel)} zones")
    
    if 'average_velocity_walk' in df_vel.columns:
        velocities = df_vel['average_velocity_walk'].dropna()
        print(f"\nVelocity statistics:")
        print(f"  Mean: {velocities.mean():.3f} m/s")
        print(f"  Median: {velocities.median():.3f} m/s")
        
        # Calculate what distances these would give
        time_hours = 5.5
        mean_distance = velocities.mean() * time_hours * 3600 / 2 / 1000
        median_distance = velocities.median() * time_hours * 3600 / 2 / 1000
        
        print(f"\nWith time = {time_hours} hours:")
        print(f"  Mean distance: {mean_distance:.2f} km")
        print(f"  Median distance: {median_distance:.2f} km")
        
        # Check if there's population data
        if 'pop_zone' in df_vel.columns:
            # Calculate population-weighted velocity
            pop_data = df_vel[['average_velocity_walk', 'pop_zone']].dropna()
            total_pop = pop_data['pop_zone'].sum()
            weighted_velocity = (pop_data['average_velocity_walk'] * pop_data['pop_zone']).sum() / total_pop
            weighted_distance = weighted_velocity * time_hours * 3600 / 2 / 1000
            
            print(f"\nPopulation-weighted:")
            print(f"  Weighted velocity: {weighted_velocity:.3f} m/s")
            print(f"  Weighted distance: {weighted_distance:.2f} km")
            
            # Check for country grouping
            if 'alpha3' in df_vel.columns or 'ISOCODE' in df_vel.columns:
                country_col = 'alpha3' if 'alpha3' in df_vel.columns else 'ISOCODE'
                
                # Calculate by country
                print(f"\nCalculating country-level distances...")
                
                # For each country, calculate weighted median
                country_results = []
                
                for country in df_vel[country_col].dropna().unique()[:10]:  # First 10 countries
                    country_data = df_vel[df_vel[country_col] == country]
                    
                    if len(country_data) > 0 and 'average_velocity_walk' in country_data.columns:
                        # Calculate max distances
                        country_data['max_distance'] = country_data['average_velocity_walk'] * time_hours * 3600 / 2 / 1000
                        
                        # Simple mean
                        mean_dist = country_data['max_distance'].mean()
                        
                        # Population weighted if available
                        if 'pop_zone' in country_data.columns:
                            pop_weights = country_data['pop_zone'].dropna()
                            if len(pop_weights) > 0:
                                weighted_dist = np.average(
                                    country_data.loc[pop_weights.index, 'max_distance'],
                                    weights=pop_weights
                                )
                            else:
                                weighted_dist = mean_dist
                        else:
                            weighted_dist = mean_dist
                        
                        country_results.append({
                            'country': country,
                            'zones': len(country_data),
                            'mean_distance': mean_dist,
                            'weighted_distance': weighted_dist
                        })
                
                if country_results:
                    results_df = pd.DataFrame(country_results)
                    print(f"\nCountry results (first 10):")
                    print(results_df.to_string(index=False))
                    
                    print(f"\nOverall country statistics:")
                    print(f"  Mean of country means: {results_df['mean_distance'].mean():.2f} km")
                    print(f"  Mean of weighted: {results_df['weighted_distance'].mean():.2f} km")
        
except Exception as e:
    print(f"Error loading velocity data: {e}")

# Check the actual country results
print("\n\n=== CHECKING ACTUAL RESULTS ===")
results_file = 'results/country_median_results.csv'
try:
    df_results = pd.read_csv(results_file)
    
    if 'weighted_med_walking' in df_results.columns:
        walking_distances = df_results['weighted_med_walking'].dropna()
        
        print(f"\nActual global model results:")
        print(f"  Mean: {walking_distances.mean():.2f} km")
        print(f"  Median: {walking_distances.median():.2f} km")
        print(f"  Min: {walking_distances.min():.2f} km")
        print(f"  Max: {walking_distances.max():.2f} km")
        
        # Show some examples
        print(f"\nExample countries:")
        for i in range(min(5, len(df_results))):
            country = df_results.iloc[i]['Entity']
            distance = df_results.iloc[i]['weighted_med_walking']
            print(f"  {country}: {distance:.2f} km")
            
except Exception as e:
    print(f"Error loading results: {e}")

print("\n\n=== HYPOTHESIS ===")
print("The 3.6x reduction could be due to:")
print("1. Population weighting toward low-velocity zones")
print("2. Country-level aggregation effects")
print("3. Median vs mean calculations")
print("4. Some zones having very low velocities")

# Final check: Look for zones with velocities around 0.374 m/s
print("\n\n=== CHECKING FOR LOW VELOCITY ZONES ===")
try:
    if 'average_velocity_walk' in df_vel.columns:
        # Find zones with velocity close to 0.374 m/s
        target_velocity = 0.374  # The implied velocity from results
        low_vel_zones = df_vel[
            (df_vel['average_velocity_walk'] > target_velocity * 0.8) & 
            (df_vel['average_velocity_walk'] < target_velocity * 1.2)
        ]
        
        print(f"Zones with velocity near {target_velocity:.3f} m/s:")
        print(f"  Count: {len(low_vel_zones)}")
        print(f"  Percentage: {len(low_vel_zones) / len(df_vel) * 100:.1f}%")
        
        if 'pop_zone' in df_vel.columns:
            total_pop = df_vel['pop_zone'].sum()
            low_vel_pop = low_vel_zones['pop_zone'].sum()
            print(f"  Population percentage: {low_vel_pop / total_pop * 100:.1f}%")
            
except Exception as e:
    print(f"Error checking low velocity zones: {e}")

print("\n" + "="*60)
print("CONCLUSION: Need to check the exact aggregation method used!")