#!/usr/bin/env python3
"""
Refactored Sensitivity Analysis Script for Water Access Mobility Model

This script performs sensitivity analysis on the mobility model for water access research.
Key improvements from original:
- Removed confusing "water_ration_kms" metric
- Reports simple one-way distance to water source (matching global model)
- Fixed water carrying capacity at 15L for all vehicles
- Water requirement per person is now a sensitivity parameter
- Clearer metric definitions and documentation
"""

import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import plotly
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Setup path to import custom modules
notebook_path = Path(__file__).resolve().parent
project_root = notebook_path.parent
sys.path.append(str(project_root))

# Import custom modules
import src.mobility_module as mm
import src.plotting_tools_water_access as pt

# Set seaborn theme
sns.set_theme()

class RefactoredSensitivityAnalyzer:
    """
    Performs sensitivity analysis on mobility models for water access research.
    
    This refactored version uses simplified, clear metrics that match the global model.
    """
    
    def __init__(self, model_type: str = "Lankford"):
        """
        Initialize the sensitivity analyzer.
        
        Args:
            model_type: Either "Lankford" for walking or "Martin" for cycling
        """
        self.model_type = model_type
        self.model_number = 3 if model_type == "Lankford" else 2
        self.filter_value = "Buckets" if model_type == "Lankford" else "Bicycle"
        
        # Fixed water parameters
        self.fixed_water_capacity = 15.0  # L (standardized for all vehicles)
        self.default_water_requirement = 15.0  # L per person per day
        
        # Graph colors for consistent plotting
        self.graph_colours = ["#3D87CB", "#F0B323", "#DC582A", "#674230", "#3A913F", "#75787B"]
        
        # Results storage
        self.full_result_dict = {}
        self.plot_dict = {}
        self.df_large = pd.DataFrame()
        
        logger.info(f"Initialized RefactoredSensitivityAnalyzer for {model_type} model")
        logger.info(f"Fixed water capacity: {self.fixed_water_capacity}L")
    
    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load parameter data and sensitivity variables from CSV files.
        
        Returns:
            Tuple of (filtered_param_df, sensitivity_df)
        """
        try:
            # File paths (relative to project root)
            project_root = Path(__file__).resolve().parent.parent
            file_path_params = project_root / "data/lookup tables/mobility-model-parameters.csv"
            file_path_sens_vars = project_root / "data/lookup tables/Sensitivity Analysis Variables.csv"
            
            # Load data
            all_hpv_param_df = pd.read_csv(file_path_params)
            sens_df = pd.read_csv(file_path_sens_vars)
            
            # Filter parameters for the selected model
            param_df = all_hpv_param_df.loc[all_hpv_param_df["Name"] == self.filter_value]
            
            # Add water requirement as a sensitivity parameter if not present
            if "Water Requirement" not in sens_df["Short Name"].values:
                water_req_row = pd.DataFrame({
                    "Short Name": ["Water Requirement"],
                    "Long Name": ["Daily water requirement per person"],
                    "Units": ["L/day"],
                    "Default Value": [15.0],
                    "Expected Min": [10.0],
                    "Expected Max": [20.0],
                    "Plotting Min": [5.0],
                    "Plotting Max": [25.0]
                })
                sens_df = pd.concat([sens_df, water_req_row], ignore_index=True)
            
            logger.info(f"Loaded {len(param_df)} parameter records and {len(sens_df)} sensitivity variables")
            return param_df, sens_df
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def initialize_model_components(self, param_df: pd.DataFrame) -> Tuple:
        """
        Initialize model components with standardized water capacity.
        """
        try:
            # Initialize model components
            mo = mm.model_options()
            mv = mm.model_variables()
            met = mm.MET_values(mv, country_weight=60, met=3.5, use_country_specific_weights=False)
            hpv = mm.HPV_variables(param_df, mv)
            mr = mm.model_results(hpv, mo)
            
            # Set model selection
            mo.model_selection = self.model_number
            
            # Override load capacity to fixed water capacity
            # This ensures all vehicles carry the same amount of water
            hpv.load_limit = np.array([[[self.fixed_water_capacity]]])
            hpv.practical_limit = np.array([[[self.fixed_water_capacity]]])
            
            logger.info("Initialized model components with fixed water capacity")
            return mo, mv, met, hpv, mr
            
        except Exception as e:
            logger.error(f"Error initializing model components: {e}")
            raise
    
    def calculate_metrics(self, velocities: List[Dict], mv) -> Dict[str, float]:
        """
        Calculate clear, simple metrics from velocity results.
        
        Args:
            velocities: List of velocity results from model runs
            mv: Model variables
            
        Returns:
            Dictionary of calculated metrics
        """
        # Extract valid velocities
        valid_velocities = [v for v in velocities if not np.isnan(v['avg_velocity'])]
        
        if not valid_velocities:
            return {
                'one_way_distance_km': np.nan,
                'round_trip_distance_km': np.nan,
                'avg_velocity_m_s': np.nan,
                'trips_per_day': np.nan,
                'time_per_trip_hours': np.nan,
                'daily_water_collected_L': np.nan
            }
        
        # Calculate average velocity across slopes
        avg_velocity = np.mean([v['avg_velocity'] for v in valid_velocities])
        
        # Calculate one-way distance to water source (matching global model)
        one_way_distance_km = avg_velocity * mv.t_hours * 3600 / 2 / 1000
        
        # Calculate round-trip distance
        round_trip_distance_km = one_way_distance_km * 2
        
        # Calculate trips needed per day
        trips_per_day = self.default_water_requirement / self.fixed_water_capacity
        
        # Time per round trip
        time_per_trip_hours = round_trip_distance_km / (avg_velocity * 3.6) if avg_velocity > 0 else np.inf
        
        # Total daily water that can be collected in available time
        max_trips_in_time = mv.t_hours / time_per_trip_hours if time_per_trip_hours > 0 else 0
        daily_water_collected = min(max_trips_in_time * self.fixed_water_capacity, 
                                   self.default_water_requirement)
        
        return {
            'one_way_distance_km': one_way_distance_km,
            'round_trip_distance_km': round_trip_distance_km,
            'avg_velocity_m_s': avg_velocity,
            'trips_per_day': trips_per_day,
            'time_per_trip_hours': time_per_trip_hours,
            'daily_water_collected_L': daily_water_collected
        }
    
    def run_direct_model_calls(self, mv, mo, met, hpv) -> List[Dict]:
        """
        Run direct mobility model calls like the global model does.
        """
        # Test slopes (matching global model approach)
        test_slopes = [0, 1, 2, 3, 4, 5]  # Extended range for better representation
        
        results = []
        
        for slope in test_slopes:
            try:
                if mo.model_selection == 2:  # Cycling (Martin model)
                    loaded_velocity, unloaded_velocity, max_load = mm.mobility_models.single_bike_run(
                        mv, mo, hpv, slope, self.fixed_water_capacity
                    )
                elif mo.model_selection == 3:  # Walking (Lankford model)
                    loaded_velocity, unloaded_velocity, max_load = mm.mobility_models.single_lankford_run(
                        mv, mo, met, hpv, slope, self.fixed_water_capacity
                    )
                else:
                    logger.warning(f"Unknown model selection: {mo.model_selection}")
                    continue
                
                # Calculate average velocity (like global model)
                avg_velocity = (loaded_velocity + unloaded_velocity) / 2
                
                results.append({
                    'slope': slope,
                    'loaded_velocity': loaded_velocity,
                    'unloaded_velocity': unloaded_velocity,
                    'avg_velocity': avg_velocity,
                    'max_load': max_load
                })
                
            except Exception as e:
                logger.error(f"Error in direct model call for slope {slope}: {e}")
                results.append({
                    'slope': slope,
                    'loaded_velocity': np.nan,
                    'unloaded_velocity': np.nan,
                    'avg_velocity': np.nan,
                    'max_load': np.nan
                })
        
        return results
    
    def apply_sensitivity_parameter(self, var_string: str, var_test: float, 
                                  hpv, mv, mo, met) -> Tuple:
        """
        Apply sensitivity parameter to model components.
        Updated to include water requirement as a parameter.
        """
        if var_string == "Coefficient of Rolling Resistance":
            hpv.Crr = np.array([[[var_test]]])
        elif var_string == "Water Requirement":
            # This affects how many trips are needed, not the mobility calculation
            self.default_water_requirement = var_test
        elif (var_string == "Practical Limit Cycling" and mo.model_selection == 2):
            # Skip - we're using fixed water capacity
            pass
        elif (var_string == "Practical Limit Walking" and mo.model_selection == 3):
            # Skip - we're using fixed water capacity
            pass
        elif var_string == "Reference Area":
            mv.A = var_test
        elif var_string == "Drag Coefficient":
            mv.C_d = var_test
        elif var_string == "Efficiency":
            mv.eta = var_test
        elif var_string == "Air Density":
            mv.ro = var_test
        elif var_string == "MET budget":
            met = mm.MET_values(mv, country_weight=60, met=var_test, use_country_specific_weights=False)
        elif var_string == "T_hours":
            mv.t_hours = var_test
        elif var_string == "HPV Weight":
            hpv.m_HPV_only = np.array([[[var_test]]])
        elif var_string == "Human Weight":
            mv.m1 = var_test
        elif var_string == "Human Power Output":
            mv.P_t = var_test
        else:
            logger.warning(f"Unknown sensitivity parameter: {var_string}")
        
        return hpv, mv, mo, met
    
    def run_single_sensitivity(self, sens_row: pd.Series, param_df: pd.DataFrame) -> pd.DataFrame:
        """
        Run sensitivity analysis for a single parameter with simplified metrics.
        """
        var_string = sens_row["Short Name"]
        var_units = sens_row["Units"]
        
        logger.info(f"Running sensitivity analysis for: {var_string}")
        
        # Create phase space
        phase_space = self.create_phase_space(sens_row)
        
        # Storage for results
        results_list = []
        
        for var_test in phase_space:
            try:
                # Initialize fresh model components
                mo, mv, met, hpv, mr = self.initialize_model_components(param_df)
                
                # Apply sensitivity parameter
                hpv, mv, mo, met = self.apply_sensitivity_parameter(
                    var_string, var_test, hpv, mv, mo, met
                )
                
                # Run model with multiple slopes
                velocities = self.run_direct_model_calls(mv, mo, met, hpv)
                
                # Calculate simplified metrics
                metrics = self.calculate_metrics(velocities, mv)
                
                # Add to results
                result_row = {
                    'Variable': var_test,
                    'Name': var_string,
                    'One-way Distance (km)': metrics['one_way_distance_km'],
                    'Round-trip Distance (km)': metrics['round_trip_distance_km'],
                    'Average Velocity (m/s)': metrics['avg_velocity_m_s'],
                    'Trips per Day': metrics['trips_per_day'],
                    'Time per Trip (hours)': metrics['time_per_trip_hours'],
                    'Daily Water Collected (L)': metrics['daily_water_collected_L']
                }
                results_list.append(result_row)
                
            except Exception as e:
                logger.error(f"Error in sensitivity calculation for {var_string} = {var_test}: {e}")
                # Fill with NaN for failed calculations
                result_row = {
                    'Variable': var_test,
                    'Name': var_string,
                    'One-way Distance (km)': np.nan,
                    'Round-trip Distance (km)': np.nan,
                    'Average Velocity (m/s)': np.nan,
                    'Trips per Day': np.nan,
                    'Time per Trip (hours)': np.nan,
                    'Daily Water Collected (L)': np.nan
                }
                results_list.append(result_row)
        
        # Create results DataFrame
        df_results = pd.DataFrame(results_list)
        
        # Add adjustment columns for plotting
        resolution = len(phase_space) - 3  # Subtract the 3 special values
        default_distance = df_results.at[resolution, 'One-way Distance (km)']
        df_results["Adjusted Distance"] = df_results["One-way Distance (km)"] - default_distance
        
        # Add data type labels
        df_results["DataType"] = "Sensitivity"
        df_results["MarkerSize"] = 10
        
        # Mark special points
        df_results.at[resolution, 'DataType'] = "Default"
        df_results.at[resolution + 1, 'DataType'] = "Minimum Expected"
        df_results.at[resolution + 2, 'DataType'] = "Maximum Expected"
        df_results.at[resolution, 'MarkerSize'] = 50
        df_results.at[resolution + 1, 'MarkerSize'] = 20
        df_results.at[resolution + 2, 'MarkerSize'] = 20
        
        return df_results
    
    def create_phase_space(self, sens_row: pd.Series, resolution: int = 7) -> np.ndarray:
        """Create phase space for sensitivity analysis."""
        plot_min = sens_row["Plotting Min"]
        plot_max = sens_row["Plotting Max"]
        minval = sens_row["Expected Min"]
        maxval = sens_row["Expected Max"]
        def_val = sens_row["Default Value"]
        
        # Create linear space
        phase_space = np.linspace(
            start=plot_min,
            stop=plot_max,
            num=resolution,
            endpoint=True
        )
        
        # Add special values for clear visualization
        phase_space = np.append(phase_space, [def_val, minval, maxval])
        
        return phase_space
    
    def run_full_sensitivity_analysis(self) -> pd.DataFrame:
        """
        Run complete sensitivity analysis for all parameters.
        """
        logger.info("Starting full sensitivity analysis with refactored metrics")
        
        # Load data
        param_df, sens_df = self.load_data()
        
        # Run analysis for each sensitivity parameter
        for i, (_, sens_row) in enumerate(sens_df.iterrows()):
            var_string = sens_row["Short Name"]
            
            # Skip old load/practical limit parameters
            if var_string in ["Load Limit", "Practical Limit Cycling", "Practical Limit Walking"]:
                logger.info(f"Skipping {var_string} (using fixed water capacity)")
                continue
            
            try:
                df_results = self.run_single_sensitivity(sens_row, param_df)
                
                # Store results
                self.full_result_dict[var_string] = df_results
                self.df_large = pd.concat([self.df_large, df_results], sort=False)
                
                logger.info(f"Completed sensitivity analysis for {var_string} ({i+1}/{len(sens_df)})")
                
            except Exception as e:
                logger.error(f"Failed sensitivity analysis for {var_string}: {e}")
                continue
        
        logger.info("Completed full sensitivity analysis")
        return self.df_large
    
    def create_summary_plot(self, mo, show_default_annotations: bool = False) -> go.Figure:
        """
        Create summary bar plot showing parameter importance with new metrics.
        """
        # Filter and prepare summary data
        summary_df = self.df_large[
            (self.df_large.DataType == "Minimum Expected") | 
            (self.df_large.DataType == "Maximum Expected") | 
            (self.df_large.DataType == "Default")
        ].copy()
        
        # Rename variables for better display
        name_mapping = {
            'T_hours': 'Time available [hours]',
            'Reference Area': 'Reference area [m²]',
            'Coefficient of Rolling Resistance': 'Rolling resistance',
            'Water Requirement': 'Water requirement [L/day]',
            'Human Weight': 'Human mass [kg]',
            'Human Power Output': 'Power output [W]',
            'MET budget': 'Energy expenditure [MET]',
            'HPV Weight': 'Vehicle mass [kg]',
            'Air Density': 'Air density [kg/m³]',
            'Drag Coefficient': 'Drag coefficient',
            'Efficiency': 'Efficiency'
        }
        summary_df['Name'] = summary_df['Name'].replace(name_mapping)
        
        # Calculate parameter importance based on distance impact
        pivot_df = summary_df.pivot(index='Name', columns='DataType', values='Adjusted Distance')
        pivot_df['abs_diff'] = np.abs(pivot_df['Maximum Expected'] - pivot_df['Minimum Expected'])
        pivot_df.reset_index(inplace=True)
        
        # Sort by importance
        pivot_df = pivot_df.sort_values('abs_diff', ascending=False)
        
        # Merge back
        summary_df = pd.merge(pivot_df[['Name', 'abs_diff']], summary_df, on='Name')
        
        # Filter based on model type
        if mo.model_selection == 2:  # Cycling
            exclude_vars = ["Energy expenditure [MET]"]
        else:  # Walking
            exclude_vars = [
                'Reference area [m²]', 'Power output [W]', 'Efficiency', 
                'Drag coefficient', 'Rolling resistance', 'Air density [kg/m³]'
            ]
        
        summary_df = summary_df[~summary_df.Name.isin(exclude_vars)]
        
        # Create plot
        filtered_summary_df = summary_df[summary_df['DataType'] != 'Default']
        color_discrete_map = {
            'Minimum Expected': '#42BFDD', 
            'Maximum Expected': '#084B83', 
            'Default': None
        }
        
        fig = px.bar(
            filtered_summary_df, 
            y="Name", 
            x="Adjusted Distance", 
            color="DataType", 
            color_discrete_map=color_discrete_map,
            labels={
                "Name": "",
                "Adjusted Distance": "Impact on One-way Distance (km)",
                "DataType": "Parameter Value"
            },
            title=f"{self.model_type} Model Sensitivity Analysis - Impact on Water Access Distance"
        )
        
        # Add baseline reference line (x=0)
        fig.add_vline(x=0, line_dash="dash", line_color="gray", line_width=2, 
                     annotation_text="Baseline", 
                     annotation_position="top")
        
        fig.update_layout(
            width=1000,
            height=600,
            font=dict(size=14),
            margin=dict(l=250, r=50, t=70, b=50)
        )
        
        return fig
    
    def save_results(self, output_dir: Optional[str] = None) -> None:
        """Save analysis results to files."""
        try:
            if output_dir is None:
                project_root = Path(__file__).resolve().parent.parent
                output_path = project_root / "results"
            else:
                output_path = Path(output_dir)
            output_path.mkdir(exist_ok=True)
            
            # Save summary data
            summary_file = output_path / f"sensitivity_analysis_{self.model_type.lower()}_refactored.csv"
            self.df_large.to_csv(summary_file, index=False)
            
            logger.info(f"Results saved to {output_path}")
            
        except Exception as e:
            logger.error(f"Error saving results: {e}")


def main():
    """Main execution function for refactored sensitivity analysis."""
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Refactored Water Access Sensitivity Analysis')
    parser.add_argument('--model', '-m', type=str, choices=['Lankford', 'Martin'], 
                       default='Lankford', help='Model type to analyze')
    parser.add_argument('--show-defaults', action='store_true', 
                       help='Show default parameter value annotations on plot')
    args = parser.parse_args()
    
    model_type = args.model
    logger.info("Starting Refactored Sensitivity Analysis")
    logger.info(f"Using model: {model_type}")
    logger.info("Key changes: Fixed water capacity, simplified metrics")
    
    try:
        # Initialize analyzer
        analyzer = RefactoredSensitivityAnalyzer(model_type=model_type)
        
        # Run full analysis
        df_results = analyzer.run_full_sensitivity_analysis()
        
        # Create summary plot
        project_root = Path(__file__).resolve().parent.parent
        param_df = pd.read_csv(project_root / "data/lookup tables/mobility-model-parameters.csv")
        mo, _, _, _, _ = analyzer.initialize_model_components(
            param_df.loc[lambda x: x["Name"] == analyzer.filter_value]
        )
        summary_fig = analyzer.create_summary_plot(mo, show_default_annotations=args.show_defaults)
        
        # Display summary plot
        summary_fig.show()
        
        # Save results
        analyzer.save_results()
        
        # Print summary statistics
        print("\n=== SUMMARY STATISTICS ===")
        print(f"Model: {model_type}")
        print(f"Fixed water capacity: {analyzer.fixed_water_capacity}L")
        
        # Get default values
        default_results = df_results[df_results['DataType'] == 'Default']
        if not default_results.empty:
            default_row = default_results.iloc[0]
            print(f"\nDefault scenario results:")
            print(f"  One-way distance: {default_row['One-way Distance (km)']:.2f} km")
            print(f"  Round-trip distance: {default_row['Round-trip Distance (km)']:.2f} km")
            print(f"  Average velocity: {default_row['Average Velocity (m/s)']:.2f} m/s")
            print(f"  Time per trip: {default_row['Time per Trip (hours)']:.2f} hours")
        
        logger.info("Refactored sensitivity analysis completed successfully")
        
    except Exception as e:
        logger.error(f"Sensitivity analysis failed: {e}")
        raise


if __name__ == "__main__":
    main()