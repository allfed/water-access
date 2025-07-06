#!/usr/bin/env python3
"""
Sensitivity Analysis Script for Water Access Mobility Model

This script performs sensitivity analysis on the mobility model for water access research.
Refactored from sensitivity_notebook.ipynb to eliminate discrepancies with the global model.

Key improvements:
- Eliminated MATLAB-style 3D matrices
- Aligned with global model function calls
- Simplified variable handling
- Added proper error handling and logging
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

class SensitivityAnalyzer:
    """
    Performs sensitivity analysis on mobility models for water access research.
    
    This class handles the analysis of how different parameters affect model outcomes,
    comparing Lankford (walking) and Martin (cycling) models.
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
        
        # Graph colors for consistent plotting
        self.graph_colours = ["#3D87CB", "#F0B323", "#DC582A", "#674230", "#3A913F", "#75787B"]
        
        # Results storage
        self.full_result_dict = {}
        self.plot_dict = {}
        self.df_large = pd.DataFrame()
        
        logger.info(f"Initialized SensitivityAnalyzer for {model_type} model")
    
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
            
            logger.info(f"Loaded {len(param_df)} parameter records and {len(sens_df)} sensitivity variables")
            return param_df, sens_df
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def initialize_model_components(self, param_df: pd.DataFrame) -> Tuple:
        """
        Initialize model components (options, variables, MET values, HPV, results).
        
        Args:
            param_df: Filtered parameter DataFrame
            
        Returns:
            Tuple of (mo, mv, met, hpv, mr)
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
            
            logger.info("Initialized model components successfully")
            return mo, mv, met, hpv, mr
            
        except Exception as e:
            logger.error(f"Error initializing model components: {e}")
            raise
    
    def create_phase_space(self, sens_row: pd.Series, resolution: int = 7) -> np.ndarray:
        """
        Create phase space for sensitivity analysis.
        
        Args:
            sens_row: Row from sensitivity DataFrame containing min/max/default values
            resolution: Number of points in linear space
            
        Returns:
            Array of values to test
        """
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
    
    def apply_sensitivity_parameter(self, var_string: str, var_test: float, 
                                  hpv, mv, mo, met) -> Tuple:
        """
        Apply sensitivity parameter to model components.
        
        This method properly handles 3D array assignments to match the expected structure
        of the mobility module. HPV variables are stored as 3D arrays with shape (n_hpv, 1, 1).
        
        Args:
            var_string: Name of the variable being tested
            var_test: Value to test
            hpv, mv, mo, met: Model components
            
        Returns:
            Updated model components
        """
        if var_string == "Coefficient of Rolling Resistance":
            # Maintain 3D array structure: (n_hpv, 1, 1)
            hpv.Crr = np.array([[[var_test]]])
        elif var_string == "Load Limit":
            hpv.load_limit = np.array([[[var_test]]])
        elif (var_string == "Practical Limit Cycling" and mo.model_selection == 2):
            hpv.practical_limit = np.array([[[var_test]]])
        elif (var_string == "Practical Limit Walking" and mo.model_selection == 3):
            hpv.practical_limit = np.array([[[var_test]]])
        elif var_string == "Reference Area":
            mv.A = var_test  # model_variables are scalars, not arrays
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
        elif var_string == "Water Ration":
            mv.waterration = var_test  # model_variables are scalars
        elif var_string == "Human Weight":
            mv.m1 = var_test  # model_variables are scalars
        elif var_string == "Human Power Output":
            mv.P_t = var_test  # model_variables are scalars
        else:
            logger.warning(f"Unknown sensitivity parameter: {var_string}")
        
        return hpv, mv, mo, met
    
    def run_direct_model_calls(self, mv, mo, met, hpv) -> List[Dict]:
        """
        Run direct mobility model calls like the global model does.
        
        This method calls single_lankford_run() or single_bike_run() directly
        for different slopes, mimicking the global model approach.
        
        Args:
            mv, mo, met, hpv: Model components
            
        Returns:
            List of dictionaries containing velocity and load results
        """
        # Test slopes (simplified for sensitivity analysis)
        test_slopes = [0, 1, 2, 3]  # degrees
        load_attempt = 15  # kg
        
        results = []
        
        for slope in test_slopes:
            try:
                if mo.model_selection == 2:  # Cycling (Martin model)
                    loaded_velocity, unloaded_velocity, max_load = mm.mobility_models.single_bike_run(
                        mv, mo, hpv, slope, load_attempt
                    )
                elif mo.model_selection == 3:  # Walking (Lankford model)
                    loaded_velocity, unloaded_velocity, max_load = mm.mobility_models.single_lankford_run(
                        mv, mo, met, hpv, slope, load_attempt
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
                    'max_load': max_load,
                    'velocity_kg': avg_velocity * max_load
                })
                
            except Exception as e:
                logger.error(f"Error in direct model call for slope {slope}: {e}")
                results.append({
                    'slope': slope,
                    'loaded_velocity': np.nan,
                    'unloaded_velocity': np.nan,
                    'avg_velocity': np.nan,
                    'max_load': np.nan,
                    'velocity_kg': np.nan
                })
        
        return results
    
    def run_single_sensitivity(self, sens_row: pd.Series, param_df: pd.DataFrame) -> pd.DataFrame:
        """
        Run sensitivity analysis for a single parameter.
        
        Args:
            sens_row: Row from sensitivity DataFrame
            param_df: Parameter DataFrame
            
        Returns:
            DataFrame with results
        """
        var_string = sens_row["Short Name"]
        var_units = sens_row["Units"]
        
        logger.info(f"Running sensitivity analysis for: {var_string}")
        
        # Create phase space
        phase_space = self.create_phase_space(sens_row)
        
        # Storage for results
        results = {
            'trip_velocity_mean': [],
            'unloaded_velocity_mean': [],
            'loaded_velocity_mean': [],
            'velocitykgs': [],
            'water_ration_kms': []
        }
        
        for var_test in phase_space:
            try:
                # Initialize fresh model components for each iteration to avoid parameter persistence
                mo, mv, met, hpv, mr = self.initialize_model_components(param_df)
                
                # Apply sensitivity parameter
                hpv, mv, mo, met = self.apply_sensitivity_parameter(
                    var_string, var_test, hpv, mv, mo, met
                )
                
                # Use direct model calls like the global model (Phase 3 implementation)
                velocities = self.run_direct_model_calls(mv, mo, met, hpv)
                
                # Calculate metrics directly from velocities
                mean_vel_kg_per_slope = np.mean([
                    v['loaded_velocity'] * v['max_load'] for v in velocities
                ])
                
                # Store results from direct model calls
                mean_loaded_velocity = np.mean([v['loaded_velocity'] for v in velocities if not np.isnan(v['loaded_velocity'])])
                mean_unloaded_velocity = np.mean([v['unloaded_velocity'] for v in velocities if not np.isnan(v['unloaded_velocity'])])
                mean_avg_velocity = np.mean([v['avg_velocity'] for v in velocities if not np.isnan(v['avg_velocity'])])
                
                # Calculate water ration kms (matching original formula)
                t_secs = mv.t_hours * 60 * 60
                water_ration_kms = mean_vel_kg_per_slope / mv.waterration * t_secs / 1000
                
                results['trip_velocity_mean'].append(mean_avg_velocity)
                results['unloaded_velocity_mean'].append(mean_unloaded_velocity)
                results['loaded_velocity_mean'].append(mean_loaded_velocity)
                results['velocitykgs'].append(mean_vel_kg_per_slope)
                results['water_ration_kms'].append(water_ration_kms)
                
            except Exception as e:
                logger.error(f"Error in sensitivity calculation for {var_string} = {var_test}: {e}")
                # Fill with NaN for failed calculations
                for key in results:
                    results[key].append(np.nan)
        
        # Create results DataFrame
        df_results = pd.DataFrame({
            'Velocity Kgs': results['velocitykgs'],
            'Water Ration Kms': results['water_ration_kms'],
            'Mean Trip Velocity': results['trip_velocity_mean'],
            'Mean Unloaded Velocity': results['unloaded_velocity_mean'],
            'Mean Loaded Velocity': results['loaded_velocity_mean'],
            'Variable': phase_space,
            'Name': var_string
        })
        
        # Add adjustment columns and metadata
        resolution = len(phase_space) - 3  # Subtract the 3 special values
        df_results["Adjusted Result"] = df_results["Velocity Kgs"] - df_results.at[resolution, 'Velocity Kgs']
        df_results["Adjusted Water Ration Kms"] = df_results["Water Ration Kms"] - df_results.at[resolution, 'Water Ration Kms']
        
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
    
    def run_full_sensitivity_analysis(self) -> pd.DataFrame:
        """
        Run complete sensitivity analysis for all parameters.
        
        Returns:
            Combined DataFrame with all results
        """
        logger.info("Starting full sensitivity analysis")
        
        # Load data
        param_df, sens_df = self.load_data()
        
        # Run analysis for each sensitivity parameter
        for i, (_, sens_row) in enumerate(sens_df.iterrows()):
            var_string = sens_row["Short Name"]
            
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
    
    def create_individual_plots(self, sens_df: pd.DataFrame) -> Dict:
        """
        Create individual plots for each sensitivity parameter.
        
        Args:
            sens_df: Sensitivity DataFrame
            
        Returns:
            Dictionary of plots
        """
        primary_graph_value = "Water Ration Kms"
        max_plot_y = self.df_large[primary_graph_value].max() * 1.10
        min_plot_y = self.df_large[primary_graph_value].min() * 0.90
        
        for i, (_, sens_row) in enumerate(sens_df.iterrows()):
            var_string = sens_row["Short Name"]
            
            if var_string not in self.full_result_dict:
                continue
                
            try:
                fig = px.scatter(
                    self.full_result_dict[var_string],
                    x="Variable",
                    y=primary_graph_value,
                    color='DataType',
                    color_discrete_sequence=self.graph_colours,
                    size='MarkerSize',
                    title=f'Effect of {var_string} on model Velocity Kgs',
                    hover_data=[
                        "Velocity Kgs",
                        'Water Ration Kms',
                        'Mean Trip Velocity',
                        'Mean Loaded Velocity',
                        'Mean Unloaded Velocity',
                    ]
                ).update_layout(
                    yaxis_title=primary_graph_value,
                    xaxis_title=f'{var_string} <i>{sens_row["Units"]}</i>',
                )
                
                fig.update_yaxes(range=[min_plot_y, max_plot_y])
                fig = pt.format_plotly_graphs(fig)
                
                self.plot_dict[var_string] = fig
                
            except Exception as e:
                logger.error(f"Error creating plot for {var_string}: {e}")
        
        return self.plot_dict
    
    def create_summary_plot(self, mo, show_default_annotations: bool = False) -> go.Figure:
        """
        Create summary bar plot showing parameter importance.
        
        Args:
            mo: Model options
            
        Returns:
            Plotly figure
        """
        # Filter and prepare summary data
        summary_df = self.df_large[
            (self.df_large.DataType == "Minimum Expected") | 
            (self.df_large.DataType == "Maximum Expected") | 
            (self.df_large.DataType == "Default")
        ].copy()
        
        # Rename variables for better display
        name_mapping = {
            'T_hours': 'Time to water [hours]',
            'Reference Area': 'Reference area [m^2]',
            'Coefficient of Rolling Resistance': 'Coefficient of rolling resistance',
            'Water Ration': 'Water ration [L/day]',
            'Load Limit': 'Rated capacity [kg]',
            'Practical Limit Cycling': 'Practical limit (cycling) [kg]',
            'Practical Limit Walking': 'Practical limit (walking) [kg]',
            'Human Weight': 'Human mass [kg]',
            'Human Power Output': 'Power output [W]',
            'MET budget': 'Energy expenditure [MET]',
            'HPV Weight': 'HPV mass [kg]',
            'Air Density': 'Air density [kg/m^3]',
            'Drag Coefficient': 'Drag coefficient',
        }
        summary_df['Name'] = summary_df['Name'].replace(name_mapping)
        
        # Calculate parameter importance
        pivot_df = summary_df.pivot(index='Name', columns='DataType', values='Adjusted Water Ration Kms')
        pivot_df['abs_diff'] = np.abs(pivot_df['Maximum Expected'] - pivot_df['Minimum Expected'])
        pivot_df.reset_index(inplace=True)
        
        # Sort by importance, but keep parameters with zero sensitivity
        pivot_df = pivot_df.sort_values('abs_diff', ascending=False)
        
        # Merge back
        summary_df = pd.merge(pivot_df[['Name', 'abs_diff']], summary_df, on='Name')
        
        # Filter based on model type (exclude non-relevant parameters)
        if mo.model_selection == 2:  # Cycling
            exclude_vars = ["Energy expenditure [MET]", 'Practical limit (walking) [kg]']
        else:  # Walking
            exclude_vars = [
                'Reference area [m^2]', 'Practical limit (cycling) [kg]', 
                'Power output [W]', 'Efficiency', 'Drag coefficient',
                'Coefficient of rolling resistance', 'Air density [kg/m^3]',
                'Rated capacity [kg]'
            ]
        
        summary_df = summary_df[~summary_df.Name.isin(exclude_vars)]
        
        # Ensure minimum bar width for visibility (add small offset for zero-sensitivity parameters)
        min_bar_width = 0.5  # Minimum visible bar width
        for idx, row in summary_df.iterrows():
            if row['abs_diff'] < min_bar_width:
                param_name = row['Name']
                default_val = summary_df[(summary_df['Name'] == param_name) & 
                                       (summary_df['DataType'] == 'Default')]['Adjusted Water Ration Kms'].iloc[0]
                
                # Add small offset to make bars visible
                summary_df.loc[(summary_df['Name'] == param_name) & 
                             (summary_df['DataType'] == 'Minimum Expected'), 
                             'Adjusted Water Ration Kms'] = default_val - min_bar_width/2
                summary_df.loc[(summary_df['Name'] == param_name) & 
                             (summary_df['DataType'] == 'Maximum Expected'), 
                             'Adjusted Water Ration Kms'] = default_val + min_bar_width/2
        
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
            x="Adjusted Water Ration Kms", 
            color="DataType", 
            color_discrete_map=color_discrete_map,
            labels={
                "Name": "",
                "Adjusted Water Ration Kms": "Adjusted Water Ration Kms [15 L.km]",
                "DataType": "Input Value"
            }
        )
        
        # Add baseline reference line (x=0)
        fig.add_vline(x=0, line_dash="dash", line_color="gray", line_width=2, 
                     annotation_text="Baseline (Default Values)", 
                     annotation_position="top")
        
        # Add annotations for values
        default_values = summary_df[summary_df['DataType'] == 'Default']
        min_expected_values = summary_df[summary_df['DataType'] == 'Minimum Expected']
        max_expected_values = summary_df[summary_df['DataType'] == 'Maximum Expected']
        
        def get_xshift(row, default_values):
            default_value = default_values[default_values['Name'] == row['Name']]['Adjusted Water Ration Kms'].values[0]
            return -25 if row['Adjusted Water Ration Kms'] < default_value else 25
        
        # Add value annotations for min/max (showing water ration impact)
        for _, row in min_expected_values.iterrows():
            fig.add_annotation(
                x=row['Adjusted Water Ration Kms'],
                y=row['Name'],
                text=f"{row['Adjusted Water Ration Kms']:.1f}",
                showarrow=False,
                font=dict(color="#42BFDD", size=12),
                xshift=get_xshift(row, default_values),
            )
        
        for _, row in max_expected_values.iterrows():
            fig.add_annotation(
                x=row['Adjusted Water Ration Kms'],
                y=row['Name'],
                text=f"{row['Adjusted Water Ration Kms']:.1f}",
                showarrow=False,
                font=dict(color="#084B83", size=12),
                xshift=get_xshift(row, default_values),
            )
        
        # Add default value annotations (centered) - optional
        if show_default_annotations:
            for _, row in default_values.iterrows():
                fig.add_annotation(
                    x=0,  # Always at baseline
                    y=row['Name'],
                    text=f"Default: {row['Variable']:.3g}",
                    showarrow=False,
                    font=dict(color="black", size=10),
                    xshift=0,
                    yshift=-15,  # Slightly below the bar
                    bgcolor="white",
                    bordercolor="gray",
                    borderwidth=1
                )
        
        # Add sensitivity indicator for very low sensitivity parameters
        for _, row in default_values.iterrows():
            if row['abs_diff'] < 0.5:  # Very low sensitivity
                fig.add_annotation(
                    x=row['Adjusted Water Ration Kms'],
                    y=row['Name'],
                    text="(Low sensitivity)",
                    showarrow=False,
                    font=dict(color="gray", size=9, style="italic"),
                    xshift=75,
                    yshift=5
                )
        
        fig.update_layout(
            width=1000,
            height=500,
            font=dict(size=16),
            margin=dict(l=275, r=50, t=70, b=50),
            title=dict(
                text=f"{self.model_type} Model Sensitivity Analysis",
                x=0.5,
                font=dict(size=18)
            )
        )
        
        return fig
    
    def save_results(self, output_dir: Optional[str] = None) -> None:
        """
        Save analysis results to files.
        
        Args:
            output_dir: Directory to save results
        """
        try:
            if output_dir is None:
                project_root = Path(__file__).resolve().parent.parent
                output_path = project_root / "results"
            else:
                output_path = Path(output_dir)
            output_path.mkdir(exist_ok=True)
            
            # Save summary data
            summary_file = output_path / f"sensitivity_analysis_{self.model_type.lower()}_summary.csv"
            self.df_large.to_csv(summary_file, index=False)
            
            # Save individual plots
            for var_name, fig in self.plot_dict.items():
                safe_name = var_name.replace(" ", "_").replace("/", "_")
                plot_file = output_path / f"sensitivity_{self.model_type.lower()}_{safe_name}.html"
                fig.write_html(str(plot_file))
            
            logger.info(f"Results saved to {output_path}")
            
        except Exception as e:
            logger.error(f"Error saving results: {e}")


def main():
    """
    Main execution function for sensitivity analysis.
    """
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Water Access Sensitivity Analysis')
    parser.add_argument('--model', '-m', type=str, choices=['Lankford', 'Martin'], 
                       default='Lankford', help='Model type to analyze')
    parser.add_argument('--show-defaults', action='store_true', 
                       help='Show default parameter value annotations on plot')
    args = parser.parse_args()
    
    model_type = args.model
    logger.info("Starting Sensitivity Analysis")
    logger.info(f"Using model: {model_type}")
    
    try:
        # Initialize analyzer
        analyzer = SensitivityAnalyzer(model_type=model_type)
        
        # Run full analysis
        df_results = analyzer.run_full_sensitivity_analysis()
        
        # Load sensitivity variables for plotting
        _, sens_df = analyzer.load_data()
        
        # Create plots
        analyzer.create_individual_plots(sens_df)
        
        # Create summary plot
        project_root = Path(__file__).resolve().parent.parent
        mo, _, _, _, _ = analyzer.initialize_model_components(
            pd.read_csv(project_root / "data/lookup tables/mobility-model-parameters.csv")
            .loc[lambda x: x["Name"] == analyzer.filter_value]
        )
        summary_fig = analyzer.create_summary_plot(mo, show_default_annotations=args.show_defaults)
        
        # Display summary plot
        summary_fig.show()
        
        # Save results
        analyzer.save_results()
        
        # Save summary figure
        project_root = Path(__file__).resolve().parent.parent
        output_file = project_root / f"results/sensitivity_analysis_{model_type.lower()}.png"
        summary_fig.write_image(output_file, width=1000, height=500)
        
        logger.info("Sensitivity analysis completed successfully")
        
    except Exception as e:
        logger.error(f"Sensitivity analysis failed: {e}")
        raise


if __name__ == "__main__":
    main() 