# Data Visualization and Plotting Module
# Generates publication-ready plots and visualizations

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import logging
from datetime import datetime
import tkinter as tk
from tkinter import ttk

# Set publication-ready style
plt.style.use('default')
sns.set_palette("husl")

class VisualizationEngine:
    """Handles all visualization and plotting functionality"""
    
    def __init__(self, output_dir: Optional[Path] = None):
        self.output_dir = output_dir or Path("visualizations")
        self.output_dir.mkdir(exist_ok=True)
        
        self.logger = logging.getLogger(__name__)
        
        # Publication settings
        self.fig_params = {
            'figure.figsize': (10, 8),
            'font.size': 12,
            'axes.titlesize': 14,
            'axes.labelsize': 12,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 10,
            'figure.dpi': 300,
            'savefig.dpi': 300,
            'savefig.bbox': 'tight',
            'savefig.pad_inches': 0.1
        }
        
        plt.rcParams.update(self.fig_params)
    
    def create_condition_comparison_plots(self, results: Dict) -> List[Path]:
        """Create comprehensive condition comparison plots"""
        
        plot_files = []
        
        if 'summary_statistics' not in results or 'condition_comparisons' not in results['summary_statistics']:
            self.logger.warning("No condition comparison data available")
            return plot_files
        
        comparisons = results['summary_statistics']['condition_comparisons']
        
        # Create plots for each quantified channel
        for channel_name, condition_data in comparisons.items():
            
            # Bar plot with error bars
            bar_plot_file = self._create_condition_bar_plot(channel_name, condition_data)
            if bar_plot_file:
                plot_files.append(bar_plot_file)
            
            # Box plot
            box_plot_file = self._create_condition_box_plot(channel_name, condition_data, results)
            if box_plot_file:
                plot_files.append(box_plot_file)
        
        # Overall summary plot
        summary_plot_file = self._create_experiment_summary_plot(results)
        if summary_plot_file:
            plot_files.append(summary_plot_file)
        
        return plot_files
    
    def _create_condition_bar_plot(self, channel_name: str, condition_data: Dict) -> Optional[Path]:
        """Create bar plot comparing conditions for a specific channel"""
        
        try:
            conditions = list(condition_data.keys())
            means = [condition_data[cond]['mean'] for cond in conditions]
            stds = [condition_data[cond]['std'] for cond in conditions]
            
            fig, ax = plt.subplots(figsize=(10, 6))
            
            bars = ax.bar(conditions, means, yerr=stds, capsize=5, 
                         alpha=0.7, color=sns.color_palette("husl", len(conditions)))
            
            ax.set_ylabel(f'CTCF ({channel_name})')
            ax.set_title(f'Condition Comparison - {channel_name}')
            ax.set_xlabel('Experimental Conditions')
            
            # Add value labels on bars
            for bar, mean, std in zip(bars, means, stds):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + std + max(means)*0.01,
                       f'{mean:.0f}±{std:.0f}',
                       ha='center', va='bottom', fontsize=10)
            
            # Add sample size annotations
            for i, (cond, bar) in enumerate(zip(conditions, bars)):
                n_cells = condition_data[cond]['n_cells']
                ax.text(bar.get_x() + bar.get_width()/2., max(means)*0.05,
                       f'n={n_cells}', ha='center', va='bottom', 
                       fontsize=9, alpha=0.7)
            
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            
            # Save plot
            filename = f"condition_comparison_{channel_name}_barplot.png"
            filepath = self.output_dir / filename
            plt.savefig(filepath, **{'dpi': 300, 'bbox_inches': 'tight'})
            plt.close()
            
            self.logger.info(f"Created bar plot: {filepath}")
            return filepath
            
        except Exception as e:
            self.logger.error(f"Error creating bar plot for {channel_name}: {e}")
            return None
    
    def _create_condition_box_plot(self, channel_name: str, condition_data: Dict, results: Dict) -> Optional[Path]:
        """Create box plot comparing conditions using individual cell data"""
        
        try:
            # Collect individual cell data
            plot_data = []
            
            for condition_name, cond_results in results['conditions'].items():
                if not cond_results['cell_data']:
                    continue
                
                df = pd.DataFrame(cond_results['cell_data'])
                ctcf_column = f"{channel_name}_ctcf"
                
                if ctcf_column in df.columns:
                    for value in df[ctcf_column]:
                        plot_data.append({
                            'Condition': condition_name,
                            'CTCF': value
                        })
            
            if not plot_data:
                self.logger.warning(f"No data available for box plot: {channel_name}")
                return None
            
            plot_df = pd.DataFrame(plot_data)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            
            sns.boxplot(data=plot_df, x='Condition', y='CTCF', ax=ax, 
                       palette="husl", showfliers=True)
            
            # Add individual points
            sns.stripplot(data=plot_df, x='Condition', y='CTCF', ax=ax,
                         size=3, alpha=0.3, color='black')
            
            ax.set_ylabel(f'CTCF ({channel_name})')
            ax.set_title(f'Distribution Comparison - {channel_name}')
            ax.set_xlabel('Experimental Conditions')
            
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            
            # Save plot
            filename = f"condition_comparison_{channel_name}_boxplot.png"
            filepath = self.output_dir / filename
            plt.savefig(filepath, **{'dpi': 300, 'bbox_inches': 'tight'})
            plt.close()
            
            self.logger.info(f"Created box plot: {filepath}")
            return filepath
            
        except Exception as e:
            self.logger.error(f"Error creating box plot for {channel_name}: {e}")
            return None
    
    def _create_experiment_summary_plot(self, results: Dict) -> Optional[Path]:
        """Create overall experiment summary visualization"""
        
        try:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('Experiment Summary', fontsize=16, fontweight='bold')
            
            # Plot 1: Cell counts per condition
            self._plot_cell_counts(ax1, results)
            
            # Plot 2: Image counts per condition  
            self._plot_image_counts(ax2, results)
            
            # Plot 3: Cell area distribution
            self._plot_cell_area_distribution(ax3, results)
            
            # Plot 4: Processing timeline
            self._plot_processing_info(ax4, results)
            
            plt.tight_layout()
            
            # Save plot
            filename = "experiment_summary.png"
            filepath = self.output_dir / filename
            plt.savefig(filepath, **{'dpi': 300, 'bbox_inches': 'tight'})
            plt.close()
            
            self.logger.info(f"Created summary plot: {filepath}")
            return filepath
            
        except Exception as e:
            self.logger.error(f"Error creating summary plot: {e}")
            return None
    
    def _plot_cell_counts(self, ax, results: Dict):
        """Plot cell counts per condition"""
        
        conditions = []
        cell_counts = []
        
        for cond_name, cond_data in results['conditions'].items():
            conditions.append(cond_name)
            cell_counts.append(len(cond_data['cell_data']))
        
        bars = ax.bar(conditions, cell_counts, alpha=0.7, color=sns.color_palette("husl", len(conditions)))
        ax.set_ylabel('Number of Cells')
        ax.set_title('Cells Analyzed per Condition')
        
        # Add value labels
        for bar, count in zip(bars, cell_counts):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + max(cell_counts)*0.01,
                   f'{count}', ha='center', va='bottom')
        
        ax.tick_params(axis='x', rotation=45)
    
    def _plot_image_counts(self, ax, results: Dict):
        """Plot image counts per condition"""
        
        conditions = []
        image_counts = []
        
        for cond_name, cond_data in results['conditions'].items():
            conditions.append(cond_name)
            image_counts.append(len(cond_data['images']))
        
        bars = ax.bar(conditions, image_counts, alpha=0.7, color=sns.color_palette("husl", len(conditions)))
        ax.set_ylabel('Number of Images')
        ax.set_title('Images Processed per Condition')
        
        # Add value labels
        for bar, count in zip(bars, image_counts):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + max(image_counts)*0.01,
                   f'{count}', ha='center', va='bottom')
        
        ax.tick_params(axis='x', rotation=45)
    
    def _plot_cell_area_distribution(self, ax, results: Dict):
        """Plot cell area distribution across conditions"""
        
        all_areas = []
        condition_labels = []
        
        for cond_name, cond_data in results['conditions'].items():
            if cond_data['cell_data']:
                df = pd.DataFrame(cond_data['cell_data'])
                areas = df['area'].values
                all_areas.extend(areas)
                condition_labels.extend([cond_name] * len(areas))
        
        if all_areas:
            plot_df = pd.DataFrame({'Condition': condition_labels, 'Cell Area': all_areas})
            sns.violinplot(data=plot_df, x='Condition', y='Cell Area', ax=ax, palette="husl")
            ax.set_title('Cell Area Distribution')
            ax.tick_params(axis='x', rotation=45)
    
    def _plot_processing_info(self, ax, results: Dict):
        """Plot processing information"""
        
        if 'experiment_info' in results:
            info = results['experiment_info']
            
            # Create simple info display
            ax.axis('off')
            
            info_text = f"""
            Processing Summary:
            
            Total Conditions: {info.get('total_conditions', 'N/A')}
            Total Images: {info.get('total_images_processed', 'N/A')}
            Duration: {info.get('duration_seconds', 0):.1f} seconds
            
            Start Time: {info.get('start_time', 'N/A')}
            """
            
            ax.text(0.1, 0.5, info_text, transform=ax.transAxes, fontsize=12,
                   verticalalignment='center', bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
    
    def create_segmentation_overlay(self, image: np.ndarray, masks: np.ndarray, 
                                   output_filename: str) -> Optional[Path]:
        """Create segmentation overlay visualization"""
        
        try:
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
            
            # Original image
            ax1.imshow(image, cmap='gray')
            ax1.set_title('Original Image')
            ax1.axis('off')
            
            # Segmentation masks
            ax2.imshow(masks, cmap='nipy_spectral')
            ax2.set_title('Segmentation Masks')
            ax2.axis('off')
            
            # Overlay
            ax3.imshow(image, cmap='gray', alpha=0.7)
            ax3.imshow(masks, cmap='nipy_spectral', alpha=0.3)
            ax3.set_title('Overlay')
            ax3.axis('off')
            
            plt.tight_layout()
            
            # Save
            filepath = self.output_dir / f"segmentation_{output_filename}.png"
            plt.savefig(filepath, **{'dpi': 300, 'bbox_inches': 'tight'})
            plt.close()
            
            return filepath
            
        except Exception as e:
            self.logger.error(f"Error creating segmentation overlay: {e}")
            return None
    
    def create_correlation_matrix(self, results: Dict) -> Optional[Path]:
        """Create correlation matrix of all quantified channels"""
        
        try:
            # Collect all quantified data
            all_data = []
            
            for cond_name, cond_data in results['conditions'].items():
                if cond_data['cell_data']:
                    df = pd.DataFrame(cond_data['cell_data'])
                    # Add condition column
                    df['condition'] = cond_name
                    all_data.append(df)
            
            if not all_data:
                self.logger.warning("No data available for correlation analysis")
                return None
            
            combined_df = pd.concat(all_data, ignore_index=True)
            
            # Get CTCF columns
            ctcf_columns = [col for col in combined_df.columns if col.endswith('_ctcf')]
            
            if len(ctcf_columns) < 2:
                self.logger.warning("Need at least 2 CTCF channels for correlation analysis")
                return None
            
            # Calculate correlation matrix
            corr_matrix = combined_df[ctcf_columns].corr()
            
            # Create heatmap
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                       square=True, ax=ax, cbar_kws={'label': 'Correlation Coefficient'})
            
            ax.set_title('Channel Correlation Matrix')
            plt.tight_layout()
            
            # Save
            filepath = self.output_dir / "correlation_matrix.png"
            plt.savefig(filepath, **{'dpi': 300, 'bbox_inches': 'tight'})
            plt.close()
            
            self.logger.info(f"Created correlation matrix: {filepath}")
            return filepath
            
        except Exception as e:
            self.logger.error(f"Error creating correlation matrix: {e}")
            return None
    
    def create_statistical_report(self, results: Dict) -> Optional[Path]:
        """Create comprehensive statistical report"""
        
        try:
            # Prepare statistical analysis
            statistical_tests = self._perform_statistical_tests(results)
            
            # Create report figure
            fig = plt.figure(figsize=(16, 20))
            
            # Title
            fig.suptitle('Statistical Analysis Report', fontsize=20, fontweight='bold')
            
            # Create subplots for different analyses
            gs = fig.add_gridspec(6, 2, hspace=0.3, wspace=0.3)
            
            # Summary statistics table
            ax1 = fig.add_subplot(gs[0, :])
            self._create_summary_table(ax1, results)
            
            # Statistical test results
            ax2 = fig.add_subplot(gs[1, :])
            self._create_statistical_test_table(ax2, statistical_tests)
            
            # Effect size plots
            ax3 = fig.add_subplot(gs[2, 0])
            ax4 = fig.add_subplot(gs[2, 1])
            self._create_effect_size_plots(ax3, ax4, statistical_tests)
            
            # Distribution plots
            ax5 = fig.add_subplot(gs[3:5, :])
            self._create_distribution_comparison(ax5, results)
            
            # Power analysis
            ax6 = fig.add_subplot(gs[5, :])
            self._create_power_analysis_summary(ax6, statistical_tests)
            
            # Save report
            filepath = self.output_dir / "statistical_report.png"
            plt.savefig(filepath, **{'dpi': 300, 'bbox_inches': 'tight'})
            plt.close()
            
            self.logger.info(f"Created statistical report: {filepath}")
            return filepath
            
        except Exception as e:
            self.logger.error(f"Error creating statistical report: {e}")
            return None
    
    def _perform_statistical_tests(self, results: Dict) -> Dict:
        """Perform statistical tests between conditions"""
        
        from scipy import stats
        
        statistical_results = {}
        
        # Get all combinations of conditions
        conditions = list(results['conditions'].keys())
        
        if len(conditions) < 2:
            return statistical_results
        
        # For each quantified channel, perform comparisons
        for cond_name, cond_data in results['conditions'].items():
            if not cond_data['cell_data']:
                continue
                
            df = pd.DataFrame(cond_data['cell_data'])
            ctcf_columns = [col for col in df.columns if col.endswith('_ctcf')]
            
            for ctcf_col in ctcf_columns:
                channel_name = ctcf_col.replace('_ctcf', '')
                
                if channel_name not in statistical_results:
                    statistical_results[channel_name] = {}
                
                # Collect data for all conditions
                condition_data = {}
                for other_cond, other_data in results['conditions'].items():
                    if other_data['cell_data']:
                        other_df = pd.DataFrame(other_data['cell_data'])
                        if ctcf_col in other_df.columns:
                            condition_data[other_cond] = other_df[ctcf_col].values
                
                # Perform pairwise comparisons
                for i, cond1 in enumerate(conditions):
                    for j, cond2 in enumerate(conditions[i+1:], i+1):
                        if cond1 in condition_data and cond2 in condition_data:
                            
                            data1 = condition_data[cond1]
                            data2 = condition_data[cond2]
                            
                            # T-test
                            t_stat, t_pval = stats.ttest_ind(data1, data2)
                            
                            # Mann-Whitney U test
                            u_stat, u_pval = stats.mannwhitneyu(data1, data2, alternative='two-sided')
                            
                            # Effect size (Cohen's d)
                            pooled_std = np.sqrt(((len(data1)-1)*np.var(data1, ddof=1) + 
                                                (len(data2)-1)*np.var(data2, ddof=1)) / 
                                               (len(data1) + len(data2) - 2))
                            cohens_d = (np.mean(data1) - np.mean(data2)) / pooled_std if pooled_std > 0 else 0
                            
                            comparison_key = f"{cond1}_vs_{cond2}"
                            statistical_results[channel_name][comparison_key] = {
                                'n1': len(data1),
                                'n2': len(data2),
                                'mean1': np.mean(data1),
                                'mean2': np.mean(data2),
                                'std1': np.std(data1, ddof=1),
                                'std2': np.std(data2, ddof=1),
                                't_statistic': t_stat,
                                't_pvalue': t_pval,
                                'u_statistic': u_stat,
                                'u_pvalue': u_pval,
                                'cohens_d': cohens_d,
                                'significant': t_pval < 0.05
                            }
        
        return statistical_results
    
    def _create_summary_table(self, ax, results: Dict):
        """Create summary statistics table"""
        
        ax.axis('off')
        ax.set_title('Summary Statistics', fontweight='bold', pad=20)
        
        # Prepare table data
        table_data = []
        headers = ['Condition', 'N Cells', 'N Images', 'Mean Area', 'Std Area']
        
        for cond_name, cond_data in results['conditions'].items():
            if cond_data['cell_data']:
                df = pd.DataFrame(cond_data['cell_data'])
                row = [
                    cond_name,
                    len(df),
                    len(cond_data['images']),
                    f"{df['area'].mean():.1f}",
                    f"{df['area'].std():.1f}"
                ]
                table_data.append(row)
        
        if table_data:
            table = ax.table(cellText=table_data, colLabels=headers,
                           cellLoc='center', loc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1, 2)
    
    def _create_statistical_test_table(self, ax, statistical_tests: Dict):
        """Create statistical test results table"""
        
        ax.axis('off')
        ax.set_title('Statistical Test Results', fontweight='bold', pad=20)
        
        if not statistical_tests:
            ax.text(0.5, 0.5, 'No statistical tests performed', 
                   ha='center', va='center', transform=ax.transAxes)
            return
        
        # Prepare table for first channel (as example)
        channel_name = list(statistical_tests.keys())[0]
        channel_tests = statistical_tests[channel_name]
        
        table_data = []
        headers = ['Comparison', 'T-test p-value', 'Mann-Whitney p-value', 'Cohen\'s d', 'Significant']
        
        for comparison, test_results in channel_tests.items():
            row = [
                comparison.replace('_vs_', ' vs '),
                f"{test_results['t_pvalue']:.4f}",
                f"{test_results['u_pvalue']:.4f}",
                f"{test_results['cohens_d']:.3f}",
                "Yes" if test_results['significant'] else "No"
            ]
            table_data.append(row)
        
        if table_data:
            table = ax.table(cellText=table_data, colLabels=headers,
                           cellLoc='center', loc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(9)
            table.scale(1, 1.5)
    
    def _create_effect_size_plots(self, ax1, ax2, statistical_tests: Dict):
        """Create effect size visualization plots"""
        
        # Placeholder implementation
        ax1.axis('off')
        ax1.set_title('Effect Sizes')
        ax1.text(0.5, 0.5, 'Effect size visualization\n(To be implemented)', 
                ha='center', va='center', transform=ax1.transAxes)
        
        ax2.axis('off')
        ax2.set_title('Power Analysis')
        ax2.text(0.5, 0.5, 'Power analysis\n(To be implemented)', 
                ha='center', va='center', transform=ax2.transAxes)
    
    def _create_distribution_comparison(self, ax, results: Dict):
        """Create distribution comparison plot"""
        
        ax.set_title('Data Distribution Comparison')
        ax.text(0.5, 0.5, 'Distribution comparison plot\n(To be implemented)', 
                ha='center', va='center', transform=ax.transAxes)
    
    def _create_power_analysis_summary(self, ax, statistical_tests: Dict):
        """Create power analysis summary"""
        
        ax.axis('off')
        ax.set_title('Power Analysis Summary')
        ax.text(0.5, 0.5, 'Power analysis summary\n(To be implemented)', 
                ha='center', va='center', transform=ax.transAxes)

class InteractiveVisualization:
    """Interactive visualization components for GUI integration"""
    
    def __init__(self, master_widget):
        self.master = master_widget
        self.logger = logging.getLogger(__name__)
    
    def create_embedded_plot(self, figure, title: str = ""):
        """Create embedded matplotlib plot in tkinter"""
        
        import tkinter as tk
        from tkinter import ttk
        
        # Create frame for the plot
        plot_frame = ttk.LabelFrame(self.master, text=title)
        plot_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Create canvas
        canvas = FigureCanvasTkAgg(figure, plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Add toolbar
        from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk
        toolbar = NavigationToolbar2Tk(canvas, plot_frame)
        toolbar.update()
        
        return canvas, toolbar
    
    def create_live_progress_plot(self):
        """Create live progress visualization during analysis"""
        
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.set_title('Analysis Progress')
        ax.set_xlabel('Time')
        ax.set_ylabel('Progress (%)')
        
        # This would be updated in real-time during analysis
        return fig, ax

# Integration with main application
def integrate_visualization_with_gui(main_app):
    """Integrate visualization capabilities with main GUI application"""

    
    # Add visualization tab to main application
    viz_frame = main_app.notebook.add_tab("Visualizations")
    
    # Create visualization engine
    viz_engine = VisualizationEngine(Path(main_app.config.get('output_directory', 'visualizations')))
    
    # Add visualization controls
    control_frame = ttk.LabelFrame(viz_frame, text="Visualization Controls")
    control_frame.pack(fill=tk.X, padx=5, pady=5)
    
    ttk.Button(control_frame, text="Generate All Plots", 
              command=lambda: generate_all_plots(viz_engine, main_app.results)).pack(side=tk.LEFT, padx=5)
    
    ttk.Button(control_frame, text="Create Custom Plot", 
              command=lambda: create_custom_plot_dialog(viz_engine)).pack(side=tk.LEFT, padx=5)
    
    # Visualization display area
    display_frame = ttk.Frame(viz_frame)
    display_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
    
    return viz_engine, display_frame

def generate_all_plots(viz_engine: VisualizationEngine, results: Dict):
    """Generate all standard plots"""
    
    if not results:
        return
    
    try:
        # Generate condition comparison plots
        viz_engine.create_condition_comparison_plots(results)
        
        # Generate correlation matrix
        viz_engine.create_correlation_matrix(results)
        
        # Generate statistical report
        viz_engine.create_statistical_report(results)
        
        print("All plots generated successfully!")
        
    except Exception as e:
        logging.error(f"Error generating plots: {e}")

def create_custom_plot_dialog(viz_engine: VisualizationEngine):
    """Create dialog for custom plot creation"""
    
    # This would open a dialog allowing users to create custom plots
    print("Custom plot dialog would open here")