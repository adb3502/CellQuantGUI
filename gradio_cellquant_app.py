#!/usr/bin/env python3
"""
CellQuantGUI - Modern Gradio Web Interface
Beautiful, web-based UI for quantitative microscopy analysis
"""

import gradio as gr
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import json
import time
import logging
import threading
from datetime import datetime
import zipfile
import io
import base64

# Import existing backend components
from analysis_pipeline import AnalysisPipeline, AnalysisParameters
from template_manager import TemplateManager, AnalysisTemplate
from quality_control import QualityControlManager
from visualization_module import VisualizationEngine

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GradioInterface:
    """Modern Gradio web interface for CellQuantGUI."""
    
    def __init__(self):
        self.current_analysis = None
        self.analysis_results = None
        self.template_manager = TemplateManager()
        self.setup_directories()
        
        # Session management for multi-user support
        self.user_sessions = {}
        
        logger.info("🚀 CellQuantGUI Gradio Interface initialized")
    
    def setup_directories(self):
        """Setup required directories."""
        self.upload_dir = Path("uploads")
        self.results_dir = Path("results") 
        self.temp_dir = Path("temp")
        
        for directory in [self.upload_dir, self.results_dir, self.temp_dir]:
            directory.mkdir(exist_ok=True)
    
    def create_interface(self) -> gr.Blocks:
        """Create the main Gradio interface."""
        
        # Custom CSS for beautiful styling
        custom_css = """
        .gradio-container {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif !important;
        }
        .tab-nav button {
            font-size: 16px !important;
            font-weight: 600 !important;
        }
        .progress-container {
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            border-radius: 10px;
            padding: 20px;
            margin: 10px 0;
        }
        .analysis-card {
            border: 2px solid #e1e5e9;
            border-radius: 12px;
            padding: 20px;
            margin: 10px 0;
            background: #f8f9fa;
        }
        .success-banner {
            background: linear-gradient(90deg, #56ab2f 0%, #a8e6cf 100%);
            border-radius: 8px;
            padding: 15px;
            color: white;
            font-weight: 600;
        }
        """
        
        with gr.Blocks(
            title="🔬 CellQuantGUI - Quantitative Microscopy Analysis",
            theme=gr.themes.Soft(
                primary_hue="blue",
                secondary_hue="green",
                neutral_hue="slate"
            ),
            css=custom_css
        ) as interface:
            
            # Header
            gr.Markdown("""
            # 🔬 CellQuantGUI - Quantitative Microscopy Analysis
            ### Professional-grade cell segmentation and fluorescence quantification
            
            **✨ Features:** Cellpose Segmentation • CTCF Quantification • Statistical Analysis • Publication-Ready Plots
            """)
            
            # Session state
            session_state = gr.State({})
            
            with gr.Tabs() as tabs:
                
                # =================== EXPERIMENT SETUP TAB ===================
                with gr.TabItem("📋 Experiment Setup", id="setup"):
                    with gr.Row():
                        with gr.Column(scale=2):
                            gr.Markdown("### 🧪 Experiment Configuration")
                            
                            experiment_name = gr.Textbox(
                                label="Experiment Name",
                                placeholder="e.g., DNA_Damage_Analysis_2024",
                                value="",
                                info="Descriptive name for your analysis"
                            )
                            
                            experiment_description = gr.Textbox(
                                label="Description", 
                                lines=3,
                                placeholder="Brief description of the experiment...",
                                info="Optional: Describe your experimental conditions"
                            )
                            
                            # Template selection
                            with gr.Row():
                                template_dropdown = gr.Dropdown(
                                    label="🎯 Analysis Template",
                                    choices=self.get_template_choices(),
                                    value="custom",
                                    info="Pre-configured analysis workflows"
                                )
                                refresh_templates_btn = gr.Button("🔄 Refresh", size="sm")
                        
                        with gr.Column(scale=1):
                            gr.Markdown("### 📊 Analysis Parameters")
                            
                            cellpose_model = gr.Dropdown(
                                label="Cellpose Model",
                                choices=["cyto2", "cyto3", "cyto", "nuclei"],
                                value="cyto2",
                                info="Cell segmentation model"
                            )
                            
                            cell_diameter = gr.Slider(
                                label="Expected Cell Diameter (pixels)",
                                minimum=10,
                                maximum=200,
                                value=30,
                                step=5,
                                info="0 for auto-detection"
                            )
                            
                            background_method = gr.Dropdown(
                                label="Background Method",
                                choices=["mode", "median", "mean"],
                                value="mode",
                                info="Background fluorescence estimation"
                            )
                
                # =================== FILE UPLOAD TAB ===================
                with gr.TabItem("📁 Upload Images", id="upload"):
                    gr.Markdown("### 📤 Upload Your Microscopy Images")
                    gr.Markdown("**Supported formats:** TIFF, PNG, JPEG | **Naming convention:** `image_C0.tiff`, `image_C1.tiff`, etc.")
                    
                    with gr.Row():
                        with gr.Column():
                            file_upload = gr.File(
                                label="Upload Images",
                                file_count="multiple",
                                file_types=[".tiff", ".tif", ".png", ".jpg", ".jpeg"],
                                height=300
                            )
                            
                            upload_btn = gr.Button(
                                "📤 Process Uploaded Files", 
                                variant="primary",
                                size="lg"
                            )
                        
                        with gr.Column():
                            upload_status = gr.Markdown("**Status:** Ready to upload")
                            
                            uploaded_files_display = gr.Dataframe(
                                headers=["Filename", "Size", "Channel", "Status"],
                                datatype=["str", "str", "str", "str"],
                                label="Uploaded Files"
                            )
                    
                    # Channel Configuration
                    gr.Markdown("### 🎨 Channel Configuration")
                    
                    with gr.Row():
                        channel_config = gr.Dataframe(
                            headers=["Channel", "Name", "Type", "Purpose", "Quantify", "Wavelength"],
                            datatype=["str", "str", "str", "str", "bool", "str"],
                            label="Channel Settings",
                            interactive=True,
                            value=[
                                ["C0", "DAPI", "nuclear", "segmentation", False, "405nm"],
                                ["C1", "Target", "cellular", "quantification", True, "488nm"],
                            ]
                        )
                    
                    auto_detect_btn = gr.Button("🔍 Auto-Detect Channels", variant="secondary")
                
                # =================== ANALYSIS TAB ===================
                with gr.TabItem("⚡ Run Analysis", id="analysis"):
                    gr.Markdown("### 🚀 Start Analysis Pipeline")
                    
                    with gr.Row():
                        with gr.Column(scale=2):
                            analysis_summary = gr.Markdown("""
                            **📋 Analysis Summary**
                            - Experiment: *Not configured*
                            - Images: *0 uploaded*
                            - Channels: *Not configured*
                            - Template: *None selected*
                            """)
                            
                            start_analysis_btn = gr.Button(
                                "🚀 Start Analysis",
                                variant="primary",
                                size="lg",
                                interactive=False
                            )
                        
                        with gr.Column(scale=1):
                            gr.Markdown("**⚙️ Advanced Options**")
                            
                            use_gpu = gr.Checkbox(
                                label="🎮 Use GPU Acceleration",
                                value=True,
                                info="Requires CUDA-capable GPU"
                            )
                            
                            batch_size = gr.Slider(
                                label="Batch Size",
                                minimum=1,
                                maximum=20,
                                value=5,
                                step=1,
                                info="Images processed simultaneously"
                            )
                    
                    # Progress Section
                    with gr.Column():
                        progress_markdown = gr.Markdown("", visible=False)
                        progress_bar = gr.Progress()
                        
                        analysis_log = gr.Textbox(
                            label="📋 Analysis Log",
                            lines=10,
                            max_lines=20,
                            interactive=False,
                            visible=False
                        )
                
                # =================== RESULTS TAB ===================
                with gr.TabItem("📊 Results & Visualization", id="results"):
                    gr.Markdown("### 📈 Analysis Results")
                    
                    # Results will be populated after analysis
                    results_status = gr.Markdown("**Status:** No analysis completed yet")
                    
                    with gr.Tabs():
                        with gr.TabItem("📋 Summary Statistics"):
                            summary_stats = gr.Dataframe(
                                label="Condition Summary",
                                interactive=False
                            )
                        
                        with gr.TabItem("💾 Cell Data"):
                            cell_data_display = gr.Dataframe(
                                label="Individual Cell Measurements",
                                interactive=False
                            )
                        
                        with gr.TabItem("📊 Visualizations"):
                            with gr.Row():
                                plot_type = gr.Dropdown(
                                    label="Plot Type",
                                    choices=["Box Plot", "Bar Plot", "Scatter Plot", "Heatmap"],
                                    value="Box Plot"
                                )
                                generate_plot_btn = gr.Button("📊 Generate Plot")
                            
                            results_plot = gr.Plot(label="Analysis Visualization")
                        
                        with gr.TabItem("📄 Quality Report"):
                            quality_report = gr.HTML(label="Quality Control Report")
                    
                    # Download Section
                    gr.Markdown("### 💾 Download Results")
                    with gr.Row():
                        download_csv_btn = gr.Button("📄 Download CSV")
                        download_plots_btn = gr.Button("📊 Download Plots") 
                        download_report_btn = gr.Button("📋 Download Report")
                        download_all_btn = gr.Button("📦 Download All", variant="primary")
                    
                    download_files = gr.File(label="Download Files", visible=False)
            
            # =================== EVENT HANDLERS ===================
            
            # Template selection
            def update_template(template_name, session):
                if template_name != "custom":
                    template = self.template_manager.get_template(template_name)
                    if template:
                        return self.apply_template_to_interface(template)
                return session
            
            template_dropdown.change(
                update_template,
                inputs=[template_dropdown, session_state],
                outputs=[session_state]
            )
            
            # File upload processing
            upload_btn.click(
                self.process_uploaded_files,
                inputs=[file_upload],
                outputs=[upload_status, uploaded_files_display, session_state]
            )
            
            # Auto-detect channels
            auto_detect_btn.click(
                self.auto_detect_channels,
                inputs=[session_state],
                outputs=[channel_config]
            )
            
            # Start analysis
            start_analysis_btn.click(
                self.run_analysis,
                inputs=[
                    experiment_name, experiment_description, cellpose_model,
                    cell_diameter, background_method, channel_config,
                    use_gpu, batch_size, session_state
                ],
                outputs=[
                    progress_markdown, analysis_log, results_status,
                    summary_stats, cell_data_display
                ]
            )
            
            # Generate plots
            generate_plot_btn.click(
                self.generate_visualization,
                inputs=[plot_type, session_state],
                outputs=[results_plot]
            )
            
            # Download handlers
            download_csv_btn.click(
                self.download_csv,
                inputs=[session_state],
                outputs=[download_files]
            )
            
            download_all_btn.click(
                self.download_all_results,
                inputs=[session_state],
                outputs=[download_files]
            )
        
        return interface
    
    def get_template_choices(self) -> List[str]:
        """Get available analysis templates."""
        templates = self.template_manager.list_templates()
        choices = ["custom"] + [t.name for t in templates]
        return choices
    
    def process_uploaded_files(self, files) -> Tuple[str, pd.DataFrame, Dict]:
        """Process uploaded image files."""
        if not files:
            return "**Status:** No files uploaded", pd.DataFrame(), {}
        
        logger.info(f"Processing {len(files)} uploaded files")
        
        # Process files and detect channels
        file_info = []
        channel_groups = {}
        
        for file in files:
            if file is None:
                continue
                
            filename = Path(file.name).name
            file_size = f"{Path(file.name).stat().st_size / 1024:.1f} KB"
            
            # Detect channel from filename
            channel = self.detect_channel_from_filename(filename)
            
            file_info.append([filename, file_size, channel, "✅ Ready"])
            
            if channel not in channel_groups:
                channel_groups[channel] = []
            channel_groups[channel].append(file.name)
        
        df = pd.DataFrame(file_info, columns=["Filename", "Size", "Channel", "Status"])
        
        status = f"**Status:** ✅ Processed {len(files)} files, detected {len(channel_groups)} channels"
        
        session_data = {
            'uploaded_files': files,
            'channel_groups': channel_groups,
            'file_info': file_info
        }
        
        return status, df, session_data
    
    def detect_channel_from_filename(self, filename: str) -> str:
        """Detect channel from filename pattern."""
        import re
        
        patterns = [
            r'[_\-]C(\d+)\.tiff?$',
            r'[_\-]c(\d+)\.tiff?$', 
            r'[_\-]CH(\d+)\.tiff?$'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, filename, re.IGNORECASE)
            if match:
                return f"C{match.group(1)}"
        
        return "Unknown"
    
    def auto_detect_channels(self, session_data: Dict) -> pd.DataFrame:
        """Auto-detect channel configuration."""
        if not session_data or 'channel_groups' not in session_data:
            return pd.DataFrame()
        
        channel_config = []
        channel_groups = session_data['channel_groups']
        
        for channel in sorted(channel_groups.keys()):
            if channel == "C0":
                config = [channel, "DAPI", "nuclear", "segmentation", False, "405nm"]
            elif channel == "C1":
                config = [channel, "Target_1", "cellular", "quantification", True, "488nm"]
            elif channel == "C2":
                config = [channel, "Target_2", "cellular", "quantification", True, "568nm"]
            elif channel == "C3":
                config = [channel, "Target_3", "cellular", "quantification", True, "647nm"]
            else:
                config = [channel, f"Channel_{channel}", "cellular", "quantification", True, ""]
            
            channel_config.append(config)
        
        return pd.DataFrame(
            channel_config,
            columns=["Channel", "Name", "Type", "Purpose", "Quantify", "Wavelength"]
        )
    
    def run_analysis(self, exp_name: str, exp_desc: str, model: str, 
                    diameter: float, bg_method: str, channel_config: pd.DataFrame,
                    use_gpu: bool, batch_size: int, session_data: Dict):
        """Run the complete analysis pipeline."""
        
        if not session_data or 'uploaded_files' not in session_data:
            return "❌ No files uploaded", "", "❌ Analysis failed: No data", pd.DataFrame(), pd.DataFrame()
        
        logger.info(f"Starting analysis: {exp_name}")
        
        try:
            # Prepare configuration
            config = {
                'experiment_name': exp_name,
                'description': exp_desc,
                'output_directory': str(self.results_dir / exp_name.replace(' ', '_')),
                'cellpose_model': model,
                'cell_diameter': diameter if diameter > 0 else None,
                'background_method': bg_method,
                'use_gpu': use_gpu,
                'batch_size': int(batch_size),
                'conditions': []
            }
            
            # Create condition from uploaded files
            condition = {
                'name': 'Analysis',
                'directory': str(self.upload_dir),
                'channels': []
            }
            
            # Process channel configuration
            for _, row in channel_config.iterrows():
                channel = {
                    'name': row['Name'],
                    'type': row['Type'],
                    'purpose': row['Purpose'],
                    'quantify': row['Quantify'],
                    'suffix': row['Channel'],
                    'wavelength': row['Wavelength']
                }
                condition['channels'].append(channel)
            
            config['conditions'] = [condition]
            
            # Progress tracking
            progress_log = []
            
            def progress_callback(message: str, percentage: int):
                progress_log.append(f"[{datetime.now().strftime('%H:%M:%S')}] {message}")
                logger.info(f"Progress: {percentage}% - {message}")
            
            # Run analysis
            pipeline = AnalysisPipeline(config, progress_callback)
            results = pipeline.run_analysis()
            
            # Store results in session
            session_data['analysis_results'] = results
            session_data['config'] = config
            
            # Generate summary statistics
            summary_data = self.generate_summary_statistics(results)
            cell_data = self.extract_cell_data(results)
            
            progress_text = "\n".join(progress_log[-10:])  # Last 10 entries
            status = f"✅ **Analysis Complete!** Processed {results.get('total_cells', 0)} cells"
            
            return "✅ Analysis completed successfully!", progress_text, status, summary_data, cell_data
            
        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            error_msg = f"❌ Analysis failed: {str(e)}"
            return error_msg, str(e), error_msg, pd.DataFrame(), pd.DataFrame()
    
    def generate_summary_statistics(self, results: Dict) -> pd.DataFrame:
        """Generate summary statistics table."""
        summary_data = []
        
        for condition_name, condition_data in results.get('conditions', {}).items():
            summary = condition_data.get('summary', {})
            
            row = {
                'Condition': condition_name,
                'Total Cells': summary.get('total_cells', 0),
                'Images': len(condition_data.get('images', [])),
                'Mean Cell Area': f"{summary.get('mean_cell_area', 0):.1f}",
            }
            
            # Add channel-specific summaries
            for key, value in summary.items():
                if key.endswith('_mean_ctcf'):
                    channel = key.replace('_mean_ctcf', '')
                    row[f'{channel} CTCF'] = f"{value:.1f}"
            
            summary_data.append(row)
        
        return pd.DataFrame(summary_data)
    
    def extract_cell_data(self, results: Dict) -> pd.DataFrame:
        """Extract individual cell data."""
        all_cell_data = []
        
        for condition_data in results.get('conditions', {}).values():
            all_cell_data.extend(condition_data.get('cell_data', []))
        
        if all_cell_data:
            return pd.DataFrame(all_cell_data)
        else:
            return pd.DataFrame()
    
    def generate_visualization(self, plot_type: str, session_data: Dict):
        """Generate visualization plot."""
        if not session_data or 'analysis_results' not in session_data:
            return None
        
        results = session_data['analysis_results']
        
        try:
            viz_engine = VisualizationEngine(self.results_dir)
            
            if plot_type == "Box Plot":
                fig = viz_engine.create_condition_comparison_plots(results, ['box_plots'])[0]
            elif plot_type == "Bar Plot":
                fig = viz_engine.create_condition_comparison_plots(results, ['bar_plots'])[0]
            elif plot_type == "Scatter Plot":
                fig = viz_engine.create_correlation_matrix(results)
            else:  # Heatmap
                fig = viz_engine.create_condition_comparison_plots(results, ['heatmaps'])[0]
            
            return fig
            
        except Exception as e:
            logger.error(f"Visualization failed: {e}")
            return None
    
    def download_csv(self, session_data: Dict):
        """Download CSV results."""
        if not session_data or 'analysis_results' not in session_data:
            return None
        
        try:
            cell_data = self.extract_cell_data(session_data['analysis_results'])
            
            # Save to temporary file
            csv_path = self.temp_dir / f"cell_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            cell_data.to_csv(csv_path, index=False)
            
            return str(csv_path)
            
        except Exception as e:
            logger.error(f"CSV download failed: {e}")
            return None
    
    def download_all_results(self, session_data: Dict):
        """Download all results as ZIP file."""
        if not session_data or 'analysis_results' not in session_data:
            return None
        
        try:
            # Create ZIP file
            zip_path = self.temp_dir / f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"
            
            with zipfile.ZipFile(zip_path, 'w') as zipf:
                # Add CSV data
                cell_data = self.extract_cell_data(session_data['analysis_results'])
                csv_buffer = io.StringIO()
                cell_data.to_csv(csv_buffer, index=False)
                zipf.writestr("cell_data.csv", csv_buffer.getvalue())
                
                # Add summary
                summary_data = self.generate_summary_statistics(session_data['analysis_results'])
                summary_buffer = io.StringIO()
                summary_data.to_csv(summary_buffer, index=False)
                zipf.writestr("summary_statistics.csv", summary_buffer.getvalue())
                
                # Add configuration
                config_str = json.dumps(session_data.get('config', {}), indent=2)
                zipf.writestr("analysis_config.json", config_str)
            
            return str(zip_path)
            
        except Exception as e:
            logger.error(f"ZIP download failed: {e}")
            return None


def create_cellquant_app():
    """Create and configure the CellQuantGUI Gradio application."""
    
    logger.info("🚀 Starting CellQuantGUI Gradio Application")
    
    # Initialize interface
    interface_manager = GradioInterface()
    app = interface_manager.create_interface()
    
    return app


if __name__ == "__main__":
    # Create the application
    app = create_cellquant_app()
    
    # Launch configuration for intranet deployment
    app.launch(
        server_name="0.0.0.0",  # Listen on all interfaces
        server_port=7860,       # Default Gradio port
        share=False,            # Don't create public link
        auth=None,              # Add authentication if needed: auth=[("username", "password")]
        show_error=True,        # Show detailed errors
        favicon_path=None,      # Add custom favicon if desired
        app_kwargs={
            "docs_url": "/docs",  # FastAPI docs endpoint
            "redoc_url": "/redoc" # Alternative docs
        }
    )