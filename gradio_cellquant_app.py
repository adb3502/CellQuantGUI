#!/usr/bin/env python3
"""
CellQuantGUI - Enhanced Gradio Web Interface
Professional UI for quantitative microscopy analysis
"""

import gradio as gr
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import json
import logging
from datetime import datetime
import tempfile
import shutil
import re
from PIL import Image
import matplotlib.pyplot as plt
import io
import base64

# Import backend components
from analysis_pipeline import AnalysisPipeline, AnalysisParameters
from template_manager import TemplateManager
from quality_control import QualityControlManager
from visualization_module import VisualizationEngine
from file_management import FileManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Professional CSS styling
CUSTOM_CSS = """
/* Modern, clean design */
.gradio-container {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif !important;
    max-width: 1400px !important;
    margin: auto !important;
}

/* Header styling */
.main-header {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    padding: 2rem;
    border-radius: 12px;
    margin-bottom: 2rem;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
}

/* Tab styling */
.tab-nav {
    background: #f8f9fa;
    border-radius: 8px;
    padding: 0.5rem;
    margin-bottom: 1rem;
}

.tab-nav button {
    font-weight: 600 !important;
    color: #4a5568 !important;
    border-radius: 6px !important;
    transition: all 0.2s ease !important;
}

.tab-nav button.selected {
    background: white !important;
    color: #667eea !important;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1) !important;
}

/* Card styling */
.analysis-card {
    background: white;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 1.5rem;
    margin-bottom: 1rem;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
}

/* Button styling */
.primary-button {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    color: white !important;
    font-weight: 600 !important;
    border: none !important;
    padding: 0.75rem 1.5rem !important;
    border-radius: 8px !important;
    transition: transform 0.2s ease !important;
}

.primary-button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4) !important;
}

/* Status badges */
.status-badge {
    padding: 0.25rem 0.75rem;
    border-radius: 9999px;
    font-size: 0.875rem;
    font-weight: 600;
    display: inline-block;
}

.status-success {
    background: #d4edda;
    color: #155724;
}

.status-warning {
    background: #fff3cd;
    color: #856404;
}

.status-error {
    background: #f8d7da;
    color: #721c24;
}

/* Progress styling */
.progress-bar {
    background: #e2e8f0 !important;
    border-radius: 9999px !important;
    overflow: hidden !important;
}

.progress-bar > div {
    background: linear-gradient(90deg, #667eea 0%, #764ba2 100%) !important;
    transition: width 0.3s ease !important;
}

/* File upload area */
.file-upload-area {
    border: 2px dashed #cbd5e0 !important;
    border-radius: 12px !important;
    background: #f7fafc !important;
    transition: all 0.2s ease !important;
}

.file-upload-area:hover {
    border-color: #667eea !important;
    background: #eef2ff !important;
}

/* Table styling */
.data-table {
    border-radius: 8px !important;
    overflow: hidden !important;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1) !important;
}

/* Info boxes */
.info-box {
    background: #eef2ff;
    border-left: 4px solid #667eea;
    padding: 1rem;
    border-radius: 4px;
    margin: 1rem 0;
}

/* Results display */
.results-container {
    background: #f8f9fa;
    border-radius: 12px;
    padding: 2rem;
    margin-top: 2rem;
}

/* Channel configuration */
.channel-config-container {
    background: white;
    border-radius: 8px;
    padding: 1rem;
    margin: 1rem 0;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
}
"""

class EnhancedGradioInterface:
    """Professional Gradio interface for CellQuantGUI"""
    
    def __init__(self):
        self.template_manager = TemplateManager()
        self.file_manager = FileManager()
        self.setup_directories()
        self.current_session = {}
        logger.info("Enhanced CellQuantGUI interface initialized")
    
    def setup_directories(self):
        """Setup required directories"""
        self.upload_dir = Path("uploads")
        self.results_dir = Path("results")
        self.temp_dir = Path("temp")
        
        for directory in [self.upload_dir, self.results_dir, self.temp_dir]:
            directory.mkdir(exist_ok=True)
    
    def create_interface(self) -> gr.Blocks:
        """Create the enhanced Gradio interface"""
        
        with gr.Blocks(
            title="CellQuantGUI - Quantitative Microscopy Analysis",
            theme=gr.themes.Soft(
                primary_hue="purple",
                secondary_hue="blue",
                neutral_hue="gray",
                font=[gr.themes.GoogleFont("Inter"), "sans-serif"]
            ),
            css=CUSTOM_CSS
        ) as interface:
            
            # Header
            with gr.Row():
                gr.HTML("""
                <div class="main-header">
                    <h1 style="margin: 0; font-size: 2rem;">🔬 CellQuantGUI</h1>
                    <p style="margin: 0.5rem 0 0 0; opacity: 0.9;">Professional Quantitative Microscopy Analysis Platform</p>
                    <p style="margin: 0.5rem 0 0 0; font-size: 0.875rem; opacity: 0.8;">
                        Powered by Cellpose • CTCF Quantification • Statistical Analysis • Publication-Ready Results
                    </p>
                </div>
                """)
            
            # Session state
            session_state = gr.State({
                'experiment_id': None,
                'uploaded_files': {},
                'channel_config': {},
                'analysis_results': None,
                'current_step': 1
            })
            
            with gr.Tabs() as tabs:
                
                # =================== EXPERIMENT SETUP TAB ===================
                with gr.TabItem("1️⃣ Experiment Setup", id=1):
                    with gr.Row():
                        with gr.Column(scale=3):
                            gr.Markdown("### 📋 Experiment Configuration")
                            
                            with gr.Group():
                                experiment_name = gr.Textbox(
                                    label="Experiment Name",
                                    placeholder="e.g., DNA_Damage_TimeCourse_2024",
                                    info="Use descriptive names without spaces"
                                )
                                
                                experiment_description = gr.Textbox(
                                    label="Description",
                                    lines=3,
                                    placeholder="Describe your experimental setup, conditions, and goals..."
                                )
                                
                                # Template selection with preview
                                with gr.Row():
                                    template_dropdown = gr.Dropdown(
                                        label="Analysis Template",
                                        choices=self.get_template_choices(),
                                        value="custom",
                                        info="Select a pre-configured workflow or create custom"
                                    )
                                    
                                    template_info = gr.Button("ℹ️ Template Info", size="sm")
                                
                                template_preview = gr.Markdown(visible=False)
                        
                        with gr.Column(scale=2):
                            gr.Markdown("### ⚙️ Analysis Parameters")
                            
                            with gr.Group():
                                with gr.Row():
                                    cellpose_model = gr.Dropdown(
                                        label="Segmentation Model",
                                        choices=["cyto2", "cyto3", "cyto", "nuclei", "custom"],
                                        value="cyto2",
                                        info="Choose based on your cell type"
                                    )
                                    
                                    use_gpu = gr.Checkbox(
                                        label="GPU Acceleration",
                                        value=True,
                                        info="~10x faster with NVIDIA GPU"
                                    )
                                
                                cell_diameter = gr.Slider(
                                    label="Expected Cell Diameter (pixels)",
                                    minimum=0,
                                    maximum=200,
                                    value=30,
                                    step=5,
                                    info="Set to 0 for automatic detection"
                                )
                                
                                with gr.Row():
                                    background_method = gr.Radio(
                                        label="Background Correction",
                                        choices=["mode", "median", "mean"],
                                        value="mode",
                                        info="Mode is best for most cases"
                                    )
                                    
                                    flow_threshold = gr.Slider(
                                        label="Flow Threshold",
                                        minimum=0.0,
                                        maximum=1.0,
                                        value=0.4,
                                        step=0.1,
                                        info="Lower = more cells detected"
                                    )
                            
                            # Quick setup guide
                            gr.HTML("""
                            <div class="info-box">
                                <strong>Quick Setup Guide:</strong><br>
                                1. Name your experiment<br>
                                2. Select or customize analysis template<br>
                                3. Adjust segmentation parameters<br>
                                4. Proceed to file upload →
                            </div>
                            """)
                
                # =================== FILE UPLOAD TAB ===================
                with gr.TabItem("2️⃣ Data Upload & Channels", id=2):
                    with gr.Row():
                        with gr.Column():
                            gr.Markdown("### 📁 Upload Microscopy Images")
                            
                            # Condition management
                            with gr.Group():
                                gr.Markdown("#### Experimental Conditions")
                                
                                condition_name = gr.Textbox(
                                    label="Condition Name",
                                    placeholder="e.g., Control, Treatment_1h, Treatment_24h"
                                )
                                
                                with gr.Row():
                                    add_condition_btn = gr.Button("➕ Add Condition", size="sm")
                                    clear_conditions_btn = gr.Button("🗑️ Clear All", size="sm")
                                
                                conditions_display = gr.Dataframe(
                                    headers=["Condition", "Files", "Status"],
                                    datatype=["str", "number", "str"],
                                    label="Configured Conditions",
                                    interactive=False
                                )
                            
                            # File upload
                            with gr.Group():
                                file_upload = gr.File(
                                    label="Upload Images for Current Condition",
                                    file_count="multiple",
                                    file_types=[".tiff", ".tif", ".png", ".jpg", ".jpeg"],
                                    elem_classes=["file-upload-area"]
                                )
                                
                                upload_status = gr.Markdown("Ready to upload files...")
                        
                        with gr.Column():
                            gr.Markdown("### 🎨 Channel Configuration")
                            
                            # Channel setup interface
                            with gr.Group():
                                channel_setup_mode = gr.Radio(
                                    label="Channel Setup Mode",
                                    choices=["Auto-detect from filenames", "Manual configuration"],
                                    value="Auto-detect from filenames"
                                )
                                
                                # Auto-detected channels display
                                detected_channels = gr.HTML(
                                    """<div class='channel-config-container'>
                                    <p>Upload files to detect channels automatically</p>
                                    </div>"""
                                )
                                
                                # Manual channel configuration
                                with gr.Group(visible=False) as manual_config_group:
                                    gr.Markdown("#### Add Channel")
                                    with gr.Row():
                                        channel_suffix = gr.Textbox(label="Channel ID", placeholder="C0")
                                        channel_name = gr.Textbox(label="Name", placeholder="DAPI")
                                        channel_type = gr.Dropdown(
                                            label="Type",
                                            choices=["nuclear", "cellular", "membrane"],
                                            value="nuclear"
                                        )
                                    
                                    with gr.Row():
                                        channel_purpose = gr.Dropdown(
                                            label="Purpose",
                                            choices=["segmentation", "quantification", "reference"],
                                            value="segmentation"
                                        )
                                        channel_wavelength = gr.Textbox(
                                            label="Wavelength",
                                            placeholder="405nm"
                                        )
                                        quantify_channel = gr.Checkbox(
                                            label="Quantify",
                                            value=False
                                        )
                                    
                                    add_channel_btn = gr.Button("Add Channel", size="sm")
                                
                                # Channel configuration table
                                channel_config_table = gr.Dataframe(
                                    headers=["Channel", "Name", "Type", "Purpose", "Quantify", "Wavelength"],
                                    datatype=["str", "str", "str", "str", "bool", "str"],
                                    label="Channel Configuration",
                                    interactive=True,
                                    value=pd.DataFrame({
                                        "Channel": ["C0"],
                                        "Name": ["DAPI"],
                                        "Type": ["nuclear"],
                                        "Purpose": ["segmentation"],
                                        "Quantify": [False],
                                        "Wavelength": ["405nm"]
                                    })
                                )
                            
                            # Validation status
                            validation_status = gr.HTML(
                                """<div class='info-box'>
                                ⚠️ Configure at least one segmentation channel and one quantification channel
                                </div>"""
                            )
                
                # =================== ANALYSIS TAB ===================
                with gr.TabItem("3️⃣ Run Analysis", id=3):
                    with gr.Row():
                        with gr.Column(scale=3):
                            gr.Markdown("### 🚀 Analysis Pipeline")
                            
                            # Pre-flight check
                            with gr.Group():
                                gr.Markdown("#### Pre-Analysis Checklist")
                                preflight_status = gr.HTML(self.generate_preflight_check({}))
                                
                                refresh_preflight_btn = gr.Button("🔄 Refresh Status", size="sm")
                            
                            # Analysis controls
                            with gr.Group():
                                gr.Markdown("#### Analysis Options")
                                
                                with gr.Row():
                                    batch_size = gr.Slider(
                                        label="Batch Size",
                                        minimum=1,
                                        maximum=20,
                                        value=5,
                                        step=1,
                                        info="Process multiple images simultaneously"
                                    )
                                    
                                    save_masks = gr.Checkbox(
                                        label="Save Segmentation Masks",
                                        value=True
                                    )
                                    
                                    generate_report = gr.Checkbox(
                                        label="Generate Quality Report",
                                        value=True
                                    )
                                
                                start_analysis_btn = gr.Button(
                                    "🚀 Start Analysis",
                                    variant="primary",
                                    size="lg",
                                    elem_classes=["primary-button"]
                                )
                        
                        with gr.Column(scale=2):
                            gr.Markdown("### 📊 Progress Monitor")
                            
                            # Progress display
                            with gr.Group():
                                analysis_status = gr.Markdown("Ready to start analysis")
                                progress_bar = gr.Progress()
                                
                                # Live stats
                                live_stats = gr.HTML(
                                    """<div style='text-align: center; padding: 1rem;'>
                                    <h4>Analysis Statistics</h4>
                                    <p>Images: 0/0 | Cells: 0 | Time: 00:00</p>
                                    </div>"""
                                )
                                
                                # Log display
                                analysis_log = gr.Textbox(
                                    label="Analysis Log",
                                    lines=10,
                                    max_lines=20,
                                    interactive=False
                                )
                
                # =================== RESULTS TAB ===================
                with gr.TabItem("4️⃣ Results & Export", id=4):
                    with gr.Row():
                        with gr.Column():
                            gr.Markdown("### 📈 Analysis Results")
                            
                            results_summary = gr.HTML(
                                """<div class='results-container'>
                                <h4>No results yet</h4>
                                <p>Complete analysis to view results</p>
                                </div>"""
                            )
                            
                            # Results tabs
                            with gr.Tabs():
                                with gr.TabItem("Summary Statistics"):
                                    summary_table = gr.Dataframe(
                                        label="Condition Summary",
                                        interactive=False
                                    )
                                
                                with gr.TabItem("Visualizations"):
                                    with gr.Row():
                                        plot_type = gr.Dropdown(
                                            label="Plot Type",
                                            choices=[
                                                "Condition Comparison",
                                                "Cell Distribution",
                                                "Correlation Matrix",
                                                "Statistical Report"
                                            ],
                                            value="Condition Comparison"
                                        )
                                        generate_plot_btn = gr.Button("Generate Plot")
                                    
                                    plot_display = gr.Image(label="Visualization")
                                
                                with gr.TabItem("Cell Data"):
                                    cell_data_table = gr.Dataframe(
                                        label="Individual Cell Measurements",
                                        interactive=False,
                                        max_rows=20
                                    )
                                
                                with gr.TabItem("Quality Report"):
                                    quality_report_display = gr.HTML()
                        
                        with gr.Column():
                            gr.Markdown("### 💾 Export Options")
                            
                            with gr.Group():
                                gr.Markdown("#### Download Results")
                                
                                export_format = gr.CheckboxGroup(
                                    label="Select Export Formats",
                                    choices=[
                                        "CSV Data",
                                        "Analysis Report (PDF)",
                                        "Plots (PNG/SVG)",
                                        "Segmentation Masks",
                                        "Complete Archive (ZIP)"
                                    ],
                                    value=["CSV Data", "Plots (PNG/SVG)"]
                                )
                                
                                export_btn = gr.Button(
                                    "📥 Export Selected",
                                    variant="primary",
                                    elem_classes=["primary-button"]
                                )
                                
                                download_links = gr.HTML()
                            
                            # Quick stats
                            with gr.Group():
                                gr.Markdown("#### Quick Statistics")
                                quick_stats_display = gr.HTML(
                                    """<div class='info-box'>
                                    Complete analysis to view statistics
                                    </div>"""
                                )
            
            # =================== EVENT HANDLERS ===================
            
            # Template selection
            def show_template_info(template_name):
                if template_name and template_name != "custom":
                    template = self.template_manager.get_template(f"builtin_{template_name.lower().replace(' ', '_')}")
                    if template:
                        info = f"""
                        ### {template.name}
                        
                        **Description:** {template.description}
                        
                        **Channels:**
                        """
                        for ch in template.channel_template:
                            info += f"\n- {ch['name']} ({ch['type']}): {ch.get('description', 'N/A')}"
                        
                        return gr.update(visible=True, value=info)
                return gr.update(visible=False)
            
            template_dropdown.change(
                show_template_info,
                inputs=[template_dropdown],
                outputs=[template_preview]
            )
            
            template_info.click(
                lambda t: gr.update(visible=not template_preview.visible),
                inputs=[template_dropdown],
                outputs=[template_preview]
            )
            
            # Condition management
            def add_condition(name, current_files, session):
                if not name:
                    return gr.update(), session, "Please enter a condition name"
                
                if 'conditions' not in session:
                    session['conditions'] = {}
                
                if current_files:
                    session['conditions'][name] = {
                        'files': current_files,
                        'status': 'Ready'
                    }
                    
                    # Update display
                    df_data = []
                    for cond_name, cond_data in session['conditions'].items():
                        df_data.append([cond_name, len(cond_data['files']), cond_data['status']])
                    
                    df = pd.DataFrame(df_data, columns=["Condition", "Files", "Status"])
                    
                    return df, session, f"✅ Added condition: {name}"
                
                return gr.update(), session, "⚠️ Please upload files first"
            
            add_condition_btn.click(
                add_condition,
                inputs=[condition_name, file_upload, session_state],
                outputs=[conditions_display, session_state, upload_status]
            )
            
            # Channel detection
            def detect_channels(files, mode):
                if not files:
                    return gr.update(), gr.update()
                
                if mode == "Auto-detect from filenames":
                    channels = self.auto_detect_channels_from_files(files)
                    
                    # Create nice display
                    html = "<div class='channel-config-container'>"
                    html += "<h4>Detected Channels:</h4>"
                    
                    for ch in channels:
                        status_color = "green" if ch['quantify'] else "gray"
                        html += f"""
                        <div style='margin: 0.5rem 0; padding: 0.5rem; background: #f8f9fa; border-radius: 4px;'>
                            <strong style='color: {status_color};'>● {ch['suffix']}</strong> - {ch['name']} 
                            ({ch['type']}) - {ch['purpose']}
                            {' - 📊 Quantify' if ch['quantify'] else ''}
                        </div>
                        """
                    
                    html += "</div>"
                    
                    # Update table
                    df = pd.DataFrame(channels)
                    
                    return html, df
                
                return gr.update(), gr.update()
            
            file_upload.change(
                detect_channels,
                inputs=[file_upload, channel_setup_mode],
                outputs=[detected_channels, channel_config_table]
            )
            
            # Channel setup mode toggle
            channel_setup_mode.change(
                lambda mode: gr.update(visible=mode == "Manual configuration"),
                inputs=[channel_setup_mode],
                outputs=[manual_config_group]
            )
            
            # Add channel manually
            def add_channel_manual(suffix, name, ch_type, purpose, wavelength, quantify, current_df):
                if not suffix or not name:
                    return current_df
                
                new_row = pd.DataFrame({
                    "Channel": [suffix],
                    "Name": [name],
                    "Type": [ch_type],
                    "Purpose": [purpose],
                    "Quantify": [quantify],
                    "Wavelength": [wavelength]
                })
                
                return pd.concat([current_df, new_row], ignore_index=True)
            
            add_channel_btn.click(
                add_channel_manual,
                inputs=[
                    channel_suffix, channel_name, channel_type,
                    channel_purpose, channel_wavelength, quantify_channel,
                    channel_config_table
                ],
                outputs=[channel_config_table]
            )
            
            # Preflight check
            def update_preflight(session):
                return self.generate_preflight_check(session)
            
            refresh_preflight_btn.click(
                update_preflight,
                inputs=[session_state],
                outputs=[preflight_status]
            )
            
            # Start analysis
            def run_analysis(
                exp_name, exp_desc, template, model, diameter,
                bg_method, flow_thresh, gpu, batch_size, save_masks,
                gen_report, channels_df, session
            ):
                if not session.get('conditions'):
                    return "❌ No conditions configured", "", {}, pd.DataFrame(), session
                
                try:
                    # Prepare configuration
                    config = self.prepare_analysis_config(
                        exp_name, exp_desc, template, model, diameter,
                        bg_method, flow_thresh, gpu, batch_size,
                        channels_df, session
                    )
                    
                    # Create progress callback
                    progress_log = []
                    
                    def progress_callback(message, percentage):
                        progress_log.append(f"[{datetime.now().strftime('%H:%M:%S')}] {message}")
                        # Update UI would happen here in real implementation
                    
                    # Run analysis
                    pipeline = AnalysisPipeline(config, progress_callback)
                    results = pipeline.run_analysis()
                    
                    # Generate visualizations
                    if results:
                        viz_engine = VisualizationEngine(Path(config['output_directory']))
                        plot_files = viz_engine.create_condition_comparison_plots(results)
                        
                        # Generate quality report
                        if gen_report:
                            qc_manager = QualityControlManager()
                            quality_report = qc_manager.generate_quality_report(results, config)
                    
                    # Store results
                    session['analysis_results'] = results
                    session['plot_files'] = plot_files if 'plot_files' in locals() else []
                    session['quality_report'] = quality_report if 'quality_report' in locals() else None
                    
                    # Update displays
                    status = "✅ Analysis completed successfully!"
                    log_text = "\n".join(progress_log[-20:])
                    summary_html = self.generate_results_summary(results)
                    summary_df = self.generate_summary_table(results)
                    
                    return status, log_text, summary_html, summary_df, session
                    
                except Exception as e:
                    logger.error(f"Analysis failed: {e}")
                    return f"❌ Analysis failed: {str(e)}", str(e), {}, pd.DataFrame(), session
            
            start_analysis_btn.click(
                run_analysis,
                inputs=[
                    experiment_name, experiment_description, template_dropdown,
                    cellpose_model, cell_diameter, background_method,
                    flow_threshold, use_gpu, batch_size, save_masks,
                    generate_report, channel_config_table, session_state
                ],
                outputs=[
                    analysis_status, analysis_log, results_summary,
                    summary_table, session_state
                ]
            )
            
            # Plot generation
            def generate_plot(plot_type, session):
                if not session.get('analysis_results'):
                    return None
                
                try:
                    results = session['analysis_results']
                    viz_engine = VisualizationEngine(self.results_dir)
                    
                    if plot_type == "Condition Comparison":
                        # Generate condition comparison plot
                        plots = viz_engine.create_condition_comparison_plots(results)
                        if plots:
                            return plots[0]
                    elif plot_type == "Correlation Matrix":
                        plot = viz_engine.create_correlation_matrix(results)
                        return plot
                    # Add other plot types...
                    
                except Exception as e:
                    logger.error(f"Plot generation failed: {e}")
                    return None
            
            generate_plot_btn.click(
                generate_plot,
                inputs=[plot_type, session_state],
                outputs=[plot_display]
            )
            
            # Export functionality
            def export_results(formats, session):
                if not session.get('analysis_results'):
                    return "<p>No results to export</p>"
                
                links = []
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                
                try:
                    if "CSV Data" in formats:
                        # Export CSV
                        csv_path = self.export_csv_data(session['analysis_results'], timestamp)
                        links.append(f'<a href="{csv_path}" download>📄 Download CSV Data</a>')
                    
                    if "Plots (PNG/SVG)" in formats:
                        # Export plots
                        if session.get('plot_files'):
                            for plot in session['plot_files']:
                                links.append(f'<a href="{plot}" download>📊 Download {plot.name}</a>')
                    
                    if "Complete Archive (ZIP)" in formats:
                        # Create ZIP archive
                        zip_path = self.create_results_archive(session, timestamp)
                        links.append(f'<a href="{zip_path}" download>📦 Download Complete Archive</a>')
                    
                    html = "<div style='line-height: 2;'>"
                    html += "<h4>Download Links:</h4>"
                    html += "<br>".join(links)
                    html += "</div>"
                    
                    return html
                    
                except Exception as e:
                    logger.error(f"Export failed: {e}")
                    return f"<p>Export failed: {str(e)}</p>"
            
            export_btn.click(
                export_results,
                inputs=[export_format, session_state],
                outputs=[download_links]
            )
        
        return interface
    
    def get_template_choices(self) -> List[str]:
        """Get available template choices"""
        templates = self.template_manager.list_templates()
        choices = ["custom"] + [t.name for t in templates]
        return choices
    
    def auto_detect_channels_from_files(self, files) -> List[Dict]:
        """Auto-detect channels from uploaded files"""
        channel_pattern = re.compile(r'[_\-]C(\d+)\.tiff?$', re.IGNORECASE)
        channels = {}
        
        for file in files:
            if file is None:
                continue
            
            filename = Path(file.name).name
            match = channel_pattern.search(filename)
            
            if match:
                channel_num = match.group(1)
                suffix = f"C{channel_num}"
                
                if suffix not in channels:
                    # Smart defaults based on channel number
                    if channel_num == "0":
                        channels[suffix] = {
                            "suffix": suffix,
                            "name": "DAPI",
                            "type": "nuclear",
                            "purpose": "segmentation",
                            "quantify": False,
                            "wavelength": "405nm"
                        }
                    else:
                        channels[suffix] = {
                            "suffix": suffix,
                            "name": f"Target_{channel_num}",
                            "type": "cellular",
                            "purpose": "quantification",
                            "quantify": True,
                            "wavelength": self.guess_wavelength(channel_num)
                        }
        
        return list(channels.values())
    
    def guess_wavelength(self, channel_num: str) -> str:
        """Guess wavelength based on channel number"""
        wavelengths = {
            "1": "488nm",
            "2": "568nm",
            "3": "647nm",
            "4": "750nm"
        }
        return wavelengths.get(channel_num, "")
    
    def generate_preflight_check(self, session: Dict) -> str:
        """Generate pre-flight check HTML"""
        checks = []
        
        # Check experiment name
        exp_name = session.get('experiment_name', '')
        checks.append({
            'item': 'Experiment Name',
            'status': '✅' if exp_name else '❌',
            'message': exp_name if exp_name else 'Not set'
        })
        
        # Check conditions
        conditions = session.get('conditions', {})
        checks.append({
            'item': 'Conditions',
            'status': '✅' if conditions else '❌',
            'message': f"{len(conditions)} conditions configured" if conditions else 'No conditions added'
        })
        
        # Check channels
        channels = session.get('channels', [])
        has_seg = any(ch.get('purpose') == 'segmentation' for ch in channels)
        has_quant = any(ch.get('quantify', False) for ch in channels)
        
        checks.append({
            'item': 'Segmentation Channel',
            'status': '✅' if has_seg else '❌',
            'message': 'Configured' if has_seg else 'Not configured'
        })
        
        checks.append({
            'item': 'Quantification Channels',
            'status': '✅' if has_quant else '❌',
            'message': f"{sum(1 for ch in channels if ch.get('quantify', False))} channels" if has_quant else 'None configured'
        })
        
        # Generate HTML
        html = "<div style='padding: 1rem;'>"
        for check in checks:
            html += f"""
            <div style='margin: 0.5rem 0;'>
                <span style='font-size: 1.2em;'>{check['status']}</span>
                <strong>{check['item']}:</strong> {check['message']}
            </div>
            """
        html += "</div>"
        
        # Overall status
        all_good = all(check['status'] == '✅' for check in checks)
        if all_good:
            html += "<div class='status-badge status-success'>✅ Ready for Analysis</div>"
        else:
            html += "<div class='status-badge status-warning'>⚠️ Configuration Incomplete</div>"
        
        return html
    
    def prepare_analysis_config(self, exp_name, exp_desc, template, model,
                              diameter, bg_method, flow_thresh, gpu,
                              batch_size, channels_df, session):
        """Prepare analysis configuration"""
        config = {
            'experiment_name': exp_name,
            'description': exp_desc,
            'output_directory': str(self.results_dir / exp_name.replace(' ', '_')),
            'cellpose_model': model,
            'cell_diameter': diameter if diameter > 0 else None,
            'background_method': bg_method,
            'flow_threshold': flow_thresh,
            'use_gpu': gpu,
            'batch_size': int(batch_size),
            'conditions': []
        }
        
        # Add conditions
        for cond_name, cond_data in session.get('conditions', {}).items():
            condition = {
                'name': cond_name,
                'directory': str(self.upload_dir),  # In real implementation, organize by condition
                'channels': []
            }
            
            # Add channels from dataframe
            for _, row in channels_df.iterrows():
                channel = {
                    'name': row['Name'],
                    'type': row['Type'],
                    'purpose': row['Purpose'],
                    'quantify': row['Quantify'],
                    'wavelength': row['Wavelength'],
                    'suffix': row['Channel']
                }
                condition['channels'].append(channel)
            
            config['conditions'].append(condition)
        
        return config
    
    def generate_results_summary(self, results: Dict) -> str:
        """Generate HTML summary of results"""
        if not results:
            return "<div class='results-container'><p>No results available</p></div>"
        
        exp_info = results.get('experiment_info', {})
        
        html = "<div class='results-container'>"
        html += "<h3>Analysis Complete! 🎉</h3>"
        html += f"<p><strong>Total Conditions:</strong> {exp_info.get('total_conditions', 0)}</p>"
        html += f"<p><strong>Images Processed:</strong> {exp_info.get('total_images_processed', 0)}</p>"
        html += f"<p><strong>Total Cells:</strong> {results.get('summary_statistics', {}).get('total_cells', 0)}</p>"
        html += f"<p><strong>Processing Time:</strong> {exp_info.get('duration_seconds', 0):.1f} seconds</p>"
        
        # Add condition details
        if results.get('conditions'):
            html += "<h4>Conditions Summary:</h4>"
            for cond_name, cond_data in results['conditions'].items():
                n_cells = len(cond_data.get('cell_data', []))
                html += f"<p>• {cond_name}: {n_cells} cells</p>"
        
        html += "</div>"
        return html
    
    def generate_summary_table(self, results: Dict) -> pd.DataFrame:
        """Generate summary statistics table"""
        if not results or not results.get('conditions'):
            return pd.DataFrame()
        
        summary_data = []
        
        for cond_name, cond_data in results['conditions'].items():
            if cond_data.get('summary'):
                row = {
                    'Condition': cond_name,
                    'Cells': cond_data['summary'].get('total_cells', 0),
                    'Mean Area': f"{cond_data['summary'].get('mean_cell_area', 0):.1f}",
                }
                
                # Add CTCF values
                for key, value in cond_data['summary'].items():
                    if key.endswith('_mean_ctcf'):
                        channel = key.replace('_mean_ctcf', '')
                        row[f'{channel} CTCF'] = f"{value:.1f}"
                
                summary_data.append(row)
        
        return pd.DataFrame(summary_data)
    
    def export_csv_data(self, results: Dict, timestamp: str) -> Path:
        """Export cell data to CSV"""
        all_cell_data = []
        
        for cond_name, cond_data in results.get('conditions', {}).items():
            for cell in cond_data.get('cell_data', []):
                cell['condition'] = cond_name
                all_cell_data.append(cell)
        
        df = pd.DataFrame(all_cell_data)
        csv_path = self.results_dir / f"cell_data_{timestamp}.csv"
        df.to_csv(csv_path, index=False)
        
        return csv_path
    
    def create_results_archive(self, session: Dict, timestamp: str) -> Path:
        """Create ZIP archive of all results"""
        import zipfile
        
        zip_path = self.results_dir / f"results_{timestamp}.zip"
        
        with zipfile.ZipFile(zip_path, 'w') as zipf:
            # Add CSV data
            if session.get('analysis_results'):
                csv_path = self.export_csv_data(session['analysis_results'], timestamp)
                zipf.write(csv_path, csv_path.name)
            
            # Add plots
            if session.get('plot_files'):
                for plot_file in session['plot_files']:
                    if plot_file.exists():
                        zipf.write(plot_file, f"plots/{plot_file.name}")
            
            # Add config
            config_data = json.dumps(session.get('config', {}), indent=2)
            zipf.writestr('analysis_config.json', config_data)
        
        return zip_path


def create_enhanced_app():
    """Create the enhanced CellQuantGUI application"""
    logger.info("Starting Enhanced CellQuantGUI")
    
    interface_manager = EnhancedGradioInterface()
    app = interface_manager.create_interface()
    
    return app


if __name__ == "__main__":
    # Create and launch the application
    app = create_enhanced_app()
    
    # Launch with proper configuration
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        auth=None,  # Add auth if needed: [("username", "password")]
        show_error=True,
        quiet=False,
        show_api=False
    )