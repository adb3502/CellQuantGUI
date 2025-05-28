# CellQuantGUI - Comprehensive Microscopy Analysis Tool
# Main application structure and core classes

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
import numpy as np
from pathlib import Path
import json
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('cellquant.log'),
        logging.StreamHandler()
    ]
)

@dataclass
class ChannelInfo:
    """Represents information about a microscopy channel"""
    name: str
    type: str  # 'nuclear', 'cellular', 'quantification', 'background'
    wavelength: Optional[str] = None
    purpose: Optional[str] = None  # 'CTCF', 'segmentation', 'reference'
    
@dataclass
class Condition:
    """Represents an experimental condition"""
    name: str
    directory: Path
    description: str = ""
    replicate_count: int = 0
    channels: List[ChannelInfo] = None
    
    def __post_init__(self):
        if self.channels is None:
            self.channels = []

@dataclass
class ExperimentConfig:
    """Complete experiment configuration"""
    name: str
    output_directory: Path
    conditions: List[Condition]
    analysis_parameters: Dict
    created_date: datetime
    
    def save_config(self, filepath: Path):
        """Save configuration to JSON file"""
        config_dict = asdict(self)
        config_dict['created_date'] = self.created_date.isoformat()
        config_dict['output_directory'] = str(self.output_directory)
        
        for condition in config_dict['conditions']:
            condition['directory'] = str(condition['directory'])
            
        with open(filepath, 'w') as f:
            json.dump(config_dict, f, indent=2)
    
    @classmethod
    def load_config(cls, filepath: Path):
        """Load configuration from JSON file"""
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        
        config_dict['created_date'] = datetime.fromisoformat(config_dict['created_date'])
        config_dict['output_directory'] = Path(config_dict['output_directory'])
        
        conditions = []
        for cond_dict in config_dict['conditions']:
            cond_dict['directory'] = Path(cond_dict['directory'])
            channels = [ChannelInfo(**ch) for ch in cond_dict['channels']]
            cond_dict['channels'] = channels
            conditions.append(Condition(**cond_dict))
        
        config_dict['conditions'] = conditions
        return cls(**config_dict)

class ImageProcessor:
    """Handles image loading and basic processing operations"""
    
    def __init__(self):
        self.supported_formats = ['.tif', '.tiff', '.png', '.jpg', '.jpeg']
        self.logger = logging.getLogger(__name__)
    
    def discover_images(self, directory: Path) -> List[Path]:
        """Discover all supported image files in directory"""
        images = []
        for ext in self.supported_formats:
            images.extend(directory.glob(f'*{ext}'))
            images.extend(directory.glob(f'*{ext.upper()}'))
        
        self.logger.info(f"Found {len(images)} images in {directory}")
        return sorted(images)
    
    def group_multichannel_images(self, image_paths: List[Path]) -> Dict[str, List[Path]]:
        """Group single-channel images by base name for multichannel analysis"""
        # This assumes naming convention like: image001_c1.tif, image001_c2.tif
        # Can be customized based on lab's naming conventions
        groups = {}
        
        for path in image_paths:
            # Extract base name (remove channel identifier)
            base_name = self._extract_base_name(path.stem)
            if base_name not in groups:
                groups[base_name] = []
            groups[base_name].append(path)
        
        return groups
    
    def _extract_base_name(self, filename: str) -> str:
        """Extract base image name, removing channel identifiers"""
        # Common patterns: _c1, _C1, _ch1, _CH1, etc.
        import re
        patterns = [r'_[cC]\d+$', r'_[cC][hH]\d+$', r'_channel\d+$']
        
        for pattern in patterns:
            if re.search(pattern, filename):
                return re.sub(pattern, '', filename)
        
        return filename

class SegmentationEngine:
    """Cellpose integration for cell segmentation"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._check_cellpose_installation()
    
    def _check_cellpose_installation(self):
        """Verify Cellpose is available"""
        try:
            from cellpose import models
            self.cellpose_available = True
            self.logger.info("Cellpose successfully imported")
        except ImportError:
            self.cellpose_available = False
            self.logger.warning("Cellpose not available. Install with: pip install cellpose")
    
    def segment_cells(self, image: np.ndarray, model_type: str = 'cyto2', 
                     diameter: Optional[float] = None) -> Tuple[np.ndarray, List]:
        """Perform cell segmentation using Cellpose"""
        if not self.cellpose_available:
            raise ImportError("Cellpose not installed")
        
        from cellpose import models
        
        model = models.Cellpose(gpu=False, model_type=model_type)
        
        masks, flows, styles, diams = model.eval(
            image, 
            diameter=diameter,
            channels=[0, 0]  # grayscale
        )
        
        # Extract cell ROIs
        rois = self._masks_to_rois(masks)
        
        self.logger.info(f"Segmented {len(rois)} cells")
        return masks, rois
    
    def _masks_to_rois(self, masks: np.ndarray) -> List[Dict]:
        """Convert segmentation masks to ROI coordinates"""
        from skimage import measure
        
        rois = []
        for region in measure.regionprops(masks):
            roi = {
                'label': region.label,
                'area': region.area,
                'centroid': region.centroid,
                'bbox': region.bbox,
                'coords': region.coords
            }
            rois.append(roi)
        
        return rois

class QuantificationEngine:
    """Handles CTCF and other quantitative measurements"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def calculate_ctcf(self, image: np.ndarray, roi_coords: np.ndarray, 
                      background_value: float) -> Dict[str, float]:
        """Calculate Corrected Total Cell Fluorescence"""
        
        # Extract pixel values within ROI
        roi_pixels = image[roi_coords[:, 0], roi_coords[:, 1]]
        
        integrated_density = np.sum(roi_pixels)
        area = len(roi_pixels)
        mean_intensity = np.mean(roi_pixels)
        
        ctcf = integrated_density - (area * background_value)
        
        return {
            'integrated_density': integrated_density,
            'area': area,
            'mean_intensity': mean_intensity,
            'background_value': background_value,
            'ctcf': ctcf
        }
    
    def estimate_background(self, image: np.ndarray, masks: np.ndarray, 
                          method: str = 'mode') -> float:
        """Estimate background fluorescence"""
        
        # Create background mask (areas not covered by cells)
        background_mask = masks == 0
        background_pixels = image[background_mask]
        
        if method == 'mode':
            # Use histogram mode as background estimate
            hist, bins = np.histogram(background_pixels, bins=50)
            mode_idx = np.argmax(hist)
            background = (bins[mode_idx] + bins[mode_idx + 1]) / 2
        elif method == 'median':
            background = np.median(background_pixels)
        elif method == 'mean':
            background = np.mean(background_pixels)
        else:
            raise ValueError(f"Unknown background estimation method: {method}")
        
        self.logger.info(f"Estimated background: {background:.2f} (method: {method})")
        return background

class MainApplication:
    """Main GUI application"""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("CellQuantGUI - Quantitative Microscopy Analysis")
        self.root.geometry("1200x800")
        
        # Initialize core components
        self.image_processor = ImageProcessor()
        self.segmentation_engine = SegmentationEngine()
        self.quantification_engine = QuantificationEngine()
        
        # Application state
        self.current_experiment = None
        self.analysis_results = {}
        
        self.setup_gui()
        self.logger = logging.getLogger(__name__)
    
    def setup_gui(self):
        """Initialize the GUI components"""
        
        # Create main menu
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="New Experiment", command=self.new_experiment)
        file_menu.add_command(label="Load Experiment", command=self.load_experiment)
        file_menu.add_command(label="Save Experiment", command=self.save_experiment)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)
        
        # Create main frame with notebook for tabs
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.notebook = ttk.Notebook(main_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # Setup tabs
        self.setup_experiment_tab()
        self.setup_analysis_tab()
        self.setup_results_tab()
    
    def setup_experiment_tab(self):
        """Setup experiment configuration tab"""
        exp_frame = ttk.Frame(self.notebook)
        self.notebook.add(exp_frame, text="Experiment Setup")
        
        # Experiment info section
        info_frame = ttk.LabelFrame(exp_frame, text="Experiment Information")
        info_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(info_frame, text="Name:").grid(row=0, column=0, sticky="w", padx=5, pady=2)
        self.exp_name_var = tk.StringVar()
        ttk.Entry(info_frame, textvariable=self.exp_name_var, width=40).grid(row=0, column=1, padx=5, pady=2)
        
        ttk.Label(info_frame, text="Output Directory:").grid(row=1, column=0, sticky="w", padx=5, pady=2)
        self.output_dir_var = tk.StringVar()
        ttk.Entry(info_frame, textvariable=self.output_dir_var, width=40).grid(row=1, column=1, padx=5, pady=2)
        ttk.Button(info_frame, text="Browse", command=self.browse_output_dir).grid(row=1, column=2, padx=5, pady=2)
        
        # Conditions section
        cond_frame = ttk.LabelFrame(exp_frame, text="Experimental Conditions")
        cond_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Condition list with scrollbar
        list_frame = ttk.Frame(cond_frame)
        list_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.conditions_listbox = tk.Listbox(list_frame, height=10)
        scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self.conditions_listbox.yview)
        self.conditions_listbox.configure(yscrollcommand=scrollbar.set)
        
        self.conditions_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Condition management buttons
        btn_frame = ttk.Frame(cond_frame)
        btn_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(btn_frame, text="Add Condition", command=self.add_condition).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text="Edit Condition", command=self.edit_condition).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text="Remove Condition", command=self.remove_condition).pack(side=tk.LEFT, padx=2)
    
    def setup_analysis_tab(self):
        """Setup analysis configuration and execution tab"""
        analysis_frame = ttk.Frame(self.notebook)
        self.notebook.add(analysis_frame, text="Analysis")
        
        # Analysis parameters
        params_frame = ttk.LabelFrame(analysis_frame, text="Analysis Parameters")
        params_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Cellpose parameters
        cellpose_frame = ttk.Frame(params_frame)
        cellpose_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(cellpose_frame, text="Cellpose Model:").grid(row=0, column=0, sticky="w", padx=5)
        self.cellpose_model_var = tk.StringVar(value="cyto2")
        model_combo = ttk.Combobox(cellpose_frame, textvariable=self.cellpose_model_var, 
                                  values=["cyto", "cyto2", "nuclei"], state="readonly")
        model_combo.grid(row=0, column=1, padx=5)
        
        ttk.Label(cellpose_frame, text="Cell Diameter (pixels):").grid(row=0, column=2, sticky="w", padx=5)
        self.cell_diameter_var = tk.StringVar(value="30")
        ttk.Entry(cellpose_frame, textvariable=self.cell_diameter_var, width=10).grid(row=0, column=3, padx=5)
        
        # Background estimation
        bg_frame = ttk.Frame(params_frame)
        bg_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(bg_frame, text="Background Method:").grid(row=0, column=0, sticky="w", padx=5)
        self.bg_method_var = tk.StringVar(value="mode")
        bg_combo = ttk.Combobox(bg_frame, textvariable=self.bg_method_var,
                               values=["mode", "median", "mean"], state="readonly")
        bg_combo.grid(row=0, column=1, padx=5)
        
        # Analysis execution
        exec_frame = ttk.LabelFrame(analysis_frame, text="Execute Analysis")
        exec_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        ttk.Button(exec_frame, text="Start Analysis", command=self.start_analysis).pack(pady=10)
        
        # Progress tracking
        self.progress_var = tk.StringVar(value="Ready to analyze")
        ttk.Label(exec_frame, textvariable=self.progress_var).pack(pady=5)
        
        self.progress_bar = ttk.Progressbar(exec_frame, length=400, mode='determinate')
        self.progress_bar.pack(pady=5)
    
    def setup_results_tab(self):
        """Setup results viewing and export tab"""
        results_frame = ttk.Frame(self.notebook)
        self.notebook.add(results_frame, text="Results")
        
        # Results summary
        summary_frame = ttk.LabelFrame(results_frame, text="Analysis Summary")
        summary_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.summary_text = tk.Text(summary_frame, height=6, wrap=tk.WORD)
        summary_scrollbar = ttk.Scrollbar(summary_frame, orient=tk.VERTICAL, command=self.summary_text.yview)
        self.summary_text.configure(yscrollcommand=summary_scrollbar.set)
        
        self.summary_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        summary_scrollbar.pack(side=tk.RIGHT, fill=tk.Y, pady=5)
        
        # Export options
        export_frame = ttk.LabelFrame(results_frame, text="Export Results")
        export_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(export_frame, text="Export CSV", command=self.export_csv).pack(side=tk.LEFT, padx=5, pady=5)
        ttk.Button(export_frame, text="Export Images", command=self.export_images).pack(side=tk.LEFT, padx=5, pady=5)
        ttk.Button(export_frame, text="Export ROIs", command=self.export_rois).pack(side=tk.LEFT, padx=5, pady=5)
    
    # Event handlers and core functionality methods
    def new_experiment(self):
        """Create a new experiment"""
        self.current_experiment = ExperimentConfig(
            name="",
            output_directory=Path.home(),
            conditions=[],
            analysis_parameters={},
            created_date=datetime.now()
        )
        self.exp_name_var.set("")
        self.output_dir_var.set(str(Path.home()))
        self.conditions_listbox.delete(0, tk.END)
        self.logger.info("Created new experiment")
    
    def load_experiment(self):
        """Load an existing experiment configuration"""
        from tkinter import filedialog
        
        config_file = filedialog.askopenfilename(
            title="Load Experiment Configuration",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        
        if config_file:
            try:
                self.current_experiment = ExperimentConfig.load_config(Path(config_file))
                
                # Update GUI with loaded data
                self.exp_name_var.set(self.current_experiment.name)
                self.output_dir_var.set(str(self.current_experiment.output_directory))
                
                # Update conditions list
                self.conditions_listbox.delete(0, tk.END)
                for condition in self.current_experiment.conditions:
                    self.conditions_listbox.insert(tk.END, f"{condition.name} ({len(condition.channels)} channels)")
                
                messagebox.showinfo("Success", f"Loaded experiment: {self.current_experiment.name}")
                self.logger.info(f"Loaded experiment from: {config_file}")
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load experiment: {e}")
                self.logger.error(f"Failed to load experiment: {e}")
    
    def save_experiment(self):
        """Save the current experiment configuration"""
        if not self.current_experiment:
            messagebox.showerror("Error", "No experiment to save")
            return
        
        from tkinter import filedialog
        
        # Update experiment with current GUI values
        self.current_experiment.name = self.exp_name_var.get()
        self.current_experiment.output_directory = Path(self.output_dir_var.get())
        
        config_file = filedialog.asksaveasfilename(
            title="Save Experiment Configuration",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        
        if config_file:
            try:
                self.current_experiment.save_config(Path(config_file))
                messagebox.showinfo("Success", f"Experiment saved to: {config_file}")
                self.logger.info(f"Saved experiment to: {config_file}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save experiment: {e}")
                self.logger.error(f"Failed to save experiment: {e}")
    
    def add_condition(self):
        """Add a new experimental condition"""
        from channel_configurator import ConditionManager
        
        condition_manager = ConditionManager(self)
        condition_manager.add_condition_dialog()
    
    def edit_condition(self):
        """Edit selected condition"""
        selection = self.conditions_listbox.curselection()
        if not selection:
            messagebox.showwarning("Warning", "Please select a condition to edit")
            return
        
        # Implementation for editing condition
        messagebox.showinfo("Info", "Edit condition functionality will be implemented")
    
    def remove_condition(self):
        """Remove selected condition"""
        selection = self.conditions_listbox.curselection()
        if not selection:
            messagebox.showwarning("Warning", "Please select a condition to remove")
            return
        
        index = selection[0]
        if self.current_experiment and index < len(self.current_experiment.conditions):
            condition_name = self.current_experiment.conditions[index].name
            
            result = messagebox.askyesno("Confirm", f"Remove condition '{condition_name}'?")
            if result:
                self.current_experiment.conditions.pop(index)
                self.conditions_listbox.delete(index)
                self.logger.info(f"Removed condition: {condition_name}")
    
    def add_condition_to_experiment(self, condition_data):
        """Add condition data to current experiment"""
        if not self.current_experiment:
            self.new_experiment()
        
        # Convert condition data to Condition object
        from cellquant_main import Condition, ChannelInfo
        
        channels = []
        for ch_data in condition_data.get('channels', []):
            channel = ChannelInfo(
                name=ch_data['name'],
                type=ch_data['type'],
                wavelength=ch_data.get('wavelength', ''),
                purpose=ch_data.get('purpose', '')
            )
            channels.append(channel)
        
        condition = Condition(
            name=condition_data['name'],
            directory=condition_data['directory'],
            description=condition_data.get('description', ''),
            channels=channels
        )
        
        self.current_experiment.conditions.append(condition)
        
        # Update GUI
        self.conditions_listbox.insert(tk.END, f"{condition.name} ({len(condition.channels)} channels)")
        self.logger.info(f"Added condition: {condition.name}")
    
    def browse_output_dir(self):
        """Browse for output directory"""
        from tkinter import filedialog
        
        directory = filedialog.askdirectory(title="Select Output Directory")
        if directory:
            self.output_dir_var.set(directory)
    
    def start_analysis(self):
        """Start the analysis pipeline"""
        if not self.current_experiment or not self.current_experiment.conditions:
            messagebox.showerror("Error", "Please configure experiment conditions first")
            return
        
        # Update experiment with current settings
        self.current_experiment.name = self.exp_name_var.get()
        self.current_experiment.output_directory = Path(self.output_dir_var.get())
        
        # Get analysis parameters from GUI
        analysis_params = {
            'cellpose_model': self.cellpose_model_var.get(),
            'cell_diameter': float(self.cell_diameter_var.get()) if self.cell_diameter_var.get() else None,
            'background_method': self.bg_method_var.get(),
        }
        self.current_experiment.analysis_parameters = analysis_params
        
        self.logger.info("Starting analysis pipeline")
        
        try:
            # Convert to dict format for AnalysisPipeline
            config_dict = {
                'experiment_name': self.current_experiment.name,
                'output_directory': str(self.current_experiment.output_directory),
                'conditions': [],
                **analysis_params
            }
            
            # Convert conditions
            for condition in self.current_experiment.conditions:
                condition_dict = {
                    'name': condition.name,
                    'directory': str(condition.directory),
                    'description': condition.description,
                    'channels': []
                }
                
                # Convert channels
                for channel in condition.channels:
                    channel_dict = {
                        'name': channel.name,
                        'type': channel.type,
                        'purpose': channel.purpose,
                        'quantify': True,  # Default for now
                        'nuclear_only': channel.type == 'nuclear',
                        'wavelength': channel.wavelength
                    }
                    condition_dict['channels'].append(channel_dict)
                
                config_dict['conditions'].append(condition_dict)
            
            # Progress callback
            def progress_callback(message, percentage):
                self.progress_var.set(message)
                self.progress_bar['value'] = percentage
                self.root.update_idletasks()
            
            # Run analysis in separate thread to avoid freezing GUI
            import threading
            from analysis_pipeline import AnalysisPipeline
            
            def run_analysis():
                try:
                    pipeline = AnalysisPipeline(config_dict, progress_callback)
                    results = pipeline.run_analysis()
                    
                    # Update results tab
                    self.analysis_results = results
                    self.update_results_display()
                    
                    messagebox.showinfo("Success", "Analysis completed successfully!")
                    
                except Exception as e:
                    messagebox.showerror("Error", f"Analysis failed: {e}")
                    self.logger.error(f"Analysis failed: {e}")
            
            analysis_thread = threading.Thread(target=run_analysis, daemon=True)
            analysis_thread.start()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to start analysis: {e}")
            self.logger.error(f"Failed to start analysis: {e}")
    
    def update_results_display(self):
        """Update the results tab with analysis results"""
        if not self.analysis_results:
            return
        
        # Update summary text
        summary_text = f"Analysis Results for: {self.analysis_results.get('experiment_info', {}).get('experiment_name', 'Unknown')}\n\n"
        
        exp_info = self.analysis_results.get('experiment_info', {})
        summary_text += f"Total Conditions: {exp_info.get('total_conditions', 0)}\n"
        summary_text += f"Total Images Processed: {exp_info.get('total_images_processed', 0)}\n"
        summary_text += f"Duration: {exp_info.get('duration_seconds', 0):.1f} seconds\n\n"
        
        # Add condition summaries
        for condition_name, condition_data in self.analysis_results.get('conditions', {}).items():
            summary_text += f"{condition_name}:\n"
            summary_text += f"  - {len(condition_data.get('cell_data', []))} cells analyzed\n"
            summary_text += f"  - {len(condition_data.get('images', []))} images processed\n\n"
        
        self.summary_text.delete(1.0, tk.END)
        self.summary_text.insert(1.0, summary_text)
    
    def export_csv(self):
        """Export results as CSV"""
        if not self.analysis_results:
            messagebox.showwarning("Warning", "No analysis results to export")
            return
        
        messagebox.showinfo("Info", "CSV export functionality will be implemented")
    
    def export_images(self):
        """Export processed images"""
        if not self.analysis_results:
            messagebox.showwarning("Warning", "No analysis results to export")
            return
        
        messagebox.showinfo("Info", "Image export functionality will be implemented")
    
    def export_rois(self):
        """Export ROI files"""
        if not self.analysis_results:
            messagebox.showwarning("Warning", "No analysis results to export")
            return
        
        messagebox.showinfo("Info", "ROI export functionality will be implemented")
        
    def run(self):
        """Start the application"""
        self.root.mainloop()

if __name__ == "__main__":
    app = MainApplication()
    app.run()