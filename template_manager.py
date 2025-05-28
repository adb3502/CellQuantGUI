# Template and Configuration Management System
# Handles saving, loading, and managing analysis templates

import json
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
import logging
from dataclasses import dataclass, asdict
import hashlib
import shutil

@dataclass 
class AnalysisTemplate:
    """Represents a reusable analysis template"""
    
    name: str
    description: str
    category: str  # 'dna_damage', 'mitochondrial', 'protein_localization', etc.
    version: str
    created_date: datetime
    modified_date: datetime
    author: str
    
    # Channel configuration template
    channel_template: List[Dict[str, Any]]
    
    # Analysis parameters template
    analysis_parameters: Dict[str, Any]
    
    # Expected file naming patterns
    file_patterns: Dict[str, str]
    
    # Usage instructions
    instructions: str
    
    # Template metadata
    tags: List[str]
    citation: Optional[str] = None
    doi: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert template to dictionary"""
        template_dict = asdict(self)
        template_dict['created_date'] = self.created_date.isoformat()
        template_dict['modified_date'] = self.modified_date.isoformat()
        return template_dict
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        """Create template from dictionary"""
        data['created_date'] = datetime.fromisoformat(data['created_date'])
        data['modified_date'] = datetime.fromisoformat(data['modified_date'])
        return cls(**data)
    
    def get_hash(self) -> str:
        """Get hash of template for versioning"""
        content = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.md5(content.encode()).hexdigest()[:8]

class TemplateManager:
    """Manages analysis templates and configurations"""
    
    def __init__(self, templates_dir: Path = None):
        self.templates_dir = templates_dir or Path("templates")
        self.templates_dir.mkdir(exist_ok=True)
        
        self.logger = logging.getLogger(__name__)
        
        # Built-in templates directory
        self.builtin_templates_dir = self.templates_dir / "builtin"
        self.user_templates_dir = self.templates_dir / "user"
        
        self.builtin_templates_dir.mkdir(exist_ok=True)
        self.user_templates_dir.mkdir(exist_ok=True)
        
        # Initialize built-in templates
        self._create_builtin_templates()
        
        # Template registry
        self.templates = {}
        self.load_all_templates()
    
    def _create_builtin_templates(self):
        """Create built-in analysis templates"""
        
        builtin_templates = [
            self._create_dna_damage_template(),
            self._create_mitochondrial_template(),
            self._create_protein_localization_template(),
            self._create_cell_cycle_template(),
            self._create_apoptosis_template()
        ]
        
        for template in builtin_templates:
            self._save_template(template, self.builtin_templates_dir, overwrite=True)
    
    def _create_dna_damage_template(self) -> AnalysisTemplate:
        """Create DNA damage analysis template"""
        
        return AnalysisTemplate(
            name="DNA Damage Foci Analysis",
            description="Standard template for analyzing DNA damage response using γH2AX and 53BP1 foci",
            category="dna_damage",
            version="1.0",
            created_date=datetime.now(),
            modified_date=datetime.now(),
            author="CellQuantGUI",
            
            channel_template=[
                {
                    "name": "DAPI",
                    "type": "nuclear",
                    "purpose": "segmentation",
                    "quantify": False,
                    "nuclear_only": True,
                    "wavelength": "405nm",
                    "description": "Nuclear staining for cell segmentation"
                },
                {
                    "name": "gamma_H2AX",
                    "type": "nuclear",
                    "purpose": "quantification",
                    "quantify": True,
                    "nuclear_only": True,
                    "wavelength": "488nm",
                    "description": "DNA double-strand break marker"
                },
                {
                    "name": "53BP1",
                    "type": "nuclear", 
                    "purpose": "quantification",
                    "quantify": True,
                    "nuclear_only": True,
                    "wavelength": "555nm",
                    "description": "DNA damage response protein"
                }
            ],
            
            analysis_parameters={
                "cellpose_model": "nuclei",
                "cell_diameter": 25,
                "use_gpu": False,
                "background_method": "mode",
                "flow_threshold": 0.4,
                "cellprob_threshold": 0.0,
                "min_cell_area": 100,
                "max_cell_area": 2000
            },
            
            file_patterns={
                "multichannel": r".*\.tif$",
                "single_channel": {
                    "DAPI": r".*[_-]dapi[_-].*\.tif$|.*[_-]hoechst[_-].*\.tif$",
                    "gamma_H2AX": r".*[_-]h2ax[_-].*\.tif$|.*[_-]gamma[_-]h2ax.*\.tif$",
                    "53BP1": r".*[_-]53bp1[_-].*\.tif$|.*[_-]bp1[_-].*\.tif$"
                }
            },
            
            instructions="""
            DNA Damage Foci Analysis Protocol:
            
            1. Image Acquisition:
               - Use consistent exposure times across conditions
               - Acquire at least 100 cells per condition
               - Include proper controls (untreated, positive control)
            
            2. Analysis Settings:
               - Nuclear segmentation using DAPI
               - Quantify γH2AX and 53BP1 foci within nuclear boundaries
               - Use mode background correction
            
            3. Quality Control:
               - Check segmentation accuracy
               - Exclude cells with poor nuclear morphology
               - Verify foci detection sensitivity
            
            4. Statistical Analysis:
               - Compare foci counts between conditions
               - Report both mean and median values
               - Use appropriate statistical tests (Mann-Whitney U, Kruskal-Wallis)
            """,
            
            tags=["dna_damage", "foci", "nuclear", "repair", "radiation"],
            citation="Rogakou EP, et al. DNA double-stranded breaks induce histone H2AX phosphorylation on serine 139. J Biol Chem. 1998;273(10):5858-68.",
            doi="10.1074/jbc.273.10.5858"
        )
    
    def _create_mitochondrial_template(self) -> AnalysisTemplate:
        """Create mitochondrial analysis template"""
        
        return AnalysisTemplate(
            name="Mitochondrial Analysis",
            description="Template for analyzing mitochondrial morphology and function",
            category="mitochondrial",
            version="1.0",
            created_date=datetime.now(),
            modified_date=datetime.now(),
            author="CellQuantGUI",
            
            channel_template=[
                {
                    "name": "DAPI",
                    "type": "nuclear",
                    "purpose": "segmentation",
                    "quantify": False,
                    "nuclear_only": True,
                    "wavelength": "405nm",
                    "description": "Nuclear staining"
                },
                {
                    "name": "MitoTracker",
                    "type": "cellular",
                    "purpose": "quantification",
                    "quantify": True,
                    "nuclear_only": False,
                    "wavelength": "594nm",
                    "description": "Mitochondrial mass/membrane potential"
                },
                {
                    "name": "Cytochrome_C",
                    "type": "cellular",
                    "purpose": "quantification",
                    "quantify": True,
                    "nuclear_only": False,
                    "wavelength": "488nm",
                    "description": "Mitochondrial respiratory complex"
                }
            ],
            
            analysis_parameters={
                "cellpose_model": "cyto2",
                "cell_diameter": 30,
                "use_gpu": False,
                "background_method": "median",
                "flow_threshold": 0.4,
                "cellprob_threshold": 0.0,
                "min_cell_area": 200,
                "max_cell_area": 5000
            },
            
            file_patterns={
                "multichannel": r".*\.tif$",
                "single_channel": {
                    "DAPI": r".*[_-]dapi[_-].*\.tif$",
                    "MitoTracker": r".*[_-]mito[_-].*\.tif$|.*[_-]mitotracker.*\.tif$",
                    "Cytochrome_C": r".*[_-]cytc[_-].*\.tif$|.*[_-]cytochrome.*\.tif$"
                }
            },
            
            instructions="""
            Mitochondrial Analysis Protocol:
            
            1. Live cell imaging with MitoTracker dyes
            2. Fixed cell analysis for cytochrome c release
            3. Quantify mitochondrial mass and distribution
            4. Analyze mitochondrial fragmentation/networking
            """,
            
            tags=["mitochondria", "metabolism", "apoptosis", "cellular"]
        )
    
    def _create_protein_localization_template(self) -> AnalysisTemplate:
        """Create protein localization template"""
        
        return AnalysisTemplate(
            name="Protein Localization Analysis",
            description="General template for protein subcellular localization studies",
            category="protein_localization",
            version="1.0",
            created_date=datetime.now(),
            modified_date=datetime.now(),
            author="CellQuantGUI",
            
            channel_template=[
                {
                    "name": "DAPI",
                    "type": "nuclear",
                    "purpose": "segmentation",
                    "quantify": False,
                    "nuclear_only": True,
                    "wavelength": "405nm",
                    "description": "Nuclear segmentation"
                },
                {
                    "name": "Target_Protein",
                    "type": "cellular",
                    "purpose": "quantification",
                    "quantify": True,
                    "nuclear_only": False,
                    "wavelength": "488nm",
                    "description": "Protein of interest"
                },
                {
                    "name": "Cellular_Marker",
                    "type": "cellular",
                    "purpose": "reference",
                    "quantify": False,
                    "nuclear_only": False,
                    "wavelength": "555nm",
                    "description": "Cellular compartment marker (optional)"
                }
            ],
            
            analysis_parameters={
                "cellpose_model": "cyto2",
                "cell_diameter": 30,
                "use_gpu": False,
                "background_method": "mode",
                "flow_threshold": 0.4,
                "cellprob_threshold": 0.0
            },
            
            file_patterns={
                "multichannel": r".*\.tif$",
                "single_channel": {
                    "DAPI": r".*[_-]dapi[_-].*\.tif$",
                    "Target_Protein": r".*[_-]target[_-].*\.tif$|.*[_-]protein[_-].*\.tif$",
                    "Cellular_Marker": r".*[_-]marker[_-].*\.tif$"
                }
            },
            
            instructions="""
            Protein Localization Analysis:
            
            1. Quantify protein expression levels
            2. Analyze subcellular distribution
            3. Calculate nuclear/cytoplasmic ratios
            4. Compare localization between conditions
            """,
            
            tags=["protein", "localization", "subcellular", "expression"]
        )
    
    def _create_cell_cycle_template(self) -> AnalysisTemplate:
        """Create cell cycle analysis template"""
        
        return AnalysisTemplate(
            name="Cell Cycle Analysis",
            description="Template for cell cycle phase analysis using DNA content and markers",
            category="cell_cycle",
            version="1.0",
            created_date=datetime.now(),
            modified_date=datetime.now(),
            author="CellQuantGUI",
            
            channel_template=[
                {
                    "name": "DAPI",
                    "type": "nuclear",
                    "purpose": "segmentation",
                    "quantify": True,
                    "nuclear_only": True,
                    "wavelength": "405nm",
                    "description": "DNA content for cell cycle phase"
                },
                {
                    "name": "Ki67",
                    "type": "nuclear",
                    "purpose": "quantification",
                    "quantify": True,
                    "nuclear_only": True,
                    "wavelength": "488nm",
                    "description": "Proliferation marker"
                },
                {
                    "name": "pH3",
                    "type": "nuclear",
                    "purpose": "quantification",
                    "quantify": True,
                    "nuclear_only": True,
                    "wavelength": "555nm",
                    "description": "Mitotic marker (phospho-histone H3)"
                }
            ],
            
            analysis_parameters={
                "cellpose_model": "nuclei",
                "cell_diameter": 20,
                "use_gpu": False,
                "background_method": "mode"
            },
            
            file_patterns={
                "single_channel": {
                    "DAPI": r".*[_-]dapi[_-].*\.tif$",
                    "Ki67": r".*[_-]ki67[_-].*\.tif$",
                    "pH3": r".*[_-]ph3[_-].*\.tif$|.*[_-]phospho.*h3.*\.tif$"
                }
            },
            
            instructions="""
            Cell Cycle Analysis Protocol:
            
            1. Quantify DNA content (DAPI intensity)
            2. Classify cells by cell cycle phase
            3. Identify proliferating cells (Ki67+)
            4. Count mitotic cells (pH3+)
            """,
            
            tags=["cell_cycle", "proliferation", "mitosis", "dna_content"]
        )
    
    def _create_apoptosis_template(self) -> AnalysisTemplate:
        """Create apoptosis analysis template"""
        
        return AnalysisTemplate(
            name="Apoptosis Analysis",
            description="Template for analyzing apoptotic cell death",
            category="apoptosis",
            version="1.0",
            created_date=datetime.now(),
            modified_date=datetime.now(),
            author="CellQuantGUI",
            
            channel_template=[
                {
                    "name": "DAPI",
                    "type": "nuclear",
                    "purpose": "segmentation",
                    "quantify": True,
                    "nuclear_only": True,
                    "wavelength": "405nm",
                    "description": "Nuclear morphology assessment"
                },
                {
                    "name": "Cleaved_Caspase3",
                    "type": "cellular",
                    "purpose": "quantification",
                    "quantify": True,
                    "nuclear_only": False,
                    "wavelength": "488nm",
                    "description": "Apoptosis execution marker"
                },
                {
                    "name": "Cytochrome_C",
                    "type": "cellular",
                    "purpose": "quantification",
                    "quantify": True,
                    "nuclear_only": False,
                    "wavelength": "555nm",
                    "description": "Mitochondrial release marker"
                }
            ],
            
            analysis_parameters={
                "cellpose_model": "cyto2",
                "cell_diameter": 25,
                "use_gpu": False,
                "background_method": "median"
            },
            
            file_patterns={
                "single_channel": {
                    "DAPI": r".*[_-]dapi[_-].*\.tif$",
                    "Cleaved_Caspase3": r".*[_-]casp3[_-].*\.tif$|.*cleaved.*caspase.*\.tif$",
                    "Cytochrome_C": r".*[_-]cytc[_-].*\.tif$"
                }
            },
            
            instructions="""
            Apoptosis Analysis Protocol:
            
            1. Assess nuclear morphology (fragmentation, condensation)
            2. Quantify cleaved caspase-3 activation
            3. Measure cytochrome c release from mitochondria
            4. Calculate apoptotic index
            """,
            
            tags=["apoptosis", "cell_death", "caspase", "mitochondria"]
        )
    
    def load_all_templates(self):
        """Load all available templates"""
        
        self.templates = {}
        
        # Load built-in templates
        for template_file in self.builtin_templates_dir.glob("*.json"):
            try:
                template = self.load_template(template_file)
                template_id = f"builtin_{template.name.lower().replace(' ', '_')}"
                self.templates[template_id] = template
            except Exception as e:
                self.logger.warning(f"Failed to load built-in template {template_file}: {e}")
        
        # Load user templates
        for template_file in self.user_templates_dir.glob("*.json"):
            try:
                template = self.load_template(template_file)
                template_id = f"user_{template.name.lower().replace(' ', '_')}"
                self.templates[template_id] = template
            except Exception as e:
                self.logger.warning(f"Failed to load user template {template_file}: {e}")
        
        self.logger.info(f"Loaded {len(self.templates)} templates")
    
    def get_template(self, template_id: str) -> Optional[AnalysisTemplate]:
        """Get template by ID"""
        return self.templates.get(template_id)
    
    def list_templates(self, category: str = None) -> List[AnalysisTemplate]:
        """List available templates, optionally filtered by category"""
        
        templates = list(self.templates.values())
        
        if category:
            templates = [t for t in templates if t.category == category]
        
        return sorted(templates, key=lambda t: t.name)
    
    def get_categories(self) -> List[str]:
        """Get list of available template categories"""
        
        categories = set(template.category for template in self.templates.values())
        return sorted(categories)
    
    def save_template(self, template: AnalysisTemplate, user_template: bool = True) -> bool:
        """Save template to file"""
        
        target_dir = self.user_templates_dir if user_template else self.builtin_templates_dir
        return self._save_template(template, target_dir)
    
    def _save_template(self, template: AnalysisTemplate, target_dir: Path, overwrite: bool = False) -> bool:
        """Internal method to save template"""
        
        try:
            filename = f"{template.name.lower().replace(' ', '_')}.json"
            filepath = target_dir / filename
            
            if filepath.exists() and not overwrite:
                self.logger.warning(f"Template file already exists: {filepath}")
                return False
            
            template_data = template.to_dict()
            
            with open(filepath, 'w') as f:
                json.dump(template_data, f, indent=2, default=str)
            
            self.logger.info(f"Template saved: {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save template: {e}")
            return False
    
    def load_template(self, filepath: Path) -> AnalysisTemplate:
        """Load template from file"""
        
        with open(filepath, 'r') as f:
            template_data = json.load(f)
        
        return AnalysisTemplate.from_dict(template_data)
    
    def delete_template(self, template_id: str) -> bool:
        """Delete a user template"""
        
        if not template_id.startswith('user_'):
            self.logger.error("Can only delete user templates")
            return False
        
        template = self.templates.get(template_id)
        if not template:
            self.logger.error(f"Template not found: {template_id}")
            return False
        
        try:
            filename = f"{template.name.lower().replace(' ', '_')}.json"
            filepath = self.user_templates_dir / filename
            
            if filepath.exists():
                filepath.unlink()
                del self.templates[template_id]
                self.logger.info(f"Template deleted: {template_id}")
                return True
            else:
                self.logger.error(f"Template file not found: {filepath}")
                return False
                
        except Exception as e:
            self.logger.error(f"Failed to delete template: {e}")
            return False
    
    def duplicate_template(self, template_id: str, new_name: str) -> Optional[str]:
        """Duplicate an existing template"""
        
        original_template = self.templates.get(template_id)
        if not original_template:
            self.logger.error(f"Template not found: {template_id}")
            return None
        
        # Create new template with modified metadata
        new_template = AnalysisTemplate(
            name=new_name,
            description=f"Copy of {original_template.description}",
            category=original_template.category,
            version="1.0",
            created_date=datetime.now(),
            modified_date=datetime.now(),
            author="User",
            channel_template=original_template.channel_template.copy(),
            analysis_parameters=original_template.analysis_parameters.copy(),
            file_patterns=original_template.file_patterns.copy(),
            instructions=original_template.instructions,
            tags=original_template.tags.copy(),
            citation=original_template.citation,
            doi=original_template.doi
        )
        
        if self.save_template(new_template):
            new_template_id = f"user_{new_name.lower().replace(' ', '_')}"
            self.templates[new_template_id] = new_template
            return new_template_id
        
        return None
    
    def create_template_from_config(self, config: Dict[str, Any], template_name: str, 
                                   template_description: str, category: str) -> Optional[str]:
        """Create a new template from an existing configuration"""
        
        # Extract channel configuration
        channel_template = []
        if 'conditions' in config and config['conditions']:
            # Use first condition as template
            first_condition = config['conditions'][0]
            if 'channels' in first_condition:
                channel_template = first_condition['channels']
        
        # Extract analysis parameters
        analysis_parameters = config.get('analysis_parameters', {})
        
        # Create template
        template = AnalysisTemplate(
            name=template_name,
            description=template_description,
            category=category,
            version="1.0",
            created_date=datetime.now(),
            modified_date=datetime.now(),
            author="User",
            channel_template=channel_template,
            analysis_parameters=analysis_parameters,
            file_patterns={},  # User will need to define these
            instructions="Custom template created from configuration",
            tags=[category]
        )
        
        if self.save_template(template):
            template_id = f"user_{template_name.lower().replace(' ', '_')}"
            self.templates[template_id] = template
            return template_id
        
        return None
    
    def apply_template_to_condition(self, template_id: str, condition_config: Dict[str, Any]) -> Dict[str, Any]:
        """Apply template configuration to a condition"""
        
        template = self.templates.get(template_id)
        if not template:
            raise ValueError(f"Template not found: {template_id}")
        
        # Create new condition config based on template
        new_config = condition_config.copy()
        
        # Apply channel template
        new_config['channels'] = []
        for channel_template in template.channel_template:
            new_config['channels'].append(channel_template.copy())
        
        return new_config
    
    def get_template_suggestions(self, image_filenames: List[str]) -> List[str]:
        """Suggest templates based on image filenames"""
        
        suggestions = []
        filename_text = ' '.join(image_filenames).lower()
        
        # Check for common keywords
        if any(keyword in filename_text for keyword in ['h2ax', 'gamma', '53bp1', 'dna', 'damage']):
            suggestions.append('builtin_dna_damage_foci_analysis')
        
        if any(keyword in filename_text for keyword in ['mito', 'cytochrome', 'mitochondr']):
            suggestions.append('builtin_mitochondrial_analysis')
        
        if any(keyword in filename_text for keyword in ['ki67', 'ph3', 'cycle']):
            suggestions.append('builtin_cell_cycle_analysis')
        
        if any(keyword in filename_text for keyword in ['casp', 'apopt', 'cleaved']):
            suggestions.append('builtin_apoptosis_analysis')
        
        # If no specific suggestions, suggest general protein localization
        if not suggestions:
            suggestions.append('builtin_protein_localization_analysis')
        
        return suggestions
    
    def export_template(self, template_id: str, export_path: Path) -> bool:
        """Export template to external file"""
        
        template = self.templates.get(template_id)
        if not template:
            self.logger.error(f"Template not found: {template_id}")
            return False
        
        try:
            template_data = template.to_dict()
            
            with open(export_path, 'w') as f:
                json.dump(template_data, f, indent=2, default=str)
            
            self.logger.info(f"Template exported to: {export_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to export template: {e}")
            return False
    
    def import_template(self, import_path: Path) -> Optional[str]:
        """Import template from external file"""
        
        try:
            with open(import_path, 'r') as f:
                template_data = json.load(f)
            
            template = AnalysisTemplate.from_dict(template_data)
            
            # Ensure it's marked as user template
            template.modified_date = datetime.now()
            
            if self.save_template(template):
                template_id = f"user_{template.name.lower().replace(' ', '_')}"
                self.templates[template_id] = template
                self.logger.info(f"Template imported: {template_id}")
                return template_id
            
            return None
            
        except Exception as e:
            self.logger.error(f"Failed to import template: {e}")
            return None
    
    def validate_template(self, template: AnalysisTemplate) -> List[str]:
        """Validate template configuration and return list of issues"""
        
        issues = []
        
        # Check required fields
        if not template.name:
            issues.append("Template name is required")
        
        if not template.description:
            issues.append("Template description is required")
        
        if not template.category:
            issues.append("Template category is required")
        
        # Check channel configuration
        if not template.channel_template:
            issues.append("At least one channel must be configured")
        else:
            # Check for segmentation channel
            seg_channels = [ch for ch in template.channel_template 
                          if ch.get('purpose') == 'segmentation']
            if not seg_channels:
                issues.append("At least one segmentation channel is recommended")
            
            # Check for quantification channels
            quant_channels = [ch for ch in template.channel_template 
                            if ch.get('quantify', False)]
            if not quant_channels:
                issues.append("At least one quantification channel is recommended")
        
        # Check analysis parameters
        required_params = ['cellpose_model', 'background_method']
        for param in required_params:
            if param not in template.analysis_parameters:
                issues.append(f"Missing required analysis parameter: {param}")
        
        return issues

# GUI Integration for Template Management
class TemplateManagerGUI:
    """GUI components for template management"""
    
    def __init__(self, parent, template_manager: TemplateManager):
        self.parent = parent
        self.template_manager = template_manager
        self.logger = logging.getLogger(__name__)
    
    def create_template_selection_dialog(self) -> Optional[str]:
        """Create dialog for selecting analysis template"""
        
        import tkinter as tk
        from tkinter import ttk
        
        dialog = tk.Toplevel(self.parent)
        dialog.title("Select Analysis Template")
        dialog.geometry("800x600")
        dialog.grab_set()
        
        selected_template = None
        
        # Create main frame
        main_frame = ttk.Frame(dialog)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Category selection
        category_frame = ttk.Frame(main_frame)
        category_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(category_frame, text="Category:").pack(side=tk.LEFT)
        
        category_var = tk.StringVar(value="All")
        categories = ["All"] + self.template_manager.get_categories()
        category_combo = ttk.Combobox(category_frame, textvariable=category_var, 
                                     values=categories, state="readonly")
        category_combo.pack(side=tk.LEFT, padx=(5, 0))
        
        # Template list
        list_frame = ttk.Frame(main_frame)
        list_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        # Create treeview for templates
        columns = ('Name', 'Category', 'Description', 'Version', 'Author')
        template_tree = ttk.Treeview(list_frame, columns=columns, show='headings', height=15)
        
        for col in columns:
            template_tree.heading(col, text=col)
            template_tree.column(col, width=120)
        
        # Scrollbar for treeview
        scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=template_tree.yview)
        template_tree.configure(yscrollcommand=scrollbar.set)
        
        template_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Template details
        details_frame = ttk.LabelFrame(main_frame, text="Template Details")
        details_frame.pack(fill=tk.X, pady=(0, 10))
        
        details_text = tk.Text(details_frame, height=8, wrap=tk.WORD)
        details_scrollbar = ttk.Scrollbar(details_frame, orient=tk.VERTICAL, command=details_text.yview)
        details_text.configure(yscrollcommand=details_scrollbar.set)
        
        details_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        details_scrollbar.pack(side=tk.RIGHT, fill=tk.Y, pady=5)
        
        def update_template_list():
            """Update template list based on category filter"""
            template_tree.delete(*template_tree.get_children())
            
            category_filter = category_var.get()
            if category_filter == "All":
                templates = self.template_manager.list_templates()
            else:
                templates = self.template_manager.list_templates(category_filter)
            
            for template in templates:
                template_tree.insert('', tk.END, values=(
                    template.name,
                    template.category,
                    template.description[:50] + "..." if len(template.description) > 50 else template.description,
                    template.version,
                    template.author
                ))
        
        def on_template_select(event):
            """Handle template selection"""
            selection = template_tree.selection()
            if selection:
                item = template_tree.item(selection[0])
                template_name = item['values'][0]
                
                # Find template by name
                for template_id, template in self.template_manager.templates.items():
                    if template.name == template_name:
                        show_template_details(template)
                        break
        
        def show_template_details(template: AnalysisTemplate):
            """Show template details"""
            details_text.delete(1.0, tk.END)
            
            details = f"""
Name: {template.name}
Category: {template.category}
Version: {template.version}
Author: {template.author}
Created: {template.created_date.strftime('%Y-%m-%d')}

Description:
{template.description}

Channels:
"""
            for i, channel in enumerate(template.channel_template, 1):
                details += f"  {i}. {channel['name']} ({channel['type']}) - {channel.get('description', 'No description')}\n"
            
            details += f"\nInstructions:\n{template.instructions}"
            
            if template.tags:
                details += f"\n\nTags: {', '.join(template.tags)}"
            
            if template.citation:
                details += f"\n\nCitation: {template.citation}"
            
            details_text.insert(1.0, details)
        
        # Bind events
        category_combo.bind('<<ComboboxSelected>>', lambda e: update_template_list())
        template_tree.bind('<<TreeviewSelect>>', on_template_select)
        
        # Initialize template list
        update_template_list()
        
        # Buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X)
        
        def on_select():
            nonlocal selected_template
            selection = template_tree.selection()
            if selection:
                item = template_tree.item(selection[0])
                template_name = item['values'][0]
                
                # Find template ID by name
                for template_id, template in self.template_manager.templates.items():
                    if template.name == template_name:
                        selected_template = template_id
                        break
                
                dialog.destroy()
            else:
                tk.messagebox.showwarning("No Selection", "Please select a template")
        
        ttk.Button(button_frame, text="Select", command=on_select).pack(side=tk.RIGHT, padx=5)
        ttk.Button(button_frame, text="Cancel", command=dialog.destroy).pack(side=tk.RIGHT, padx=5)
        
        dialog.wait_window()
        return selected_template

# Example usage
def demonstrate_template_system():
    """Demonstrate the template management system"""
    
    # Initialize template manager
    template_manager = TemplateManager()
    
    print("CellQuantGUI Template System Demo")
    print("=" * 40)
    
    # List available templates
    print("\nAvailable Templates:")
    for category in template_manager.get_categories():
        print(f"\n{category.upper()}:")
        templates = template_manager.list_templates(category)
        for template in templates:
            print(f"  - {template.name} (v{template.version})")
    
    # Show template suggestions for sample filenames
    sample_filenames = [
        "experiment_001_dapi.tif",
        "experiment_001_h2ax.tif", 
        "experiment_001_53bp1.tif"
    ]
    
    print(f"\nSample filenames: {sample_filenames}")
    suggestions = template_manager.get_template_suggestions(sample_filenames)
    print(f"Suggested templates: {suggestions}")
    
    # Show template details
    if suggestions:
        template_id = suggestions[0]
        template = template_manager.get_template(template_id)
        if template:
            print(f"\nTemplate Details - {template.name}:")
            print(f"Description: {template.description}")
            print(f"Channels: {len(template.channel_template)}")
            for ch in template.channel_template:
                print(f"  - {ch['name']}: {ch['type']} ({ch['purpose']})")

if __name__ == "__main__":
    demonstrate_template_system()