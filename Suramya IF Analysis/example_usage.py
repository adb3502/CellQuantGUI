# Example Usage and Integration Script
# Demonstrates how to use the CellQuantGUI system

import sys
from pathlib import Path
import json
import logging
from datetime import datetime

# Add the main application modules to path
# sys.path.append('path/to/your/cellquant_modules')

# Import the main components
from cellquant_main import MainApplication, ExperimentConfig, Condition, ChannelInfo
from analysis_pipeline import AnalysisPipeline, AnalysisParameters

def setup_logging():
    """Setup logging configuration"""
    
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # Create logs directory
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Setup file and console logging
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        handlers=[
            logging.FileHandler(log_dir / f"cellquant_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
            logging.StreamHandler(sys.stdout)
        ]
    )

def create_example_experiment_config():
    """Create an example experiment configuration for testing"""
    
    # Define example channels for different experimental scenarios
    
    # Scenario 1: DNA damage study (γH2AX + 53BP1 + DAPI)
    dna_damage_channels = [
        ChannelInfo(
            name="DAPI",
            type="nuclear",
            purpose="segmentation",
            wavelength="405nm"
        ),
        ChannelInfo(
            name="gamma_H2AX",
            type="nuclear", 
            purpose="quantification",
            wavelength="488nm"
        ),
        ChannelInfo(
            name="53BP1",
            type="nuclear",
            purpose="quantification", 
            wavelength="555nm"
        )
    ]
    
    # Scenario 2: Mitochondrial study (MitoTracker + Cytochrome C + DAPI)
    mitochondrial_channels = [
        ChannelInfo(
            name="DAPI",
            type="nuclear",
            purpose="segmentation",
            wavelength="405nm"
        ),
        ChannelInfo(
            name="MitoTracker",
            type="cellular",
            purpose="quantification",
            wavelength="594nm"
        ),
        ChannelInfo(
            name="Cytochrome_C",
            type="cellular", 
            purpose="quantification",
            wavelength="488nm"
        )
    ]
    
    # Create example conditions
    conditions = [
        Condition(
            name="Control",
            directory=Path("data/control"),
            description="Control condition - untreated cells",
            channels=dna_damage_channels
        ),
        Condition(
            name="Treatment_1Gy",
            directory=Path("data/treatment_1gy"),
            description="Cells treated with 1 Gy ionizing radiation",
            channels=dna_damage_channels
        ),
        Condition(
            name="Treatment_2Gy", 
            directory=Path("data/treatment_2gy"),
            description="Cells treated with 2 Gy ionizing radiation",
            channels=dna_damage_channels
        )
    ]
    
    # Create experiment configuration
    experiment = ExperimentConfig(
        name="DNA_Damage_Response_Study",
        output_directory=Path("results/dna_damage_study"),
        conditions=conditions,
        analysis_parameters=AnalysisParameters.get_default_config(),
        created_date=datetime.now()
    )
    
    return experiment

def create_batch_analysis_config():
    """Create configuration for batch analysis without GUI"""
    
    # This demonstrates how to set up analysis programmatically
    config = {
        'experiment_name': 'Automated_Batch_Analysis',
        'output_directory': 'results/batch_analysis',
        'conditions': [
            {
                'name': 'Control',
                'directory': 'data/control',
                'description': 'Control samples',
                'channels': [
                    {
                        'name': 'DAPI',
                        'type': 'nuclear',
                        'purpose': 'segmentation',
                        'quantify': False,
                        'nuclear_only': True,
                        'wavelength': '405nm',
                        'source': 'single_file',
                        'filepath': 'dapi.tif'
                    },
                    {
                        'name': 'Target_Protein',
                        'type': 'cellular',
                        'purpose': 'quantification', 
                        'quantify': True,
                        'nuclear_only': False,
                        'wavelength': '488nm',
                        'source': 'single_file',
                        'filepath': 'target.tif'
                    }
                ]
            },
            {
                'name': 'Treatment',
                'directory': 'data/treatment',
                'description': 'Treated samples',
                'channels': [
                    {
                        'name': 'DAPI',
                        'type': 'nuclear',
                        'purpose': 'segmentation',
                        'quantify': False,
                        'nuclear_only': True,
                        'wavelength': '405nm',
                        'source': 'single_file',
                        'filepath': 'dapi.tif'
                    },
                    {
                        'name': 'Target_Protein',
                        'type': 'cellular',
                        'purpose': 'quantification',
                        'quantify': True,
                        'nuclear_only': False,
                        'wavelength': '488nm',
                        'source': 'single_file',
                        'filepath': 'target.tif'
                    }
                ]
            }
        ],
        # Analysis parameters
        'cellpose_model': 'cyto2',
        'cell_diameter': 30,
        'use_gpu': False,
        'background_method': 'mode',
        'flow_threshold': 0.4,
        'cellprob_threshold': 0.0
    }
    
    return config

def run_gui_application():
    """Run the main GUI application"""
    
    print("Starting CellQuantGUI application...")
    print("Features available:")
    print("- Multi-condition experiment setup")
    print("- Automatic channel detection and configuration") 
    print("- Cellpose integration for cell segmentation")
    print("- CTCF quantification with background correction")
    print("- Statistical analysis and comparison")
    print("- Export to CSV, images, and ROI files")
    print()
    
    try:
        app = MainApplication()
        app.run()
    except Exception as e:
        logging.error(f"GUI application failed: {e}")
        print(f"Error starting GUI: {e}")

def run_batch_analysis():
    """Run analysis in batch mode without GUI"""
    
    print("Running batch analysis...")
    
    # Create configuration
    config = create_batch_analysis_config()
    
    # Progress callback for monitoring
    def progress_callback(message, percentage):
        if percentage >= 0:
            print(f"[{percentage:3d}%] {message}")
        else:
            print(f"[ERR] {message}")
    
    try:
        # Initialize and run analysis pipeline
        pipeline = AnalysisPipeline(config, progress_callback)
        results = pipeline.run_analysis()
        
        print("\nAnalysis completed successfully!")
        print(f"Results saved to: {config['output_directory']}")
        print(f"Total conditions processed: {results['experiment_info']['total_conditions']}")
        print(f"Total images processed: {results['experiment_info']['total_images_processed']}")
        print(f"Analysis duration: {results['experiment_info']['duration_seconds']:.1f} seconds")
        
        return results
        
    except Exception as e:
        logging.error(f"Batch analysis failed: {e}")
        print(f"Batch analysis failed: {e}")
        return None

def run_configuration_example():
    """Demonstrate experiment configuration creation and saving"""
    
    print("Creating example experiment configuration...")
    
    try:
        # Create example experiment
        experiment = create_example_experiment_config()
        
        # Save configuration to file
        config_file = Path("configs/example_experiment.json")
        config_file.parent.mkdir(exist_ok=True)
        
        experiment.save_config(config_file)
        print(f"Configuration saved to: {config_file}")
        
        # Load and verify configuration
        loaded_experiment = ExperimentConfig.load_config(config_file)
        print(f"Configuration loaded successfully!")
        print(f"Experiment: {loaded_experiment.name}")
        print(f"Conditions: {len(loaded_experiment.conditions)}")
        
        for condition in loaded_experiment.conditions:
            print(f"  - {condition.name}: {len(condition.channels)} channels")
        
        return experiment
        
    except Exception as e:
        logging.error(f"Configuration example failed: {e}")
        print(f"Configuration example failed: {e}")
        return None

def demonstrate_channel_types():
    """Demonstrate different channel configuration scenarios"""
    
    print("\nChannel Configuration Examples:")
    print("=" * 50)
    
    # Example 1: Basic DAPI + Target protein
    print("\n1. Basic two-channel setup:")
    basic_channels = [
        {
            'name': 'DAPI',
            'type': 'nuclear',
            'purpose': 'segmentation',
            'quantify': False,
            'nuclear_only': True,
            'wavelength': '405nm'
        },
        {
            'name': 'Target_Protein',
            'type': 'cellular', 
            'purpose': 'quantification',
            'quantify': True,
            'nuclear_only': False,
            'wavelength': '488nm'
        }
    ]
    
    for ch in basic_channels:
        print(f"   {ch['name']}: {ch['type']} channel for {ch['purpose']}")
    
    # Example 2: DNA damage foci analysis
    print("\n2. DNA damage foci analysis:")
    foci_channels = [
        {
            'name': 'DAPI',
            'type': 'nuclear',
            'purpose': 'segmentation',
            'quantify': False,
            'nuclear_only': True,
            'wavelength': '405nm'
        },
        {
            'name': 'gamma_H2AX',
            'type': 'nuclear',
            'purpose': 'quantification',  # Could be extended to 'foci_detection'
            'quantify': True,
            'nuclear_only': True,
            'wavelength': '488nm'
        },
        {
            'name': '53BP1',
            'type': 'nuclear',
            'purpose': 'quantification',  # Could be extended to 'foci_detection'
            'quantify': True,
            'nuclear_only': True,
            'wavelength': '555nm'
        }
    ]
    
    for ch in foci_channels:
        print(f"   {ch['name']}: {ch['type']} channel for {ch['purpose']}")
        if ch['quantify']:
            print(f"      -> Will calculate CTCF (nuclear: {ch['nuclear_only']})")
    
    # Example 3: Mitochondrial analysis
    print("\n3. Mitochondrial analysis:")
    mito_channels = [
        {
            'name': 'DAPI',
            'type': 'nuclear',
            'purpose': 'segmentation',
            'quantify': False,
            'nuclear_only': True,
            'wavelength': '405nm'
        },
        {
            'name': 'MitoTracker',
            'type': 'cellular',
            'purpose': 'quantification',
            'quantify': True,
            'nuclear_only': False,
            'wavelength': '594nm'
        },
        {
            'name': 'Cytochrome_C',
            'type': 'cellular',
            'purpose': 'quantification',
            'quantify': True,
            'nuclear_only': False,
            'wavelength': '488nm'
        }
    ]
    
    for ch in mito_channels:
        print(f"   {ch['name']}: {ch['type']} channel for {ch['purpose']}")

def check_dependencies():
    """Check if required dependencies are installed"""
    
    print("Checking dependencies...")
    print("-" * 30)
    
    dependencies = {
        'tkinter': 'GUI framework',
        'numpy': 'Numerical computing',
        'pandas': 'Data analysis',
        'PIL': 'Image loading (Pillow)',
        'skimage': 'Image processing (scikit-image)',
        'cellpose': 'Cell segmentation'
    }
    
    missing_deps = []
    
    for dep, description in dependencies.items():
        try:
            if dep == 'tkinter':
                import tkinter
            elif dep == 'numpy':
                import numpy
            elif dep == 'pandas':
                import pandas
            elif dep == 'PIL':
                from PIL import Image
            elif dep == 'skimage':
                from skimage import io
            elif dep == 'cellpose':
                from cellpose import models
            
            print(f"✓ {dep:12} - {description}")
            
        except ImportError:
            print(f"✗ {dep:12} - {description} (MISSING)")
            missing_deps.append(dep)
    
    if missing_deps:
        print(f"\nMissing dependencies: {', '.join(missing_deps)}")
        print("\nInstall missing dependencies with:")
        
        if 'PIL' in missing_deps:
            print("  pip install pillow")
        if 'skimage' in missing_deps:
            print("  pip install scikit-image")
        if 'cellpose' in missing_deps:
            print("  pip install cellpose")
        if 'pandas' in missing_deps:
            print("  pip install pandas")
        if 'numpy' in missing_deps:
            print("  pip install numpy")
            
        return False
    
    print("\nAll dependencies are available!")
    return True

def main():
    """Main function to demonstrate different usage modes"""
    
    # Setup logging
    setup_logging()
    
    print("CellQuantGUI - Quantitative Microscopy Analysis Tool")
    print("=" * 55)
    print()
    print("This tool provides comprehensive analysis of microscopy images including:")
    print("• Automated cell segmentation using Cellpose")
    print("• CTCF (Corrected Total Cell Fluorescence) quantification")
    print("• Multi-condition experimental comparisons") 
    print("• Support for both GUI and batch processing modes")
    print("• Extensible architecture for foci detection and 3D/4D analysis")
    print()
    
    # Check dependencies
    if not check_dependencies():
        print("\nPlease install missing dependencies before proceeding.")
        return
    
    # Demonstrate channel configurations
    demonstrate_channel_types()
    
    print("\nUsage Options:")
    print("-" * 20)
    print("1. GUI Mode - Full interactive interface")
    print("2. Batch Mode - Programmatic analysis")  
    print("3. Configuration Example - Setup demonstration")
    print("4. Exit")
    
    while True:
        try:
            choice = input("\nSelect option (1-4): ").strip()
            
            if choice == '1':
                print("\nStarting GUI mode...")
                run_gui_application()
                break
                
            elif choice == '2':
                print("\nStarting batch analysis...")
                results = run_batch_analysis()
                if results:
                    print("Batch analysis completed successfully!")
                break
                
            elif choice == '3':
                print("\nRunning configuration example...")
                config = run_configuration_example()
                if config:
                    print("Configuration example completed!")
                
            elif choice == '4':
                print("Exiting...")
                break
                
            else:
                print("Invalid choice. Please select 1-4.")
                
        except KeyboardInterrupt:
            print("\n\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()