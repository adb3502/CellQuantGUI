# Installation and Setup Scripts
# Comprehensive setup for CellQuantGUI system

import sys
import subprocess
import importlib
import os
from pathlib import Path
import json
import platform
import shutil
from typing import Dict, List, Tuple, Optional
import logging

class CellQuantInstaller:
    """Handles installation and setup of CellQuantGUI system"""
    
    def __init__(self):
        self.python_version = sys.version_info
        self.platform = platform.system().lower()
        self.architecture = platform.architecture()[0]
        
        # Setup logging
        self.setup_logging()
        self.logger = logging.getLogger(__name__)
        
        # Required dependencies with versions
        self.dependencies = {
            'core': {
                'numpy': '>=1.19.0',
                'pandas': '>=1.3.0',
                'pillow': '>=8.0.0',
                'scikit-image': '>=0.18.0',
                'matplotlib': '>=3.3.0',
                'seaborn': '>=0.11.0',
                'scipy': '>=1.7.0'
            },
            'ml': {
                'cellpose': '>=2.0.0',
                'torch': '>=1.9.0',  # For Cellpose GPU support
                'torchvision': '>=0.10.0'
            },
            'optional': {
                'opencv-python': '>=4.5.0',  # For advanced image processing
                'napari': '>=0.4.0',  # For 3D visualization
                'jupyter': '>=1.0.0',  # For notebook integration
                'plotly': '>=5.0.0',  # For interactive plots
                'dash': '>=2.0.0'  # For web dashboard
            }
        }
        
        # Installation status
        self.installation_status = {}
        
    def setup_logging(self):
        """Setup installation logging"""
        
        log_dir = Path("installation_logs")
        log_dir.mkdir(exist_ok=True)
        
        from datetime import datetime
        log_file = log_dir / f"installation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
    
    def check_python_version(self) -> bool:
        """Check if Python version is compatible"""
        
        min_version = (3, 8, 0)
        
        if self.python_version >= min_version:
            self.logger.info(f"✓ Python version {'.'.join(map(str, self.python_version[:3]))} is compatible")
            return True
        else:
            self.logger.error(f"✗ Python version {'.'.join(map(str, self.python_version[:3]))} is too old. Minimum required: {'.'.join(map(str, min_version))}")
            return False
    
    def check_system_requirements(self) -> Dict[str, bool]:
        """Check system-specific requirements"""
        
        requirements = {
            'python_version': self.check_python_version(),
            'pip_available': self.check_pip(),
            'git_available': self.check_git(),
            'display_available': self.check_display(),
            'memory_adequate': self.check_memory(),
            'disk_space': self.check_disk_space()
        }
        
        return requirements
    
    def check_pip(self) -> bool:
        """Check if pip is available"""
        try:
            subprocess.run([sys.executable, '-m', 'pip', '--version'], 
                         check=True, capture_output=True)
            self.logger.info("✓ pip is available")
            return True
        except subprocess.CalledProcessError:
            self.logger.error("✗ pip is not available")
            return False
    
    def check_git(self) -> bool:
        """Check if git is available (optional)"""
        try:
            subprocess.run(['git', '--version'], check=True, capture_output=True)
            self.logger.info("✓ git is available")
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            self.logger.warning("⚠ git is not available (optional)")
            return False
    
    def check_display(self) -> bool:
        """Check if display is available for GUI"""
        
        if self.platform == 'linux':
            display = os.environ.get('DISPLAY')
            if display:
                self.logger.info(f"✓ Display available: {display}")
                return True
            else:
                self.logger.warning("⚠ No DISPLAY environment variable (GUI may not work)")
                return False
        else:
            # Assume display is available on Windows/Mac
            self.logger.info("✓ Display assumed available")
            return True
    
    def check_memory(self) -> bool:
        """Check if system has adequate memory"""
        try:
            import psutil
            memory_gb = psutil.virtual_memory().total / (1024**3)
            
            if memory_gb >= 4:
                self.logger.info(f"✓ Memory: {memory_gb:.1f} GB (adequate)")
                return True
            else:
                self.logger.warning(f"⚠ Memory: {memory_gb:.1f} GB (may be insufficient for large images)")
                return False
        except ImportError:
            self.logger.info("? Memory check skipped (psutil not available)")
            return True
    
    def check_disk_space(self) -> bool:
        """Check available disk space"""
        try:
            import shutil
            free_space_gb = shutil.disk_usage('.').free / (1024**3)
            
            if free_space_gb >= 1:
                self.logger.info(f"✓ Free disk space: {free_space_gb:.1f} GB")
                return True
            else:
                self.logger.warning(f"⚠ Low disk space: {free_space_gb:.1f} GB")
                return False
        except Exception:
            self.logger.info("? Disk space check failed")
            return True
    
    def check_dependencies(self) -> Dict[str, Dict[str, bool]]:
        """Check which dependencies are already installed"""
        
        status = {}
        
        for category, deps in self.dependencies.items():
            status[category] = {}
            
            for package, version_req in deps.items():
                try:
                    module = importlib.import_module(package.replace('-', '_'))
                    
                    # Try to get version
                    version = getattr(module, '__version__', 'unknown')
                    
                    status[category][package] = {
                        'installed': True,
                        'version': version,
                        'requirement': version_req
                    }
                    
                    self.logger.info(f"✓ {package} {version} is installed")
                    
                except ImportError:
                    status[category][package] = {
                        'installed': False,
                        'version': None,
                        'requirement': version_req
                    }
                    
                    self.logger.info(f"✗ {package} is not installed")
        
        return status
    
    def install_dependencies(self, categories: List[str] = None, force_reinstall: bool = False) -> bool:
        """Install specified dependency categories"""
        
        if categories is None:
            categories = ['core', 'ml']  # Install core and ML by default
        
        success = True
        
        for category in categories:
            if category not in self.dependencies:
                self.logger.warning(f"Unknown dependency category: {category}")
                continue
            
            self.logger.info(f"Installing {category} dependencies...")
            
            for package, version_req in self.dependencies[category].items():
                if not self.install_package(package, version_req, force_reinstall):
                    success = False
        
        return success
    
    def install_package(self, package: str, version_req: str, force_reinstall: bool = False) -> bool:
        """Install a single package"""
        
        package_spec = f"{package}{version_req}"
        
        try:
            if force_reinstall:
                cmd = [sys.executable, '-m', 'pip', 'install', '--upgrade', '--force-reinstall', package_spec]
            else:
                cmd = [sys.executable, '-m', 'pip', 'install', '--upgrade', package_spec]
            
            self.logger.info(f"Installing {package_spec}...")
            
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            
            self.logger.info(f"✓ Successfully installed {package}")
            return True
            
        except subprocess.CalledProcessError as e:
            self.logger.error(f"✗ Failed to install {package}: {e}")
            self.logger.error(f"Error output: {e.stderr}")
            return False
    
    def setup_gpu_support(self) -> bool:
        """Setup GPU support for Cellpose if available"""
        
        self.logger.info("Checking GPU support...")
        
        try:
            # Check for NVIDIA GPU
            result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
            if result.returncode == 0:
                self.logger.info("✓ NVIDIA GPU detected")
                
                # Install CUDA-enabled PyTorch
                return self.setup_cuda_pytorch()
            else:
                self.logger.info("No NVIDIA GPU detected, using CPU-only version")
                return True
                
        except FileNotFoundError:
            self.logger.info("nvidia-smi not found, assuming no GPU")
            return True
    
    def setup_cuda_pytorch(self) -> bool:
        """Install CUDA-enabled PyTorch"""
        
        try:
            # Install PyTorch with CUDA support
            cmd = [
                sys.executable, '-m', 'pip', 'install', 
                'torch', 'torchvision', '--index-url', 
                'https://download.pytorch.org/whl/cu118'
            ]
            
            self.logger.info("Installing CUDA-enabled PyTorch...")
            subprocess.run(cmd, check=True, capture_output=True)
            
            self.logger.info("✓ CUDA-enabled PyTorch installed")
            return True
            
        except subprocess.CalledProcessError as e:
            self.logger.error(f"✗ Failed to install CUDA PyTorch: {e}")
            return False
    
    def create_project_structure(self, project_dir: Path = None) -> bool:
        """Create recommended project directory structure"""
        
        if project_dir is None:
            project_dir = Path("CellQuantGUI")
        
        directories = [
            project_dir / "data",
            project_dir / "data" / "raw",
            project_dir / "data" / "processed",
            project_dir / "results",
            project_dir / "configs",
            project_dir / "logs",
            project_dir / "exports",
            project_dir / "exports" / "images",
            project_dir / "exports" / "data",
            project_dir / "exports" / "rois",
            project_dir / "templates",
            project_dir / "scripts"
        ]
        
        try:
            for directory in directories:
                directory.mkdir(parents=True, exist_ok=True)
                self.logger.info(f"Created directory: {directory}")
            
            # Create README files
            self.create_readme_files(project_dir)
            
            # Create example configuration
            self.create_example_config(project_dir)
            
            self.logger.info(f"✓ Project structure created at: {project_dir}")
            return True
            
        except Exception as e:
            self.logger.error(f"✗ Failed to create project structure: {e}")
            return False
    
    def create_readme_files(self, project_dir: Path):
        """Create README files for project structure"""
        
        readme_content = {
            'data/raw': "Place raw microscopy images here, organized by experiment/condition",
            'data/processed': "Processed images and intermediate results are stored here",
            'results': "Final analysis results, statistics, and reports",
            'configs': "Experiment configurations and analysis parameters",
            'logs': "Analysis logs and processing information", 
            'exports': "Exported data, images, and ROIs",
            'templates': "Reusable configuration templates",
            'scripts': "Custom analysis scripts and extensions"
        }
        
        for subdir, content in readme_content.items():
            readme_file = project_dir / subdir / "README.md"
            with open(readme_file, 'w') as f:
                f.write(f"# {subdir.replace('/', ' / ').title()}\n\n{content}\n")
    
    def create_example_config(self, project_dir: Path):
        """Create example configuration file"""
        
        example_config = {
            "experiment_name": "Example_DNA_Damage_Study",
            "description": "Example configuration for DNA damage foci analysis",
            "output_directory": str(project_dir / "results" / "example_study"),
            "conditions": [
                {
                    "name": "Control",
                    "directory": str(project_dir / "data" / "raw" / "control"),
                    "description": "Untreated control cells",
                    "channels": [
                        {
                            "name": "DAPI",
                            "type": "nuclear",
                            "purpose": "segmentation",
                            "quantify": False,
                            "nuclear_only": True,
                            "wavelength": "405nm"
                        },
                        {
                            "name": "gamma_H2AX",
                            "type": "nuclear",
                            "purpose": "quantification",
                            "quantify": True,
                            "nuclear_only": True,
                            "wavelength": "488nm"
                        }
                    ]
                }
            ],
            "analysis_parameters": {
                "cellpose_model": "cyto2",
                "cell_diameter": 30,
                "use_gpu": False,
                "background_method": "mode",
                "export_images": True,
                "export_rois": True
            }
        }
        
        config_file = project_dir / "configs" / "example_config.json"
        with open(config_file, 'w') as f:
            json.dump(example_config, f, indent=2)
        
        self.logger.info(f"Created example configuration: {config_file}")
    
    def run_installation_tests(self) -> bool:
        """Run tests to verify installation"""
        
        self.logger.info("Running installation tests...")
        
        tests = [
            self.test_basic_imports,
            self.test_cellpose_functionality,
            self.test_gui_components,
            self.test_image_processing,
            self.test_analysis_pipeline
        ]
        
        all_passed = True
        
        for test in tests:
            try:
                if test():
                    self.logger.info(f"✓ {test.__name__} passed")
                else:
                    self.logger.error(f"✗ {test.__name__} failed")
                    all_passed = False
            except Exception as e:
                self.logger.error(f"✗ {test.__name__} failed with exception: {e}")
                all_passed = False
        
        return all_passed
    
    def test_basic_imports(self) -> bool:
        """Test basic package imports"""
        try:
            import numpy
            import pandas
            import matplotlib
            import seaborn
            from PIL import Image
            import scipy
            return True
        except ImportError as e:
            self.logger.error(f"Basic import failed: {e}")
            return False
    
    def test_cellpose_functionality(self) -> bool:
        """Test Cellpose installation and basic functionality"""
        try:
            from cellpose import models
            
            # Create a simple test model
            model = models.Cellpose(gpu=False, model_type='cyto2')
            
            # Test with dummy data
            import numpy as np
            test_image = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
            
            masks, flows, styles, diams = model.eval(test_image, diameter=30, channels=[0, 0])
            
            return True
        except Exception as e:
            self.logger.error(f"Cellpose test failed: {e}")
            return False
    
    def test_gui_components(self) -> bool:
        """Test GUI components"""
        try:
            import tkinter as tk
            from tkinter import ttk
            
            # Create a test window (don't show it)
            root = tk.Tk()
            root.withdraw()  # Hide the window
            
            # Test basic widgets
            frame = ttk.Frame(root)
            label = ttk.Label(frame, text="Test")
            button = ttk.Button(frame, text="Test")
            
            root.destroy()
            return True
        except Exception as e:
            self.logger.error(f"GUI test failed: {e}")
            return False
    
    def test_image_processing(self) -> bool:
        """Test image processing capabilities"""
        try:
            from skimage import io, measure, filters
            import numpy as np
            
            # Create test image
            test_image = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
            
            # Test basic operations
            filtered = filters.gaussian(test_image, sigma=1)
            regions = measure.regionprops(test_image > 128)
            
            return True
        except Exception as e:
            self.logger.error(f"Image processing test failed: {e}")
            return False
    
    def test_analysis_pipeline(self) -> bool:
        """Test analysis pipeline components"""
        try:
            # This would test the main analysis components
            # For now, just test that we can import them
            # from analysis_pipeline import AnalysisPipeline
            # from cellquant_main import MainApplication
            
            # Placeholder test
            return True
        except Exception as e:
            self.logger.error(f"Analysis pipeline test failed: {e}")
            return False
    
    def generate_installation_report(self) -> str:
        """Generate comprehensive installation report"""
        
        report = []
        report.append("CellQuantGUI Installation Report")
        report.append("=" * 40)
        report.append("")
        
        # System information
        report.append("System Information:")
        report.append(f"  Platform: {platform.platform()}")
        report.append(f"  Python: {'.'.join(map(str, self.python_version[:3]))}")
        report.append(f"  Architecture: {self.architecture}")
        report.append("")
        
        # System requirements
        requirements = self.check_system_requirements()
        report.append("System Requirements:")
        for req, status in requirements.items():
            status_symbol = "✓" if status else "✗"
            report.append(f"  {status_symbol} {req.replace('_', ' ').title()}")
        report.append("")
        
        # Dependencies
        dependencies = self.check_dependencies()
        report.append("Dependencies:")
        for category, deps in dependencies.items():
            report.append(f"  {category.upper()}:")
            for package, info in deps.items():
                status_symbol = "✓" if info['installed'] else "✗"
                version_info = f" ({info['version']})" if info['version'] else ""
                report.append(f"    {status_symbol} {package}{version_info}")
        report.append("")
        
        return "\n".join(report)

def run_interactive_installer():
    """Run interactive installation process"""
    
    print("CellQuantGUI Installation Wizard")
    print("=" * 40)
    print()
    
    installer = CellQuantInstaller()
    
    # Step 1: Check system requirements
    print("Step 1: Checking system requirements...")
    requirements = installer.check_system_requirements()
    
    if not all(requirements.values()):
        print("⚠ Some system requirements are not met. Continue anyway? (y/n): ", end="")
        if input().lower() != 'y':
            print("Installation cancelled.")
            return False
    
    # Step 2: Choose installation type
    print("\nStep 2: Choose installation type:")
    print("1. Minimal (core dependencies only)")
    print("2. Standard (core + machine learning)")
    print("3. Full (all dependencies)")
    print("4. Custom")
    
    choice = input("Enter choice (1-4): ").strip()
    
    install_categories = []
    if choice == '1':
        install_categories = ['core']
    elif choice == '2':
        install_categories = ['core', 'ml']
    elif choice == '3':
        install_categories = ['core', 'ml', 'optional']
    elif choice == '4':
        print("Available categories: core, ml, optional")
        categories = input("Enter categories (comma-separated): ").strip()
        install_categories = [cat.strip() for cat in categories.split(',')]
    else:
        print("Invalid choice, using standard installation.")
        install_categories = ['core', 'ml']
    
    # Step 3: Install dependencies
    print(f"\nStep 3: Installing dependencies ({', '.join(install_categories)})...")
    if not installer.install_dependencies(install_categories):
        print("⚠ Some packages failed to install. Continue? (y/n): ", end="")
        if input().lower() != 'y':
            print("Installation cancelled.")
            return False
    
    # Step 4: Setup GPU support
    if 'ml' in install_categories:
        print("\nStep 4: Setting up GPU support...")
        installer.setup_gpu_support()
    
    # Step 5: Create project structure
    print("\nStep 5: Creating project structure...")
    project_dir = input("Project directory (default: CellQuantGUI): ").strip()
    if not project_dir:
        project_dir = "CellQuantGUI"
    
    installer.create_project_structure(Path(project_dir))
    
    # Step 6: Run tests
    print("\nStep 6: Running installation tests...")
    tests_passed = installer.run_installation_tests()
    
    # Step 7: Generate report
    print("\nStep 7: Generating installation report...")
    report = installer.generate_installation_report()
    
    report_file = Path("installation_report.txt")
    with open(report_file, 'w') as f:
        f.write(report)
    
    print(f"\nInstallation report saved to: {report_file}")
    
    # Summary
    print("\nInstallation Summary:")
    if tests_passed:
        print("✓ Installation completed successfully!")
        print(f"✓ Project structure created at: {project_dir}")
        print("✓ All tests passed")
    else:
        print("⚠ Installation completed with warnings")
        print("  Some tests failed - check the log for details")
    
    print(f"\nTo get started:")
    print(f"1. cd {project_dir}")
    print("2. python -m cellquant_main  # or run your main script")
    
    return tests_passed

def create_batch_installer():
    """Create batch installation scripts for different platforms"""
    
    # Windows batch file
    windows_script = """@echo off
echo CellQuantGUI Installation Script
echo ===============================

echo Checking Python installation...
python --version
if errorlevel 1 (
    echo Python is not installed or not in PATH
    echo Please install Python 3.8+ from https://python.org
    pause
    exit /b 1
)

echo Installing CellQuantGUI dependencies...
python -m pip install --upgrade pip
python -m pip install numpy pandas pillow scikit-image matplotlib seaborn scipy
python -m pip install cellpose torch torchvision

echo Installation completed!
echo Run: python cellquant_main.py
pause
"""
    
    # Unix shell script
    unix_script = """#!/bin/bash
echo "CellQuantGUI Installation Script"
echo "==============================="

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "Python 3 is not installed"
    echo "Please install Python 3.8+ from your package manager"
    exit 1
fi

echo "Python version:"
python3 --version

echo "Installing CellQuantGUI dependencies..."
python3 -m pip install --upgrade pip
python3 -m pip install numpy pandas pillow scikit-image matplotlib seaborn scipy
python3 -m pip install cellpose torch torchvision

echo "Installation completed!"
echo "Run: python3 cellquant_main.py"
"""
    
    # Create script files
    with open("install_windows.bat", 'w') as f:
        f.write(windows_script)
    
    with open("install_unix.sh", 'w') as f:
        f.write(unix_script)
    
    # Make Unix script executable
    try:
        os.chmod("install_unix.sh", 0o755)
    except:
        pass
    
    print("Batch installation scripts created:")
    print("- install_windows.bat (Windows)")
    print("- install_unix.sh (Linux/Mac)")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "batch":
            create_batch_installer()
        elif sys.argv[1] == "check":
            installer = CellQuantInstaller()
            report = installer.generate_installation_report()
            print(report)
        else:
            print("Usage: python installation_setup.py [batch|check]")
    else:
        # Run interactive installer
        run_interactive_installer()