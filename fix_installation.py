#!/usr/bin/env python3
"""
Quick fix script for CellQuantGUI installation issues
Run this after the main installation to fix common problems
"""

import subprocess
import sys
import logging
from pathlib import Path

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def install_missing_packages():
    """Install missing required packages"""
    print("🔧 Installing missing packages...")
    
    required_packages = [
        'pillow>=8.0.0',
        'scikit-image>=0.18.0', 
        'opencv-python>=4.5.0'
    ]
    
    for package in required_packages:
        try:
            print(f"Installing {package}...")
            subprocess.run([sys.executable, '-m', 'pip', 'install', package], 
                         check=True, capture_output=True)
            print(f"✅ {package} installed successfully")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install {package}: {e}")

def check_tkinter():
    """Check tkinter availability and provide instructions"""
    print("\n🖥️ Checking GUI support...")
    
    try:
        import tkinter
        print("✅ tkinter is available")
        return True
    except ImportError:
        print("❌ tkinter not available")
        print("\n🔧 To fix tkinter on Linux:")
        print("Ubuntu/Debian: sudo apt-get install python3-tk")
        print("CentOS/RHEL: sudo yum install tkinter tk-devel")
        print("Fedora: sudo dnf install tkinter tk-devel")
        return False

def test_cellpose():
    """Test Cellpose with updated API"""
    print("\n🧬 Testing Cellpose...")
    
    try:
        import cellpose
        from cellpose import models
        
        print(f"Cellpose version: {cellpose.__version__}")
        
        # Test model creation with new API
        if hasattr(models, 'CellposeModel'):
            model = models.CellposeModel(gpu=False, model_type='cyto2')
            print("✅ Cellpose 4.x API working")
        elif hasattr(models, 'Cellpose'):
            model = models.Cellpose(gpu=False, model_type='cyto2') 
            print("✅ Cellpose 2.x/3.x API working")
        else:
            print("⚠️ Unknown Cellpose API - may need manual fixes")
        
        return True
        
    except Exception as e:
        print(f"❌ Cellpose test failed: {e}")
        return False

def test_basic_functionality():
    """Test basic functionality"""
    print("\n🧪 Testing basic functionality...")
    
    # Test imports
    test_imports = [
        ('numpy', 'np'),
        ('pandas', 'pd'), 
        ('matplotlib.pyplot', 'plt'),
        ('PIL', 'Image'),
        ('skimage', 'io')
    ]
    
    all_good = True
    
    for module, alias in test_imports:
        try:
            if alias:
                exec(f"import {module} as {alias}")
            else:
                exec(f"import {module}")
            print(f"✅ {module}")
        except ImportError as e:
            print(f"❌ {module}: {e}")
            all_good = False
    
    return all_good

def create_simple_launcher():
    """Create a simple launcher script"""
    print("\n📝 Creating launcher script...")
    
    launcher_code = '''#!/usr/bin/env python3
"""
Simple CellQuantGUI Launcher
"""

import sys
import os
from pathlib import Path

def main():
    print("🔬 Starting CellQuantGUI...")
    
    try:
        # Add current directory to path
        sys.path.insert(0, str(Path(__file__).parent))
        
        # Try to import and run main application
        from cellquant_main import MainApplication
        
        print("✅ Main application imported successfully")
        app = MainApplication()
        print("🚀 Launching GUI...")
        app.run()
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("\\n🔧 Troubleshooting:")
        print("1. Make sure all files are in the same directory")
        print("2. Run: python fix_installation.py")
        print("3. Check that all dependencies are installed")
        
    except Exception as e:
        print(f"❌ Error starting application: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
'''
    
    launcher_file = Path("launch_cellquant.py")
    with open(launcher_file, 'w') as f:
        f.write(launcher_code)
    
    # Make executable on Unix systems
    try:
        import stat
        launcher_file.chmod(launcher_file.stat().st_mode | stat.S_IEXEC)
    except:
        pass
    
    print(f"✅ Created launcher: {launcher_file}")

def main():
    print("🔧 CellQuantGUI Installation Fix")
    print("=" * 40)
    
    setup_logging()
    
    # Fix missing packages
    install_missing_packages()
    
    # Check GUI support
    tkinter_ok = check_tkinter()
    
    # Test Cellpose
    cellpose_ok = test_cellpose()
    
    # Test basic functionality  
    basic_ok = test_basic_functionality()
    
    # Create launcher
    create_simple_launcher()
    
    # Final status
    print("\\n" + "=" * 40)
    print("🎯 Installation Fix Summary:")
    print(f"GUI Support (tkinter): {'✅' if tkinter_ok else '❌'}")
    print(f"Cellpose: {'✅' if cellpose_ok else '⚠️'}")
    print(f"Basic functionality: {'✅' if basic_ok else '❌'}")
    
    if tkinter_ok and basic_ok:
        print("\\n🚀 Ready to start!")
        print("Run: python launch_cellquant.py")
    else:
        print("\\n⚠️ Some issues remain - check messages above")
    
    print("\\n📁 Your project structure:")
    project_files = [
        "cellquant_main.py",
        "analysis_pipeline.py", 
        "launch_cellquant.py",
        "data/",
        "results/",
        "configs/"
    ]
    
    for file in project_files:
        path = Path(file)
        if path.exists():
            print(f"✅ {file}")
        else:
            print(f"❌ {file} (missing)")

if __name__ == "__main__":
    main()