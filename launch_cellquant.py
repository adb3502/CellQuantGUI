#!/usr/bin/env python3
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
        print("\n🔧 Troubleshooting:")
        print("1. Make sure all files are in the same directory")
        print("2. Run: python fix_installation.py")
        print("3. Check that all dependencies are installed")
        
    except Exception as e:
        print(f"❌ Error starting application: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
