#!/usr/bin/env python3
"""
Simple launcher script for the 2D to 3D Converter GUI application.
This script checks dependencies and launches the GUI with proper error handling.
"""

import sys
import subprocess
import importlib


def check_dependencies():
    """Check if all required dependencies are installed."""
    required_modules = [
        'tkinter',
        'cv2',
        'numpy',
        'PIL',
        'open3d'
    ]
    
    missing_modules = []
    
    for module in required_modules:
        try:
            if module == 'cv2':
                importlib.import_module('cv2')
            elif module == 'PIL':
                importlib.import_module('PIL')
            elif module == 'open3d':
                importlib.import_module('open3d')
            else:
                importlib.import_module(module)
        except ImportError:
            missing_modules.append(module)
    
    return missing_modules


def install_dependencies():
    """Install missing dependencies."""
    print("Installing missing dependencies...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Dependencies installed successfully!")
        return True
    except subprocess.CalledProcessError:
        print("❌ Failed to install dependencies automatically.")
        print("Please run: pip install -r requirements.txt")
        return False


def main():
    """Main launcher function."""
    print("🚀 2D to 3D Converter GUI Launcher")
    print("=" * 40)
    
    # Check dependencies
    print("Checking dependencies...")
    missing_modules = check_dependencies()
    
    if missing_modules:
        print(f"❌ Missing modules: {', '.join(missing_modules)}")
        print("\nInstalling dependencies...")
        
        if not install_dependencies():
            print("\n❌ Please install dependencies manually and try again.")
            input("Press Enter to exit...")
            return
        
        # Check again after installation
        missing_modules = check_dependencies()
        if missing_modules:
            print(f"❌ Still missing modules: {', '.join(missing_modules)}")
            print("Please install them manually.")
            input("Press Enter to exit...")
            return
    
    print("✅ All dependencies are available!")
    
    # Launch GUI
    print("\n🎯 Launching GUI application...")
    try:
        from gui_app import main as gui_main
        gui_main()
    except ImportError as e:
        print(f"❌ Failed to import GUI: {e}")
        print("Make sure gui_app.py is in the current directory.")
        input("Press Enter to exit...")
    except Exception as e:
        print(f"❌ GUI error: {e}")
        print("Please check the error and try again.")
        input("Press Enter to exit...")


if __name__ == "__main__":
    main()
