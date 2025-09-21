# src/setup_env.py
import os
import sys
from pathlib import Path

def setup_environment():
    """Setup environment for both local and cloud environments."""
    
    # Fix OpenMP issues (common on macOS)
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
    
    # Add src to path if not already there
    src_path = str(Path(__file__).parent.parent.absolute())
    if src_path not in sys.path:
        sys.path.insert(0, src_path)
    
    print(f"   Environment setup complete")
    print(f"   Working directory: {os.getcwd()}")
    print(f"   Src path: {src_path}")

def safe_import(module_name, package_name=None):
    """Safely import module with error handling."""
    try:
        if package_name:
            module = __import__(package_name, fromlist=[module_name])
        else:
            module = __import__(module_name)
        return module
    except ImportError as e:
        print(f" Failed to import {module_name}: {e}")
        raise

# Auto-setup when module is imported
setup_environment()

