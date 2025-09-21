# test_setup.py
def test_universal_import():
    """Test that imports work in any environment."""
    try:
        # Try package import
        from src.utils import get_data
        print("Package import successful")
    except ImportError:
        # Try direct import
        import sys
        import os
        sys.path.append("src")
        from utils import get_data
        print(" Direct import successful")
    
    # Test function
    data = get_data(100)
    print(f"Function test successful: {data.shape}")

if __name__ == "__main__":
    test_universal_import()

