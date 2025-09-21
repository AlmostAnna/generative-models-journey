"""
Generative Journey Package
"""

__version__ = "0.1.0"

# Auto-setup environment
from .setup_env import setup_environment
setup_environment()

# Convenience imports
try:
    from .utils import get_data, plot_real_data
    from .diffusion import get_alpha_bar, positional_encoding, ScoreModel, ddim_sample
except ImportError as e:
    print(f"\n src package import failed: {e}")
    pass  # Handle cases where dependencies aren't available yet

__all__ = [
    'get_data',
    'plot_real_data',
    'get_alpha_bar',
    'positional_encoding',
    'ScoreModel',
    'ddim_sample',
]