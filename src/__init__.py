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
    from .ebm import annealed_langevin_sampler, EnergyModel, TGMixtureEnergy, sample_two_mixture
    from .mcmc import two_moons_logp_circular, grad_logp_circular, two_moons_model, banana_logp, banana_grad_logp, banana_model, generate_true_banana_samples, run_smc, run_mala, run_hmc_adaptive, run_nuts, compute_validation_metrics, run_experiment
    from .visualization import plot_energy_landscape_unified, plot_true_contour_and_samples
    from .training import train_model, train_ebm_model
    from .diagnostics import plot_diagnostics_grid_az
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
    'EnergyModel',
    'annealed_langevin_sampler',
    'TGMixtureEnergy',
    'sample_two_mixture',
    'two_moons_logp_circular',
    'grad_logp_circular',
    'two_moons_model',
    'banana_logp',
    'banana_grad_logp',
    'banana_model',
    'generate_true_banana_samples',
    'run_smc',
    'run_mala',
    'run_hmc_adaptive',
    'run_nuts',
    'compute_validation_metrics',
    'run_experiment',
    'plot_energy_landscape_unified',
    'plot_true_contour_and_samples',
    'train_model',
    'train_ebm_model',
    'plot_diagnostics_grid_az',
]