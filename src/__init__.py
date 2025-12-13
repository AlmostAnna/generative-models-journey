"""Generative Models Journey Package."""

__version__ = "0.1.0"

# Auto-setup environment
from .setup_env import setup_environment

setup_environment()

# Convenience imports
try:
    from .diagnostics import plot_diagnostics_grid_az
    from .diffusion import ScoreModel, ddim_sample, get_alpha_bar, positional_encoding
    from .ebm import (
        EnergyModel,
        TGMixtureEnergy,
        annealed_langevin_sampler,
        sample_two_mixture,
    )
    from .mcmc import (
        banana_grad_logp,
        banana_logp,
        banana_model,
        compute_validation_metrics,
        generate_true_banana_samples,
        grad_logp_circular,
        run_experiment,
        run_hmc_adaptive,
        run_mala,
        run_nuts,
        run_smc,
        two_moons_logp_circular,
        two_moons_model,
    )
    from .training import train_ebm_model, train_model
    from .utils import find_project_root, get_data
    from .visualization import (
        get_color_palette,
        plot_chain_trajectories,
        plot_comparison,
        plot_energy_landscape_unified,
        plot_historgam,
        plot_real_and_generated,
        plot_real_data,
        plot_true_contour_and_samples,
    )
except ImportError as e:
    print(f"\n src package import failed: {e}")
    pass  # Handle cases where dependencies aren't available yet

__all__ = [
    "get_data",
    "find_project_root",
    "plot_real_data",
    "get_alpha_bar",
    "positional_encoding",
    "ScoreModel",
    "ddim_sample",
    "EnergyModel",
    "annealed_langevin_sampler",
    "TGMixtureEnergy",
    "sample_two_mixture",
    "two_moons_logp_circular",
    "grad_logp_circular",
    "two_moons_model",
    "banana_logp",
    "banana_grad_logp",
    "banana_model",
    "generate_true_banana_samples",
    "run_smc",
    "run_mala",
    "run_hmc_adaptive",
    "run_nuts",
    "compute_validation_metrics",
    "run_experiment",
    "get_color_palette",
    "plot_real_and_generated",
    "plot_energy_landscape_unified",
    "plot_true_contour_and_samples",
    "plot_chain_trajectories",
    "plot_historgam",
    "plot_comparison",
    "train_model",
    "train_ebm_model",
    "plot_diagnostics_grid_az",
]
