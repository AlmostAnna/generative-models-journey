"""
Energy-Based Model (EBM) components for the generative models journey.

This module defines the core components for training and sampling from
Energy-Based Models, including the energy function, Langevin samplers,
and training objectives.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm


# EBM model
class EnergyModel(nn.Module):
    """
    A simple Multi-Layer Perceptron (MLP) used as an Energy-Based Model.

    The model outputs an energy value for an input x. Lower energy values
    correspond to higher probability under the model's distribution p(x) ∝ exp(-E(x)).
    """

    def __init__(self, input_dim: int = 2, hidden_dim: int = 64, output_dim: int = 1):
        """
        Initialize the EnergyModel.

        Args:
            input_dim (int): Dimensionality of the input data (e.g., 2 for 2D points). Defaults to 2.
            hidden_dim (int): Number of neurons in the hidden layers. Defaults to 64.
            output_dim (int): Dimensionality of the output (should be 1 for scalar energy). Defaults to 1.
        """
        super().__init__()
        self.net = nn.Sequential(
            spectral_norm(nn.Linear(input_dim, hidden_dim)),
            nn.ReLU(),
            spectral_norm(nn.Linear(hidden_dim, hidden_dim)),
            nn.ReLU(),
            spectral_norm(nn.Linear(hidden_dim, output_dim)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the energy for input x.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_dim).

        Returns:
            torch.Tensor: Energy values of shape (batch_size, output_dim).
        """
        return self.net(x).squeeze(-1)  # scalar energy


# Annealed Langevin Sampler
def annealed_langevin_sampler(
    model: EnergyModel,
    x_init: torch.Tensor,
    n_steps: int = 100,
    noise_scale: float = 1.0,
    burn_in: int = 50,
):
    """Perform Annealed Langevin dynamics sampling with variable step size and noise.

    This sampler uses a schedule for step size and noise, often starting with larger
    steps/noise and annealing (decreasing) them over time, which can help escape local
    minima and improve mixing compared to standard Langevin dynamics.

    Args:
        ebm_model (EnergyModel): The trained EBM to sample from.
        x_init (torch.Tensor): Initial state for the sampler, shape (batch_size, input_dim).
        n_steps (int): Total number of Langevin steps. Defaults to 100.
        noise_scale (float): Initial scale factor for the noise. Defaults to 2.0.
        burn_in (int): Number of initial steps to discard (burn-in period). Defaults to 50.

    Returns:
        torch.Tensor: Samples collected *after* the burn-in period,
            shape (n_samples_after_burn_in, batch_size, input_dim).
            If n_steps <= burn_in, returns an empty tensor.
    """
    x = x_init.detach().clone()  # Start without gradient history
    samples = []

    for i in range(n_steps):
        # Just Langevin: constant step, constant noise
        # step_size = 0.1
        # current_noise_std = noise_scale

        # Annealed Langevin: variate noise based on the step
        # Variate step size
        t = i / n_steps
        step_size = 0.1 * (1 - t)
        current_noise_std = noise_scale * (1 - t)

        # Enable gradient tracking for this step
        with torch.enable_grad():
            x = x.requires_grad_(True)

            energy = model(x).sum()

            # Compute gradient
            grad = torch.autograd.grad(energy, x, retain_graph=False)[0]  # Free memory

        # Update
        noise = torch.randn_like(x) * np.sqrt(2 * step_size) * current_noise_std
        x = x.detach() - step_size * grad.detach() + noise  # Detach for next step

        if i >= burn_in:  # only save after burn-in
            samples.append(x.detach().clone())
    return torch.stack(samples)


# Two Gaussians Mixture Model
class TGMixtureEnergy(nn.Module):
    """
    A toy Energy-Based Model representing a mixture of two specific 2D Gaussians.

    This model defines an energy function corresponding to a distribution
    that is a mixture of two isotropic Gaussians centered at (-2, 0) and (2, 0).
    It's designed to demonstrate sampling challenges (like mode collapse or poor mixing)
    even when the energy function perfectly represents the target distribution.
    The energy is calculated as E(x) = -log(p(x)), where p(x) is the mixture density.
    """

    def __init__(self):
        """
        Initialize the Two-Gaussian Mixture Energy model.

        The centers of the Gaussians are fixed within the forward method.
        """
        super().__init__()

        # Centers are fixed: (-2, 0) and (2, 0)
        self.center1 = torch.tensor([-2.0, 0.0])
        self.center2 = torch.tensor([2.0, 0.0])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the energy for input points x under the two-Gaussian mixture model.

        The energy is derived from the negative log-likelihood of the mixture:
        p(x) = 0.5 * N(x | c1, I) + 0.5 * N(x | c2, I)
        E(x) = -log(p(x))
        where N(c, I) is an isotropic Gaussian centered at c with identity covariance.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 2).

        Returns:
            torch.Tensor: Energy values of shape (batch_size,).
                          Lower values correspond to higher probability regions (near centers).
        """
        # Calculate squared distances from x to each center
        d1 = torch.sum((x - self.center1) ** 2, dim=1)  # Distance to (-2, 0)
        d2 = torch.sum((x - self.center2) ** 2, dim=1)  # Distance to (2, 0)

        # Calculate the unnormalized probability density (sum of two Gaussian kernels)
        # Using exp(-0.5 * d) corresponds to the Gaussian exponent part
        unnormalized_prob = torch.exp(-0.5 * d1) + torch.exp(-0.5 * d2)

        # Add a small constant (1e-8) to prevent taking log(0), which would lead to inf/nan
        prob = unnormalized_prob + 1e-8

        # Return the negative log probability, which is the energy
        return -torch.log(prob)


def sample_two_mixture(n_samples=1000):
    """Sample from Two Gaussians Mixture Model."""
    modes = torch.randint(0, 2, (n_samples,))
    centers = torch.tensor([[-2.0, 0.0], [2.0, 0.0]])  # Now distance = 4
    x = centers[modes]
    x += 0.5 * torch.randn_like(x)  # Smaller noise: σ = 0.5
    return x
