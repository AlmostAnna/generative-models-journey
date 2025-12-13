"""
Diffusion Model Implementation and Sampling.

This module provides components for score-based diffusion probabilistic models.
It includes:
- ScoreModel: A neural network to estimate the score (gradient of log probability)
  for a given data point and timestep.
- ddim_sample: A function to perform Denoising Diffusion Implicit Modeling (DDIM)
  sampling, generating data samples from learned noise.

These components are primarily used for training and sampling from generative
models on 2D benchmark datasets like two-moons.
"""

import math

import torch
import torch.nn as nn


def get_alpha_bar(t: torch.Tensor) -> torch.Tensor:
    """
    Cosine noise schedule from DDIM (2020).

    Args:
        t: (B,) tensor of time values in [0, 1], where:
           t ≈ 0 → data, t ≈ 1 → noise

    Returns:
        (B, 1) tensor of alpha_bar(t)
    """
    s = 0.008
    return (torch.cos((t + s) / (1 + s) * torch.pi / 2) ** 2).unsqueeze(1)


def positional_encoding(t: torch.Tensor, dim=16) -> torch.Tensor:
    """
    Positional encoding for time t ∈ [0,1].

    Args:
        t: (B,) tensor of time values
        dim: embedding dimension (must be even)

    Returns:
        (B, dim) tensor
    """
    device = t.device
    B = t.size(0)
    pe = torch.zeros(B, dim, device=device)
    position = t.float().unsqueeze(-1)  # [B, 1]
    div_term = torch.exp(
        torch.arange(0, dim, 2, device=device).float() * (-math.log(10000.0) / dim)
    )
    div_term = div_term.view(1, -1)  # [1, dim//2]

    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe


class ScoreModel(nn.Module):
    """
    A neural network to predict the score (gradient of log probability).

    This model takes a 2D data point and a timestep, embeds the timestep,
    concatenates them, and passes the combined vector through a multi-layer perceptron
    to output the estimated score vector (same dimensionality as the input data).
    """

    def __init__(self, time_dim: int = 16):
        """
        Initialize the ScoreModel.

        Args:
            time_dim: The dimensionality of the initial positional encoding
                      for the timestep `t`. Defaults to 16.
        """
        super().__init__()
        self.time_dim = time_dim
        self.t_proj = nn.Sequential(
            nn.Linear(time_dim, 32), nn.ReLU(), nn.Linear(32, 32)
        )
        self.net = nn.Sequential(
            nn.Linear(2 + 32, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 2),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the ScoreModel.

        Predicts the score (negative gradient of log probability w.r.t. x)
        for the given data points `x` conditioned on timesteps `t`.

        Args:
            x: Input data points. Shape [B, 2] where B is the batch size.
               Represents coordinates in the 2D space (e.g., the two-moons space).
            t: Timesteps associated with each data point. Shape [B, ].
               Values are typically in the range [0, T] where T is the total
               number of diffusion steps.

        Returns:
            Estimated score vectors. Shape [B, 2], representing the predicted
            gradient direction for each input point `x` at its corresponding
            timestep `t`.
        """
        # t: [B], x: [B, 2]
        t_emb = positional_encoding(t, self.time_dim)  # [B, time_dim]
        t_emb = self.t_proj(t_emb)  # [B, 32]

        x_t = torch.cat([x, t_emb], dim=1)  # [B, 2+32]
        return self.net(x_t)


# Diffusion sampling
@torch.no_grad()
def ddim_sample(
    model: "ScoreModel", steps: int = 100, n: int = 1000, device: str = "cpu"
) -> torch.Tensor:
    """
    Denoising Diffusion Implicit Modeling (DDIM) sampling.

    This function iteratively denoises random noise over a specified number of
    timesteps to generate samples from the data distribution learned by the model.

    Args:
        model: The trained diffusion model (e.g., ScoreModel) used to predict
               the score (or noise) at each timestep. It must have a `forward`
               method compatible with the sampler (e.g., accepting (x_t, t)).
        steps: The total number of denoising steps. More steps generally lead
               to higher quality samples but take longer. Defaults to 100.
        n: The number of samples to generate in the batch. Defaults to 1000.
        device: The device ('cpu' or 'cuda') on which to perform the sampling.
                Defaults to 'cpu'.
        initial_noise: Optional initial noise tensor of shape [n, data_dim].
                       If provided, this tensor is used as the starting point
                       (x_T) for the denoising process. If None (default),
                       random noise is generated internally.

    Returns:
        A tensor of generated samples. Shape [n, data_dim], where `n` is the
        requested number of samples, and `data_dim` is the dimensionality of
        the data the model was trained on (e.g., [1000, 2] for 2D data).
    """
    model.eval()
    ts = torch.linspace(1.0, 0.001, steps).to(device)  # Move to device
    x = torch.randn(n, 2).to(device)

    for i in range(steps):
        # t = torch.ones(n).to(device) * ts[i]
        t = torch.ones(n, device=device) * ts[i]  # [n], on device
        # Predict noise
        noise_pred = model(x, t)

        # Compute alpha_bar
        alpha_bar = get_alpha_bar(t).squeeze(1)  # [n]

        # Predict x0
        x0_pred = (x - torch.sqrt(1 - alpha_bar).unsqueeze(1) * noise_pred) / (
            torch.sqrt(alpha_bar).unsqueeze(1) + 1e-8
        )

        # Clip to avoid explosion
        x0_pred = torch.clamp(x0_pred, -3.5, 3.5)

        # If we're at the last step, return x0
        if i == steps - 1:
            x = x0_pred
            break

        # Next t
        t_next = torch.ones(n, device=device) * ts[i + 1]
        alpha_bar_next = get_alpha_bar(t_next).squeeze(1)

        # Deterministic step:
        # x_{t-1} = sqrt(abar_prev) * x0 + sqrt(1-abar_prev) * noise_pred
        # Recover x0 from noisy x
        x = (
            torch.sqrt(alpha_bar_next).unsqueeze(1) * x0_pred
            + torch.sqrt(1 - alpha_bar_next).unsqueeze(1) * noise_pred
        )

    model.train()
    return x
