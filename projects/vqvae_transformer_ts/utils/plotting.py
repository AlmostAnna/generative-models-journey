"""
Plotting functions.

This module contains implementation of plotting functions used
in the project.
"""

import os
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.plot_style import colors  # noqa: F401 # Side effect: initializes plot style


def plot_continuous_samples(
    reconstructions: np.ndarray,
    n_samples: int = 10,
    save_path: Optional[str] = None,
) -> None:
    """
    Plot continuous time series samples.

    Args:
        reconstructions: Array of shape (n_samples, T, D) in the original scale.
        n_samples: Number of samples to plot.
        save_path: Path to save the plot image.
    """
    assert reconstructions.shape[0] >= n_samples

    plt.figure(figsize=(10, 2 * n_samples))

    for i in range(n_samples):
        plt.subplot(n_samples, 1, i + 1)
        plt.plot(reconstructions[i, :, 0], "o-")
        plt.title(f"Continuous Sample {i+1}")
        plt.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
    plt.close()


def plot_vqvae_ts_samples(
    reconstructions: np.ndarray,
    token_sequences: list[torch.Tensor],
    n_samples: int = 10,
    save_path: Optional[str] = None,
) -> None:
    """
    Plot VQVAE time series samples.

    Args:
        reconstructions: Array of shape (n_samples, T, D) in the original scale.
        token_sequences: List of tokens used for generation.
        n_samples: Number of samples to plot.
        save_path: Path to save the plot image.
    """
    assert reconstructions.shape[0] >= n_samples

    plt.figure(figsize=(10, 2 * n_samples))
    for i in range(n_samples):
        plt.subplot(n_samples, 1, i + 1)
        plt.plot(reconstructions[i, :, 0], "o-")
        plt.title(f"Sample {i+1} | Tokens: {token_sequences[i].tolist()}")
        plt.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150)
    plt.close()
