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


def plot_continuous_samples(samples, scaler, save_path, n_samples: int = 10) -> None:
    """Plot continuous time series samples."""
    plt.figure(figsize=(12, 2 * n_samples))

    for i in range(min(n_samples, len(samples))):
        # Reshape flat sample to [16, 3]
        flat_sample = samples[i].numpy()
        ts = flat_sample.reshape(1, -1)
        ts_orig = scaler.inverse_transform(ts).reshape(-1, 3)

        plt.subplot(n_samples, 1, i + 1)
        plt.plot(ts_orig[:, 0], "o-")
        plt.title(f"Continuous Sample {i+1}")
        plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_vqvae_ts_samples(
    reconstructions: np.ndarray,
    token_sequences: list[torch.Tensor],
    n_samples: int = 10,
    save_path: Optional[str] = None,
) -> None:
    """Plot VQVAE time series samples."""
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
