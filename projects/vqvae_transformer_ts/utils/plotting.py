"""
Plotting functions.

This module contains implementation of plotting functions used
in the project.
"""

import os
from typing import Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.plot_style import colors  # noqa: F401 # Side effect: initializes plot style


def _validate_and_convert_data(data: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
    """Ensure data is a numpy array."""
    if torch.is_tensor(data):
        data = data.numpy()
    if not isinstance(data, np.ndarray):
        raise TypeError(f"Data must be a numpy array or torch tensor, got {type(data)}")
    return data


def plot_time_series_samples(
    data: Union[np.ndarray, torch.Tensor],
    n_samples: int = 10,
    channels_to_plot: list[int] | str = [0],  # Can be [0], [0, 1, 2], or 'all'
    title_prefix: str = "Sample",
    save_path: Optional[str] = None,
    figsize_per_sub: tuple[float, float] = (10, 2),  # Width, Height per subplot row
) -> None:
    """
    Plot time series samples.

    Args:
        data: Array of shape (n_samples, T, D) or (n_samples, T) in the original scale.
              If shape is (n_samples, T), it's treated as single-channel.
        n_samples: Number of samples to plot (up to the size of the first dimension).
        channels_to_plot: List of channel indices to plot for each sample, or 'all'.
                         Only used if D > 1. Default is [0].
        title_prefix: Prefix for the subplot titles (e.g., "Continuous Sample", "VQ-VAE Sample").
        save_path: Path to save the plot image.
        figsize_per_sub: Base width and height (in inches) allocated per subplot row.
    """
    data = _validate_and_convert_data(data)

    if data.ndim == 2:  # Shape (N, T)
        data = data[:, :, np.newaxis]  # Add channel dim -> (N, T, 1)
        D = 1
    elif data.ndim == 3:  # Shape (N, T, D)
        D = data.shape[2]
    else:
        raise ValueError(
            f"Data must be 2D (N, T) or 3D (N, T, D), got shape {data.shape}"
        )

    N, T, _ = data.shape
    n_samples = min(n_samples, N)  # Don't try to plot more than available

    if channels_to_plot == "all":
        channels_to_plot = list(range(D))
    else:
        # Validate channel indices
        if any(c < 0 or c >= D for c in channels_to_plot):
            raise ValueError(
                f"Invalid channel index in channels_to_plot {channels_to_plot} for data with {D} channels."
            )

    n_channels_to_plot = len(channels_to_plot)

    total_subplots_needed = n_samples * n_channels_to_plot

    # Adjust figure size based on the total number of subplots needed
    total_height = total_subplots_needed * figsize_per_sub[1]
    total_width = figsize_per_sub[0]  # Keep width constant
    plt.figure(figsize=(total_width, total_height))

    plot_index = 1
    for i in range(n_samples):
        for ch_idx in channels_to_plot:
            # Use the total number of subplots needed for the grid definition
            plt.subplot(total_subplots_needed, 1, plot_index)
            # Plot the specific channel for the specific sample
            plt.plot(
                data[i, :, ch_idx],
                marker="o",
                markersize=2,
                linestyle="-",
                linewidth=0.8,
            )
            plt.title(f"{title_prefix} {i+1}, Channel {ch_idx}")
            plt.grid(True, alpha=0.3)
            plt.ylabel(f"Value (Ch. {ch_idx})")
            plot_index += 1  # Increment for the next subplot position

    plt.xlabel("Time Step")
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150)  # Increased DPI for potentially more subplots
    plt.close()


def plot_continuous_samples(
    reconstructions: Union[np.ndarray, torch.Tensor],
    n_samples: int = 10,
    save_path: Optional[str] = None,
) -> None:
    """
    Plot continuous time series samples (plots channel 0).

    (Kept for backward compatibility, now calls the generic function)
    """
    plot_time_series_samples(
        data=reconstructions,
        n_samples=n_samples,
        channels_to_plot=[0],  # Always plot channel 0 for this specific function
        title_prefix="Continuous Sample",
        save_path=save_path,
    )


def plot_vqvae_ts_samples(
    reconstructions: Union[np.ndarray, torch.Tensor],
    token_sequences: Optional[list[torch.Tensor]] = None,
    n_samples: int = 10,
    save_path: Optional[str] = None,
) -> None:
    """
    Plot VQVAE time series samples (plots channel 0).

    (Kept for backward compatibility, now calls the generic function)
    """
    title_prefix = "VQ-VAE Sample"
    if token_sequences is not None:
        pass  # Just use title_prefix

    plot_time_series_samples(
        data=reconstructions,
        n_samples=n_samples,
        channels_to_plot=[0],  # Always plot channel 0 for this specific function
        title_prefix=title_prefix,
        save_path=save_path,
    )
