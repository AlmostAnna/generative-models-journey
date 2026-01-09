"""
Preview plot generator.

This module contains implementation of various functions needed to create
a preview plot for project's REAME. The plot shows some samples generated
by models implemented in the project.
"""

import matplotlib.pyplot as plt
import torch

from src.plot_style import colors

from .data.generate_synthetic import generate_dataset
from .utils.generation_pipeline import (
    generate_continuous_samples_from_model,
    generate_vqvae_samples_from_model,
)


def create_preview_plot():
    """Create plot with 3 samples from each model."""
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
        }
    )

    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=(8, 4))
    fig.suptitle(
        "VQ-VAE + Transformer vs Continuous Transformer\nGenerated Time Series Samples",
        fontsize=12,
        y=0.98,
    )

    device = torch.device("cpu")

    # Get first values for continuous generation
    X_real, labels_real, _ = generate_dataset(n_per_class=100, T=16, D=3)
    X_real = torch.tensor(X_real, dtype=torch.float32)  # Keep on CPU for now
    X_flat = X_real.reshape(X_real.size(0), -1)  # [N, 48]
    X_flat = X_flat.to(device)

    first_vals = X_flat[:, 0].cpu().numpy()  # [N,] — first value of each sequence

    # Generate 3 samples from each model
    n_samples = 3
    T = 16
    D = 3
    # seq_length = 48
    vqvae_samples = []
    cont_samples = []

    # Generate samples from both models
    print("Generating samples from the models...")
    vqvae_samples = generate_vqvae_samples_from_model(n_samples, device=device)

    cont_samples = generate_continuous_samples_from_model(
        n_samples,
        T,
        D,
        first_vals=first_vals,
        d_model=32,
        n_heads=2,
        n_layers=2,
        dropout=0.0,
        device=device,
    )

    # Plot samples
    for i in range(n_samples):
        # VQ-VAE samples (top row)
        axes[0, i].plot(
            vqvae_samples[i, :, 0],
            "o-",
            color=colors["primary"],
            markersize=3,
            linewidth=1.5,
        )
        axes[0, i].set_title(f"VQ-VAE Sample {i+1}", fontsize=9)
        axes[0, i].set_xticks([0, 8, 15])
        axes[0, i].set_yticks([])
        axes[0, i].grid(True, alpha=0.3, linewidth=0.5)

        # Continuous samples (bottom row)
        axes[1, i].plot(
            cont_samples[i, :, 0],
            "s-",
            color=colors["secondary"],
            markersize=3,
            linewidth=1.5,
        )
        axes[1, i].set_title(f"Continuous Sample {i+1}", fontsize=9)
        axes[1, i].set_xticks([0, 8, 15])
        axes[1, i].set_yticks([])
        axes[1, i].grid(True, alpha=0.3, linewidth=0.5)

    # Adjust layout
    plt.tight_layout(pad=1.5)

    # Save with web optimization
    plt.savefig(
        "plots/comparison_preview.png",
        dpi=150,
        bbox_inches="tight",
        facecolor="white",
        edgecolor="none",
    )
    plt.close()

    print("Preview plot saved to plots/comparison_preview.png")


if __name__ == "__main__":
    create_preview_plot()
