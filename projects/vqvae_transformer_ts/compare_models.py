"""
Model Comparison Script.

This module provides functions to run comparison of VQ-VAE + Transformer
and Continuous Transformer models.
"""

import argparse

import joblib
import numpy as np
import torch
from data.generate_synthetic import generate_dataset

from .utils.data_processing import inverse_scale_samples
from .utils.evaluation import diversity_score, fid_score, regime_accuracy_from_flat
from .utils.generation_pipeline import (
    generate_continuous_samples_from_model,
    generate_vqvae_samples_from_model,
)


def samples_to_flat_array(samples):
    """Convert list of samples to [N, 48] array."""
    # Each sample is [16, 3]
    return np.array([s.flatten() for s in samples])


def main(args):
    """Calculate varous metrics for each model."""
    device = torch.device("cpu")  # CPU-only

    # Load real data
    print("1. Generating synthetic data for comparison...")
    X_real, labels_real, _ = generate_dataset(
        n_per_class=args.n_per_class, T=args.seq_len, D=args.n_channels
    )
    X_real = torch.tensor(X_real, dtype=torch.float32)  # Keep on CPU for now
    X_flat = X_real.reshape(X_real.size(0), -1)  # [N, 48]
    X_flat = X_flat.to(device)

    first_vals = X_flat[:, 0].cpu().numpy()  # [N,] — first value of each sequence

    n_samples = args.n_samples

    # Generate samples from both models
    print("2. Generating samples from the models and rescaling them...")
    vqvae_samples = generate_vqvae_samples_from_model(n_samples, device=device)

    cont_samples = generate_continuous_samples_from_model(
        n_samples,
        T=args.seq_len,
        D=args.n_channels,
        first_vals=first_vals,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        dropout=0.0,
        device=device,
    )

    vqvae_array = samples_to_flat_array(vqvae_samples)
    cont_array = samples_to_flat_array(cont_samples)
    X_full_np = X_flat.cpu().numpy()

    # Load the scaler that was fitted during training and saved
    vqvae_scaler = joblib.load("artifacts/vqvae_scaler.pkl")
    cont_scaler = joblib.load("artifacts/continuous_transformer_scaler.pkl")

    # Compute metrics
    print("3. Computing metrics...")

    print(
        "Regime Distribution Match (VQ-VAE):",
        regime_accuracy_from_flat(
            vqvae_array, inverse_scale_samples(vqvae_scaler, X_full_np), labels_real
        ),
    )
    print(
        "Regime Distribution Match (Continuous):",
        regime_accuracy_from_flat(
            cont_array, inverse_scale_samples(cont_scaler, X_full_np), labels_real
        ),
    )

    print("Diversity (VQ-VAE):", diversity_score(vqvae_array))
    print(
        "Diversity (Real for VQ-VAE comparison):",
        diversity_score(
            inverse_scale_samples(vqvae_scaler, X_flat[:n_samples].cpu().numpy())
        ),
    )
    print("Diversity (Continuous):", diversity_score(cont_array))
    print(
        "Diversity (Real for Continuous comparison):",
        diversity_score(
            inverse_scale_samples(cont_scaler, X_flat[:n_samples].cpu().numpy())
        ),
    )

    print(
        "FID (VQ-VAE):",
        fid_score(
            inverse_scale_samples(vqvae_scaler, X_flat[:n_samples].cpu().numpy()),
            vqvae_array,
        ),
    )
    print(
        "FID (Continuous):",
        fid_score(
            inverse_scale_samples(cont_scaler, X_flat[:n_samples].cpu().numpy()),
            cont_array,
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generative Transformers Comparison")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_per_class", type=int, default=3000)
    parser.add_argument("--seq_len", type=int, default=16)
    parser.add_argument("--n_channels", type=int, default=3)
    parser.add_argument("--d_model", type=int, default=32)
    parser.add_argument("--n_heads", type=int, default=2)
    parser.add_argument("--n_layers", type=int, default=2)
    parser.add_argument("--n_samples", type=int, default=3000)

    args = parser.parse_args()
    main(args)
