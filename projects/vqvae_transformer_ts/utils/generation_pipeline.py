"""
Sampling pipelines.

This module contains convinience functions implementing
load models->sample->inverse transform to the original scale pipeline.
"""

import joblib
import numpy as np
import torch

from ..models.continuous_transformer import ContinuousTimeSeriesTransformer
from ..models.transformer import TimeSeriesTransformer
from ..models.vqvae import VQVAETimeSeries
from .data_processing import inverse_scale_samples
from .generation import generate_continuous_samples, generate_vqvae_samples


def generate_vqvae_samples_from_model(
    n_samples: int,
    device: torch.device = torch.device("cpu"),
    model_path: str = "artifacts/vqvae.pth",
    transformer_path: str = "artifacts/transformer.pth",
    scaler_path: str = "artifacts/vqvae_scaler.pkl",
    n_tokens: int = 4,
) -> np.ndarray:
    """Generate VQVAE Transformer samples from pretrained model files."""
    # === Load VQ-VAE + Transformer ===
    vqvae = VQVAETimeSeries(n_codes=64, code_dim=16, n_tokens=n_tokens).to(device)
    vqvae.load_state_dict(torch.load(model_path, map_location=device))
    vqvae.eval()

    transformer = TimeSeriesTransformer(n_codes=64, n_tokens=n_tokens).to(device)
    transformer.load_state_dict(torch.load(transformer_path, map_location=device))
    transformer.eval()

    # Load the scaler that was fitted during training and saved
    scaler = joblib.load(scaler_path)

    # Generate VQ samples (in scaled domain)
    scaled_samples, tokens = generate_vqvae_samples(
        vqvae,
        transformer,
        n_samples=n_samples,
        n_tokens=n_tokens,
        device=device,
    )

    # Inverse scale to original domain
    original_scale_samples = inverse_scale_samples(scaler, scaled_samples)

    return original_scale_samples


def generate_continuous_samples_from_model(
    n_samples: int,
    T: int,
    D: int,
    first_vals,
    device: torch.device = torch.device("cpu"),
    model_path: str = "artifacts/continuous_transformer.pth",
    scaler_path: str = "artifacts/continuous_transformer_scaler.pkl",
    d_model: int = 64,  # Default or pass as needed
    n_heads: int = 4,
    n_layers: int = 4,
    dropout: float = 0.0,
) -> np.ndarray:
    """Generate Continuous Transformer samples from pretrained model files."""
    model_for_sampling = ContinuousTimeSeriesTransformer(
        T * D, d_model, n_heads, n_layers, dropout=dropout
    ).to(device)
    model_for_sampling.load_state_dict(torch.load(model_path, map_location=device))
    model_for_sampling.eval()

    # Load the scaler that was fitted during training and saved
    scaler = joblib.load(scaler_path)

    # Generate samples (in scaled domain)
    scaled_samples = generate_continuous_samples(
        model_for_sampling,
        n_samples=n_samples,
        seq_len=T,
        n_channels=D,
        first_vals=first_vals,
        device=device,
    )

    # Inverse scale to original domain
    original_scale_samples = inverse_scale_samples(scaler, scaled_samples)

    return original_scale_samples
