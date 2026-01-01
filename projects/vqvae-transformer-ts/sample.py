"""
VQVAE-TS sampling.

This module contains implementation of sampling from VQ-VAE + Transformer
model.
"""

import numpy as np
import torch
import torch.nn.functional as F


def sample_sequence(
    model,
    n_tokens: int = 4,
    temperature: float = 1.0,
    device: torch.device = torch.device("cpu"),
):
    """Sample n_tokens from the model."""
    model.eval()
    with torch.no_grad():
        # Sample first token uniformly
        probs = torch.ones(model.n_codes, device=device)  # uniform probabilities
        first_token = torch.multinomial(probs, 1).unsqueeze(0)  # [1, 1]
        seq = first_token

        # Autoregressive sampling
        for _ in range(1, n_tokens):
            logits = model(seq)  # [1, L, n_codes]
            next_logits = logits[0, -1, :]  # [n_codes]

            if temperature != 1.0:
                next_logits = next_logits / temperature

            # Safe softmax
            probs = F.softmax(next_logits, dim=-1)

            # Safety check (shouldn't be needed, but just in case)
            if torch.isnan(probs).any() or (probs < 0).any():
                probs = torch.ones_like(probs) / probs.numel()

            next_token = torch.multinomial(probs, 1).unsqueeze(0)  # [1, 1]
            seq = torch.cat([seq, next_token], dim=1)

        return seq.squeeze(0)  # [n_tokens]


def tokens_to_time_series(tokens, vqvae):
    """Decode tokens back to time series."""
    with torch.no_grad():
        # tokens: [4] → [1, 4]
        z_q = vqvae.vq_layer.codebook(tokens.unsqueeze(0))  # [1, 4, 16]
        recon = vqvae.decode(z_q)  # [1, 16, 3]
        return recon.squeeze(0).numpy()


def sample_and_decode(
    vqvae,
    transformer,
    scaler,
    n_samples: int = 10,
    n_tokens: int = 4,
    temperature: float = 1.0,
    device: torch.device = torch.device("cpu"),
) -> tuple[np.ndarray, list[torch.Tensor]]:
    """
    Generate and decode n_samples time series using VQ-VAE + Transformer.

    Returns:
        reconstructions: np.ndarray of shape (n_samples, T, D)
        token_sequences: List of token tensors, length n_samples
    """
    vqvae.eval()
    transformer.eval()

    reconstructions = []
    token_sequences = []

    with torch.no_grad():
        for _ in range(n_samples):
            tokens = sample_sequence(
                transformer, n_tokens, temperature=temperature, device=device
            )
            z_q = vqvae.vq_layer.codebook(
                tokens.unsqueeze(0).to(device)
            )  # (1, n_tokens, code_dim)
            recon = vqvae.decode(z_q).cpu().numpy()[0]  # (T, D)

            # Inverse transform to original scale
            recon_orig = scaler.inverse_transform(recon.reshape(1, -1)).reshape(
                -1, 3
            )  # assume D=3
            reconstructions.append(recon_orig)
            token_sequences.append(tokens.cpu())

    return np.stack(reconstructions, axis=0), token_sequences  # (n_samples, T, 3)
