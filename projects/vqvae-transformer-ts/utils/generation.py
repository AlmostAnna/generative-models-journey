"""
Sampling functions.

This module contains various functions generating time series by sampling tokens
from a transformer (and decoding them through a VQ-VAE in the VQ-VAE+Transformer
case).
"""

import numpy as np
import torch
import torch.nn.functional as F


def sample_continuous(model, first_vals, seq_len=48, temperature=0.1, device="cpu"):
    """Sample from continuous autoregressive model."""
    model.eval()
    with torch.no_grad():
        first_val = np.random.choice(first_vals)
        seq = torch.tensor([[first_val]], device=device)
        # Start with scalar 0.0
        # seq = torch.tensor([[0.0]], device=device)  # [1, 1]

        for step in range(1, seq_len):
            pred = model(seq)  # [1, L]
            next_mean = pred[0, -1]  # scalar

            # if step == 1:  # Print first prediction
            #    print(f"Step 1: input={seq.flatten()}, pred={next_mean.item():.3f}")

            if temperature > 0:
                # Generate SCALAR noise
                noise = torch.randn_like(next_mean) * temperature
                next_val = next_mean + noise
            else:
                next_val = next_mean

            # next_val is scalar → unsqueeze to [1, 1]
            next_val = next_val.unsqueeze(0).unsqueeze(0)  # [1, 1]
            seq = torch.cat([seq, next_val], dim=1)  # [1, L+1]

        return seq.squeeze(0).cpu()  # [L]


def generate_continuous_samples(
    model,  # The loaded model object
    n_samples: int,
    seq_len: int,  # T
    n_channels: int,  # D
    first_vals,
    temperature: float = 1.0,
    device: torch.device = torch.device("cpu"),
) -> np.ndarray:  # Returns scaled samples
    """
    Generate time series samples using a pre-trained Continuous Transformer.

    Args:
        model: The pre-trained Continuous Transformer model.
        n_samples: Number of samples to generate.
        seq_len: Length of the sequence to generate.
        n_channels: Number of channels.
        first_vals: Initial values for the sequence.
        temperature: Sampling temperature or noise level.
        device: The device to run the model on.

    Returns:
        A numpy array containing the generated samples in the scaled domain
        (shape (n_samples, seq_len, n_channels)).
    """
    model.eval()

    scaled_samples = []  # Store scaled outputs

    with torch.no_grad():
        for _ in range(n_samples):
            sample_flat_scaled = sample_continuous(
                model,
                first_vals,
                seq_len=seq_len * n_channels,
                temperature=temperature,
                device=device,
            )
            scaled_samples.append(sample_flat_scaled)

    all_samples_flat = np.stack(scaled_samples, axis=0)  # Shape: (n_samples, T*D)
    all_samples_reshaped = all_samples_flat.reshape(n_samples, seq_len, n_channels)

    return all_samples_reshaped  # Shape: (n_samples, T, D)


def sample_sequence(transformer, n_tokens=4, temperature=1.0, device="cpu"):
    """Sample token sequence from discrete Transformer."""
    transformer.eval()
    with torch.no_grad():
        probs = torch.ones(transformer.n_codes, device=device)
        first = torch.multinomial(probs, 1).unsqueeze(0)
        seq = first

        for _ in range(1, n_tokens):
            logits = transformer(seq)
            next_logits = logits[0, -1, :] / temperature
            probs = F.softmax(next_logits, dim=-1)
            if torch.isnan(probs).any() or (probs < 0).any():
                probs = torch.ones_like(probs) / probs.numel()
            next_token = torch.multinomial(probs, 1).unsqueeze(0)
            seq = torch.cat([seq, next_token], dim=1)
        return seq.squeeze(0)


def tokens_to_time_series(tokens, vqvae, device="cpu"):
    """Decode tokens to time series."""
    with torch.no_grad():
        z_q = vqvae.vq_layer.codebook(tokens.unsqueeze(0).to(device))
        recon = vqvae.decode(z_q)
        return recon.squeeze(0).cpu().numpy()


def generate_vqvae_samples(
    vqvae,
    transformer,
    n_samples: int,
    n_tokens: int,
    temperature: float = 1.0,
    device: torch.device = torch.device("cpu"),
) -> tuple[np.ndarray, list[torch.Tensor]]:
    """
    Generate time series samples using a pre-trained VQ-VAE and Transformer.

    Args:
        vqvae: The pre-trained VQ-VAE model.
        transformer: The pre-trained Transformer model.
        n_samples: Number of samples to generate.
        n_tokens: Number of tokens per sample.
        temperature: Sampling temperature for the transformer.
        device: The device to run the models on.

    Returns:
        A numpy array containing the generated samples in the scaled domain
        (e.g., shape (n_samples, T*D) or (n_samples, T, D) depending on VAE output).
    """
    vqvae.eval()
    transformer.eval()

    scaled_reconstructions = []  # Store scaled outputs
    token_sequences = []  # Store tokens

    with torch.no_grad():
        for _ in range(n_samples):
            tokens = sample_sequence(
                transformer, n_tokens, temperature=temperature, device=device
            )
            z_q = vqvae.vq_layer.codebook(tokens.unsqueeze(0).to(device))
            recon_scaled = (
                vqvae.decode(z_q).cpu().numpy()[0]
            )  # Get scaled output from VAE decoder
            scaled_reconstructions.append(recon_scaled)
            token_sequences.append(tokens.cpu())

    return np.stack(scaled_reconstructions, axis=0), token_sequences
