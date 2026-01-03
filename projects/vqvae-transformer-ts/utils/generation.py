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
    scaler,  # Pass scaler if needed for inverse transform before return
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
        scaler: The scaler used for inverse transform.
        n_samples: Number of samples to generate.
        n_tokens: Number of tokens per sample.
        temperature: Sampling temperature for the transformer.
        device: The device to run the models on.

    Returns:
        A tuple containing the generated samples (e.g., shape (n_samples, T, D))
        and the corresponding token sequences.
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
            z_q = vqvae.vq_layer.codebook(tokens.unsqueeze(0).to(device))
            recon = vqvae.decode(z_q).cpu().numpy()[0]

            # Inverse transform to original scale if needed before returning
            recon_orig = scaler.inverse_transform(recon.reshape(1, -1)).reshape(
                -1, 3
            )  # Adjust shape as needed
            reconstructions.append(recon_orig)
            token_sequences.append(tokens.cpu())

    return np.stack(reconstructions, axis=0), token_sequences
