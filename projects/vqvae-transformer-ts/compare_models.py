import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics.pairwise import pairwise_distances
from scipy.linalg import sqrtm

from data.generate_synthetic import generate_dataset

from models.vqvae import VQVAETimeSeries
from models.transformer import TimeSeriesTransformer
from models.continuous_transformer import ContinuousTimeSeriesTransformer

from utils.generation import sample_sequence, tokens_to_time_series, sample_continuous


def generate_vqvae_samples(n_samples, device):
    # === Load VQ-VAE + Transformer ===
    vqvae = VQVAETimeSeries(n_codes=64, code_dim=16, n_tokens=4).to(device)
    vqvae.load_state_dict(torch.load("vqvae.pth", map_location=device))
    vqvae.eval()
 
    transformer = TimeSeriesTransformer(n_codes=64, n_tokens=4).to(device)
    transformer.load_state_dict(torch.load("transformer.pth", map_location=device))
    transformer.eval()
 
    # Generate VQ samples
    vqvae_samples = []
    for _ in range(n_samples):
        tokens = sample_sequence(transformer, n_tokens=4, temperature=1.0)
        ts = tokens_to_time_series(tokens, vqvae)
        vqvae_samples.append(ts)
    return vqvae_samples

def generate_continuous_samples(n_samples, seq_len, first_vals, d_model, n_heads, n_layers, dropout, device):
    # For sampling, create a fresh model and load weights
    model_for_sampling = ContinuousTimeSeriesTransformer(
        seq_len,
        d_model,
        n_heads,
        n_layers,
        dropout=0.0
    ).to(device)
    model_for_sampling.load_state_dict(torch.load("continuous_transformer.pth"))
    model_for_sampling.eval()
 
    generated_samples = []
    for _ in range(n_samples):
        sample_flat = sample_continuous(
            model_for_sampling,
            first_vals,
            seq_len,
            temperature=1.0,  # noise for diversity
            device=device
        )
        generated_samples.append(sample_flat)

    return generated_samples

def samples_to_flat_array(samples, is_vqvae=False):
    """Convert list of samples to [N, 48] array."""
    if is_vqvae:
        # Each sample is [16, 3]
        return np.array([s.flatten() for s in samples])
    else:
        # Each sample is [48]
        return np.array(samples)

def diversity_score(samples_array):
    distances = pairwise_distances(samples_array, metric='euclidean')
    return np.mean(distances)  # higher = more diverse


def fid_score(real_samples_array, gen_samples_array, eps=1e-6):
    # Compute means
    mu1 = np.mean(real_samples_array, axis=0)
    mu2 = np.mean(gen_samples_array, axis=0)
    ssdiff = np.sum((mu1 - mu2) ** 2)

    # Compute covariances
    sigma1 = np.cov(real_samples_array, rowvar=False)
    sigma2 = np.cov(gen_samples_array, rowvar=False)

    # Add regularization to diagonals (prevents singularity)
    sigma1 += np.eye(sigma1.shape[0]) * eps
    sigma2 += np.eye(sigma2.shape[0]) * eps

    # Compute sqrtm with error handling
    try:
        covmean = sqrtm(sigma1 @ sigma2)
        if np.iscomplexobj(covmean):
            covmean = covmean.real
        # Ensure covmean is finite
        if not np.isfinite(covmean).all():
            print("Warning: covmean contains non-finite values. Using fallback.")
            covmean = np.zeros_like(covmean)
    except Exception as e:
        print(f"sqrtm failed: {e}. Using fallback.")
        covmean = np.zeros_like(sigma1)

    fid = ssdiff + np.trace(sigma1 + sigma2 - 2 * covmean)
    return fid



def autocorr(x, lag=1):
    return np.corrcoef(x[:-lag], x[lag:])[0, 1]

def main(args):
    device = torch.device("cpu")  # CPU-only

    # Load real data
    print("1. Generating synthetic data for comparison...")
    X_real, labels_real, scaler = generate_dataset(
          n_per_class=args.n_per_class,
          T=args.seq_len,
          D=args.n_channels
    )
    X_real = torch.tensor(X_real, dtype=torch.float32)  # Keep on CPU for now
    X_flat = X_real.reshape(X_real.size(0), -1)         # [N, 48]
    X_flat = X_flat.to(device)

    first_vals = X_flat[:, 0].cpu().numpy()  # [N,] — first value of each sequence

    n_samples = args.n_samples

    # Generate samples from both models
    print("2. Generating samples from the models...")
    vqvae_samples = generate_vqvae_samples(n_samples, device=device)
    
    s_l=args.seq_len * args.n_channels
    cont_samples = generate_continuous_samples(n_samples, seq_len=s_l, first_vals=first_vals, d_model=args.d_model, n_heads=args.n_heads, n_layers=args.n_layers, dropout=0.0, device=device)
 
    vqvae_array = samples_to_flat_array(vqvae_samples, is_vqvae=True)
    cont_array = samples_to_flat_array(cont_samples, is_vqvae=False)

    # Compute metrics
    print("3. Computing metrics...")
    print("Diversity (VQ-VAE):", diversity_score(vqvae_array))
    print("Diversity (Continuous):", diversity_score(cont_array))
    real_div = diversity_score(X_flat[:n_samples].cpu().numpy())  # compare same N
    print(f"Diversity (Real): {real_div:.3f}")
    
    print("FID (VQ-VAE):", fid_score(X_flat[:n_samples].cpu().numpy(), vqvae_array))
    print("FID (Continuous):", fid_score(X_flat[:n_samples].cpu().numpy(), cont_array))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generative Transformers Comparison")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_per_class", type=int, default=3000)
    parser.add_argument("--seq_len", type=int, default=16)
    parser.add_argument("--n_channels", type=int, default=3)
    parser.add_argument("--d_model", type=int, default=32)
    parser.add_argument("--n_heads", type=int, default=2)
    parser.add_argument("--n_layers", type=int, default=2)
    parser.add_argument("--n_samples", type=int, default=1000)
    
    args = parser.parse_args()
    main(args)



