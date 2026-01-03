"""
Training and evaluating VQ-VAE Transformer.

This module contains implementation of VQ-VAE-TS model pipeline.
"""

import argparse
import os

import numpy as np
import torch
from data.generate_synthetic import generate_dataset
from train_transformer import train_transformer
from train_vqvae import train_vqvae

from utils.generation import generate_vqvae_samples
from utils.plotting import plot_vqvae_ts_samples


def main(args):
    """Run VQ-VAE Transformer pipeline."""
    # Reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # === Step 1: Generate Data ===
    print("1. Generating synthetic data...")
    X, labels, scaler = generate_dataset(
        n_per_class=args.n_per_class, T=args.seq_len, D=args.n_channels
    )
    X = torch.tensor(X, dtype=torch.float32).to(device)

    # === Step 2: Train VQ-VAE ===
    print("2. Training VQ-VAE...")
    vqvae, history = train_vqvae(
        X,
        n_codes=args.n_codes,
        code_dim=args.code_dim,
        n_tokens=args.n_tokens,
        n_epochs=args.vqvae_epochs,
        batch_size=args.vqvae_batch_size,
        learning_rate=args.vqvae_lr,
        beta=args.beta,
        device=device,
        checkpoint_interval=10,
        pth_path="vqvae.pth",
    )

    # === Step 3: Extract Tokens ===
    print("3. Extracting tokens...")
    with torch.no_grad():
        _, _, indices = vqvae(X)
        token_dataset = indices.long().cpu()  # [N, n_tokens]

    # === Step 4: Train Transformer Prior ===
    print("4. Training Transformer prior...")
    transformer, history = train_transformer(
        token_dataset,
        n_codes=args.n_codes,
        n_tokens=args.n_tokens,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        n_epochs=args.trans_epochs,
        batch_size=args.trans_batch_size,
        learning_rate=args.trans_lr,
        device=device,
        checkpoint_interval=10,
        pth_path="transformer.pth",
    )

    # === Step 5: Generate ===
    print("5. Generating new time series...")
    recon_s, tokens = generate_vqvae_samples(
        vqvae,
        transformer,
        scaler,
        n_samples=10,
        n_tokens=args.n_tokens,
        device=device,
    )

    # === Step 6: Plot ===
    os.makedirs("plots", exist_ok=True)
    plot_vqvae_ts_samples(
        recon_s,
        tokens,
        n_samples=10,
        save_path="plots/generated_vqvae_ts_samples.png",
    )

    print(
        "Pipeline complete! Generated samples saved to plots/generated_vqvae_ts_samples.png"  # noqa:E501
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="VQ-VAE + Transformer for Time Series Generation"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_per_class", type=int, default=3000)
    parser.add_argument("--seq_len", type=int, default=16)
    parser.add_argument("--n_channels", type=int, default=3)
    parser.add_argument("--n_codes", type=int, default=64)
    parser.add_argument("--code_dim", type=int, default=16)
    parser.add_argument("--n_tokens", type=int, default=4)
    parser.add_argument("--vqvae_epochs", type=int, default=100)
    parser.add_argument("--vqvae_batch_size", type=int, default=64)
    parser.add_argument("--vqvae_lr", type=float, default=1e-3)
    parser.add_argument("--beta", type=float, default=0.01)
    parser.add_argument("--d_model", type=int, default=32)
    parser.add_argument("--n_heads", type=int, default=2)
    parser.add_argument("--n_layers", type=int, default=2)
    parser.add_argument("--trans_epochs", type=int, default=100)
    parser.add_argument("--trans_batch_size", type=int, default=128)
    parser.add_argument("--trans_lr", type=float, default=3e-4)

    args = parser.parse_args()
    main(args)
