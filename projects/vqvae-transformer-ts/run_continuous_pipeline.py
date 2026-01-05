"""
Training and evaluating Continuous Transformer.

This module contains implementation of continuous transformer pipeline.
"""

import argparse
import os

import joblib
import numpy as np
import torch
from data.generate_synthetic import generate_dataset
from models.continuous_transformer import ContinuousTimeSeriesTransformer
from train_continuous import train_continuous

from utils.generation import sample_continuous
from utils.plotting import plot_continuous_samples


def main(args):
    """Run Continuous Transformer pipeline."""
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
    X = torch.tensor(X, dtype=torch.float32)  # Keep on CPU for now
    X_flat = X.reshape(X.size(0), -1)  # [N, 48]
    X_flat = X_flat.to(device)

    # === Step 2: Train Continuous Transformer ===
    print("2. Training Continuous Transformer (JiT-style)...")
    os.makedirs("artifacts", exist_ok=True)
    model, history = train_continuous(
        X_flat,
        scaler=scaler,
        seq_len=args.seq_len * args.n_channels,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        n_epochs=args.cont_epochs,
        batch_size=args.cont_batch_size,
        learning_rate=args.cont_lr,
        device=device,
        checkpoint_interval=10,
        pth_path="artifacts/continuous_transformer.pth",
        plot_recon=True,
    )

    # Save the scaler that was fitted on the training data
    joblib.dump(scaler, "artifacts/continuous_transformer_scaler.pkl")
    print("Model and Scaler saved.")

    # Diagnostic: test model on a short sequence
    print("Testing model on short sequence...")
    model.eval()
    with torch.no_grad():
        test_input = torch.tensor([[0.5]], device=device)  # [1, 1]
        output = model(test_input)
        print("Input:", test_input)
        print("Output:", output)
        print("Output contains NaN:", torch.isnan(output).any().item())

    # === Step 3: Generate & Plot ===
    print("3. Generating new time series (continuous)...")
    os.makedirs("plots", exist_ok=True)

    # Load the scaler that was fitted during training and saved
    scaler = joblib.load("artifacts/continuous_transformer_scaler.pkl")
    print("Scaler loaded from 'artifacts/continuous_transformer_scaler.pkl'.")

    # For sampling, create a fresh model and load weights
    model_for_sampling = ContinuousTimeSeriesTransformer(
        seq_len=48,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        dropout=0.0,
    ).to(device)
    model_for_sampling.load_state_dict(
        torch.load("artifacts/continuous_transformer.pth")
    )
    model_for_sampling.eval()

    generated_samples = []
    first_vals = X_flat[:, 0].cpu().numpy()  # [N,] — first value of each sequence

    for _ in range(10):
        sample_flat = sample_continuous(
            model_for_sampling,
            first_vals,
            seq_len=args.seq_len * args.n_channels,
            temperature=1.0,  # noise for diversity
            device=device,
        )
        generated_samples.append(sample_flat)
    # Plot
    plot_continuous_samples(
        generated_samples, scaler, "plots/generated_samples_continuous.png"
    )

    print(
        "Continuous pipeline complete! Samples saved to plots/generated_samples_continuous.png"  # noqa:E501
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Continuous Transformer for Time Series (JiT-style)"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_per_class", type=int, default=3000)
    parser.add_argument("--seq_len", type=int, default=16)
    parser.add_argument("--n_channels", type=int, default=3)
    parser.add_argument("--d_model", type=int, default=32)
    parser.add_argument("--n_heads", type=int, default=2)
    parser.add_argument("--n_layers", type=int, default=2)
    parser.add_argument("--cont_epochs", type=int, default=100)
    parser.add_argument("--cont_batch_size", type=int, default=64)
    parser.add_argument("--cont_lr", type=float, default=3e-4)

    args = parser.parse_args()
    main(args)
