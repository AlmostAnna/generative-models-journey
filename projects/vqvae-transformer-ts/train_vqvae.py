"""
Training VQ-VAE tokenizer.

This module contains implementation of VQ-VAE model training.
"""

import torch
import torch.nn.functional as F
from models.vqvae import VQVAETimeSeries


def train_vqvae(
    X,
    n_codes=64,
    code_dim=16,
    n_tokens=4,
    n_epochs=50,
    batch_size=64,
    learning_rate=1e-3,
    beta=0.01,
    device="cpu",
    checkpoint_interval=10,
    pth_path="vqvae.pth",
):
    """Train the vqvae model."""
    # Initialize model
    model = VQVAETimeSeries(n_codes=n_codes, code_dim=code_dim, n_tokens=n_tokens).to(
        device
    )

    # Initialize optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    print(f"Starting training on {device}...")
    print(f"Epochs: {n_epochs}, Batch size: {batch_size}")

    for epoch in range(n_epochs):
        idx = torch.randperm(X.size(0))
        epoch_loss = 0.0

        for i in range(0, X.size(0), batch_size):
            batch = X[idx[i : i + batch_size]]

            recon, z_q, indices = model(batch)
            z_e = model.encode(batch)  # Need z_e for losses

            recon_loss = F.mse_loss(recon, batch)
            codebook_loss = F.mse_loss(z_q, z_e.detach())  # update codebook
            commit_loss = F.mse_loss(z_e, z_q.detach())  # update encoder
            # After computing losses
            if epoch < 5:
                # Warm-up: no commitment loss for first 5 epochs
                loss = recon_loss + codebook_loss
            else:
                loss = recon_loss + codebook_loss + beta * commit_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()

        if epoch % checkpoint_interval == 0:
            print(f"Epoch {epoch}, Loss: {epoch_loss:.4f}")

    print("Training complete!")
    # Save the result
    if pth_path:
        torch.save(model.state_dict(), pth_path)
        print(f"VQ-VAE model saved to {pth_path}")
    return model, {"final_loss": loss.item()}
