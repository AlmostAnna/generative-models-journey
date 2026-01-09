"""
Training Continuous Transformer.

This module contains implementation of continuous transformer training.
"""

import torch
import torch.nn.functional as F

from .models.continuous_transformer import ContinuousTimeSeriesTransformer


def train_continuous(
    X_flat,
    scaler=None,
    seq_len=48,
    d_model=32,
    n_heads=2,
    n_layers=2,
    n_epochs=50,
    batch_size=64,
    learning_rate=3e-4,
    device="cpu",
    checkpoint_interval=10,
    pth_path="continuous_transformer.pth",
    plot_recon=False,
):
    """Train the continuous time series transformer model."""
    # Initialize model
    model = ContinuousTimeSeriesTransformer(
        seq_len=seq_len, d_model=d_model, n_heads=n_heads, n_layers=n_layers
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Prepare autoregressive inputs/targets
    input_seq = X_flat[:, :-1]  # [N, 47]
    target_seq = X_flat[:, 1:]  # [N, 47]

    print(f"Starting continuous transformer training on {device}...")
    print(f"Epochs: {n_epochs}, Batch size: {batch_size}, Seq len: {seq_len}")

    for epoch in range(n_epochs):
        idx = torch.randperm(input_seq.size(0))
        epoch_loss = 0.0

        for i in range(0, input_seq.size(0), batch_size):
            inp = input_seq[idx[i : i + batch_size]].to(device)  # [B, 47]
            tgt = target_seq[idx[i : i + batch_size]].to(device)  # [B, 47]

            pred = model(inp)  # [B, 47]
            loss = F.mse_loss(pred, tgt)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()

        if epoch % checkpoint_interval == 0:
            print(f"Epoch {epoch}, Loss: {epoch_loss:.4f}")

    print("Continuous transformer training complete!")
    if pth_path:
        torch.save(model.state_dict(), pth_path)
        print(f"Model saved to {pth_path}")

    if plot_recon and scaler is not None:
        print("Generating reconstruction diagnostic...")
        model.eval()
        with torch.no_grad():
            test_input = X_flat[:1, :-1].to(device)  # [1, 47]
            pred = model(test_input)
            full_pred = torch.cat([test_input, pred[:, -1:]], dim=1)  # [1, 48]

            # Inverse transform
            orig = scaler.inverse_transform(
                X_flat[0].cpu().numpy().reshape(1, -1)
            ).reshape(16, 3)
            recon = scaler.inverse_transform(
                full_pred.cpu().numpy().reshape(1, -1)
            ).reshape(16, 3)

            # Plot
            import matplotlib.pyplot as plt

            plt.figure(figsize=(8, 3))
            plt.plot(orig[:, 0], "o-", label="Original")
            plt.plot(recon[:, 0], "s--", label="Reconstruction")
            plt.legend()
            plt.title("Training Reconstruction (Continuous)")
            plt.savefig("plots/continuous_recon.png")
            plt.close()
            print("Reconstruction plot saved to plots/continuous_recon.png")

    return model, {"final_loss": loss.item()}
