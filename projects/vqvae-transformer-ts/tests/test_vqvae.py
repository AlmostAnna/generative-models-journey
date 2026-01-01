"""
Tests for VQ-VAE model.

This module contains tests to ensure basic correctness of VQ-VAE implemetation.
"""

import torch
from models.vqvae import VectorQuantizer, VQVAETimeSeries


def test_vector_quantizer_shapes():
    """Test that VectorQuantizer returns correct shapes."""
    B, L, code_dim, n_codes = 4, 2, 16, 64
    z_e = torch.randn(B, L, code_dim)
    vq = VectorQuantizer(n_codes, code_dim)
    z_q, indices = vq(z_e)

    assert z_q.shape == (B, L, code_dim)
    assert indices.shape == (B, L)
    assert indices.min() >= 0
    assert indices.max() < n_codes


def test_vector_quantizer_nearest_code():
    """Test that VQ layer correctly finds nearest codebook entry."""
    vq = VectorQuantizer(n_codes=2, code_dim=2)

    # Set known codebook
    with torch.no_grad():
        vq.codebook.weight[:] = torch.tensor(
            [[0.0, 0.0], [2.0, 0.0]], dtype=torch.float  # code 0  # code 1
        )

    # Input closer to code 0
    z_e = torch.tensor([[[0.1, 0.0]]])
    _, indices = vq(z_e)

    assert indices.item() == 0, f"Expected code 0, got {indices.item()}"


def test_vector_quantizer_straight_through():
    """Test that gradients flow through VQ layer (straight-through estimator)."""
    vq = VectorQuantizer(n_codes=4, code_dim=2)
    z_e = torch.randn(1, 1, 2, requires_grad=True)

    z_q, _ = vq(z_e)
    loss = z_q.sum()
    loss.backward()

    # Should have gradients despite discrete operation
    assert z_e.grad is not None
    assert not torch.allclose(z_e.grad, torch.zeros_like(z_e.grad))


def test_vqvae_shapes():
    """Test full VQ-VAE input-output shapes."""
    B, T, D = 8, 16, 3
    model = VQVAETimeSeries(T=T, D=D, n_tokens=2, code_dim=16, n_codes=64)
    x = torch.randn(B, T, D)

    recon, z_q, indices = model(x)

    assert recon.shape == (B, T, D)
    assert z_q.shape == (B, 2, 16)
    assert indices.shape == (B, 2)


def test_vqvae_reconstruction():
    """Test that model can overfit a single sample (basic sanity check)."""
    torch.manual_seed(0)
    model = VQVAETimeSeries(T=8, D=2, n_tokens=1, n_codes=8, code_dim=8, hidden_dim=32)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    x = torch.randn(1, 8, 2)

    for _ in range(200):
        optimizer.zero_grad()
        recon, _, _ = model(x)
        loss = torch.nn.functional.mse_loss(recon, x)
        loss.backward()
        optimizer.step()

    final_loss = torch.nn.functional.mse_loss(recon, x).item()
    assert final_loss < 0.1, f"Reconstruction failed: loss={final_loss:.4f}"


def test_codebook_usage():
    """Test that multiple codes are used (not all indices the same)."""
    model = VQVAETimeSeries(n_tokens=2, n_codes=16, code_dim=8, T=16, D=3)
    x = torch.randn(32, 16, 3)
    with torch.no_grad():
        _, _, indices = model(x)
    unique_codes = torch.unique(indices)
    assert (
        unique_codes.numel() > 1
    ), "All inputs mapped to same code — codebook collapse likely."
