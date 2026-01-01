"""
Tests for the TimeSeriesTransformer model.

This module contains tests to ensure basic correctness of the Transformer prior implementation.
"""

import torch
import torch.nn.functional as F
from models.transformer import TimeSeriesTransformer


def test_transformer_initialization():
    """Test if the model initializes correctly with default and custom parameters."""
    # Test default initialization
    model = TimeSeriesTransformer()
    assert model.n_codes == 64
    assert model.n_tokens == 4
    assert model.layers.__len__() == 2

    # Test custom initialization
    custom_model = TimeSeriesTransformer(
        n_codes=128, n_tokens=8, d_model=64, n_heads=4, n_layers=3
    )
    assert custom_model.n_codes == 128
    assert custom_model.n_tokens == 8
    assert custom_model.layers.__len__() == 3


def test_transformer_forward_shape():
    """Test that the forward pass returns the correct output shape."""
    model = TimeSeriesTransformer(
        n_codes=64, n_tokens=4, d_model=32, n_heads=2, n_layers=2
    )

    batch_size = 5
    seq_len = 3  # Less than n_tokens to test partial sequences
    codes = torch.randint(0, model.n_codes, (batch_size, seq_len))

    logits = model(codes)

    # Expected shape: [Batch, SeqLen, n_codes]
    expected_shape = (batch_size, seq_len, model.n_codes)
    assert logits.shape == expected_shape


def test_transformer_forward_deterministic():
    """Test that the same input produces the same output."""
    model = TimeSeriesTransformer(
        n_codes=16, n_tokens=3, d_model=16, n_heads=2, n_layers=1
    )
    model.eval()  # Set to eval mode to disable dropout if present

    codes = torch.randint(0, model.n_codes, (2, 2))  # [Batch=2, Seq=2]

    # Run forward pass twice
    out1 = model(codes)
    out2 = model(codes)

    # Outputs should be identical
    assert torch.allclose(out1, out2, atol=1e-6)


def test_transformer_forward_causal_masking():
    """
    Test that the model respects causal masking.

    Do that by ensuring that the output for
    a position does not change if future positions in the input are altered,
    assuming the past and current positions are the same.
    Note: LayerNorm applied to the full sequence length might cause very small
    differences even with correct masking. Using a slightly larger tolerance.
    """
    model = TimeSeriesTransformer(
        n_codes=10, n_tokens=5, d_model=16, n_heads=2, n_layers=1
    )
    model.eval()

    # Input sequence: [a, b, c, d, e]
    input_seq = torch.tensor([[0, 1, 2, 3, 4]], dtype=torch.long)  # Ensure dtype
    print(f"Input sequence: {input_seq}")

    # Modify only the last token to get [a, b, c, d, e_new]
    input_seq_modified = torch.tensor(
        [[0, 1, 2, 3, 8]], dtype=torch.long
    )  # Ensure dtype
    print(f"Input sequence modified: {input_seq_modified}")

    # Get outputs
    out_original = model(input_seq)
    out_modified = model(input_seq_modified)

    print(f"Output Original shape: {out_original.shape}")
    print(f"Output Modified shape: {out_modified.shape}")
    print(f"Output Original (first 4, first 5 logits):\n{out_original[0, :4, :5]}")
    print(f"Output Modified (first 4, first 5 logits):\n{out_modified[0, :4, :5]}")

    # The output for the first 4 positions should be *very* similar
    # because the inputs up to position 3 are the same, and the mask prevents
    # the prediction at position i from seeing positions > i.
    # Allow for potential small differences due to LayerNorm over variable lengths.
    tolerance = 1e-4  # Increased from 1e-6
    assert torch.allclose(
        out_original[0, :4, :], out_modified[0, :4, :], atol=tolerance
    ), (
        f"Outputs for first 4 positions differ more than tolerance {tolerance}. "
        f"Max difference: {(out_original[0, :4, :] - out_modified[0, :4, :]).abs().max().item()}"
    )

    # The output for the last position (index 4) should be different
    # because the input token at that position changed.
    # Use a larger tolerance as logits might differ significantly
    assert not torch.allclose(out_original[0, 4, :], out_modified[0, 4, :], atol=1e-3)


def test_transformer_logits_reasonable_values():
    """Test that the output logits are finite numbers (no inf/nan)."""
    model = TimeSeriesTransformer(
        n_codes=8, n_tokens=2, d_model=16, n_heads=2, n_layers=1
    )
    model.eval()

    codes = torch.randint(0, model.n_codes, (1, 2))

    logits = model(codes)

    assert torch.isfinite(logits).all(), "Logits contain inf or nan values!"


def test_transformer_loss_calculation_integration():
    """
    Test integration with loss calculation.

    Also ensure shapes match
    expected targets for cross-entropy.
    """
    model = TimeSeriesTransformer(
        n_codes=10, n_tokens=4, d_model=16, n_heads=2, n_layers=1
    )
    batch_size = 3
    seq_len = model.n_tokens  # Use full length for input/target alignment

    # Simulate input and target sequences for autoregressive training
    # Input: [c1, c2, c3] -> Target: [c2, c3, c4]
    input_codes = torch.randint(0, model.n_codes, (batch_size, seq_len - 1))
    target_codes = torch.randint(0, model.n_codes, (batch_size, seq_len - 1))

    logits = model(input_codes)  # Shape: [B, L, n_codes] where L = seq_len - 1

    # Calculate loss
    # Reshape logits and targets for cross_entropy: [N, C] and [N]
    loss = F.cross_entropy(
        logits.reshape(-1, model.n_codes),  # [B*L, n_codes]
        target_codes.reshape(-1),  # [B*L]
    )

    # Loss should be a scalar and finite
    assert loss.dim() == 0
    assert torch.isfinite(loss)
