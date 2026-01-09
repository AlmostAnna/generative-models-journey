"""
Tests for the ContinuousTimeSeriesTransformer model.

This module contains tests to ensure basic correctness of the
Continuous Transformer implementation.
"""

import torch

from ..models.continuous_transformer import ContinuousTimeSeriesTransformer


def test_init_default_params():
    """Test initialization with default parameters."""
    model = ContinuousTimeSeriesTransformer()

    assert model.seq_len == 48
    assert model.input_proj.in_features == 1
    assert model.input_proj.out_features == 32
    assert model.pos_emb.num_embeddings == 48
    assert model.pos_emb.embedding_dim == 32
    assert len(model.layers) == 2
    assert model.head.in_features == 32
    assert model.head.out_features == 1


def test_init_custom_params():
    """Test initialization with custom parameters."""
    model = ContinuousTimeSeriesTransformer(
        seq_len=24, d_model=16, n_heads=4, n_layers=3, dropout=0.1
    )

    assert model.seq_len == 24
    assert model.input_proj.out_features == 16
    assert model.pos_emb.num_embeddings == 24
    assert model.pos_emb.embedding_dim == 16
    assert len(model.layers) == 3
    assert model.head.in_features == 16


def test_forward_shape():
    """Test that the forward pass produces the correct output shape."""
    model = ContinuousTimeSeriesTransformer(
        seq_len=10, d_model=16, n_heads=2, n_layers=1
    )

    batch_size = 5
    seq_len = 8
    x = torch.randn(batch_size, seq_len)

    output = model(x)

    # Output shape should be the same as input shape
    assert output.shape == (batch_size, seq_len)


def test_forward_deterministic_with_seed():
    """Test that the forward pass is deterministic given a fixed seed."""
    torch.manual_seed(42)
    model = ContinuousTimeSeriesTransformer(
        seq_len=10, d_model=16, n_heads=2, n_layers=1
    )
    x = torch.randn(2, 5)

    torch.manual_seed(42)  # Reset seed before first forward pass
    output1 = model(x)

    torch.manual_seed(42)  # Reset seed before second forward pass
    output2 = model(x)

    assert torch.allclose(output1, output2, atol=1e-6)


def test_forward_no_nan_or_inf():
    """Test that the forward pass does not produce NaN or Inf values."""
    model = ContinuousTimeSeriesTransformer(
        seq_len=10, d_model=16, n_heads=2, n_layers=1
    )
    x = torch.randn(2, 5)  # Use a sequence shorter than seq_len

    output = model(x)

    assert not torch.isnan(output).any()
    assert not torch.isinf(output).any()


def test_forward_autoregressive_masking_logic():
    """
    Test the autoregressive property.

    Output at position i should depend only on inputs at positions
    [0, ..., i]. This is a simplified check: ensure that changing a future input
    does not change an output based on past inputs.
    """
    model = ContinuousTimeSeriesTransformer(
        seq_len=10, d_model=16, n_heads=2, n_layers=1
    )

    # Create two input sequences that differ only in the last element
    x1 = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    x2 = torch.tensor([[1.0, 2.0, 3.0, 99.0]])  # Only last element differs

    with torch.no_grad():  # Ensure no gradients affect the test
        out1 = model(x1)
        out2 = model(x2)

    # The outputs for the first three positions should be identical
    # because the inputs up to those positions are identical
    # and the model is autoregressive.
    assert torch.allclose(out1[0, :3], out2[0, :3], atol=1e-5)


def test_forward_output_values_reasonable_range():
    """
    Test that output values are within a reasonable range given normalized inputs.

    This is a basic sanity check. Values might still be large if the model learns
    to amplify signals, but extremely large values (e.g., > 1000) are suspicious.
    """
    model = ContinuousTimeSeriesTransformer(
        seq_len=10, d_model=16, n_heads=2, n_layers=1
    )
    # Use a standard normal input which is typical for normalized data
    x = torch.randn(2, 5)

    with torch.no_grad():  # Ensure no gradients affect the test
        output = model(x)

    # Check that no output value is extremely large (e.g., > 100 in absolute value)
    assert torch.all(torch.abs(output) <= 100.0)


def test_forward_batch_processing():
    """Test that the model processes batches correctly."""
    model = ContinuousTimeSeriesTransformer(
        seq_len=10, d_model=16, n_heads=2, n_layers=1
    )

    batch_size = 3
    seq_len = 6
    x = torch.randn(batch_size, seq_len)

    output = model(x)

    assert output.shape[0] == batch_size
    assert output.shape[1] == seq_len
    # Check that the processing is not identical across batches
    # (unless inputs are identical, which they are not here)
    # This is a weak check for batch independence.
    assert not torch.allclose(output[0], output[1], atol=1e-5) or not torch.allclose(
        output[1], output[2], atol=1e-5
    )
