"""
Tests for VQ-VAE sampling.

This module contains tests to ensure basic correctness
of sampling from VQ-VAE+Transformer model.
"""

from unittest.mock import Mock

import numpy as np
import pytest
import torch

from utils.generation import generate_vqvae_samples


# --- Mock Model Components for Testing ---
class MockVQVAE:  # (torch.nn.Module): # Inherit from nn.Module to get .eval(), .train()
    """Mock VQ-VAE tokenizer."""

    def __init__(self, codebook_size=64, code_dim=16):
        """Initialize a codebook."""
        # super().__init__() # Call parent init
        # Simulate a codebook: (codebook_size, code_dim)
        self.vq_layer = Mock()
        self.vq_layer.codebook = torch.nn.Embedding(codebook_size, code_dim)
        # Initialize with random values for realism
        torch.nn.init.normal_(self.vq_layer.codebook.weight)

    def decode(self, z_q):
        """Simulate decoding."""
        # Simulate decoding: (batch, seq_len, code_dim) -> (batch, seq_len, data_dim)
        batch_size, seq_len, code_dim = z_q.shape
        # A simple mock decoding (e.g., a linear layer simulation)
        # In reality, this would be the VQ-VAE's decoder network
        # Here, we just return a tensor of shape (batch, seq_len, 3)
        # Ensure the output is detached from the computation graph for numpy conversion
        return torch.randn(batch_size, seq_len, 3)

    def eval(self):
        """Mock eval method."""
        return self

    def train(self, mode=True):
        """Mock train method."""
        return self


class MockTransformer:  # (torch.nn.Module): # Inherit from nn.Module to get .eval(), .train()
    """Mock Transformer."""

    def __init__(self, n_codes=64, n_tokens=4):
        """Initialize a mock transformer."""
        # super().__init__() # Call parent init
        self.n_codes = n_codes
        self.n_tokens = n_tokens

    def __call__(self, seq):
        """
        Mock the transformer's forward pass.

        seq: (batch_size, current_seq_len) -> torch.long (token indices)
        Expected output: logits (batch_size, current_seq_len, vocab_size=n_codes)
        """
        batch_size, current_seq_len = seq.shape
        # Return logits of the correct shape
        # Use randn for more realistic logit values (can be positive/negative)
        return torch.randn(batch_size, current_seq_len, self.n_codes)

    def eval(self):
        """Mock eval method."""
        return self

    def train(self, mode=True):
        """Mock train method."""
        return self


class MockScaler:
    """Mock Scaler."""

    def inverse_transform(self, data):
        """Return the same unchanged data."""
        return data


# --- Functional Tests ---
@pytest.fixture
def mock_models_and_scaler():
    """Provide mock VQ-VAE, Transformer, and Scaler for tests."""
    vqvae = MockVQVAE()
    transformer = MockTransformer()
    scaler = MockScaler()
    device = torch.device("cpu")
    return vqvae, transformer, scaler, device


def test_returns_correct_types_and_shapes(mock_models_and_scaler):
    """Test that the function returns a tuple of np.ndarray and List[torch.Tensor] with correct shapes."""
    vqvae, transformer, scaler, device = mock_models_and_scaler
    n_samples = 5
    n_tokens = 4

    samples, tokens_list = generate_vqvae_samples(
        vqvae, transformer, scaler, n_samples, n_tokens, device=device
    )

    assert isinstance(
        samples, np.ndarray
    ), f"Expected np.ndarray for samples, got {type(samples)}"
    assert isinstance(
        tokens_list, list
    ), f"Expected list for tokens, got {type(tokens_list)}"
    assert all(
        isinstance(t, torch.Tensor) for t in tokens_list
    ), "All items in tokens_list should be torch.Tensor"

    expected_samples_shape = (
        n_samples,
        n_tokens,
        3,
    )  # Adjust D if your decode output changes
    assert (
        samples.shape == expected_samples_shape
    ), f"Expected samples shape {expected_samples_shape}, got {samples.shape}"
    assert (
        len(tokens_list) == n_samples
    ), f"Expected {n_samples} token sequences, got {len(tokens_list)}"


def test_returns_correct_number_of_samples(mock_models_and_scaler):
    """Test that the number of returned samples matches n_samples."""
    vqvae, transformer, scaler, device = mock_models_and_scaler
    n_samples = 1
    n_tokens = 4

    samples, tokens_list = generate_vqvae_samples(
        vqvae, transformer, scaler, n_samples, n_tokens, device=device
    )

    assert (
        len(samples) == n_samples
    ), f"Expected {n_samples} samples, got {len(samples)}"
    assert (
        len(tokens_list) == n_samples
    ), f"Expected {n_samples} token lists, got {len(tokens_list)}"

    n_samples = 10
    samples, tokens_list = generate_vqvae_samples(
        vqvae, transformer, scaler, n_samples, n_tokens, device=device
    )

    assert (
        len(samples) == n_samples
    ), f"Expected {n_samples} samples, got {len(samples)}"
    assert (
        len(tokens_list) == n_samples
    ), f"Expected {n_samples} token lists, got {len(tokens_list)}"


def test_handles_different_n_tokens(mock_models_and_scaler):
    """Test that the function works with different n_tokens values."""
    vqvae, transformer, scaler, device = mock_models_and_scaler
    n_samples = 2

    for n_tokens in [2, 8, 16]:
        # Create a new transformer mock instance for each test case to ensure n_tokens is set correctly
        # if sample_sequence logic depends on the initial n_tokens.
        # However, n_tokens is passed to sample_sequence, so the mock's initial n_tokens might be less relevant
        # for the sampling loop itself, but the vocab size (n_codes) is.
        # The mock transformer's n_codes is used for the output shape of its __call__.
        mock_trans_for_case = MockTransformer(n_codes=64, n_tokens=n_tokens)
        samples, tokens_list = generate_vqvae_samples(
            vqvae, mock_trans_for_case, scaler, n_samples, n_tokens, device=device
        )
        # Check shape based on n_tokens
        expected_samples_shape = (n_samples, n_tokens, 3)  # Adjust D if needed
        assert (
            samples.shape == expected_samples_shape
        ), f"Shape mismatch for n_tokens={n_tokens}"
        assert len(tokens_list) == n_samples
        # The length of each token tensor in tokens_list will be n_tokens
        assert all(len(t) == n_tokens for t in tokens_list)
