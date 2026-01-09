"""
Tests for data generation.

This module contains tests to ensure basic correctness
of generated data.
"""

import numpy as np

from ..data.generate_synthetic import generate_dataset


def test_generate_dataset():
    """Test shapes, values and labels for the data set."""
    X, labels, scaler = generate_dataset(n_per_class=10, T=16, D=3)

    # Shape checks
    assert X.shape == (30, 16, 3)  # 3 classes × 10
    assert labels.shape == (30,)

    # Value checks
    assert X.dtype == np.float32
    assert X.min() >= -5.0
    assert X.max() <= 5.0

    # Label checks
    assert set(np.unique(labels)) == {0, 1, 2}
