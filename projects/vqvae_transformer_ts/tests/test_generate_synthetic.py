"""
Tests for data generation.

This module contains tests to ensure basic correctness
of generated data.
"""

import numpy as np
import pytest

from ..data.generate_synthetic import (
    generate_dataset,
    generate_mean_reverting,
    generate_spike,
    generate_trending,
)


@pytest.mark.parametrize(
    "func", [generate_trending, generate_mean_reverting, generate_spike]
)
def test_generate_functions_shapes(func):
    """Test that individual generation functions return correct shapes."""
    T, D = 16, 3
    output = func(T=T)
    assert output.shape == (
        T,
        D,
    ), f"Function {func.__name__} returned shape {output.shape}, expected {(T, D)}"


INITIAL_PRICE = 100.0
EXPECTED_RANGE_BUFFER = 5.0


def test_generate_trending_value_ranges():
    """Test the approximate value ranges of generate_trending."""
    T = 16
    trending_data = generate_trending(T=T)
    x1_trending = trending_data[:, 0]

    # Assuming GBM with low drift/vol starting at INITIAL_PRICE
    assert x1_trending.min() > 0, "generate_trending x1 should be positive"
    assert (
        x1_trending.min() >= INITIAL_PRICE - EXPECTED_RANGE_BUFFER
    ), f"generate_trending x1 min ({x1_trending.min()}) out of expected range [{INITIAL_PRICE - EXPECTED_RANGE_BUFFER}, {INITIAL_PRICE + EXPECTED_RANGE_BUFFER}]"
    assert (
        x1_trending.max() <= INITIAL_PRICE + EXPECTED_RANGE_BUFFER
    ), f"generate_trending x1 max ({x1_trending.max()}) out of expected range [{INITIAL_PRICE - EXPECTED_RANGE_BUFFER}, {INITIAL_PRICE + EXPECTED_RANGE_BUFFER}]"

    x2_trending = trending_data[:, 1]
    assert np.allclose(
        x2_trending, 0.2, atol=0.2
    ), "generate_trending x2 out of expected range (too far from 0.2)"

    x3_trending = trending_data[:, 2]
    assert np.all(x3_trending > 0), "generate_trending x3 should be positive"


def test_generate_mean_reverting_value_ranges():
    """Test the approximate value ranges of generate_mean_reverting (now equity Heston-like)."""
    T = 16
    mr_data = generate_mean_reverting(T=T)
    x1_mr = mr_data[:, 0]

    # Assuming Heston-like with mean rev around initial price
    assert x1_mr.min() > 0, "generate_mean_reverting x1 should be positive"
    assert (
        x1_mr.min() >= INITIAL_PRICE - EXPECTED_RANGE_BUFFER
    ), f"generate_mean_reverting x1 min ({x1_mr.min()}) out of expected range [{INITIAL_PRICE - EXPECTED_RANGE_BUFFER}, {INITIAL_PRICE + EXPECTED_RANGE_BUFFER}]"
    assert (
        x1_mr.max() <= INITIAL_PRICE + EXPECTED_RANGE_BUFFER
    ), f"generate_mean_reverting x1 max ({x1_mr.max()}) out of expected range [{INITIAL_PRICE - EXPECTED_RANGE_BUFFER}, {INITIAL_PRICE + EXPECTED_RANGE_BUFFER}]"


def test_generate_spike_value_ranges():
    """Test the approximate value ranges of generate_spike (now equity with jumps)."""
    T = 16
    spike_data = generate_spike(T=T)
    x1_spike = spike_data[:, 0]

    # Assuming GBM + jumps starting around INITIAL_PRICE
    MIN_EXPECTED_SPIKE = INITIAL_PRICE * np.exp(-2.0)
    MAX_EXPECTED_SPIKE = INITIAL_PRICE * np.exp(2.0)

    assert x1_spike.min() > 0, "generate_spike x1 should be positive"
    assert (
        x1_spike.min() >= MIN_EXPECTED_SPIKE
    ), f"generate_spike x1 min ({x1_spike.min()}) out of expected range [{MIN_EXPECTED_SPIKE}, {MAX_EXPECTED_SPIKE}]"
    assert (
        x1_spike.max() <= MAX_EXPECTED_SPIKE
    ), f"generate_spike x1 max ({x1_spike.max()}) out of expected range [{MIN_EXPECTED_SPIKE}, {MAX_EXPECTED_SPIKE}]"


def test_generate_dataset_output_shapes():
    """Test the output shapes of generate_dataset."""
    n_per_class = 100
    T, D = 16, 3

    X, labels, scaler = generate_dataset(n_per_class=n_per_class, T=T, D=D)

    expected_N = n_per_class * 3  # 3 classes

    assert X.shape == (
        expected_N,
        T,
        D,
    ), f"X shape {X.shape} does not match expected {(expected_N, T, D)}"
    assert labels.shape == (
        expected_N,
    ), f"Labels shape {labels.shape} does not match expected {(expected_N,)}"
    # Scaler should have T*D features (flattened sequence length)
    assert (
        scaler.mean_.shape[0] == T * D
    ), f"Scaler mean shape {scaler.mean_.shape} does not match expected ({T * D},)"
    assert (
        scaler.scale_.shape[0] == T * D
    ), f"Scaler scale shape {scaler.scale_.shape} does not match expected ({T * D},)"


def test_generate_dataset_labels():
    """Test that labels are assigned correctly."""
    n_per_class = 50
    T, D = 16, 3

    X, labels, scaler = generate_dataset(n_per_class=n_per_class, T=T, D=D)

    expected_N = n_per_class * 3
    assert len(labels) == expected_N

    # Labels should be [0]*n_per_class + [1]*n_per_class + [2]*n_per_class
    expected_labels = np.concatenate(
        [np.full(n_per_class, 0), np.full(n_per_class, 1), np.full(n_per_class, 2)]
    )
    np.testing.assert_array_equal(
        labels, expected_labels, err_msg="Labels do not match expected sequence"
    )


def test_generate_dataset_standardization_and_clipping():
    """Test the standardization and clipping process."""
    n_per_class = 200
    T, D = 16, 3

    print("\n--- Testing generate_dataset standardization and clipping (pytest) ---")

    X, labels, scaler = generate_dataset(n_per_class=n_per_class, T=T, D=D)

    N = X.shape[0]

    print(f"Generated dataset shape: {X.shape}")
    print(f"Scaler mean range: [{scaler.mean_.min():.3f}, {scaler.mean_.max():.3f}]")
    print(f"Scaler std range: [{scaler.scale_.min():.3f}, {scaler.scale_.max():.3f}]")

    X_flat_after_proc = X.reshape(N, -1)  # Shape: (N, T*D)
    print(
        f"Final processed data (X) range (channel 0): [{X_flat_after_proc[:, :T].min():.3f}, {X_flat_after_proc[:, :T].max():.3f}]"
    )  # Channel 0
    print(
        f"Final processed data (X) range (channel 1): [{X_flat_after_proc[:, T:2*T].min():.3f}, {X_flat_after_proc[:, T:2*T].max():.3f}]"
    )  # Channel 1
    print(
        f"Final processed data (X) range (channel 2): [{X_flat_after_proc[:, 2*T:].min():.3f}, {X_flat_after_proc[:, 2*T:].max():.3f}]"
    )  # Channel 2
    print(
        f"Final processed data (X) overall range: [{X_flat_after_proc.min():.3f}, {X_flat_after_proc.max():.3f}]"
    )

    # Assert final data is within the clip range
    assert (
        X_flat_after_proc.min() >= -5.0
    ), "Processed data has values below -5.0 after clipping"
    assert (
        X_flat_after_proc.max() <= 5.0
    ), "Processed data has values above 5.0 after clipping"

    print("--- End of standardization and clipping test (pytest) ---\n")
