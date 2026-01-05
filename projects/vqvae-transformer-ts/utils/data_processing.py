"""
Data processing functions.

This module contains functions transforming samples from the models.
"""

import numpy as np


def inverse_scale_samples(scaler, scaled_samples: np.ndarray) -> np.ndarray:
    """
    Apply the scaler's inverse transform.

    Applies the scaler's inverse transform to convert scaled samples back
    to the original data scale.

    Args:
        scaler: The fitted scaler object (e.g., sklearn StandardScaler, MinMaxScaler).
        scaled_samples: A numpy array of samples in the scaled domain,
                        shape (n_samples, ...).

    Returns:
        A numpy array of samples in the original data scale,
        shape (n_samples, ...).
    """
    original_shape = scaled_samples.shape
    # Reshape to 2D if necessary for sklearn scalers (samples, features)
    if scaled_samples.ndim > 2:
        scaled_samples_flat = scaled_samples.reshape(scaled_samples.shape[0], -1)
    else:
        scaled_samples_flat = scaled_samples

    # Apply inverse transform
    original_samples_flat = scaler.inverse_transform(scaled_samples_flat)

    # Reshape back to original shape
    original_samples = original_samples_flat.reshape(original_shape)

    return original_samples
