"""
Metrics for model evaluation.

This module contains various metrics to access quality of samples
created by the models in the project.
"""

import numpy as np
from scipy.linalg import sqrtm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics.pairwise import pairwise_distances


def extract_features_from_flat(flat_samples, T=16, D=3):
    """
    Extract regime features from flat [N, T*D] arrays.

    Args:
        flat_samples: array of shape [N, 48]
        T: sequence length (default 16)
        D: number of channels (default 3)
    """
    features = []
    for flat_sample in flat_samples:
        # Reshape to [T, D]
        sample_2d = flat_sample.reshape(T, D)  # [16, 3]
        x = sample_2d[:, 0]  # First channel (trend/spike)

        features.append(
            [
                np.std(x),  # volatility
                np.max(x) - np.min(x),  # range
                np.polyfit(np.arange(len(x)), x, 1)[0],  # trend slope
                (
                    np.argmax(np.abs(x - np.mean(x))) if len(x) > 0 else 0
                ),  # spike location
                np.max(np.abs(x)),  # max deviation
            ]
        )
    return np.array(features)


def train_regime_classifier_from_flat(flat_real_samples, real_labels, T=16, D=3):
    """Train classifier on real data (flat format)."""
    features = extract_features_from_flat(flat_real_samples, T, D)
    classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    classifier.fit(features, real_labels)
    return classifier


def regime_accuracy_from_flat(
    gen_flat_samples, real_flat_samples, real_labels, T=16, D=3
):
    """Measure regime distribution match for flat-format samples."""
    # Train classifier on REAL data
    classifier = train_regime_classifier_from_flat(real_flat_samples, real_labels, T, D)

    # Extract features from GENERATED samples
    gen_features = extract_features_from_flat(gen_flat_samples, T, D)

    # Predict regimes for generated samples
    pred_labels = classifier.predict(gen_features)

    # Compare regime distributions
    real_dist = np.bincount(real_labels, minlength=3) / len(real_labels)
    gen_dist = np.bincount(pred_labels, minlength=3) / len(pred_labels)

    # Use cosine similarity (higher = better match)
    from scipy.spatial.distance import cosine

    similarity = 1 - cosine(real_dist, gen_dist)
    return similarity


def diversity_score(samples_array):
    """Measure diversity score(higher = more diverse)."""
    distances = pairwise_distances(samples_array, metric="euclidean")
    return np.mean(distances)


def fid_score(real_samples_array, gen_samples_array, eps=1e-6):
    """Measure Frechet inception distance."""
    # Compute means
    mu1 = np.mean(real_samples_array, axis=0)
    mu2 = np.mean(gen_samples_array, axis=0)
    ssdiff = np.sum((mu1 - mu2) ** 2)

    # Compute covariances
    sigma1 = np.cov(real_samples_array, rowvar=False)
    sigma2 = np.cov(gen_samples_array, rowvar=False)

    # Add regularization to diagonals (prevents singularity)
    sigma1 += np.eye(sigma1.shape[0]) * eps
    sigma2 += np.eye(sigma2.shape[0]) * eps

    # Compute sqrtm with error handling
    try:
        covmean = sqrtm(sigma1 @ sigma2)
        if np.iscomplexobj(covmean):
            covmean = covmean.real
        # Ensure covmean is finite
        if not np.isfinite(covmean).all():
            print("Warning: covmean contains non-finite values. Using fallback.")
            covmean = np.zeros_like(covmean)
    except Exception as e:
        print(f"sqrtm failed: {e}. Using fallback.")
        covmean = np.zeros_like(sigma1)

    fid = ssdiff + np.trace(sigma1 + sigma2 - 2 * covmean)
    return fid
