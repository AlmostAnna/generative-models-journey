"""
Training data simulator.

This module contains code for generation of synthetic time series dataset
combining 3 types of regimes:
    'Trending' - slow linear drift + low noise,
    'Mean-Reverting' - oscillation around zero,
    'Volatile Spike' - mostly flat, then sudden spike in middle.
"""

import numpy as np
from sklearn.preprocessing import StandardScaler

# Define parameters for equity-like behavior
# GBM parameters
mu = 0.05 / 252  # Annualized drift per day
sigma_base = 0.2 / np.sqrt(252)  # Annualized vol per sqrt(day)
initial_price = 100.0  # Starting price

# Jump parameters (for spike-like function)
jump_intensity = 1.0 / 100  # Expected frequency (e.g., 1 every 100 steps)
jump_mean = 0.0  # Average jump size
jump_std = 0.3  # Jump size volatility (log scale)


def generate_equity_gbm(T=16, dt=1.0 / 252):  # dt in years
    """Generate equity-like price path using discretized GBM."""
    S = np.zeros(T)
    S[0] = initial_price

    dW = np.random.normal(0, np.sqrt(dt), size=T - 1)

    for t in range(1, T):
        # S_{t+dt} = S_t * exp((mu - 0.5*sigma^2)*dt + sigma*dW)
        S[t] = S[t - 1] * np.exp(
            (mu - 0.5 * sigma_base**2) * dt + sigma_base * dW[t - 1]
        )

    return S


def generate_equity_mean_rev_vol(T=16, dt=1.0 / 252):
    """Generate equity-like price path with mean-reverting volatility."""
    S = np.zeros(T)
    v = np.zeros(T)  # Instantaneous variance
    S[0] = initial_price
    v[0] = sigma_base**2

    kappa = 1.5 / 252  # Speed of mean reversion for variance
    theta = sigma_base**2  # Long-run average variance
    sigma_v = 0.5 / (252**0.5)  # Vol of vol
    rho = -0.7  # Correlation between price and vol shocks

    dW_S = np.random.normal(0, np.sqrt(dt), size=T - 1)  # Price shock
    dW_v_un = np.random.normal(0, np.sqrt(dt), size=T - 1)  # Uncorr vol shock
    # Correlate shocks
    dW_v = rho * dW_S + np.sqrt(1 - rho**2) * dW_v_un

    for t in range(1, T):
        dv = kappa * (theta - v[t - 1]) * dt + sigma_v * np.sqrt(v[t - 1]) * dW_v[t - 1]
        v[t] = max(v[t - 1] + dv, 1e-6)  # Ensure > 0

        S[t] = S[t - 1] * np.exp(
            (mu - 0.5 * v[t - 1]) * dt + np.sqrt(v[t - 1]) * dW_S[t - 1]
        )

    return S


def generate_equity_with_jumps(T=16, dt=1.0 / 252):
    """Generate equity-like price path with occasional jumps."""
    S = np.zeros(T)
    S[0] = initial_price

    dW = np.random.normal(0, np.sqrt(dt), size=T - 1)
    # Bernoulli trial
    jump_occurred = np.random.binomial(1, jump_intensity * dt, size=T - 1)

    for t in range(1, T):
        price_inc = (mu - 0.5 * sigma_base**2) * dt + sigma_base * dW[t - 1]

        # Add jump if occurred
        if jump_occurred[t - 1]:
            jump_size = np.random.normal(jump_mean, jump_std)
            price_inc += jump_size

        S[t] = S[t - 1] * np.exp(price_inc)

    return S


def generate_trending(T=16, dt=1.0 / 252):
    """Generate equity-like trending type samples."""
    x1 = generate_equity_gbm(T, dt)
    x2 = 0.2 + np.random.normal(0, 0.05, T)  # TODO: model vol as mean-reverting?
    x3 = np.random.exponential(1, T)  # TODO: model volume dynamics?
    return np.stack([x1, x2, x3], axis=-1)


def generate_mean_reverting(T=16, dt=1.0 / 252):
    """Generate equity-like mean-reverting type samples."""
    x1 = generate_equity_mean_rev_vol(T, dt)
    x2 = 0.2 + np.random.normal(0, 0.05, T)
    x3 = np.random.exponential(1, T)
    return np.stack([x1, x2, x3], axis=-1)


def generate_spike(T=16, dt=1.0 / 252):
    """Generate equity-like volatile spike type samples."""
    x1 = generate_equity_with_jumps(T, dt)
    x2 = 0.2 + np.random.normal(0, 0.05, T)
    x3 = np.random.exponential(1, T)
    return np.stack([x1, x2, x3], axis=-1)


def generate_dataset(n_per_class, T, D):
    """Generate normalized dataset of combined samples."""
    data = []
    labels = []
    types = [generate_trending, generate_mean_reverting, generate_spike]

    for cls_id, gen_func in enumerate(types):
        for _ in range(n_per_class):
            seq = gen_func(T=16)
            data.append(seq)
            labels.append(cls_id)

    data = np.array(data, dtype=np.float32)  # Shape: [9000, 16, 3]
    labels = np.array(labels)

    # Normalize
    N = data.shape[0]
    scaler = StandardScaler()
    data = scaler.fit_transform(data.reshape(N, -1)).reshape(N, T, D).astype(np.float32)

    # Clip extremes
    X = np.clip(data, -5.0, 5.0).astype(np.float32)

    return X, labels, scaler
