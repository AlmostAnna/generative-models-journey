import numpy as np
from sklearn.preprocessing import StandardScaler

def generate_trending(T=16):
    trend = np.linspace(0, 1, T) * 0.5
    noise = np.random.normal(0, 0.1, T)
    x1 = trend + noise                    # price-like
    x2 = 0.2 + np.random.normal(0, 0.05, T)  # vol
    x3 = np.random.exponential(1, T)     # volume
    return np.stack([x1, x2, x3], axis=-1)

def generate_mean_reverting(T=16):
    x1 = np.zeros(T)
    for t in range(1, T):
        x1[t] = x1[t-1] - 0.2 * x1[t-1] + np.random.normal(0, 0.1)
    x2 = 0.2 + np.random.normal(0, 0.05, T)
    x3 = np.random.exponential(1, T)
    return np.stack([x1, x2, x3], axis=-1)

def generate_spike(T=16):
    x1 = np.random.normal(0, 0.1, T)
    spike_at = T // 2
    x1[spike_at] += np.random.choice([1.5, -1.5])  # random up/down spike
    x2 = 0.2 + np.random.normal(0, 0.05, T)
    x3 = np.random.exponential(1, T)
    return np.stack([x1, x2, x3], axis=-1)

def generate_dataset(n_per_class, T, D):
    # Generate dataset
    data = []
    labels = []
    types = [generate_trending, generate_mean_reverting, generate_spike]

    for cls_id, gen_func in enumerate(types):
        for _ in range(n_per_class):
            seq = gen_func(T=16)
            data.append(seq)
            labels.append(cls_id)

    data = np.array(data, dtype=np.float32) # Shape: [9000, 16, 3]
    labels = np.array(labels)
    
    # Normalize
    N = data.shape[0]
    scaler = StandardScaler()
    data = scaler.fit_transform(data.reshape(N, -1)).reshape(N, T, D).astype(np.float32)

    # Clip extremes
    X = np.clip(data, -5.0, 5.0).astype(np.float32)
    
    return X, labels, scaler