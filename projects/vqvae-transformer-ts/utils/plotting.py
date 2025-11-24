import matplotlib.pyplot as plt
from src.plot_style import colors


def plot_continuous_samples(samples, scaler, save_path, n_samples=10):
    """Plot continuous time series samples"""
    plt.figure(figsize=(12, 2 * n_samples))
    
    for i in range(min(n_samples, len(samples))):
        # Reshape flat sample to [16, 3]
        flat_sample = samples[i].numpy()
        ts = flat_sample.reshape(1, -1)
        ts_orig = scaler.inverse_transform(ts).reshape(-1, 3)
        
        plt.subplot(n_samples, 1, i+1)
        plt.plot(ts_orig[:, 0], 'o-')
        plt.title(f"Continuous Sample {i+1}")
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

