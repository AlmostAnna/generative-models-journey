import numpy as np
import matplotlib.pyplot as plt
import torch

from data.generate_synthetic import generate_dataset

from models.vqvae import VQVAETimeSeries
from models.transformer import TimeSeriesTransformer
from models.continuous_transformer import ContinuousTimeSeriesTransformer

from utils.generation import sample_sequence, tokens_to_time_series, sample_continuous

from src.plot_style import colors


def generate_vqvae_samples(n_samples, device):
    # === Load VQ-VAE + Transformer ===
    vqvae = VQVAETimeSeries(n_codes=64, code_dim=16, n_tokens=4).to(device)
    vqvae.load_state_dict(torch.load("vqvae.pth", map_location=device))
    vqvae.eval()

    transformer = TimeSeriesTransformer(n_codes=64, n_tokens=4).to(device)
    transformer.load_state_dict(torch.load("transformer.pth", map_location=device))
    transformer.eval()

    # Generate VQ samples
    vqvae_samples = []
    for _ in range(n_samples):
        tokens = sample_sequence(transformer, n_tokens=4, temperature=1.0)
        ts = tokens_to_time_series(tokens, vqvae)
        vqvae_samples.append(ts)
    return vqvae_samples

def generate_continuous_samples(n_samples, seq_len, first_vals, d_model, n_heads, n_layers, dropout, device):
    model_for_sampling = ContinuousTimeSeriesTransformer(
        seq_len,
        d_model,
        n_heads,
        n_layers,
        dropout=0.0
    ).to(device)
    model_for_sampling.load_state_dict(torch.load("continuous_transformer.pth"))
    model_for_sampling.eval()

    generated_samples = []
    for _ in range(n_samples):
        sample_flat = sample_continuous(
            model_for_sampling,
            first_vals,
            seq_len,
            temperature=1.0,  # noise for diversity
            device=device
        )
        ts_cont = sample_flat.numpy().reshape(16, 3)
        generated_samples.append(ts_cont)

    return generated_samples



def create_preview_plot():
    plt.rcParams.update({
        'font.size': 10,
        'axes.linewidth': 0.8,
        'xtick.major.width': 0.8,
        'ytick.major.width': 0.8
    })
    
    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=(8, 4))
    fig.suptitle('VQ-VAE + Transformer vs Continuous Transformer\nGenerated Time Series Samples', 
                 fontsize=12, y=0.98)
    
    device = torch.device('cpu')
    
    # Get scaler for inverse transform and first values for continuous generation
    X_real, labels_real, scaler = generate_dataset(
          n_per_class=100,
          T=16,
          D=3
    )
    X_real = torch.tensor(X_real, dtype=torch.float32)  # Keep on CPU for now
    X_flat = X_real.reshape(X_real.size(0), -1)         # [N, 48]
    X_flat = X_flat.to(device)

    first_vals = X_flat[:, 0].cpu().numpy()  # [N,] — first value of each sequence

    # Generate 3 samples from each model
    n_samples = 3
    seq_length = 48
    vqvae_samples = []
    cont_samples = []
    
    # Generate samples from both models
    print("Generating samples from the models...")
    vqvae_samples = generate_vqvae_samples(n_samples, device=device)
    
    cont_samples = generate_continuous_samples(n_samples, seq_len=seq_length, first_vals=first_vals, d_model=32, n_heads=2, n_layers=2, dropout=0.0, device=device)

    # Plot samples
    for i in range(n_samples):
        # VQ-VAE samples (top row)
        vq_ts = scaler.inverse_transform(vqvae_samples[i].reshape(1, -1)).reshape(16, 3)
        axes[0, i].plot(vq_ts[:, 0], 'o-', color=colors['primary'], markersize=3, linewidth=1.5)
        axes[0, i].set_title(f'VQ-VAE Sample {i+1}', fontsize=9)
        axes[0, i].set_xticks([0, 8, 15])
        axes[0, i].set_yticks([])
        axes[0, i].grid(True, alpha=0.3, linewidth=0.5)
        
        # Continuous samples (bottom row)
        cont_ts = scaler.inverse_transform(cont_samples[i].reshape(1, -1)).reshape(16, 3)
        axes[1, i].plot(cont_ts[:, 0], 's-', color=colors['secondary'], markersize=3, linewidth=1.5)
        axes[1, i].set_title(f'Continuous Sample {i+1}', fontsize=9)
        axes[1, i].set_xticks([0, 8, 15])
        axes[1, i].set_yticks([])
        axes[1, i].grid(True, alpha=0.3, linewidth=0.5)
    
    # Adjust layout
    plt.tight_layout(pad=1.5)
    
    # Save with web optimization
    plt.savefig('plots/comparison_preview.png', 
                dpi=150, 
                bbox_inches='tight', 
                facecolor='white',
                edgecolor='none')
    plt.close()
    
    print("Preview plot saved to plots/comparison_preview.png")

if __name__ == "__main__":
    create_preview_plot()

