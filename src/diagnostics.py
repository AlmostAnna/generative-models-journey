import numpy as np
import arviz as az
import matplotlib.pyplot as plt

import pandas as pd

def plot_diagnostics_grid_az(sampler_samples_dict, max_lag=50, title=None):
    n_samplers = len(sampler_samples_dict)
    
    # Create a big figure with 2 columns (for x0, x1) and n_samplers rows
    fig, axes = plt.subplots(n_samplers, 2, figsize=(10, 3 * n_samplers), 
                             sharex=True)
    if n_samplers == 1:
        axes = axes.reshape(1, -1)
    
    for idx, (name, samples) in enumerate(sampler_samples_dict.items()):
        # Convert to InferenceData: (chains=1, samples, dim=2)
        idata = az.convert_to_inference_data(samples[None, :, :])
        
        # Get the row of axes for this sampler
        ax_row = axes[idx] if n_samplers > 1 else axes
        
        # Plot autocorrelation for each coordinate separately
        az.plot_autocorr(
            data=idata,
            var_names=["x"],
            coords={"x_dim_0": 0},  # Select first coordinate (x0)
            ax=ax_row[0],
            max_lag=max_lag
        )
        
        az.plot_autocorr(
            data=idata,
            var_names=["x"],
            coords={"x_dim_0": 1},  # Select second coordinate (x1)
            ax=ax_row[1],
            max_lag=max_lag
        )
        
        # Add title to the row
        ax_row[0].set_title(f"{name} - $x_1$")
        ax_row[1].set_title(f"{name} - $x_2$")

    # Label common x-axis
    fig.text(0.5, 0.02, "   ", ha='center', fontsize=12)
    # Add overall title if provided
    if title:
        fig.suptitle(title, fontsize=14, y=0.98)
    
    plt.tight_layout()
    if title:
        plt.subplots_adjust(top=0.93, bottom=0.08)  # Adjust for both suptitle and xlabel
    else:
        plt.subplots_adjust(bottom=0.08)  # Make room for xlabel
    plt.show()


def plot_diagnostics_grid_np(sampler_samples_dict, max_lag=50, title=None):
    n_samplers = len(sampler_samples_dict)
    
    # Create a big figure with 2 columns (for x0, x1) and n_samplers rows
    fig, axes = plt.subplots(n_samplers, 2, figsize=(10, 2.5 * n_samplers), 
                             sharex=True)
    if n_samplers == 1:
        axes = axes.reshape(1, -1)
    
    for idx, (name, samples) in enumerate(sampler_samples_dict.items()):
        # Get the row of axes for this sampler
        ax_row = axes[idx] if n_samplers > 1 else axes
        
        # Calculate autocorrelation manually for each coordinate
        coord0_samples = samples[:, 0]  # First coordinate
        coord1_samples = samples[:, 1]  # Second coordinate
        
        # Calculate autocorrelation using numpy
        def autocorr(x, max_lag):
            result = []
            mean_x = np.mean(x)
            var_x = np.var(x)
            for lag in range(max_lag + 1):
                if lag == 0:
                    result.append(1.0)
                else:
                    correlation = np.corrcoef(x[:-lag], x[lag:])[0, 1]
                    result.append(correlation)
            return np.array(result)
        
        # Calculate autocorrelations
        acf0 = autocorr(coord0_samples, max_lag)
        acf1 = autocorr(coord1_samples, max_lag)
        
        # Plot manually
        lags = np.arange(max_lag + 1)
        ax_row[0].plot(lags, acf0, 'b-', alpha=0.7)
        ax_row[0].axhline(y=0, color='k', linestyle='--', alpha=0.3)
        ax_row[0].fill_between(lags, -1/np.sqrt(len(coord0_samples)), 
                              1/np.sqrt(len(coord0_samples)), alpha=0.2, color='gray')
        ax_row[0].set_xlim(0, max_lag)
        ax_row[0].set_ylim(-0.2, 1.0)
        ax_row[0].grid(True, alpha=0.3)
        
        ax_row[1].plot(lags, acf1, 'b-', alpha=0.7)
        ax_row[1].axhline(y=0, color='k', linestyle='--', alpha=0.3)
        ax_row[1].fill_between(lags, -1/np.sqrt(len(coord1_samples)), 
                              1/np.sqrt(len(coord1_samples)), alpha=0.2, color='gray')
        ax_row[1].set_xlim(0, max_lag)
        ax_row[1].set_ylim(-0.2, 1.0)
        ax_row[1].grid(True, alpha=0.3)
        
        # Add title to the row
        ax_row[0].set_title(f"{name} - $x_1$")
        ax_row[1].set_title(f"{name} - $x_2$")

    # Label common x-axis
    fig.text(0.5, 0.01, "   ", ha='center', fontsize=12)
    # Add overall title if provided
    if title:
        fig.suptitle(title, fontsize=14, y=0.98)
    
    plt.tight_layout()
    if title:
        plt.subplots_adjust(top=0.93, bottom=0.08)  # Adjust for both suptitle and xlabel
    else:
        plt.subplots_adjust(bottom=0.08)  # Make room for xlabel
    plt.show()

