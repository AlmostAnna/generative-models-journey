"""
MCMC Diagnostics Toolkit
========================

A collection of functions for evaluating MCMC samplers on multi-modal distributions.
Designed with the lessons from the 'Sampling Is Not Solved' notebook.

Key principles:
- Never trust ESS alone
- Visualize everything
- Assume silent failure until proven otherwise
"""

import numpy as np
import arviz as az
import matplotlib.pyplot as plt
from scipy.stats import wasserstein_distance

import pandas as pd

def compute_ess_safe(samples, method="bulk"):
    """
    Compute ESS with safety: clips to number of samples.
    
    Args:
        samples: (N, D) array of samples
        method: "bulk", "tail", or "quantile"
    
    Returns:
        float: min ESS across dimensions, clipped at N
    """
    n_samples = len(samples)
    idata = az.convert_to_inference_data(samples[None, :, :])  # (chain=1, sample, dim)
    ess_raw = az.ess(idata, method=method)["x"].min().item()  # worst-case dim
    return min(ess_raw, n_samples)


def compute_w1_distance(true_samples, sampler_samples, n=1000):
    """
    Compute 1-Wasserstein distance between two samples (simplified).
    
    Args:
        true_samples: (N1, D)
        sampler_samples: (N2, D)
        n: number of points to compare
    
    Returns:
        float: sum of 1D W₁ distances
    """
    n = min(n, len(true_samples), len(sampler_samples))
    w1 = 0.0
    for i in range(true_samples.shape[1]):
        w1 += wasserstein_distance(
            true_samples[:n, i],
            sampler_samples[:n, i]
        )
    return w1


def analyze_chain_diversity(sampler_samples_dict, threshold=1.0):
    """
    Check if chains explored diverse regions.
    
    Args:
        sampler_samples_dict: dict of name -> (N, 2) samples
        threshold: min distance between chain means to count as "diverse"
    
    Returns:
        dict: per-sampler diversity report
    """
    report = {}
    for name, samples in sampler_samples_dict.items():
        mean_x = samples.mean(axis=0)
        std_x = samples.std(axis=0)
        
        # If you ran multiple runs, split them
        if len(samples) > 1500:
            mid = len(samples) // 2
            chain1_mean = samples[:mid].mean(axis=0)
            chain2_mean = samples[mid:].mean(axis=0)
            dist_between = np.linalg.norm(chain1_mean - chain2_mean)
            diverse = dist_between > threshold
        else:
            chain1_mean = chain2_mean = mean_x
            dist_between = 0.0
            diverse = False
        
        report[name] = {
            'overall_mean': mean_x,
            'chain_separation': dist_between,
            'diverse': diverse,
            'std': std_x
        }
    return report


def full_diagnostic_report(true_samples, sampler_samples_dict, tag=""):
    """
    Print a complete diagnostic summary.
    """
    print(f"\n{'='*40}")
    print(f"DIAGNOSTIC REPORT {tag}")
    print(f"{'='*40}")
    
    results = {}
    for name, samples in sampler_samples_dict.items():
        ess = compute_ess_safe(samples)
        w1 = compute_w1_distance(true_samples, samples)
        results[name] = {'ESS': ess, 'W1': w1}
        print(f"{name:15} ESS: {ess:6.1f}  W1: {w1:5.3f}")
    
    # Chain diversity
    print(f"\nChain Diversity Check (separation > 1.0?):")
    diversity = analyze_chain_diversity(sampler_samples_dict)
    for name, d in diversity.items():
        sep = d['chain_separation']
        mark = "Yes" if d['diverse'] else "No"
        print(f"  {name}: {sep:.3f} {mark}")
    
    return results

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

