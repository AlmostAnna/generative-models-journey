import numpy as np
import torch
import matplotlib.pyplot as plt
from cycler import cycler

from pathlib import Path
from scipy.stats import gaussian_kde

# Handle imports
try:
    from .plot_style import colors
    from .utils import get_data
    from .diffusion import ddim_sample
    from .ebm import annealed_langevin_sampler
except ImportError:
    from plot_style import colors
    from utils import get_data
    from diffusion import ddim_sample
    from ebm import annealed_langevin_sampler



def get_color_palette():
    """Returns the configured color palette."""
    return colors

def plot_real_data(x_real, title="Two-Moons Dataset (Normalized)", save_path=None):
    plt.figure(figsize=(6,6))
    plt.scatter(x_real[:, 0], x_real[:, 1], s=10, c=colors['real_data'], alpha=0.6, linewidth=0)
    plt.title(title, fontsize=14)
    plt.xlabel("$x_1$", fontsize=12)
    plt.ylabel("$x_2$", fontsize=12)
    plt.xlim(-3, 3)
    plt.ylim(-3, 3)
    plt.grid(True, alpha=0.3)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.tight_layout()
    if save_path: 
        plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.show()

def plot_real_and_generated(real_data, gen_data, distr_title, real_title, gen_title):
    # Create figure and axes objects
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    # Add the main title for the entire figure
    fig.suptitle(distr_title, fontsize=16)

    # Plot on specific axes
    ax1.scatter(real_data[:,0], real_data[:,1], s=5, c=colors['real_data'], alpha=0.5)
    ax1.set_title(real_title)

    ax2.scatter(gen_data[:,0], gen_data[:,1], s=5, c=colors['generated_data'], alpha=0.5)
    ax2.set_title(gen_title)

    fig.tight_layout(rect=[0, 0, 1, 0.95]) # Leave space at the top
    plt.show()

def plot_comparison(data_2_plot, main_title=""):
    """
    Plots scatter plots for multiple datasets in subplots.

    Args:
        data_2_plot (dict): A dictionary where keys are titles and values are data arrays (e.g., torch tensors or numpy).
        main_title (str, optional): The main title for the entire figure. Defaults to "".
    """
    # Create figure and axes objects
    fig, axs = plt.subplots(ncols=len(data_2_plot), nrows=1,
                            figsize=(12, 4), layout='constrained')

    # Add the main title for the entire figure
    fig.suptitle(main_title, fontsize=16)

    global_color_cycle_list = plt.rcParams['axes.prop_cycle'].by_key()['color']
    # Create an iterator over this list
    color_cycle_iterator = iter(global_color_cycle_list)
    
    for ax, (title, samples) in zip(axs, data_2_plot.items()):
        # Convert samples to numpy if they are torch tensors
        x_data = samples[:, 0].numpy() if hasattr(samples, 'numpy') else samples[:, 0]
        y_data = samples[:, 1].numpy() if hasattr(samples, 'numpy') else samples[:, 1]

        if title == "Real Data":
            # Use specific color for Real Data
            ax.scatter(x_data, y_data, s=5, c=get_color_palette()['real_data'], alpha=0.6)
        else:
            #ax.scatter(x_data, y_data, s=5, alpha=0.6) # Uses global cycle
            # Explicitly get the next color from the global cycle list
            try:
                next_color = next(color_cycle_iterator)
            except StopIteration:
                # If we run out of colors in the list, loop back to the beginning
                color_cycle_iterator = iter(global_color_cycle_list) # Reset iterator
                next_color = next(color_cycle_iterator)             # Get first color again

            # Use the explicitly fetched color for this scatter plot
            ax.scatter(x_data, y_data, s=5, c=next_color, alpha=0.6)

        ax.set_title(title)
    plt.show() # Display the plot

def plot_true_contour_and_samples(true_samples, sampler_samples_dict, name=''):
    n_samplers = len(sampler_samples_dict)
    total_plots = n_samplers + 1  # +1 for true distribution plot
    
    # Calculate grid dimensions
    n_cols = 2
    n_rows = (total_plots + 1) // 2  # This ensures we have enough rows
    
    # Create a grid for contour
    x_min, x_max = true_samples[:, 0].min() - 1, true_samples[:, 0].max() + 1
    y_min, y_max = true_samples[:, 1].min() - 1, true_samples[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), 
                         np.linspace(y_min, y_max, 100))
    grid_points = np.c_[xx.ravel(), yy.ravel()]

    # Use KDE to estimate density
    kde = gaussian_kde(true_samples.T)
    Z = kde(grid_points.T).reshape(xx.shape)

    # Create subplots with dynamic sizing
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 5*n_rows))
    
    # Handle case where there's only one row or one subplot
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    elif total_plots == 1:  # Only true samples, no samplers
        axes = axes.reshape(1, -1)
    axes = axes.ravel()

    # True samples (contour + scatter) - always in first subplot
    ax = axes[0]
    ax.contour(xx, yy, Z, levels=10, alpha=0.5, colors=colors['true_contour'], linewidths=0.5)
    ax.scatter(true_samples[:, 0], true_samples[:, 1], alpha=0.3, s=1, c=colors['real_data'], label='True Samples')
    # ax.scatter(true_samples[:, 0], true_samples[:, 1], alpha=0.3, s=1, c='tab:blue', label='True Samples') # Uses default cycle's blue
    
    ax.set_title(f"True Distribution ({name})")
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.axis("equal")

    # Plot each sampler
    for idx, (name, samples) in enumerate(sampler_samples_dict.items()):
        ax = axes[idx + 1]  # +1 because first subplot is for true distribution
        ax.contour(xx, yy, Z, levels=10, alpha=0.5, colors=colors['true_contour'], linewidths=0.5)
        ax.scatter(samples[:, 0], samples[:, 1], alpha=0.6, s=10, label=name)
        ax.set_title(f"{name} Samples vs True Contour")
        ax.set_xlabel("$x_1$")
        ax.set_ylabel("$x_2$")
        ax.axis("equal")

    # Hide any unused subplots
    for idx in range(total_plots, len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()
    plt.show()    

def plot_chain_trajectories(trace, title="MCMC Chain Trajectories"):
    plt.figure(figsize=(8, 6))
    
    # Plot trajectory of first 10 chains
    for i in range(min(10, trace.shape[1])):
        plt.plot(trace[:, i, 0], trace[:, i, 1], '-', alpha=0.6, linewidth=1)

    # Mark start and end
    plt.scatter(trace[0, :, 0], trace[0, :, 1], s=30, label='Start', zorder=5)
    plt.scatter(trace[-1, :, 0], trace[-1, :, 1], s=30, label='End', zorder=5)

    plt.scatter([-2, 2], [0, 0], s=100, marker='x', zorder=5, label='True Modes')

    plt.title(title)
    plt.xlabel('$x_1$'); plt.ylabel('$x_2$')
    plt.legend()
    plt.xlim(-4, 4); plt.ylim(-4, 4)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.tight_layout()
    plt.show()

    return trace

def plot_historgam(times):
    colors = get_color_palette()

    plt.figure(figsize=(8, 4))
    if len(times) > 0:
        plt.hist(times, bins=30, alpha=0.7, color=colors['neutral'], edgecolor='black')
        mean_cross = times.mean()
        plt.axvline(mean_cross, color=colors['accent1'], linestyle='--', label=f'Mean = {mean_cross:.1f}')
    else:
        plt.text(0.5, 0.5, 'No chains crossed', transform=plt.gca().transAxes, 
                 ha='center', va='center', fontsize=14, color=colors['dark'])

    plt.title("First Crossing Time into Opposite Mode")
    plt.xlabel("Step")
    plt.ylabel("Number of Chains")
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_energy_landscape_unified(model, data=None, title="Energy Landscape", 
                                style='filled', colormap='coolwarm', show_data=True):
    """
    Unified energy landscape plotting function
    
    Parameters:
    - style: 'filled', 'lines', or 'both'
    - colormap: matplotlib colormap name
    - show_data: whether to overlay data points if provided
    """
    xx = torch.linspace(-4, 4, 100)
    yy = torch.linspace(-4, 4, 100)
    X, Y = torch.meshgrid(xx, yy, indexing='ij')
    grid = torch.stack([X.ravel(), Y.ravel()], dim=1)
    
    with torch.no_grad():
        energies = model(grid).reshape(100, 100)
    
    plt.figure(figsize=(6, 6))
    
    if style in ['filled', 'both']:
        # Filled contours
        cs_fill = plt.contourf(X.numpy(), Y.numpy(), energies.numpy(), 
                              levels=50, cmap=colormap, alpha=0.8)
        if style == 'filled':
            plt.colorbar(cs_fill, label='Energy')
    
    if style in ['lines', 'both']:
        # Contour lines
        levels = np.linspace(energies.min(), energies.max(), 15)
        cs_lines = plt.contour(X.numpy(), Y.numpy(), energies.numpy(), 
                              levels=levels, alpha=0.7, colors=colors['true_contour'], linewidths=0.5)
    
    if show_data and data is not None:
        plt.scatter(data[:,0], data[:,1], s=3, c=colors['real_data'], alpha=0.6, 
                   label='True Data', zorder=5)
    
    plt.title(title)
    plt.xlabel('$x_1$'); plt.ylabel('$x_2$')
    if show_data and data is not None:
        plt.legend()
    plt.gca().set_aspect('equal', adjustable='box')
    plt.tight_layout()
    plt.show()


def plot_real_vs_generated(
    model, 
    device, 
    step, 
    n_samples=1000, 
    loss=None,
    loss_history=None,
    step_history=None,
    figures_path="figures",
    generated_data=None,  # accept pre-generated data
    sampling_function=None,  # accept sampling function
    enforce_positive_y=False
):
    """
    Sample from model and plot real vs generated + loss curve.
    """
    model.eval() if model else None

    # Get real data
    x_real = get_data(n_samples).cpu().numpy()

    # Generate samples
    if generated_data is not None:
        # Use pre-generated data
        x_gen = generated_data
    elif model is not None and sampling_function is not None:
        # Use provided sampling function
        with torch.no_grad():
            x_gen = sampling_function(model, n_samples, device).cpu().numpy()
    elif model is not None:
        # Default: use DDIM sampling (backward compatibility)
        with torch.no_grad():
            x_gen = ddim_sample(model, steps=50, n=n_samples, device=device).cpu().numpy()
    else:
        raise ValueError("Must provide either generated_data or model with sampling function")

    # Create 2x2 plot
    fig = plt.figure(figsize=(12, 8))
    
    # Real data
    plt.subplot(2, 2, 1)
    plt.scatter(x_real[:, 0], x_real[:, 1], s=8, c=colors['real_data'], alpha=0.7)
    plt.xlim(-4, 4)
    plt.ylim(-4, 4)
    plt.title("Real Data")
    plt.gca().set_aspect('equal', adjustable='box')
    plt.grid(True, alpha=0.3)

    # Generated data
    plt.subplot(2, 2, 2)
    plt.scatter(x_gen[:, 0], x_gen[:, 1], s=8, c=colors['generated_data'], alpha=0.7)
    plt.xlim(-4, 4)
    plt.ylim(-4, 4)
    plt.title(f"Generated (Step {step})")
    plt.gca().set_aspect('equal', adjustable='box')
    plt.grid(True, alpha=0.3)

    # Loss curve
    plt.subplot(2, 1, 2)  # Span full width
    if loss_history and len(loss_history) > 1:
        plt.plot(step_history, loss_history, color=colors['loss_curve'], linewidth=2, label='Training Loss')
        if step_history and loss_history:
            plt.scatter(step_history[-1], loss_history[-1], color=colors['loss_curve'], zorder=5)
    plt.xlabel("Training Step")
    plt.ylabel("Loss")
    plt.title("Loss Curve")
    plt.grid(True, alpha=0.3)
    if enforce_positive_y:
        plt.ylim(bottom=0)
    if loss_history and len(loss_history) > 1:
        plt.legend()

    plt.tight_layout()
    
    # Save figure
    figures_path = Path(figures_path)
    figures_path.mkdir(exist_ok=True)
    plt.savefig(figures_path / f"step_{step:05d}.png", dpi=120, bbox_inches='tight')
    plt.show()
    
    model.train() if model else None

def plot_loss_curve(loss_history, step_history, save_path=None, enforce_positive_y=False):
    """Plot just the loss curve."""
    plt.figure(figsize=(10, 6))
    plt.plot(step_history, loss_history, color=colors['loss_curve'], linewidth=2)
    plt.xlabel("Training Step")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.grid(True, alpha=0.3)
    if enforce_positive_y:
        plt.ylim(bottom=0)
    
    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.show()

