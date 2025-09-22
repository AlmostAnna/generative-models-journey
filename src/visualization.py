import torch
import matplotlib.pyplot as plt
from pathlib import Path

# Handle imports
try:
    from .utils import get_data
    from .diffusion import ddim_sample
except ImportError:
    from utils import get_data
    from diffusion import ddim_sample

def plot_real_vs_generated(
    model, 
    device, 
    step, 
    n_samples=1000, 
    loss=None,
    loss_history=None,
    step_history=None,
    figures_path="figures"
):
    """
    Sample from model and plot real vs generated + loss curve.
    """
    model.eval()
    
    # Get real data
    x_real = get_data(n_samples).cpu().numpy()
    
    # Generate samples
    with torch.no_grad():
        x_gen = ddim_sample(model, steps=50, n=n_samples, device=device).cpu().numpy()

    # Create 2x2 plot
    fig = plt.figure(figsize=(12, 8))
    
    # Real data
    plt.subplot(2, 2, 1)
    plt.scatter(x_real[:, 0], x_real[:, 1], s=8, c='blue', alpha=0.7)
    plt.xlim(-4, 4)
    plt.ylim(-4, 4)
    plt.title("Real Data")
    plt.gca().set_aspect('equal', adjustable='box')
    plt.grid(True, alpha=0.3)

    # Generated data
    plt.subplot(2, 2, 2)
    plt.scatter(x_gen[:, 0], x_gen[:, 1], s=8, c='red', alpha=0.7)
    plt.xlim(-4, 4)
    plt.ylim(-4, 4)
    plt.title(f"Generated (Step {step})")
    plt.gca().set_aspect('equal', adjustable='box')
    plt.grid(True, alpha=0.3)

    # Loss curve
    plt.subplot(2, 1, 2)  # Span full width
    if loss_history and len(loss_history) > 1:
        plt.plot(step_history, loss_history, color='green', linewidth=2, label='Training Loss')
        if step_history and loss_history:
            plt.scatter(step_history[-1], loss_history[-1], color='green', zorder=5)
    plt.xlabel("Training Step")
    plt.ylabel("Loss")
    plt.title("Loss Curve")
    plt.grid(True, alpha=0.3)
    plt.ylim(bottom=0)
    if loss_history and len(loss_history) > 1:
        plt.legend()

    plt.tight_layout()
    
    # Save figure
    figures_path = Path(figures_path)
    figures_path.mkdir(exist_ok=True)
    plt.savefig(figures_path / f"step_{step:05d}.png", dpi=120, bbox_inches='tight')
    plt.show()
    
    model.train()

def plot_loss_curve(loss_history, step_history, save_path=None):
    """Plot just the loss curve."""
    plt.figure(figsize=(10, 6))
    plt.plot(step_history, loss_history, color='green', linewidth=2)
    plt.xlabel("Training Step")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.grid(True, alpha=0.3)
    plt.ylim(bottom=0)
    
    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.show()

