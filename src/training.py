"""
Training Module and Command-Line Interface for Generative Models.

This module provides functions to train different types of generative models,
specifically diffusion models and energy-based models (EBMs), on a 2D dataset
(e.g., two-moons).

It includes:
- `train_model`: Function to train a diffusion model using score matching.
- `train_ebm_model`: Function to train an Energy-Based Model using contrastive
  divergence and Langevin dynamics.
- `gradient_penalty`: Helper function for EBM training.
- `import_dependencies`: Helper function to manage optional dependencies like
  `tqdm` gracefully.

When run as a script (e.g., `python training.py diffusion`), it provides a
command-line interface to execute the training functions with configurable
parameters like number of epochs, batch size, learning rate, and device.

Example Usage as a Script:
    # Train a diffusion model
    python training.py diffusion --epochs 10000 --batch-size 512 --lr 1e-3 --device cuda

    # Train an EBM model, disabling visualization
    python training.py ebm --epochs 2000 --no-visualize --checkpoint-interval 200
"""

from pathlib import Path
from typing import Any, Dict, Tuple

import torch
import torch.nn as nn
import torch.optim as optim


# Handle imports gracefully
def import_dependencies():
    """Import dependencies with proper error handling."""
    global tqdm, get_data, get_alpha_bar, ddim_sample, ScoreModel
    global annealed_langevin_sampler, EnergyModel, plot_real_vs_generated

    # Try to import tqdm
    try:
        from tqdm import tqdm
    except ImportError:
        # Fallback progress bar
        def tqdm(iterable, **kwargs):
            return iterable

        print("tqdm not available, using simple iteration")

    # Import other dependencies
    try:
        from .diffusion import ScoreModel, ddim_sample, get_alpha_bar
        from .ebm import EnergyModel, annealed_langevin_sampler
        from .utils import get_data
        from .visualization import plot_real_vs_generated
    except ImportError:
        # Fallback for direct imports
        from diffusion import ScoreModel, ddim_sample, get_alpha_bar
        from ebm import EnergyModel, annealed_langevin_sampler
        from utils import get_data
        from visualization import plot_real_vs_generated


# Call import function
import_dependencies()


def train_model(
    n_epochs: int = 20000,
    batch_size: int = 1024,
    learning_rate: float = 1e-4,
    device: str = "cpu",
    checkpoint_interval: int = 1000,
    visualize: bool = True,
    figures_path: str = "figures",
) -> Tuple[nn.Module, Dict[str, Any]]:
    """
    Train the diffusion model using score matching.

    Args:
        n_epochs: Number of training steps/epochs. Defaults to 20000.
        batch_size: Size of the training batch. Defaults to 1024.
        learning_rate: Learning rate for the Adam optimizer. Defaults to 1e-4.
        device: Device ('cpu' or 'cuda') for training. Defaults to 'cpu'.
        checkpoint_interval: How often (in steps) to print progress and
                             optionally visualize results. Defaults to 1000.
        visualize: Whether to generate plots during training. Defaults to True.
        figures_path: Directory to save generated plots. Defaults to 'figures'.

    Returns:
        A tuple containing:
        - The trained ScoreModel instance.
        - A dictionary with training history: 'loss_history', 'step_history',
          and 'final_loss'.
    """
    # Initialize model and optimizer
    score_model = ScoreModel()
    optimizer = optim.Adam(score_model.parameters(), lr=learning_rate)

    # Training history
    loss_history = []
    step_history = []

    # Create figures directory
    if visualize:
        figures_path = Path(figures_path)
        figures_path.mkdir(exist_ok=True)

    print(f"Starting training on {device}...")
    print(f"Epochs: {n_epochs}, Batch size: {batch_size}")

    for step in tqdm(
        range(n_epochs), desc="Training", disable=not hasattr(tqdm, "__name__")
    ):
        # Get data
        x_real = get_data(batch_size).to(device)

        # Random noise level t ∈ [0,1]
        t = torch.rand(batch_size, device=device)

        # Add noise
        alpha_bar = get_alpha_bar(t)
        noise = torch.randn_like(x_real)
        x_t = torch.sqrt(alpha_bar) * x_real + torch.sqrt(1 - alpha_bar) * noise

        # Predict noise
        pred_noise = score_model(x_t, t)
        loss = (noise - pred_noise).pow(2).mean()

        # Backprop
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(score_model.parameters(), max_norm=1.0)
        optimizer.step()

        # Store loss
        if step % 100 == 0:
            loss_history.append(loss.item())
            step_history.append(step)

        # Checkpoint and visualize
        if step % checkpoint_interval == 0:
            print(f"Step {step} | Loss: {loss.item():.3f}")

            if visualize:
                plot_real_vs_generated(
                    model=score_model,
                    device=device,
                    step=step,
                    n_samples=1000,
                    loss=loss.item(),
                    loss_history=loss_history,
                    step_history=step_history,
                    figures_path=figures_path,
                )

    return score_model, {
        "loss_history": loss_history,
        "step_history": step_history,
        "final_loss": loss.item(),
    }


def gradient_penalty(model: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Calculate gradient penalty for EBM training."""
    x.requires_grad_(True)
    e = model(x).sum()
    grad = torch.autograd.grad(e, x, create_graph=True)[0]
    return ((grad.norm(2, dim=1) - 1) ** 2).mean()


def train_ebm_model(
    n_epochs: int = 2000,
    batch_size: int = 512,
    learning_rate: float = 1e-4,
    weight_decay: float = 1e-4,
    lambda_gp: float = 0.05,
    device: str = "cpu",
    checkpoint_interval: int = 500,
    visualize: bool = True,
    figures_path: str = "figures",
) -> Tuple[nn.Module, Dict[str, Any]]:
    """
    Train Energy-Based Model using contrastive divergence and Langevin dynamics.

    Args:
        n_epochs: Number of training steps/epochs. Defaults to 2000.
        batch_size: Size of the training batch. Defaults to 512.
        learning_rate: Learning rate for the Adam optimizer. Defaults to 1e-4.
        weight_decay: Weight decay (L2 penalty) for the optimizer.
                      Defaults to 1e-4.
        lambda_gp: Weight for the gradient penalty term. Defaults to 0.05.
        device: Device ('cpu' or 'cuda') for training. Defaults to 'cpu'.
        checkpoint_interval: How often (in steps) to print progress and
                             optionally visualize results. Defaults to 500.
        visualize: Whether to generate plots during training. Defaults to True.
        figures_path: Directory to save generated plots. Defaults to 'figures'.

    Returns:
        A tuple containing:
        - The trained EnergyModel instance.
        - A dictionary with training history: 'loss_history', 'step_history',
          and 'final_loss'.
    """
    # Initialize model and optimizer
    model = EnergyModel().to(device)
    nn.init.zeros_(model.net[-1].bias)
    optimizer = optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )

    loss_history = []
    step_history = []

    print(f"Starting EBM training on {device}...")

    for step in tqdm(range(n_epochs), desc="Training EBM"):
        x_real = get_data(batch_size).to(device)  # Already normalized

        # Positive phase: lower energy of real data
        loss_real = model(x_real).mean()

        # Negative phase: raise energy of model samples
        x_fake = torch.randn_like(x_real) * 2  # Broad initialization

        with torch.no_grad():
            # Run short Langevin and discard initial samples
            x_fake = annealed_langevin_sampler(
                model, x_fake, n_steps=30, noise_scale=1.0, burn_in=10
            )

        # Now compute energy on x_fake — this creates a new gradient path
        loss_fake = model(x_fake).mean()

        gp = gradient_penalty(model, x_real)
        loss = loss_real - loss_fake + lambda_gp * gp  # regularization

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # Store loss
        if step % 100 == 0:
            loss_history.append(loss.item())
            step_history.append(step)

        # Checkpoint
        if step % checkpoint_interval == 0:
            print(
                f"Step {step} | GP: {gp.item():.4f}, scaled: {lambda_gp * gp.item():.4f}"  # noqa: E501
            )
            print(f"Main loss: {(loss_real - loss_fake).item():.4f}")
            print(
                f"EBM Loss: {loss.item():.3f}, E_real: {loss_real.item():.3f}, E_fake: {loss_fake.item():.3f}"  # noqa: E501
            )

            if visualize:
                # Generate samples for visualization
                with torch.no_grad():
                    x_gen = annealed_langevin_sampler(
                        model,
                        torch.randn(1000, 2).to(device) * 2,
                        n_steps=30,
                        noise_scale=1.0,
                        burn_in=10,
                    )

                plot_real_vs_generated(
                    model=None,
                    device=device,
                    step=step,
                    n_samples=1000,
                    loss=loss.item(),
                    loss_history=loss_history,
                    step_history=step_history,
                    figures_path=figures_path,
                    generated_data=x_gen,
                )

    return model, {
        "loss_history": loss_history,
        "step_history": step_history,
        "final_loss": loss.item(),
    }


# Command line interface
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train generative models")
    parser.add_argument(
        "model_type", choices=["diffusion", "ebm"], help="Type of model to train"
    )
    parser.add_argument("--epochs", type=int, default=20000, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=1024, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--device", type=str, default="cpu", help="Device to use")
    parser.add_argument(
        "--no-visualize", action="store_true", help="Disable visualization"
    )
    parser.add_argument(
        "--checkpoint-interval", type=int, default=1000, help="Checkpoint interval"
    )

    args = parser.parse_args()
    if args.model_type == "diffusion":
        model, history = train_model(
            n_epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            device=args.device,
            checkpoint_interval=args.checkpoint_interval,
            visualize=not args.no_visualize,
        )
        print("Training completed!")
        print(f"Final loss: {history['final_loss']:.3f}")
    elif args.model_type == "ebm":
        model, history = train_ebm_model(
            n_epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            device=args.device,
            checkpoint_interval=args.checkpoint_interval,
            visualize=not args.no_visualize,
        )
        print("Training completed!")
        print(f"Final loss: {history['final_loss']:.3f}")
