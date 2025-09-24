import sys
from pathlib import Path

import torch
import torch.optim as optim
import torch.nn as nn


# Handle imports gracefully
def import_dependencies():
    """Import dependencies with proper error handling."""
    global tqdm, get_data, get_alpha_bar, ddim_sample, ScoreModel, annealed_langevin_sampler, EnergyModel, plot_real_vs_generated
    
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
        from .utils import get_data
        from .diffusion import get_alpha_bar, ddim_sample, ScoreModel
        from .ebm import annealed_langevin_sampler, EnergyModel
        from .visualization import plot_real_vs_generated
    except ImportError:
        # Fallback for direct imports
        from utils import get_data
        from diffusion import get_alpha_bar, ddim_sample, ScoreModel
        from ebm import annealed_langevin_sampler, EnergyModel
        from visualization import plot_real_vs_generated

# Call import function
import_dependencies()

def train_model(
    n_epochs=20000,
    batch_size=1024,
    learning_rate=1e-4,
    device='cpu',
    checkpoint_interval=1000,
    visualize=True,
    figures_path="figures"
):
    """
    Train the diffusion model.
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
    
    for step in tqdm(range(n_epochs), desc="Training", disable=not hasattr(tqdm, '__name__')):
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
                    figures_path=figures_path
                )
    
    return score_model, {
        'loss_history': loss_history,
        'step_history': step_history,
        'final_loss': loss.item()
    }

def gradient_penalty(model, x):
    x.requires_grad_(True)
    e = model(x).sum()
    grad = torch.autograd.grad(e, x, create_graph=True)[0]
    return ((grad.norm(2, dim=1) - 1)**2).mean()


def train_ebm_model(
    n_epochs=2000,
    batch_size=512,
    learning_rate=1e-4,
    weight_decay=1e-4,
    lambda_gp=0.05,
    device='cpu',
    checkpoint_interval=500,
    visualize=True,
    figures_path="figures"
):
    """
        Train Energy-Based Model.
    """
    
    # Initialize model and optimizer
    model = EnergyModel().to(device)
    nn.init.zeros_(model.net[-1].bias)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
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
            x_fake = annealed_langevin_sampler(model, x_fake, n_steps=30, noise_scale=1.0, burn_in=10)

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
            print(f"Step {step} | GP: {gp.item():.4f}, scaled: {lambda_gp * gp.item():.4f}")
            print(f"Main loss: {(loss_real - loss_fake).item():.4f}")
            print(f"EBM Loss: {loss.item():.3f}, E_real: {loss_real.item():.3f}, E_fake: {loss_fake.item():.3f}")
            
            if visualize:
                # Generate samples for visualization
                with torch.no_grad():
                    x_gen = annealed_langevin_sampler(model, torch.randn(1000, 2).to(device) * 2, 
                                                    n_steps=30, noise_scale=1.0, burn_in=10)
                
                plot_real_vs_generated(
                    model=None,
                    device=device,
                    step=step,
                    n_samples=1000,
                    loss=loss.item(),
                    loss_history=loss_history,
                    step_history=step_history,
                    figures_path=figures_path,
                    generated_data = x_gen
                )
    
    return model, {
        'loss_history': loss_history,
        'step_history': step_history,
        'final_loss': loss.item()
    }

# Command line interface
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train generative models')
    parser.add_argument('model_type', choices=['diffusion', 'ebm'], 
                       help='Type of model to train')
    parser.add_argument('--epochs', type=int, default=20000, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=1024, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--device', type=str, default='cpu', help='Device to use')
    parser.add_argument('--no-visualize', action='store_true', help='Disable visualization')
    parser.add_argument('--checkpoint-interval', type=int, default=1000, help='Checkpoint interval')
    
    args = parser.parse_args()
    if args.model_type == 'diffusion':
        model, history = train_model(
            n_epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            device=args.device,
            checkpoint_interval=args.checkpoint_interval,
            visualize=not args.no_visualize
        )
        print("Training completed!")
        print(f"Final loss: {history['final_loss']:.3f}")
    elif args.model_type == 'ebm':
        model, history = train_ebm_model(
            n_epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            device=args.device,
            checkpoint_interval=args.checkpoint_interval,
            visualize=not args.no_visualize
        )
        print("Training completed!")
        print(f"Final loss: {history['final_loss']:.3f}")
