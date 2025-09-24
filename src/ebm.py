import numpy as np

import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm

#EBM model
class EnergyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            spectral_norm(nn.Linear(2, 64)),  
            nn.ReLU(),
            spectral_norm(nn.Linear(64, 64)),
            nn.ReLU(),
            spectral_norm(nn.Linear(64, 1))
        )
    
    def forward(self, x):
        return self.net(x).squeeze(-1) # scalar energy

#Annealed Langevin Sampler
def annealed_langevin_sampler(model, x_init, n_steps=100, noise_scale=1.0, burn_in=50):
    x = x_init.detach().clone()  # Start without gradient history
    samples = []

    for i in range(n_steps):
        # Just Langevin: constant step, constant noise
        #step_size = 0.1
        #current_noise_std = noise_scale

        # Annealed Langevin: variate noise based on the step
        #Variate step size
        t = i / n_steps
        step_size = 0.1 * (1 - t)
        current_noise_std = noise_scale * (1 - t)

        # Enable gradient tracking for this step
        with torch.enable_grad():
            x = x.requires_grad_(True)
            
            energy = model(x).sum()

            # Compute gradient
            grad = torch.autograd.grad(energy, x, retain_graph=False)[0]  # Free memory

        # Update
        noise = torch.randn_like(x) * np.sqrt(2 * step_size) * current_noise_std
        x = x.detach() - step_size * grad.detach() + noise  # Detach for next step

        if i >= burn_in:  # only save after burn-in
            samples.append(x.detach().clone())
    return torch.stack(samples)

# Two Gaussians Mixture Model
class TGMixtureEnergy(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        # Two Gaussians at (-2,0) and (2,0)
        d1 = torch.sum((x - torch.tensor([-2., 0.]))**2, dim=1)
        d2 = torch.sum((x - torch.tensor([2., 0.]))**2, dim=1)
        return -torch.log(torch.exp(-0.5 * d1) + torch.exp(-0.5 * d2) + 1e-8)

# Sample from Two Gaussians Mixture Model
def sample_two_mixture(n_samples=1000):
    modes = torch.randint(0, 2, (n_samples,))
    centers = torch.tensor([[-2.0, 0.0], [2.0, 0.0]])  # Now distance = 4
    x = centers[modes]
    x += 0.5 * torch.randn_like(x)  # Smaller noise: σ = 0.5
    return x    


