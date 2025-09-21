import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
import math

def get_alpha_bar(t):
    """
    Cosine noise schedule from DDIM (2020).
    
    Args:
        t: (B,) tensor of time values in [0, 1], where:
           t ≈ 0 → data, t ≈ 1 → noise
    
    Returns:
        (B, 1) tensor of alpha_bar(t)
    """
    s = 0.008
    return (torch.cos((t + s) / (1 + s) * torch.pi / 2) ** 2).unsqueeze(1)


def positional_encoding(t, dim=16):
    """
    Positional encoding for time t ∈ [0,1].
    
    Args:
        t: (B,) tensor of time values
        dim: embedding dimension (must be even)
    
    Returns:
        (B, dim) tensor
    """   
    device = t.device
    B = t.size(0)
    pe = torch.zeros(B, dim, device=device)
    position = t.float().unsqueeze(-1)  # [B, 1]
    div_term = torch.exp(torch.arange(0, dim, 2, device=device).float() * (-math.log(10000.0) / dim))
    div_term = div_term.view(1, -1)  # [1, dim//2]

    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe


class ScoreModel(nn.Module):
    def __init__(self, time_dim=16):
        super().__init__()
        self.time_dim = time_dim
        self.t_proj = nn.Sequential(
            nn.Linear(time_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 32)
        )
        self.net = nn.Sequential(
            nn.Linear(2 + 32, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )
    
    def forward(self, x, t):
        # t: [B], x: [B, 2]
        t_emb = positional_encoding(t, self.time_dim)  # [B, time_dim]
        t_emb = self.t_proj(t_emb)  # [B, 32]
        
        x_t = torch.cat([x, t_emb], dim=1)  # [B, 2+32]
        return self.net(x_t)

# Diffusion sampling
@torch.no_grad()
def ddim_sample(model, steps=100, n=1000, device='cpu'):
    model.eval()
    ts = torch.linspace(1.0, 0.001, steps).to(device)  # Move to device
    x = torch.randn(n, 2).to(device)

    for i in range(steps):
        #t = torch.ones(n).to(device) * ts[i]
        t = torch.ones(n, device=device) * ts[i]  # [n], on device
        # Predict noise
        noise_pred = model(x, t)

        # Compute alpha_bar
        alpha_bar = get_alpha_bar(t).squeeze(1)  # [n]

        # Predict x0
        x0_pred = (x - torch.sqrt(1 - alpha_bar).unsqueeze(1) * noise_pred) / (torch.sqrt(alpha_bar).unsqueeze(1) + 1e-8)

        # Clip to avoid explosion
        x0_pred = torch.clamp(x0_pred, -3.5, 3.5)

        # If we're at the last step, return x0
        if i == steps - 1:
            x = x0_pred
            break

        # Next t
        t_next = torch.ones(n, device=device) * ts[i+1]
        alpha_bar_next = get_alpha_bar(t_next).squeeze(1)


        # Deterministic step: x_{t-1} = sqrt(abar_prev) * x0 + sqrt(1-abar_prev) * noise_pred
        # Recover x0 from noisy x
        x = torch.sqrt(alpha_bar_next).unsqueeze(1) * x0_pred + \
            torch.sqrt(1 - alpha_bar_next).unsqueeze(1) * noise_pred
        
    model.train()
    return x