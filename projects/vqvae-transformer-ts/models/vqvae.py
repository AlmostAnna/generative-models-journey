import torch
import torch.nn as nn
import torch.nn.functional as F


# VQ layer
class VectorQuantizer(nn.Module):
    def __init__(self, n_codes, code_dim):
        super().__init__()
        self.n_codes = n_codes
        self.code_dim = code_dim
        self.codebook = nn.Embedding(n_codes, code_dim)

    
    def forward(self, z_e):
        # z_e: [B, L, code_dim]
        # codebook: [n_codes, code_dim]
    
        # Expand z_e and codebook to [B, L, n_codes, code_dim]
        z_e_exp = z_e.unsqueeze(-2)                     # [B, L, 1, code_dim]
        codebook_exp = self.codebook.weight.unsqueeze(0).unsqueeze(0)  # [1, 1, n_codes, code_dim]
    
        # Compute squared distances
        distances = (z_e_exp - codebook_exp).pow(2).sum(dim=-1)  # [B, L, n_codes]
    
        indices = distances.argmin(dim=-1)  # [B, L]
        z_q = self.codebook(indices)       # [B, L, code_dim]
    
        # Straight-through
        z_q = z_e + (z_q - z_e).detach()
        return z_q, indices


class VQVAETimeSeries(nn.Module):
    def __init__(self, T=16, D=3, n_codes=64, code_dim=16, n_tokens=2):
        super().__init__()
        self.n_tokens = n_tokens
        self.code_dim = code_dim
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(T * D, 128),
            nn.ReLU(),
            nn.Linear(128, n_tokens * code_dim)
        )
        
        # Vector Quantizer
        self.vq_layer = VectorQuantizer(n_codes, code_dim)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(n_tokens * code_dim, 128),
            nn.ReLU(),
            nn.Linear(128, T * D)
        )
        
        self.T, self.D = T, D

    def encode(self, x):
        z_e = self.encoder(x.reshape(x.size(0), -1))
        z_e = z_e.view(x.size(0), self.n_tokens, self.code_dim)
        # Normalize to prevent explosion
        z_e = F.layer_norm(z_e, normalized_shape=(self.code_dim,))
        return z_e

    def decode(self, z_q):
        # z_q: [B, n_tokens, code_dim]
        out = self.decoder(z_q.view(z_q.size(0), -1))
        return out.view(z_q.size(0), self.T, self.D)

    def forward(self, x):
        z_e = self.encode(x)           # continuous latent
        z_q, indices = self.vq_layer(z_e)  # z_q: quantized, indices: [B, n_tokens]
        recon = self.decode(z_q)
        return recon, z_q, indices
