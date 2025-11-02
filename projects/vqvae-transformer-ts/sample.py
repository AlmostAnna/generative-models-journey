import torch
import torch.optim as optim
import torch.nn as nn

import numpy as np
import torch.nn.functional as F

import matplotlib.pyplot as plt

def sample_sequence(model, n_tokens=4, temperature=1.0, device="cpu"):
    model.eval()
    with torch.no_grad():
        # Sample first token uniformly
        probs = torch.ones(model.n_codes, device=device)  # uniform probabilities
        first_token = torch.multinomial(probs, 1).unsqueeze(0)  # [1, 1]
        seq = first_token
        
        # Autoregressive sampling
        for _ in range(1, n_tokens):
            logits = model(seq)  # [1, L, n_codes]
            next_logits = logits[0, -1, :]  # [n_codes]
            
            if temperature != 1.0:
                next_logits = next_logits / temperature
            
            # Safe softmax
            probs = F.softmax(next_logits, dim=-1)
            
            # Safety check (shouldn't be needed, but just in case)
            if torch.isnan(probs).any() or (probs < 0).any():
                probs = torch.ones_like(probs) / probs.numel()
            
            next_token = torch.multinomial(probs, 1).unsqueeze(0)  # [1, 1]
            seq = torch.cat([seq, next_token], dim=1)
        
        return seq.squeeze(0)  # [n_tokens]

# Decode tokens back to time series
def tokens_to_time_series(tokens, vqvae):
    with torch.no_grad():
        # tokens: [4] → [1, 4]
        z_q = vqvae.vq_layer.codebook(tokens.unsqueeze(0))  # [1, 4, 16]
        recon = vqvae.decode(z_q)  # [1, 16, 3]
        return recon.squeeze(0).numpy()

def sample_and_decode(vqvae, transformer, scaler, n_samples=10, n_tokens=4, device="cpu", save_path="generated.png"):
    vqvae.eval(); transformer.eval()
    plt.figure(figsize=(12, 2*n_samples))
    
    for i in range(n_samples):
        tokens = sample_sequence(transformer, n_tokens, temperature=1.0, device=device)
        with torch.no_grad():
            z_q = vqvae.vq_layer.codebook(tokens.unsqueeze(0).to(device))
            recon = vqvae.decode(z_q).cpu().numpy()[0]
        
        recon_orig = scaler.inverse_transform(recon.reshape(1, -1)).reshape(-1, 3) #16, 3/-1, 3
        plt.subplot(n_samples, 1, i+1)
        plt.plot(recon_orig[:, 0], 'o-')
        plt.title(f"Sample {i+1} | Tokens: {tokens.cpu().tolist()}")
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
        