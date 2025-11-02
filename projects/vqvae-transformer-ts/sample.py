import torch
import torch.optim as optim
import torch.nn as nn

import numpy as np
import torch.nn.functional as F

# Access local project code
from models.transformer import TimeSeriesTransformer


def sample_sequence(model, n_tokens=4, temperature=1.0):
    model.eval()
    with torch.no_grad():
        device = next(model.parameters()).device
        
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
        