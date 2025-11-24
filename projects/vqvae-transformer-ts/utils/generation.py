import torch
import torch.nn.functional as F
import numpy as np

def sample_sequence(transformer, n_tokens=4, temperature=1.0, device="cpu"):
    """Sample token sequence from discrete Transformer"""
    transformer.eval()
    with torch.no_grad():
        probs = torch.ones(transformer.n_codes, device=device)
        first = torch.multinomial(probs, 1).unsqueeze(0)
        seq = first
        
        for _ in range(1, n_tokens):
            logits = transformer(seq)
            next_logits = logits[0, -1, :] / temperature
            probs = F.softmax(next_logits, dim=-1)
            if torch.isnan(probs).any() or (probs < 0).any():
                probs = torch.ones_like(probs) / probs.numel()
            next_token = torch.multinomial(probs, 1).unsqueeze(0)
            seq = torch.cat([seq, next_token], dim=1)
        return seq.squeeze(0)

def tokens_to_time_series(tokens, vqvae, device="cpu"):
    """Decode tokens to time series"""
    with torch.no_grad():
        z_q = vqvae.vq_layer.codebook(tokens.unsqueeze(0).to(device))
        recon = vqvae.decode(z_q)
        return recon.squeeze(0).cpu().numpy()


def sample_continuous(model, first_vals, seq_len=48, temperature=0.1, device="cpu"):
    """Sample from continuous autoregressive model"""
    model.eval()
    with torch.no_grad():
        first_val = np.random.choice(first_vals)
        seq = torch.tensor([[first_val]], device=device)
        # Start with scalar 0.0
        #seq = torch.tensor([[0.0]], device=device)  # [1, 1]
        
        for step in range(1, seq_len):
            pred = model(seq)  # [1, L]
            next_mean = pred[0, -1]  # scalar

            #if step == 1:  # Print first prediction
            #    print(f"Step 1: input={seq.flatten()}, pred={next_mean.item():.3f}")

            if temperature > 0:
                # Generate SCALAR noise
                noise = torch.randn_like(next_mean) * temperature
                next_val = next_mean + noise
            else:
                next_val = next_mean
            
            # next_val is scalar → unsqueeze to [1, 1]
            next_val = next_val.unsqueeze(0).unsqueeze(0)  # [1, 1]
            seq = torch.cat([seq, next_val], dim=1)  # [1, L+1]
        
        return seq.squeeze(0).cpu()  # [L]

