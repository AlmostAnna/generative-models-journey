import torch
import torch.nn as nn
import torch.nn.functional as F

class TimeSeriesTransformer(nn.Module):
    def __init__(self, n_codes=64, n_tokens=4, d_model=32, n_heads=2, n_layers=2):
        super().__init__()
        self.n_tokens = n_tokens
        self.n_codes = n_codes
        
        # Token embedding + positional encoding
        self.tok_emb = nn.Embedding(n_codes, d_model)
        self.pos_emb = nn.Embedding(n_tokens, d_model)
                
        # Transformer layers
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=d_model * 2,
                dropout=0.1,
                batch_first=True
            ) for _ in range(n_layers)
        ])
        
        # Output head
        self.head = nn.Linear(d_model, n_codes)
    
        # Causal mask (upper triangular)
        self.register_buffer('mask', torch.tril(torch.ones(n_tokens, n_tokens)))

    def forward(self, codes):
    
        # codes: [B, L] where L <= n_tokens
        B, L = codes.shape
        
        # Embed tokens and positions
        tok_emb = self.tok_emb(codes)  # [B, L, d_model]
        pos_emb = self.pos_emb(torch.arange(L, device=codes.device))  # [L, d_model]
        x = tok_emb + pos_emb  # [B, L, d_model]
        
        # Apply causal mask
        mask = self.mask[:L, :L]  # [L, L]
        
        # Pass through transformer layers
        for layer in self.layers:
            x = layer(x, src_mask=mask)
        
        # Predict next token logits
        logits = self.head(x)  # [B, L, n_codes]
        return logits