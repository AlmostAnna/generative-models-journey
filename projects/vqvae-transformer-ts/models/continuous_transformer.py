import torch
import torch.nn as nn

class ContinuousTimeSeriesTransformer(nn.Module):
    def __init__(self, seq_len=48, d_model=32, n_heads=2, n_layers=2, dropout=0.0):
        super().__init__()
        self.seq_len = seq_len
        self.input_proj = nn.Linear(1, d_model)
        self.pos_emb = nn.Embedding(seq_len, d_model)

        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=d_model * 2,
                dropout=dropout,
                batch_first=True
            ) for _ in range(n_layers)
        ])

        self.head = nn.Linear(d_model, 1)

    def forward(self, x):
        B, L = x.shape
        x = x.unsqueeze(-1)  # [B, L, 1]
        x = self.input_proj(x)  # [B, L, d_model]
        x = x + self.pos_emb(torch.arange(L, device=x.device))  # [B, L, d_model]

        # Create proper causal mask (additive, with -inf)
        mask = torch.triu(torch.ones(L, L, device=x.device) * float('-inf'), diagonal=1)

        for layer in self.layers:
            x = layer(x, src_mask=mask)

        return self.head(x).squeeze(-1)  # [B, L]

