import sys
from pathlib import Path

import torch
import torch.optim as optim
import torch.nn as nn

import numpy as np
import torch.nn.functional as F

# Access local project code
from models.transformer import TimeSeriesTransformer

def train_model(
    token_dataset,
    n_codes=64,
    n_tokens=4,
    d_model=32,
    n_heads=2,
    n_layers=2
    n_epochs=50,
    batch_size=128,
    learning_rate=3e-4,
    beta = 0.01,
    device='cpu',
    checkpoint_interval=10,
    debug=False,
    pth_path="transformer.pth" 
):
    """
    Train the time series transformer model.
    """

    # Initialize model
    model = TimeSeriesTransformer(n_codes=n_codes, n_tokens=n_tokens, d_model=d_model, n_heads=n_heads, n_layers=n_layers)

    # Initialize optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    
    print(f"Starting training on {device}...")
    print(f"Epochs: {n_epochs}, Batch size: {batch_size}")

    for epoch in range(n_epochs):
        idx = torch.randperm(token_dataset.size(0))
        epoch_loss = 0.0
    
        for i in range(0, token_dataset.size(0), batch_size):
            batch = token_dataset[idx[i:i+batch_size]]  # [B, 4]
        
            # Autoregressive: input = [c1,c2,c3], target = [c2,c3,c4]
            input_seq = batch[:, :-1]   # [B, 3]
            target_seq = batch[:, 1:]   # [B, 3]
        
            logits = model(input_seq)   # [B, 3, 64]
            loss = criterion(logits.reshape(-1, 64), target_seq.reshape(-1))
        
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
            epoch_loss += loss.item()
    
        if epoch % checkpoint_interval == 0:
            print(f"Epoch {epoch}, Loss: {epoch_loss:.4f}")
    print("Training complete!")
    # Save the result
    if(pth_path):
        torch.save(model.state_dict(), pth_path)
        print(f"Time series transformer model saved to {pth_path}")
    return model, {
        'final_loss': loss.item()
    }
