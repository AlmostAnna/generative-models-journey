"""
Continuous Time Series Transformer Model Implementation.

This module provides implementation for the Continuous
Time Series Transformer.
"""

import torch
import torch.nn as nn


class ContinuousTimeSeriesTransformer(nn.Module):
    """A Transformer-based model for autoregressive time series generation without tokenization.

    This model processes raw time series values directly (hence "Just Image Transformers" style for time series),
    using a causal (autoregressive) Transformer architecture. It predicts the next value in a sequence
    based on all previous values, operating in a continuous space rather than using discrete tokens.

    The model takes a sequence of scalar values and autoregressively predicts the next value.
    It uses learned positional embeddings and causal masking to ensure predictions only depend
    on past and present values, not future ones.

    Args:
        seq_len (int, optional): The maximum expected length of input sequences.
            Used for initializing positional embeddings and pre-defining the maximum mask size.
            Defaults to 48.
        d_model (int, optional): The dimensionality of the model's internal representations.
            Defaults to 32.
        n_heads (int, optional): The number of attention heads in the Transformer layers.
            Defaults to 2.
        n_layers (int, optional): The number of Transformer encoder layers.
            Defaults to 2.
        dropout (float, optional): The dropout probability applied within the Transformer layers.
            Should typically be 0.0 for inference. Defaults to 0.0.
    """

    def __init__(
        self,
        seq_len: int = 48,
        d_model: int = 32,
        n_heads: int = 2,
        n_layers: int = 2,
        dropout: float = 0.0,
    ):
        """
        Initialize the TimeSeriesTransformer.

        Args:
            seq_len (int):  The maximum expected length of input sequences.
                Used for initializing positional embeddings
                and pre-defining the maximum mask size.
                Default is 48.
            d_model (int): The dimensionality of the model's internal representations.
                Default is 32.
            n_heads (int): The number of attention heads in the Transformer layers.
                Default is 2.
            n_layers (int): The number of Transformer encoder layers stacked.
                Default is 2.
            dropout (float): The dropout probability applied within the Transformer layers.
                Should typically be 0.0 for inference. Defaults to 0.0.
        """
        super().__init__()
        self.seq_len = seq_len

        # Project the input scalar value to the model's internal embedding dimension
        self.input_proj = nn.Linear(1, d_model)

        # Learnable positional embeddings to provide information about the position
        # in the sequence
        self.pos_emb = nn.Embedding(seq_len, d_model)

        # Stack of Transformer encoder layers
        # Note: Uses nn.TransformerEncoderLayer which is typically for self-attention.
        # In the autoregressive setting, causal masking ensures it functions like a decoder layer.
        self.layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=d_model,
                    nhead=n_heads,
                    dim_feedforward=d_model * 2,
                    dropout=dropout,
                    batch_first=True,
                )
                for _ in range(n_layers)
            ]
        )

        # Final linear layer to project back to a scalar prediction
        self.head = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for autoregressive time series prediction.

        Processes the input sequence `x` and predicts the next value for each position
        in the sequence, conditioned on all previous values (causal masking).

        Args:
            x (torch.Tensor): Input tensor of shape `(batch_size, sequence_length)`.
                Represents a batch of time series sequences where each element is a scalar value.

        Returns:
            torch.Tensor: Output tensor of shape `(batch_size, sequence_length)`.
                Each element `output[b, i]` is the model's prediction for the value
                that should come *after* `x[b, i-1]` in the sequence. For an input of
                length L, it effectively predicts the next L values autoregressively.
        """
        B, L = x.shape

        # Add a dimension for the linear projection: [B, L] -> [B, L, 1]
        x = x.unsqueeze(-1)  # [B, L, 1]

        # Project the scalar input value to the model's internal embedding dimension
        x = self.input_proj(x)  # [B, L, d_model]

        # Add learned positional embeddings
        x = x + self.pos_emb(torch.arange(L, device=x.device))  # [B, L, d_model]

        # Create causal mask: upper triangular part is -inf, lower triangular part is 0
        # This ensures that when predicting output[b, i], the attention at position i
        # only considers positions 0 through i.
        mask = torch.triu(torch.ones(L, L, device=x.device) * float("-inf"), diagonal=1)

        # Pass through the Transformer layers with the causal mask
        for layer in self.layers:
            x = layer(x, src_mask=mask)

        # Project the final hidden states back to a scalar prediction
        output = self.head(x)  # [B, L, 1]

        # Remove the last dimension to get the final output shape [B, L]
        return output.squeeze(-1)  # [B, L]
