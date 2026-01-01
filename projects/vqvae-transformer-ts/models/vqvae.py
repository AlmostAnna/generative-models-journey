"""
VQ VAE Model Implementation.

This module provides implementation for VQ VAE Tokenizer.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# VQ layer
class VectorQuantizer(nn.Module):
    """Discretizes continuous latent vectors by mapping them to the nearest code in a learnable codebook.

    This layer implements the vector quantization step from the VQ-VAE model
    (van den Oord et al., 2017, see https://arxiv.org/abs/1711.00937 for more details).
    It takes a continuous latent embedding and replaces it with the nearest embedding from a finite
    codebook of `n_codes` vectors, each of dimension `code_dim`.

    The forward pass returns both the quantized tensor and the indices of the selected codes.
    A straight-through estimator is used to allow gradients to flow through the discrete operation.

    Attributes:
        codebook (nn.Embedding): Embedding layer representing the codebook.
        n_codes (int): Number of discrete codes in the codebook.
        code_dim (int): Dimensionality of each code vector.
    """

    def __init__(self, n_codes: int, code_dim: int):
        """Initialize the vector quantizer.

        Args:
            n_codes (int): Number of discrete codes (size of the codebook).
            code_dim (int): Dimensionality of each code vector.
        """
        super().__init__()
        self.n_codes = n_codes
        self.code_dim = code_dim
        self.codebook = nn.Embedding(n_codes, code_dim)

    def forward(self, z_e: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Quantize continuous latent embeddings to nearest codebook vectors.

        Args:
            z_e (torch.Tensor): Continuous input embeddings of shape `(B, L, code_dim)`.

        Returns:
            tuple[torch.Tensor, torch.Tensor]:
                - z_q (torch.Tensor): Quantized embeddings of same shape as input.
                - indices (torch.Tensor): Indices of selected codes of shape `(B, L)`.
        """
        # z_e: [B, L, code_dim]
        # codebook: [n_codes, code_dim]

        # Expand z_e and codebook to [B, L, n_codes, code_dim]
        z_e_exp = z_e.unsqueeze(-2)  # [B, L, 1, code_dim]
        codebook_exp = self.codebook.weight.unsqueeze(0).unsqueeze(
            0
        )  # [1, 1, n_codes, code_dim]

        # Compute squared distances
        distances = (z_e_exp - codebook_exp).pow(2).sum(dim=-1)  # [B, L, n_codes]

        indices = distances.argmin(dim=-1)  # [B, L]
        z_q = self.codebook(indices)  # [B, L, code_dim]

        # Straight-through
        z_q = z_e + (z_q - z_e).detach()
        return z_q, indices


class VQVAETimeSeries(nn.Module):
    """Vector-Quantized Variational Autoencoder for multivariate time series.

    This model compresses short time series into a discrete sequence of latent tokens using a
    vector quantizer, then reconstructs the input from these tokens. It is designed for low-dimensional,
    short-horizon time series (e.g., T=16, D=3) and serves as a tokenizer for downstream generative models
    such as Transformers.

    The architecture uses MLP-based encoder and decoder for simplicity and fast CPU training.

    Attributes:
        encoder (nn.Sequential): Maps flattened time series to continuous latent space.
        vq_layer (VectorQuantizer): Discretizes latent vectors.
        decoder (nn.Sequential): Reconstructs time series from quantized latents.
        T (int): Sequence length.
        D (int): Number of channels (features).
        n_tokens (int): Number of discrete tokens per sequence.
        code_dim (int): Dimensionality of each token embedding.
    """

    def __init__(
        self,
        T: int = 16,
        D: int = 3,
        n_codes: int = 64,
        code_dim: int = 16,
        n_tokens: int = 2,
        hidden_dim: int = 128,
    ):
        """Initialize the VQ-VAE for time series.

        Args:
            T (int): Length of input time series (default: 16).
            D (int): Number of channels/features (default: 3).
            n_codes (int): Size of the discrete codebook (default: 64).
            code_dim (int): Dimensionality of each code vector (default: 16).
            n_tokens (int): Number of discrete tokens to represent each sequence (default: 2).
            hidden_dim (int): Width of hidden layers in encoder/decoder (default: 128).
        """
        super().__init__()
        self.n_tokens = n_tokens
        self.code_dim = code_dim

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(T * D, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_tokens * code_dim),
        )

        # Vector Quantizer
        self.vq_layer = VectorQuantizer(n_codes, code_dim)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(n_tokens * code_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, T * D),
        )

        self.T, self.D = T, D

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input time series into continuous latent space.

        Args:
            x (torch.Tensor): Input time series of shape `(B, T, D)`.

        Returns:
            torch.Tensor: Continuous latents of shape `(B, n_tokens, code_dim)`.
        """
        z_e = self.encoder(x.reshape(x.size(0), -1))
        z_e = z_e.view(x.size(0), self.n_tokens, self.code_dim)
        # Normalize to prevent explosion
        z_e = F.layer_norm(z_e, normalized_shape=(self.code_dim,))
        return z_e

    def decode(self, z_q: torch.Tensor) -> torch.Tensor:
        """Decode quantized latents into reconstructed time series.

        Args:
            z_q (torch.Tensor): Quantized latents of shape `(B, n_tokens, code_dim)`.

        Returns:
            torch.Tensor: Reconstructed time series of shape `(B, T, D)`.
        """
        # z_q: [B, n_tokens, code_dim]
        out = self.decoder(z_q.view(z_q.size(0), -1))
        return out.view(z_q.size(0), self.T, self.D)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass: encode → quantize → decode.

        Args:
            x (torch.Tensor): Input time series of shape `(B, T, D)`.

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                - recon (torch.Tensor): Reconstructed time series `(B, T, D)`.
                - z_q (torch.Tensor): Quantized latents `(B, n_tokens, code_dim)`.
                - indices (torch.Tensor): Token indices `(B, n_tokens)`.
        """
        z_e = self.encode(x)  # continuous latent
        z_q, indices = self.vq_layer(z_e)  # z_q: quantized, indices: [B, n_tokens]
        recon = self.decode(z_q)
        return recon, z_q, indices
