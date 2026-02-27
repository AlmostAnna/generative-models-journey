"""
Long Sequence Generation Script.

This module provides pipeline to generate long sequences using VQ-VAE + Transformer
and Continuous Transformer models.
"""

import argparse

import joblib
import numpy as np
import torch
import torch.nn.functional as F

from .models.continuous_transformer import ContinuousTimeSeriesTransformer
from .models.transformer import TimeSeriesTransformer
from .models.vqvae import VQVAETimeSeries
from .utils.generation import tokens_to_time_series


def generate_long_sequence_vqvae_transformer(
    vqvae_model, transformer_model, scaler, n_samples, target_seq_len, device, temp=1.0
):
    """
    Generate long sequences using the VQ-VAE + Transformer model autoregressively.

    Args:
        vqvae_model: The trained VQVAETimeSeries model (encoder & decoder).
        transformer_model: The trained TimeSeriesTransformer model (prior over latents).
        scaler: The scaler fitted on the original training data.
        n_samples: Number of sequences to generate.
        target_seq_len: Desired length of the final output sequence (e.g., 253).
        device: Torch device.
        temp: Temperature for sampling from the transformer logits (controls randomness).

    Returns:
        torch.Tensor: Generated sequences of shape (n_samples, target_seq_len, D).
    """
    vqvae_model.eval()
    transformer_model.eval()

    with torch.no_grad():
        T_vqvae = vqvae_model.T  # Length handled by VQ-VAE (default=16)
        n_tokens_per_chunk = vqvae_model.n_tokens  # Latent length per VQ-VAE chunk
        max_transformer_len = (
            transformer_model.n_tokens
        )  # Max length transformer can handle

        # Ensure VQ-VAE chunk length matches Transformer's max length for this setup
        assert (
            n_tokens_per_chunk == max_transformer_len
        ), f"VQ-VAE n_tokens ({n_tokens_per_chunk}) must match Transformer n_tokens ({max_transformer_len}) for this generation logic."

        # Calculate number of VQ-VAE chunks needed
        n_chunks_needed = int(np.ceil(target_seq_len / T_vqvae))
        print(
            f"[VQ-VAE+T] Target length: {target_seq_len}, VQ-VAE T: {T_vqvae}, Chunks needed: {n_chunks_needed}"
        )

        all_generated_samples = []

        for sample_idx in range(n_samples):
            print(f"[VQ-VAE+T] Generating sample {sample_idx + 1}/{n_samples}")

            generated_chunks_list = []

            for chunk_idx in range(n_chunks_needed):
                print(f"  Generating chunk {chunk_idx + 1}/{n_chunks_needed}")

                # Initialize context for the current chunk
                current_chunk_context = torch.empty(
                    (1, 0), dtype=torch.long, device=device
                )  # Shape [1, 0]

                chunk_codes = []
                for i in range(n_tokens_per_chunk):
                    # Get logits for the next token in the current chunk
                    transformer_output = transformer_model(
                        current_chunk_context
                    )  # Shape [1, current_ctx_len, n_codes]
                    if transformer_output.size(1) == 0:
                        dummy_input = torch.zeros(
                            (1, 1), dtype=torch.long, device=device
                        )  # Dummy token ID 0
                        transformer_output = transformer_model(
                            dummy_input
                        )  # Shape [1, 1, n_codes]
                        next_token_logits = transformer_output[:, 0, :] / temp
                        # Sample the next token
                        next_token_probs = F.softmax(
                            next_token_logits, dim=-1
                        )  # Shape [1, n_codes]
                        next_token_id = torch.multinomial(
                            next_token_probs, num_samples=1
                        )  # Shape [1, 1]
                        # Append to context and chunk
                        current_chunk_context = torch.cat(
                            [dummy_input, next_token_id], dim=1
                        )  # Shape [1, 2] - update context for next iter if needed within chunk
                        chunk_codes.append(next_token_id.squeeze())
                        continue

                    next_token_logits = (
                        transformer_output[:, -1, :] / temp
                    )  # Scale by temperature, Shape [1, n_codes]

                    # Sample the next token
                    next_token_probs = F.softmax(
                        next_token_logits, dim=-1
                    )  # Shape [1, n_codes]
                    next_token_id = torch.multinomial(
                        next_token_probs, num_samples=1
                    )  # Shape [1, 1]

                    # Append the new token to the context and the chunk list
                    current_chunk_context = torch.cat(
                        [current_chunk_context, next_token_id], dim=1
                    )  # Shape [1, current_ctx_len + 1]
                    chunk_codes.append(
                        next_token_id.squeeze()
                    )  # Shape [1] after squeeze

                # Concatenate codes for the current chunk
                chunk_indices_tensor = torch.stack(chunk_codes, dim=0).unsqueeze(
                    0
                )  # Shape [1, n_tokens_per_chunk]

                decoded_chunk_raw = tokens_to_time_series(
                    chunk_indices_tensor.squeeze(0), vqvae_model, device=device
                )
                decoded_chunk_tensor = torch.tensor(
                    decoded_chunk_raw, dtype=torch.float32, device=device
                ).unsqueeze(
                    0
                )  # Add batch dim -> [1, T_vqvae, D_vqvae]

                generated_chunks_list.append(decoded_chunk_tensor)

            # Concatenate along the time dimension (dim=1)
            full_sequence_raw = torch.cat(
                generated_chunks_list, dim=1
            )  # Shape [1, n_chunks_needed * T_vqvae, D_vqvae]

            # Truncate to the exact target length
            actual_generated_len = full_sequence_raw.shape[1]
            if actual_generated_len > target_seq_len:
                truncated_sequence = full_sequence_raw[
                    :, :target_seq_len, :
                ]  # Shape [1, target_seq_len, D_vqvae]
            elif actual_generated_len == target_seq_len:
                truncated_sequence = (
                    full_sequence_raw  # Shape [1, target_seq_len, D_vqvae]
                )
            else:
                raise RuntimeError(
                    f"Generated length ({actual_generated_len}) is less than target ({target_seq_len}). This indicates an error in chunk calculation."
                )

            all_generated_samples.append(
                truncated_sequence.cpu()
            )  # Move to CPU for storage

        # Concatenate all generated samples along the batch dimension
        final_samples = torch.cat(
            all_generated_samples, dim=0
        )  # Shape [n_samples, target_seq_len, D_vqvae]

        print(f"[VQ-VAE+T] Final generated shape: {final_samples.shape}")

    return final_samples


def generate_long_sequence_continuous_transformer(
    model, scaler, n_samples, target_seq_len, device, temp=1.0, seed_seq_len=10
):
    """
    Generate long sequences using the Continuous Transformer model autoregressively.

    Args:
        model: The trained ContinuousTimeSeriesTransformer model.
        scaler: The scaler fitted on the original training data.
        n_samples: Number of sequences to generate.
        target_seq_len: Desired length of the final output sequence (e.g., 253).
        device: Torch device.
        temp: Temperature for sampling (usually less applicable for continuous outputs).
        seed_seq_len: Length of the initial seed sequence to start generation.

    Returns:
        torch.Tensor: Generated sequences of shape (n_samples, target_seq_len).
                      Assumes the model generates scalar sequences (single channel).
    """
    model.eval()

    with torch.no_grad():
        all_generated_samples = []

        for sample_idx in range(n_samples):
            print(f"[Continuous-T] Generating sample {sample_idx + 1}/{n_samples}")

            # The model handles sequences up to its initialized seq_len.
            model_max_len = model.seq_len

            # Determine effective seed length
            effective_seed_len = min(seed_seq_len, model_max_len, target_seq_len)

            # Initialize seed sequence
            if target_seq_len <= model_max_len:
                # If target is short enough, seed with target length
                seed_values = torch.randn(1, target_seq_len, device=device)
            else:
                # If target is longer, seed with the effective length
                seed_values = torch.randn(1, effective_seed_len, device=device)

            current_sequence = seed_values  # Start with the seed [1, S]

            current_len = current_sequence.shape[1]
            target_len_reached = False

            while not target_len_reached:
                input_start_idx = max(0, current_sequence.shape[1] - model_max_len)
                input_to_model = current_sequence[
                    :, input_start_idx:
                ]  # Shape [1, min(current_len, model_max_len)]

                # Get model prediction [1, input_seq_len]
                model_output = model(input_to_model)

                # model_output[:, -1] is the prediction for the value after input_to_model[:, -1].
                next_predicted_value = model_output[:, -1:]  # Shape [1, 1]

                # Append the predicted value
                current_sequence = torch.cat(
                    [current_sequence, next_predicted_value], dim=1
                )  # Shape [1, current_len + 1]
                current_len += 1

                # Check termination
                if current_len >= target_seq_len:
                    target_len_reached = True
                print(f"    Generated step {current_len}/{target_seq_len}")

            # Extract the desired target length
            generated_sample = current_sequence[
                :, :target_seq_len
            ]  # Shape [1, target_seq_len]

            all_generated_samples.append(generated_sample.cpu())  # Move to CPU

        final_samples = torch.cat(
            all_generated_samples, dim=0
        )  # Shape [n_samples, target_seq_len]
        print(f"[Continuous-T] Final generated shape: {final_samples.shape}")

    return final_samples


def main():
    """Run generation pipeline based on arguments."""
    parser = argparse.ArgumentParser(
        description="Generate Long Time Series using Pre-trained Models"
    )
    parser.add_argument(
        "--model_type",
        choices=["vqvae_transformer", "continuous_transformer"],
        required=True,
        help="Type of model to use for generation",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=2000,
        help="Number of time series samples to generate",
    )
    parser.add_argument(
        "--target_seq_len",
        type=int,
        required=True,
        help="Target length of each generated time series sample (e.g., 253 for daily over 1 year)",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to save generated samples (.pt)",
    )

    # Arguments specific to VQ-VAE + Transformer
    vqvae_group = parser.add_argument_group("VQ-VAE+Transformer Args")
    vqvae_group.add_argument(
        "--vqvae_ckpt_path", type=str, help="Path to trained VQ-VAE .pth file"
    )
    vqvae_group.add_argument(
        "--transformer_ckpt_path",
        type=str,
        help="Path to trained Transformer .pth file",
    )
    vqvae_group.add_argument("--n_codes", type=int, help="VQ-VAE codebook size")
    vqvae_group.add_argument("--code_dim", type=int, help="VQ-VAE code dimension")
    vqvae_group.add_argument(
        "--n_tokens",
        type=int,
        help="VQ-VAE latent sequence length (n_tokens during training)",
    )
    vqvae_group.add_argument(
        "--T_vqvae", type=int, help="VQ-VAE input sequence length T"
    )  # Need T to calculate chunks
    vqvae_group.add_argument(
        "--D_vqvae", type=int, help="VQ-VAE input channels D"
    )  # Need D for output shape verification
    vqvae_group.add_argument(
        "--d_model", type=int, help="Transformer d_model", default=32
    )
    vqvae_group.add_argument(
        "--n_heads", type=int, help="Transformer n_heads", default=2
    )
    vqvae_group.add_argument(
        "--n_layers", type=int, help="Transformer n_layers", default=2
    )

    # Arguments specific to Continuous Transformer
    cont_group = parser.add_argument_group("Continuous Transformer Args")
    cont_group.add_argument(
        "--continuous_ckpt_path",
        type=str,
        help="Path to trained Continuous Transformer .pth file",
    )
    cont_group.add_argument(
        "--cont_d_model", type=int, help="Continuous Transformer d_model"
    )
    cont_group.add_argument(
        "--cont_n_heads", type=int, help="Continuous Transformer n_heads"
    )
    cont_group.add_argument(
        "--cont_n_layers", type=int, help="Continuous Transformer n_layers"
    )
    cont_group.add_argument(
        "--cont_seq_len",
        type=int,
        help="Continuous Transformer initialized sequence length",
    )  # Need this for model init and sliding window
    cont_group.add_argument(
        "--seed_seq_len",
        type=int,
        default=10,
        help="Length of initial seed sequence for continuous transformer",
    )

    # Shared arguments
    parser.add_argument(
        "--scaler_path", type=str, required=True, help="Path to fitted scaler .pkl file"
    )
    parser.add_argument(
        "--device", type=str, default="cpu", help="Device to run generation on"
    )
    parser.add_argument("--temp", type=float, default=1.0, help="Sampling temperature")

    args = parser.parse_args()
    device = torch.device(args.device)

    # Load scaler
    print(f"Loading scaler from {args.scaler_path}...")
    scaler = joblib.load(args.scaler_path)

    if args.model_type == "vqvae_transformer":
        if not (args.vqvae_ckpt_path and args.transformer_ckpt_path):
            parser.error(
                "--vqvae_ckpt_path and --transformer_ckpt_path are required for model_type 'vqvae_transformer'"
            )

        print("Loading VQ-VAE and Transformer models...")
        # Load VQ-VAE
        vqvae = VQVAETimeSeries(
            T=args.T_vqvae,
            D=args.D_vqvae,
            n_codes=args.n_codes,
            code_dim=args.code_dim,
            n_tokens=args.n_tokens,
        )
        vqvae.load_state_dict(torch.load(args.vqvae_ckpt_path, map_location=device))
        vqvae = vqvae.to(device)

        # Load Transformer
        transformer = TimeSeriesTransformer(
            n_codes=args.n_codes,
            n_tokens=args.n_tokens,
            d_model=args.d_model,
            n_heads=args.n_heads,
            n_layers=args.n_layers,
        )
        transformer.load_state_dict(
            torch.load(args.transformer_ckpt_path, map_location=device)
        )
        transformer = transformer.to(device)

        print(
            f"Generating {args.n_samples} long sequences using VQ-VAE + Transformer..."
        )
        generated_data = generate_long_sequence_vqvae_transformer(
            vqvae_model=vqvae,
            transformer_model=transformer,
            scaler=scaler,
            n_samples=args.n_samples,
            target_seq_len=args.target_seq_len,
            device=device,
            temp=args.temp,
        )

    elif args.model_type == "continuous_transformer":
        if not args.continuous_ckpt_path:
            parser.error(
                "--continuous_ckpt_path is required for model_type 'continuous_transformer'"
            )

        print("Loading Continuous Transformer model...")
        # Load Continuous Transformer
        model = ContinuousTimeSeriesTransformer(
            seq_len=args.cont_seq_len,  # Must match the length used during training
            d_model=args.cont_d_model,
            n_heads=args.cont_n_heads,
            n_layers=args.cont_n_layers,
            dropout=0.0,
        )
        model.load_state_dict(
            torch.load(args.continuous_ckpt_path, map_location=device)
        )
        model = model.to(device)

        print(
            f"Generating {args.n_samples} long sequences using Continuous Transformer..."
        )
        generated_data = generate_long_sequence_continuous_transformer(
            model=model,
            scaler=scaler,
            n_samples=args.n_samples,
            target_seq_len=args.target_seq_len,
            device=device,
            temp=args.temp,
            seed_seq_len=args.seed_seq_len,
        )

    else:
        raise ValueError(f"Unknown model_type: {args.model_type}")

    print(f"Saving generated data to {args.output_path}...")
    torch.save(generated_data, args.output_path)
    print(
        f"Successfully saved {generated_data.shape[0]} samples of shape {generated_data.shape} to {args.output_path}"
    )


if __name__ == "__main__":
    main()
