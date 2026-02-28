"""Convert Data For QuantLab Script."""

import argparse

import torch


def main():
    """Convert Data for QuantLab."""
    parser = argparse.ArgumentParser(description="Convert Data for QuantLab.")
    parser.add_argument(
        "--file_path", type=str, required=True, help="Path to the generated .pt file."
    )
    parser.add_argument("--save_path", type=str, help="Path to save the output.")

    args = parser.parse_args()

    print(f"Loading generated data from {args.file_path}...")
    try:
        generated_data = torch.load(args.file_path)  # Shape: (N, T, D) or (N, T)
        print(f"Loaded data shape: {generated_data.shape}")

        generated_price_paths = generated_data[:, :, 0]
        torch.save(generated_price_paths, args.save_path)
        print(f"Result saved to {args.save_path}")

    except FileNotFoundError:
        print(f"File {args.file_path} not found.")
        return
    except Exception as e:
        print(f"Error loading file: {e}")
        return

    print("Data conversion complete.")


if __name__ == "__main__":
    main()
