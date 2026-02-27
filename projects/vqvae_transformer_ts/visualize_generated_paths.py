"""Long Sequence Visualization Script."""

import argparse

import torch

from projects.vqvae_transformer_ts.utils.plotting import plot_time_series_samples


def main():
    """Visualize generated time series paths."""
    parser = argparse.ArgumentParser(
        description="Visualize generated time series paths."
    )
    parser.add_argument(
        "--file_path", type=str, required=True, help="Path to the generated .pt file."
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=5,
        help="Number of samples to plot (default: 5).",
    )
    parser.add_argument(
        "--plot_channels",
        nargs="+",
        type=str,
        default=["0"],
        help="Channel(s) to plot. Use 'all' for all channels, or list indices like '0' '1'. Default: ['0']",
    )
    parser.add_argument(
        "--title_prefix", type=str, default="Generated", help="Prefix for plot titles."
    )
    parser.add_argument(
        "--save_path", type=str, help="Path to save the plot image (optional)."
    )

    args = parser.parse_args()

    # Load the generated data
    print(f"Loading generated data from {args.file_path}...")
    try:
        generated_data = torch.load(args.file_path)  # Shape: (N, T, D) or (N, T)
        print(f"Loaded data shape: {generated_data.shape}")
    except FileNotFoundError:
        print(f"File {args.file_path} not found.")
        return
    except Exception as e:
        print(f"Error loading file: {e}")
        return

    # Process channels_to_plot argument
    if "all" in args.plot_channels:
        channels_to_plot = "all"
    else:
        try:
            channels_to_plot = [int(c) for c in args.plot_channels]
        except ValueError:
            print(
                f"Error: Could not parse channel indices from {args.plot_channels}. Use integers or 'all'."
            )
            return

    # Use the refactored plotting function
    print(f"Plotting {args.num_samples} samples, channels: {channels_to_plot}...")
    plot_time_series_samples(
        data=generated_data,
        n_samples=args.num_samples,
        channels_to_plot=channels_to_plot,
        title_prefix=args.title_prefix,
        save_path=args.save_path,
    )

    print("Plotting complete.")
    if args.save_path:
        print(f"Plot saved to {args.save_path}")


if __name__ == "__main__":
    main()
