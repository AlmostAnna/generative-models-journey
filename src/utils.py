import glob  
import os
import sys
from pathlib import Path
import torch
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
# Optional: imageio for GIFs
#try:
#    import imageio
#except ImportError:
#    print("imageio not installed — animations disabled")


def get_data(n_samples=1024, normalize=True):
    """
    Generate two-moons dataset.
    Returns: (N, 2) tensor
    """
    x, _ = make_moons(n_samples=n_samples, noise=0.1, random_state=42)
    x = torch.tensor(x, dtype=torch.float32)
    
    if normalize:
        x = (x - x.mean(dim=0)) / x.std(dim=0)
    
    return x


def find_project_root(marker_files=['README.md', 'requirements.txt', '.git']):
    """Find project root by looking for common marker files/directories"""
    current_path = Path.cwd()
    for parent in [current_path] + list(current_path.parents):
        if any((parent / marker).exists() for marker in marker_files):
            return parent
    return current_path  # fallback to current directory



def save_training_gif(location, giffile="diffusion_training.gif"):
    # Find all saved images
    print("Looking for png files in ", location)

    png_files = sorted(glob.glob(location+"/step_*.png"))
    print("png_files from training:", png_files)
    
    if png_files:
    # Build GIF
#        with imageio.get_writer(giffile, mode='I', fps=1) as writer:
#            for filename in png_files:
#                image = imageio.v2.imread(filename)
#                writer.append_data(image)
        print(f"Training animation saved: {giffile}")
    else:
        print("No images found. Did you run training?")