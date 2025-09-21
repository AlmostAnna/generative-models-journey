import glob  
import os
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

# Plot real data
def plot_real_data(x_real, title="Two-Moons Dataset (Normalized)", save_path=None):
    plt.figure(figsize=(6,6))
    plt.scatter(x_real[:, 0], x_real[:, 1], s=10, c='black', alpha=0.6, linewidth=0)
    plt.title(title, fontsize=14)
    plt.xlabel("$x_1$", fontsize=12)
    plt.ylabel("$x_2$", fontsize=12)
    plt.xlim(-3, 3)
    plt.ylim(-3, 3)
    plt.grid(True, alpha=0.3)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.tight_layout()
    if save_path: 
        plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.show()

def save_gif():
    # TODO need to save all pictures in ../figures
    # Find all saved images
    # TODO do we need to remove existing images from the previous run early in the set up stage? 
    png_files = sorted(glob.glob("step_*.png"))

    if png_files:
    # Build GIF
#        with imageio.get_writer("diffusion_training.gif", mode='I', fps=1) as writer:
#            for filename in png_files:
#                image = imageio.v2.imread(filename)
#                writer.append_data(image)
        print("✅ Training animation saved: diffusion_training.gif")
    else:
        print("No images found. Did you run training?")