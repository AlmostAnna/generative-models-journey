# From MCMC to Diffusion: A Journey Through Generative Modeling

> *"How do we generate data from complex distributions? This project traces the evolution of generative methods — from classical MCMC to modern diffusion — using the two-moons dataset as a unifying playground."*

This repository is a **hands-on exploration** of how we model and generate data when the distribution is multi-modal, non-convex, and challenging to sample from.

We follow a clear arc:
1. **MCMC** (MALA, HMC): Sampling from a known distribution
2. **Energy-Based Models (EBM)**: Learning a distribution, but struggling to sample
3. **Diffusion Models**: Learning to *reconstruct* data by reversing noise

All experiments are conducted on the **two-moons dataset**, a simple 2D benchmark that reveals deep truths about sampling, optimization, and structure.

**Goal**: Not just to implement models — but to understand *why* some succeed where others fail.

---

## Table of Contents

### [MCMC: Why HMC Outperforms MALA](notebooks/mala-vs-hmc-story.ipynb) *(Coming Soon)*
- Compares **Metropolis-Adjusted Langevin (MALA)** and **Hamiltonian Monte Carlo (HMC)**

### [EBM: The Struggle to Learn the Two Moons](notebooks/ebm-story.ipynb) *(Coming Soon)*
- Trains an Energy-Based Model with Langevin, Contrastive Divergence, and Score Matching
- Demonstrates poor mixing and mode collapse
- Asks: *"Why is sampling so hard?"*

### [Diffusion: From Noise to Structure](notebooks/diffusion-story.ipynb)
- Trains a score-based diffusion model with DDIM sampling
- Shows how samples emerge from noise, step by step
- Explains why diffusion avoids the pitfalls of EBM


---

## Quick Start

```bash
# Clone the repo
git clone https://github.com/your-username/generative-models-journey.git
cd generative-models-journey

# Install dependencies
# For conda users(recommended):
conda env create -f environment.yml
conda activate generative-journey
python -m ipykernel install --user --name generative-journey

# For pip users:
python -m venv generative-journey
source generative-journey/bin/activate  # Linux/Mac
# generative-journey\Scripts\activate  # Windows
pip install -r requirements.txt

# Launch Jupyter lab
jupyter lab
```
Then open any notebook in notebooks/.
