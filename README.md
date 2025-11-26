# Generative Models Journey

> *"How do we generate data from complex distributions? This project traces the evolution of generative methods — from classical MCMC to modern diffusion — and now extends to structured sequential data."*

This repository documents my **hands-on exploration** of generative modeling across two complementary domains:

## Part 1: Foundations — The Two Moons Playground

All foundational experiments use the **two-moons dataset**, a simple 2D benchmark that reveals deep truths about sampling, optimization, and structure.

We follow a clear arc:
1. **MCMC** (MALA, HMC): Sampling from a known distribution  
2. **Energy-Based Models (EBM)**: Learning a distribution, but struggling to sample  
3. **Diffusion Models**: Learning to *reconstruct* data by reversing noise  

**Goal**: Not just to implement models — but to understand *why* some succeed where others fail.

### [Sampling Is Not Solved: Silent Failures in Modern MCMC](notebooks/mcmc-story.ipynb)
- Compares MCMC methods on challenging distributions (banana and two-moons)

### [EBM: The Struggle to Learn the Two Moons](notebooks/ebm-story.ipynb)
- Trains an Energy-Based Model with Langevin and Contrastive Divergence
- Demonstrates poor mixing and mode collapse  
- Asks: *"Why is sampling so hard?"*

### [Diffusion: From Noise to Structure](notebooks/diffusion-story.ipynb)
- Trains a score-based diffusion model with DDIM sampling
- Shows how samples emerge from noise, step by step  
- Explains why diffusion avoids the pitfalls of EBM

---

## Part 2: Extension — Structured Sequential Data

Building on these foundations, I've extended the exploration to **time series with distinct regimes** (trending, mean-reverting, spiking):

### [VQ-VAE + Transformer vs Continuous Autoregression](projects/vqvae-transformer-ts/README.md)
- Compares discrete tokenization vs continuous autoregressive approaches
- **Key finding**: Continuous modeling better captures regime structure and generates more diverse samples
- Includes quantitative evaluation: regime accuracy, diversity, and FID metrics

This represents a natural progression: from **2D distributions** to **structured sequential data** with temporal dependencies.

---

## Project Organization
generative-models-journey/ 
├── notebooks/ # Part 1: Two-moons foundational work 
├── projects/ # Part 2: Sequential data extensions
│ └── vqvae-transformer-ts/ # Transformer-based time series generation 
└── README.md # This file


Each project maintains the same philosophy: **understanding through implementation**, with careful attention to **what works, what doesn't, and why**.

---

*This repository reflects my belief that generative modeling is not just about building bigger models — but about understanding the fundamental tradeoffs between different approaches to learning and sampling from complex distributions.*


