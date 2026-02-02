# References

This document lists key papers relevant to the methods explored in this generative models journey, covering Energy-Based Models (EBM), Diffusion Models, and Transformer-based Generative Models.

## Energy-Based Models (EBM)

- **A Tutorial on Energy-Based Learning** (LeCun et al., 2006) - Provides a foundational understanding of EBMs, their formulation, and training principles.
- **Regularized Learning of Generalized Linear Models** (Jaakkola & Haussler, 1999) - Discusses contrastive divergence and related techniques often used for training EBMs.
- **Deep Learning: An Introduction** (Goodfellow et al., 2016) - Chapter 20 covers undirected graphical models, including Boltzmann machines, which are a type of EBM.

## Diffusion Models

- **Denoising Diffusion Probabilistic Models** (Ho et al., 2020) - Introduces DDPMs, establishing the foundational framework for training and sampling via forward and reverse diffusion processes. [[Paper](https://arxiv.org/abs/2006.11239)]
- **Denoising Diffusion Implicit Models** (Song et al., 2020) - Presents DDIMs, offering a faster sampling method compared to DDPMs by using a non-Markovian process. [[Paper](https://arxiv.org/abs/2010.02502)]
- **Score-Based Generative Modeling through Stochastic Differential Equations** (Song et al., 2020) - Unifies score-based and diffusion models under the SDE framework, introducing predictor-corrector sampling schemes. [[Paper](https://arxiv.org/abs/2011.13456)]

## Transformer-Based Generative Models

### Vector Quantized Variational Autoencoders (VQ-VAEs)

- **Neural Discrete Representation Learning** (van den Oord et al., 2017) - Introduces the VQ-VAE architecture, enabling the learning of discrete latent representations. [[Paper](https://arxiv.org/abs/1711.00937)]
- **Generating Diverse High-Fidelity Images with VQ-VAE-2** (Razavi et al., 2019) - Presents VQ-VAE-2, an improved version demonstrating high-quality image generation. [[Paper](https://arxiv.org/abs/1906.00446)]

### Transformers for Sequential/Time-Series Generation

- **Attention Is All You Need** (Vaswani et al., 2017) - Introduces the Transformer architecture, crucial for many subsequent generative models on sequential data. [[Paper](https://arxiv.org/abs/1706.03762)]
- **Generating Long Sequences with Sparse Transformers** (Parrish et al., 2022) - Explores efficient Transformer variants for modeling long sequences. [[Paper](https://arxiv.org/abs/1904.10509)]
- **Zero-Shot Text-to-Image Generation** (Ramesh et al., 2021) - Example application using VQ-VAE combined with a Transformer decoder (often used in models like DALL-E). [[Paper](https://arxiv.org/abs/2102.12092)]
- **Music Transformer** (Huang et al., 2018) - Example application of Transformers for music generation. [[Paper](https://arxiv.org/abs/1809.04281)]
