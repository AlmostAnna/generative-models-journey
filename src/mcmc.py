"""
MCMC samplers.

A collection of functions implementing some MCMC samplers and diagnostics.
"""

import warnings

import arviz as az
import numpy as np
import pandas as pd
import pyro
import pyro.distributions as dist
import torch
from pyro.infer import MCMC, NUTS
from scipy.stats import wasserstein_distance
from tqdm import tqdm

warnings.filterwarnings("ignore")

# Log_ps and gradients for MALA and SMC, models for HMC/NUTS


# --- TWO MOONS DISTRIBUTION --- #
def two_moons_logp_circular(x, noise_scale: float = 0.1):
    """Log-density that matches make_moons (circular arcs)."""
    x1, x2 = x[..., 0], x[..., 1]

    # Moon 1: upper semicircle: (x1)^2 + (x2)^2 = 1, x1 in [-1,1], x2 > 0
    # Distance to circle: (x1^2 + x2^2 - 1)^2
    dist1 = (x1**2 + x2**2 - 1) ** 2

    # Moon 2: lower semicircle centered at (1, 0): (x1-1)^2 + x2^2 = 1
    dist2 = ((x1 - 1) ** 2 + x2**2 - 1) ** 2

    # Only want upper for moon1, lower for moon2
    # Use directional penalties
    penalty1 = torch.where(x2 >= 0, dist1, dist1 + 100)  # penalize if x2 < 0 for moon1
    penalty2 = torch.where(x2 <= 0, dist2, dist2 + 100)  # penalize if x2 > 0 for moon2

    # Combine
    precision = 1.0 / (2 * noise_scale**2)
    logp1 = -precision * penalty1
    logp2 = -precision * penalty2

    return torch.logsumexp(torch.stack([logp1, logp2]), dim=0)


def grad_logp_circular(x, noise_scale: float = 0.1):
    """Gradient for circular moons logp."""
    x = x.detach().requires_grad_(True)
    logp = two_moons_logp_circular(x.unsqueeze(0), noise_scale).squeeze()
    grad = torch.autograd.grad(logp, x, retain_graph=False)[0]
    return grad


def two_moons_model(x=None):
    """Circular moons model."""
    x = pyro.sample("x", dist.Uniform(-3.0, 3.0).expand([2]).to_event(1))
    lp = two_moons_logp_circular(x)
    pyro.factor("likelihood", lp)


# --- BANANA DISTRIBUTION --- #
def banana_logp(x, b=1.0):
    """Log-density that matches banana distribution."""
    x1 = x[..., 0]
    x2 = x[..., 1]
    logp = -0.5 * (x1**2 + (x2 - b * x1**2) ** 2)
    if x.shape[-1] > 2:
        logp += -0.5 * np.sum(x[..., 2:] ** 2, axis=-1)
    return logp


def banana_grad_logp(x, b=1.0):
    """Gradient of log-density for banana distribution."""
    x = x.detach().requires_grad_(True)
    logp = banana_logp(x.unsqueeze(0), b).squeeze()
    grad = torch.autograd.grad(logp, x, retain_graph=False)[0]
    return grad


def banana_model(x=None):
    """Banana distribution model."""
    x = pyro.sample("x", dist.Uniform(-3.0, 3.0).expand([2]).to_event(1))
    lp = banana_logp(x)
    pyro.factor("likelihood", lp)


def generate_true_banana_samples(n=2000, b=1.0, d=2):
    """Generate ground truth banana samples."""
    samples = np.random.normal(size=(n, d))
    samples[:, 1] = np.random.normal(loc=b * samples[:, 0] ** 2, scale=1.0)
    return samples


# Samplers
# Sequential Monte-Carlo
def run_smc(logp_fn, num_particles=1000):
    """Sequential Monte-Carlo Sampler."""
    # Sample from prior (uniform in [-3, 3])
    samples = torch.zeros(num_particles, 2)
    log_weights = torch.zeros(num_particles)

    # Initialize particles
    samples = (torch.rand(num_particles, 2) - 0.5) * 6.0  # [-3, 3]

    # Compute unnormalized log-weights
    log_weights = logp_fn(samples)

    # Normalize weights
    weights = torch.softmax(log_weights, dim=0)

    # Resample particles based on weights
    indices = torch.multinomial(weights, num_particles, replacement=True)
    resampled = samples[indices]

    return resampled.numpy()


# MALA (Metropolis-Adjusted Langevin Algorithm)
def run_mala(logp_fn, grad_logp_fn, num_samples=1000, step_size=0.02):
    """MALA (Metropolis-Adjusted Langevin Algorithm) Sampler."""
    samples = torch.zeros(num_samples, 2)
    x = torch.randn(2) * 0.1

    for i in tqdm(range(num_samples), desc="MALA"):
        grad = grad_logp_fn(x)
        # Langevin proposal
        # x_prop = x + step_size * grad + torch.sqrt(2 * step_size) * torch.randn(2)
        x_prop = (
            x
            + step_size * grad
            + torch.sqrt(torch.tensor(2 * step_size)) * torch.randn(2)
        )

        # Metropolis-Hastings correction
        log_prob_curr = logp_fn(x)
        log_prob_prop = logp_fn(x_prop)

        # Reverse proposal mean
        grad_prop = grad_logp_fn(x_prop)
        x_rev_mean = x_prop + step_size * grad_prop
        x_curr_mean = x + step_size * grad

        # Gaussian proposal log-ratios
        log_q_prop = -0.5 * ((x - x_rev_mean) ** 2).sum() / (2 * step_size)
        log_q_curr = -0.5 * ((x_prop - x_curr_mean) ** 2).sum() / (2 * step_size)

        log_accept = log_prob_prop - log_prob_curr + log_q_prop - log_q_curr
        if torch.log(torch.rand(1)) < log_accept:
            x = x_prop

        samples[i] = x.clone()

    return samples.numpy()


# Adaptive HMC
def run_hmc_adaptive(logp_fn, grad_logp_fn, num_samples=1000, warmup_steps=200):
    """Adaptive Hamiltonian Monte Carlo Sampler."""
    samples = torch.zeros(num_samples, 2)
    x = torch.randn(2) * 0.1

    # Dual averaging parameters (like NUTS)
    step_size = 0.01
    avg_accept = 0.0
    log_step_size = torch.log(torch.tensor(step_size))
    log_step_size_avg = 0.0
    mu = torch.log(torch.tensor(10 * step_size))
    t0 = 10
    gamma = 0.05

    num_steps = 10
    accept_count = 0

    for i in tqdm(range(num_samples + warmup_steps), desc="HMC Adaptive"):
        x0 = x.clone()
        p0 = torch.randn_like(x)
        H0 = 0.5 * p0.pow(2).sum() + logp_fn(x0)

        # Current step size (adaptive)
        current_step = torch.exp(log_step_size).item()

        # Leapfrog
        p = p0 + 0.5 * current_step * grad_logp_fn(x)
        for _ in range(num_steps):
            x = x + current_step * p
            p = p + current_step * grad_logp_fn(x)
        p = p + 0.5 * current_step * grad_logp_fn(x)

        H_new = 0.5 * p.pow(2).sum() + logp_fn(x)
        dH = H_new - H0

        # Accept/reject
        if torch.log(torch.rand(1)) < -dH:
            accept_count += 1
            avg_accept = avg_accept + (1.0 - avg_accept) / (i + 1)
        else:
            x = x0
            avg_accept = avg_accept + (0.0 - avg_accept) / (i + 1)

        # Adapt step size during warmup
        if i < warmup_steps:
            # Dual averaging
            eta = 1.0 / (i + t0)
            log_step_size = mu - avg_accept * gamma / eta**0.5
            log_step_size_avg = eta * log_step_size + (1 - eta) * log_step_size_avg

        if i >= warmup_steps:
            samples[i - warmup_steps] = x.clone()

    final_step = torch.exp(log_step_size_avg).item()
    print(
        f"Final step size: {final_step:.4f}, Accept rate: {accept_count / num_samples:.3f}"
    )
    return samples.numpy()


# HMC(NUTS), requires Pyro
def run_nuts(model, num_samples=400, warmup=200, num_runs=2):
    """No-U-Turn Sampler(NUTS) using Pyro."""
    all_samples = []
    chain_means = []

    for i in range(num_runs):
        pyro.set_rng_seed(42 + i * 100)
        kernel = NUTS(model, adapt_step_size=True, target_accept_prob=0.8)
        mcmc = MCMC(kernel, num_samples=num_samples, warmup_steps=warmup)
        mcmc.run()
        samples = mcmc.get_samples()["x"].numpy()
        all_samples.append(samples)

        # Check where each chain spends most time
        mean_x = samples.mean(axis=0)
        chain_means.append(mean_x)
        print(f"Chain {i+1} mean: {mean_x}")

    combined = np.vstack(all_samples)
    print(f"Combined mean: {combined.mean(axis=0)}")
    print(f"Chain means spread: {np.std(chain_means, axis=0)}")

    return combined


# Metrics
def compute_ess(samples):
    """Compute Effective Sample Size (ESS) using ArviZ."""
    # Convert to ArviZ format
    idata = az.convert_to_inference_data(samples[None, :, :])  # (chain, sample, dim)
    ess = az.ess(idata)["x"].mean().item()
    return ess


def compute_validation_metrics(true_samples, sampler_dict):
    """Compute metrics comparing samplers to true distribution."""
    results = {}
    for name, samples in sampler_dict.items():
        # ESS
        ess = compute_ess(samples)

        # Wasserstein distance (simplified: just compare first 1000 samples)
        n = min(len(true_samples), len(samples), 1000)
        emd = wasserstein_distance(
            true_samples[:n, 0], samples[:n, 0]
        ) + wasserstein_distance(true_samples[:n, 1], samples[:n, 1])

        results[name] = {"ESS": ess, "W1_to_True": emd}

    return results


# Specially for multimodal distributions


def compute_ess_multimodal(samples, chains=2):
    """
    Compute ESS for multi-modal sampling.

    Args:
        samples: combined samples from multiple chains.
        chains: number of chains.
    """
    n_per_chain = len(samples) // chains
    ess_per_chain = []

    for i in range(chains):
        chain_samples = samples[i * n_per_chain : (i + 1) * n_per_chain]
        # Convert to arviz format: (chain, sample, var)
        idata = az.convert_to_inference_data(chain_samples[None, :, :])
        chain_ess = az.ess(idata)["x"].mean().item()
        ess_per_chain.append(chain_ess)

    # Return mean ESS per chain
    return np.mean(ess_per_chain)


def analyze_multimodal_performance(samples, chains=2):
    """Analyze multi-modal sampling performance."""
    n_per_chain = len(samples) // chains

    results = {}
    for i in range(chains):
        chain_samples = samples[i * n_per_chain : (i + 1) * n_per_chain]
        chain_ess = compute_ess(chain_samples)
        chain_mean = chain_samples.mean(axis=0)

        results[f"Chain_{i+1}"] = {
            "ESS": chain_ess,
            "Mean": chain_mean,
            "Mode": "Upper" if chain_mean[1] > 0 else "Lower",
        }

    # Combined analysis
    mode_separation = np.std([r["Mean"][1] for r in results.values()])

    return results, mode_separation


# Experiment


def run_experiment(samplers):
    """Execute and collect samples."""
    sampler_samples = {}
    results = {}

    for name, runner in samplers.items():
        print(f"\nRunning {name}...")
        samples = runner()
        sampler_samples[name] = samples

        ess = compute_ess(samples)
        results[name] = {"ESS": ess}
        print(f"{name} ESS: {ess:.1f}")

    # Print summary
    print("\n" + "=" * 30)
    print(pd.DataFrame(results).T)
    return sampler_samples, results
