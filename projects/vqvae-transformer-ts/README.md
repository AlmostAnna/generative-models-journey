# Time Series Generation: Discrete vs Continuous Transformers

![VQ-VAE Sample](plots/vqvae_sample.png)
![Continuous Sample](plots/continuous_sample.png)

We compare two modern generative approaches for time series:
- **Discrete**: VQ-VAE + Transformer (tokenized)
- **Continuous**: JiT-style autoregressive Transformer (raw values)

## Key Finding

> **Continuous modeling better captures regime structure and generates more diverse samples** — despite being simpler (no tokenizer needed).

## Quantitative Results

| Metric | VQ-VAE + Transformer | Continuous Transformer |
|--------|----------------------|------------------------|
| Regime Distribution Match | 0.780 | **0.890** |
| Diversity | 7.35 | **10.51** |
| FID (lower = better) | 24.49 | **23.79** |

## Try It Yourself

```bash
# Train VQ-VAE + Transformer
python run_pipeline.py

# Train Continuous Transformer  
python run_continuous_pipeline.py

# Compare results
python compare_models.py

