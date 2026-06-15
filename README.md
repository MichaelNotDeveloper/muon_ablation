# Muon Optimizer Ablation

Language-model training codebase for comparing **Muon** vs **AdamW** across several architectures.

Supported models:

- **LSTM** language model
- **Transformer** encoder LM
- **NanoGPT** (GPT-style)

Default dataset: [stas/openwebtext-10k](https://huggingface.co/datasets/stas/openwebtext-10k) (10K examples, auto-downloaded on first run).

For the full OpenWebText corpus, override the dataset config:

```bash
uv run train.py --config-name lstm datasets=openwebtext
```

## Installation

1. Install [uv](https://docs.astral.sh/uv/):

   ```bash
   pip install uv
   ```

2. Sync dependencies:

   ```bash
   uv sync
   ```

## Training

Three Hydra configs (one per architecture), default optimizer is AdamW:

```bash
uv run train.py --config-name lstm
uv run train.py --config-name transformer
uv run train.py --config-name nanogpt
```

Switch optimizer via CLI:

```bash
uv run train.py --config-name lstm optimizer=muon
uv run train.py --config-name transformer optimizer=muon
```

Useful overrides:

```bash
uv run train.py --config-name lstm \
  optimizer=muon \
  trainer.n_epochs=20 \
  trainer.compute_topsubspace_metrics=true \
  trainer.topsubspace_k=5 \
  writer.run_name=my_run \
  writer.mode=offline
```

### Analysis metrics toggles

Controlled via `trainer` config:

| Flag | Default | Description |
|------|---------|-------------|
| `trainer.compute_matrix_metrics` | `true` | SVD-based weight metrics (condition number, orthogonality error, stable/effective rank, …) |
| `trainer.compute_topsubspace_metrics` | `false` | Top-subspace concentration (`rho_k`, `concentration_c`) from kate2-style analysis |
| `trainer.topsubspace_k` | `5` | Number of top singular directions used for subspace metrics |

Examples:

```bash
# matrix metrics only (default)
uv run train.py --config-name lstm trainer.compute_matrix_metrics=true trainer.compute_topsubspace_metrics=false

# top-subspace metrics only
uv run train.py --config-name lstm trainer.compute_matrix_metrics=false trainer.compute_topsubspace_metrics=true

# both
uv run train.py --config-name lstm trainer.compute_matrix_metrics=true trainer.compute_topsubspace_metrics=true
```

## Demo notebook

See [`scripts/demo.ipynb`](scripts/demo.ipynb) for a walkthrough that clones the repo and runs all **3 architectures × 2 optimizers** (6 setups).

## Suggested experiments

- **Learning-rate sweep**: `optimizer.lr=1e-4,3e-4,1e-3` or `optimizer.muon.lr=...`
- **Muon projection**: `optimizer.muon.projection=exact` vs `optimizer.muon.projection=ns`
- **Momentum / Nesterov**: `optimizer.muon.momentum=0.9 optimizer.muon.nesterov=true`
- **Matrix metrics**: compare `condition_number_weighted_mean` and `orthogonality_error_weighted_mean`
- **Top-subspace metrics**: compare `rho_k_weighted_mean` and `concentration_c_weighted_mean` between AdamW and Muon
- **Scale**: use `datasets=openwebtext`, increase `trainer.n_epochs`, model size in `src/configs/model/`

## Project layout

```
src/
  configs/          # Hydra configs (lstm, transformer, nanogpt, optimizer, …)
  model/            # LSTM, Transformer, NanoGPT
  optimizers/       # Muon, AdamW Python implementations
  datasets/         # OpenWebText-10k (default) and full OpenWebText
  metrics/          # Perplexity, matrix metrics, top-subspace metrics
  trainer/          # Training loop (AMP, schedulers, checkpointing)
train.py
scripts/
  demo.ipynb        # end-to-end demo (clone + 6 training runs)
```

## License

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](/LICENSE)
