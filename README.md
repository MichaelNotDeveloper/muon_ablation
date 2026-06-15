# Muon Optimizer Ablation

Language-model training codebase for comparing **Muon** vs **AdamW** across several architectures.

Supported models:

- **LSTM** language model
- **Transformer** encoder LM
- **NanoGPT** (GPT-style)

Dataset: [OpenWebText](https://huggingface.co/datasets/Skylion007/openwebtext) (downloaded automatically via HuggingFace on first run).

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
uv run train.py --config-name lstm optimizers=muon
uv run train.py --config-name transformer optimizers=muon
```

Useful overrides:

```bash
uv run train.py --config-name lstm \
  optimizers=muon \
  trainer.n_epochs=20 \
  datasets.train.download_limit=50000 \
  writer.run_name=my_run \
  writer.mode=offline
```

## Demo notebook

See [`scripts/demo.ipynb`](scripts/demo.ipynb) for a walkthrough that clones the repo and runs all **3 architectures × 2 optimizers** (6 setups).

## Suggested experiments

- **Learning-rate sweep**: `optimizers.lr=1e-4,3e-4,1e-3` or `optimizers.muon.lr=...`
- **Muon projection**: `optimizers.muon.projection=exact` vs `optimizers.muon.projection=ns`
- **Momentum / Nesterov**: `optimizers.muon.momentum=0.9 optimizers.muon.nesterov=true`
- **Matrix metrics**: compare `condition_number_weighted_mean` and `orthogonality_error_weighted_mean` in W&B logs
- **Scale**: increase `datasets.train.download_limit`, `trainer.n_epochs`, model size in `src/configs/model/`

## Project layout

```
src/
  configs/          # Hydra configs (lstm, transformer, nanogpt, optimizers, …)
  model/            # LSTM, Transformer, NanoGPT
  optimizers/       # Muon, AdamW builders, projections
  datasets/         # OpenWebText download + text dataset
  metrics/          # Perplexity, matrix metrics
  trainer/          # Training loop (AMP, schedulers, checkpointing)
train.py
scripts/
  demo.ipynb        # end-to-end demo (clone + 6 training runs)
```

## License

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](/LICENSE)
