import torch

from src.optimizers.multi import MultiOptimizer
from src.optimizers.muon import Muon


def split_matrix_params(model):
    matrix_params = []
    other_params = []
    for param in model.parameters():
        if not param.requires_grad:
            continue
        if param.ndim == 2:
            matrix_params.append(param)
        else:
            other_params.append(param)
    return matrix_params, other_params


def build_muon_hybrid(
    model,
    muon=None,
    adam=None,
):
    """
    Build Muon+AdamW hybrid optimizer for language models.

    Muon updates 2D parameters; remaining params use AdamW.
    """
    muon_cfg = dict(
        lr=1e-2,
        momentum=0.0,
        nesterov=False,
        projection="exact",
        ns_steps=5,
        eps=1e-7,
    )
    muon_cfg.update(muon or {})

    adam_cfg = dict(lr=1e-3, betas=(0.9, 0.999), weight_decay=0.01)
    adam_cfg.update(adam or {})
    adam_cfg["betas"] = tuple(adam_cfg["betas"])

    matrix_params, other_params = split_matrix_params(model)
    optimizers = [Muon(matrix_params, **muon_cfg)]
    if other_params:
        optimizers.append(torch.optim.AdamW(other_params, **adam_cfg))
    return MultiOptimizer(optimizers)


# Backward-compatible alias used by older configs.
build_language_model_optimizer = build_muon_hybrid
