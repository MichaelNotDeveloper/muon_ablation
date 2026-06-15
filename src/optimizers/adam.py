import torch


def build_adamw(
    model,
    lr=1e-3,
    betas=(0.9, 0.999),
    weight_decay=0.01,
    **kwargs,
):
    """
    Build an AdamW optimizer for all trainable model parameters.
    """
    params = [p for p in model.parameters() if p.requires_grad]
    return torch.optim.AdamW(
        params,
        lr=lr,
        betas=tuple(betas),
        weight_decay=weight_decay,
        **kwargs,
    )
