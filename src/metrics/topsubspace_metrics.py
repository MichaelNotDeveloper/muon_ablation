"""
Top-subspace concentration metrics adapted from kate2/nanogpt_layerwise_hessian_top.py.

Uses the top-k singular-vector subspace of each 2D weight matrix as a proxy
for the Hessian top eigenspace during training.
"""

import numpy as np
import torch


def _to_float32_cpu(weight: torch.Tensor) -> torch.Tensor:
    return weight.detach().float().cpu()


def rho_k(d: torch.Tensor, u_flat: torch.Tensor) -> float:
    """
    Fraction of ||d||^2 captured by the subspace spanned by columns of u_flat.
    """
    d_flat = d.reshape(-1).float()
    denom = torch.dot(d_flat, d_flat).clamp_min(1e-30)
    coeffs = u_flat.float().T @ d_flat
    return float((coeffs.square().sum() / denom).detach().cpu())


def concentration_c(d: torch.Tensor, u_flat: torch.Tensor) -> float:
    """
    Normalized top-subspace concentration: rho_k / (k / p).
    """
    p = int(d.numel())
    k = int(u_flat.shape[1])
    return rho_k(d, u_flat) / max(float(k) / float(p), 1e-30)


def top_k_subspace_basis(weight: torch.Tensor, k: int) -> torch.Tensor:
    """
    Top-k left singular vectors of a 2D matrix, shape (numel_rows, k).
    """
    mat = _to_float32_cpu(weight)
    u, _, _ = torch.linalg.svd(mat, full_matrices=False)
    k_eff = min(k, u.shape[1])
    return u[:, :k_eff]


def topsubspace_metrics(weight: torch.Tensor, k: int = 5):
    basis = top_k_subspace_basis(weight, k)
    k_eff = basis.shape[1]
    rho = rho_k(weight, basis)
    concentration = concentration_c(weight, basis)
    return {
        "topsubspace_k": k_eff,
        "rho_k": rho,
        "concentration_c": concentration,
    }


def collect_weighted_topsubspace_metrics(model, k: int = 5):
    rows = []
    for name, param in model.named_parameters():
        if param.ndim != 2:
            continue
        stats = topsubspace_metrics(param, k=k)
        stats["name"] = name
        stats["numel"] = int(param.numel())
        rows.append(stats)

    summary = {
        "topsubspace_count": len(rows),
        "topsubspace_metrics_by_parameter": rows,
        "topsubspace_k": k,
    }

    metric_names = ("rho_k", "concentration_c")
    if not rows:
        for metric_name in metric_names:
            summary[f"{metric_name}_weighted_mean"] = float("nan")
        return summary

    weights = np.array([row["numel"] for row in rows], dtype=np.float64)
    for metric_name in metric_names:
        values = np.array([row[metric_name] for row in rows], dtype=np.float64)
        summary[f"{metric_name}_weighted_mean"] = float(np.average(values, weights=weights))

    return summary
