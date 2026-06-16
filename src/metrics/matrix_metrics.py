import math

import numpy as np
import torch


def _to_float32_cpu(weight: torch.Tensor) -> torch.Tensor:
    return weight.detach().float().cpu()


def matrix_metrics(weight):
    mat = _to_float32_cpu(weight)
    rows, cols = mat.shape

    singular_values = torch.linalg.svdvals(mat)
    singular_values = torch.clamp(singular_values, min=1e-12)

    spectral_norm = singular_values.max().item()
    min_singular = singular_values.min().item()
    fro_norm = torch.linalg.matrix_norm(mat, ord="fro").item()

    stable_rank = (fro_norm**2) / max(spectral_norm**2, 1e-12)

    probs = singular_values / singular_values.sum()
    entropy = -(probs * probs.log()).sum().item()
    effective_rank = math.exp(entropy)

    normalized = mat / max(spectral_norm, 1e-12)
    if rows >= cols:
        gram = normalized.T @ normalized
        identity = torch.eye(cols, dtype=mat.dtype, device=mat.device)
    else:
        gram = normalized @ normalized.T
        identity = torch.eye(rows, dtype=mat.dtype, device=mat.device)

    orthogonality_error = torch.linalg.matrix_norm(gram - identity, ord="fro").item()
    orthogonality_error /= math.sqrt(identity.size(0))

    return {
        "condition_number": spectral_norm / max(min_singular, 1e-12),
        "orthogonality_error": orthogonality_error,
        "stable_rank": stable_rank,
        "effective_rank": effective_rank,
        "spectral_norm": spectral_norm,
        "min_singular": min_singular,
    }


def collect_weighted_matrix_metrics(model):
    rows = []
    for name, param in model.named_parameters():
        if param.ndim != 2:
            continue
        stats = matrix_metrics(param)
        stats["name"] = name
        stats["numel"] = int(param.numel())
        rows.append(stats)

    summary = {
        "matrix_count": len(rows),
        "matrix_metrics_by_parameter": rows,
    }

    metric_names = (
        "condition_number",
        "orthogonality_error",
        "stable_rank",
        "effective_rank",
        "spectral_norm",
        "min_singular",
    )

    if not rows:
        for metric_name in metric_names:
            summary[f"{metric_name}_weighted_mean"] = float("nan")
        return summary

    weights = np.array([row["numel"] for row in rows], dtype=np.float64)
    for metric_name in metric_names:
        values = np.array([row[metric_name] for row in rows], dtype=np.float64)
        summary[f"{metric_name}_weighted_mean"] = float(np.average(values, weights=weights))

    return summary
