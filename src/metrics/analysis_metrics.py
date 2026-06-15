from src.metrics.matrix_metrics import collect_weighted_matrix_metrics
from src.metrics.topsubspace_metrics import collect_weighted_topsubspace_metrics


def collect_epoch_analysis_metrics(model, config):
    """
    Collect optional matrix / top-subspace metrics based on trainer config.
    """
    trainer_cfg = config.trainer
    logs = {}

    if trainer_cfg.get("compute_matrix_metrics", True):
        matrix_summary = collect_weighted_matrix_metrics(model)
        for metric_name, value in matrix_summary.items():
            if metric_name.endswith("_weighted_mean"):
                logs[metric_name] = value

    if trainer_cfg.get("compute_topsubspace_metrics", False):
        topsubspace_summary = collect_weighted_topsubspace_metrics(
            model,
            k=trainer_cfg.get("topsubspace_k", 5),
        )
        for metric_name, value in topsubspace_summary.items():
            if metric_name.endswith("_weighted_mean"):
                logs[metric_name] = value

    return logs
