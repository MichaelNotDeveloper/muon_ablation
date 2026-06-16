import os
from pathlib import Path

import hydra
import matplotlib.pyplot as plt
import wandb

from src.utils.io_utils import ROOT_PATH


DEFAULT_LRS = [1e-5, 2e-5, 5e-5, 1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2]


def build_run_names(lrs):
    return {
        f"{model_name}_{lr}"
        for model_name in ["lstm", "transformer", "nanogpt"]
        for lr in lrs
    }


def _resolve_entity(api, config):
    entity = config.writer.get("entity")
    if entity:
        return entity
    entity = os.getenv("WANDB_ENTITY")
    if entity:
        return entity
    entity = getattr(api, "default_entity", None)
    if entity:
        return entity
    entity = getattr(getattr(api, "api", None), "default_entity", None)
    if entity:
        return entity
    raise ValueError(
        "Could not resolve W&B entity. Set writer.entity in config or WANDB_ENTITY env var."
    )


@hydra.main(version_base=None, config_path="../src/configs", config_name="lstm")
def main(config):
    run_names = build_run_names(DEFAULT_LRS)
    project = config.writer.project_name

    api = wandb.Api()
    entity = _resolve_entity(api, config)
    runs = [run for run in api.runs(f"{entity}/{project}") if run.name in run_names]

    plt.figure(figsize=(14, 8))
    seen = 0
    for run in runs:
        history = run.history(
            keys=["epoch_elapsed_seconds_epoch", "train_loss_epoch", "val_loss_epoch"],
            pandas=True,
        )
        history = history.dropna(subset=["epoch_elapsed_seconds_epoch"])
        if history.empty:
            continue

        x = history["epoch_elapsed_seconds_epoch"].values
        if "train_loss_epoch" in history.columns:
            plt.plot(
                x,
                history["train_loss_epoch"].values,
                alpha=0.6,
                linestyle="-",
                label=f"{run.name} train",
            )
        if "val_loss_epoch" in history.columns:
            plt.plot(
                x,
                history["val_loss_epoch"].values,
                alpha=0.6,
                linestyle="--",
                label=f"{run.name} val",
            )
        seen += 1

    plt.xlabel("Elapsed time since training start (seconds)")
    plt.ylabel("Loss")
    plt.title("Train/Val Loss vs Elapsed Time")
    if seen <= 12:
        plt.legend(fontsize=8)
    plt.grid(True, alpha=0.3)

    save_dir = ROOT_PATH / config.trainer.save_dir
    save_dir.mkdir(parents=True, exist_ok=True)
    output_path = save_dir / "loss_vs_time.png"
    plt.savefig(output_path, dpi=180, bbox_inches="tight")
    print(f"Plotted {seen} runs.")
    print(f"Saved figure to {output_path}")


if __name__ == "__main__":
    main()
