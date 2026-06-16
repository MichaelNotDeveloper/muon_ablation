import os
from pathlib import Path

import hydra
import matplotlib
import wandb

from src.utils.io_utils import ROOT_PATH

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def build_run_names():
    return {"adamw_time", "muon_time"}


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
    run_names = build_run_names()
    project = config.writer.project_name

    api = wandb.Api()
    entity = _resolve_entity(api, config)
    runs = [run for run in api.runs(f"{entity}/{project}") if run.name in run_names]

    plt.figure(figsize=(14, 8))
    seen = 0
    for run in runs:
        history = run.history(pandas=True)
        time_key = next(
            (
                key
                for key in ("epoch_elapsed_seconds_epoch", "epoch_elapsed_seconds")
                if key in history.columns
            ),
            None,
        )
        if time_key is None:
            continue
        history = history.dropna(subset=[time_key])
        if history.empty:
            continue

        x = history[time_key].values
        train_key = next(
            (
                key
                for key in ("train_loss_epoch", "loss_train", "loss")
                if key in history.columns
            ),
            None,
        )
        val_key = next(
            (
                key
                for key in ("val_loss_epoch", "loss_val", "val_loss")
                if key in history.columns
            ),
            None,
        )
        if train_key is not None:
            plt.plot(
                x,
                history[train_key].values,
                alpha=0.6,
                linestyle="-",
                label=f"{run.name} train",
            )
        if val_key is not None:
            plt.plot(
                x,
                history[val_key].values,
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
    plt.close()

    mode = config.writer.get("mode", "online")
    if mode != "offline":
        log_run = wandb.init(
            project=project,
            entity=entity,
            name="loss_vs_time_plot",
            job_type="analysis",
            reinit=True,
        )
        wandb.log({"loss_vs_time": wandb.Image(str(output_path))})
        log_run.finish()

    print(f"Plotted {seen} runs.")
    print(f"Saved figure to {output_path}")


if __name__ == "__main__":
    main()
