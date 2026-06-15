import warnings

import hydra
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from src.datasets.data_utils import get_dataloaders
from src.trainer import Trainer
from src.utils.init_utils import (
    select_most_suitable_gpu,
    set_random_seed,
    setup_saving_and_logging,
)
from src.utils.torch_utils import set_tf32_allowance

warnings.filterwarnings("ignore", category=UserWarning)


@hydra.main(version_base=None, config_path="src/configs", config_name="lstm")
def main(config):
    set_random_seed(
        config.trainer.seed, config.trainer.get("save_reproducibility", True)
    )
    set_tf32_allowance(config.trainer.get("tf32_allowance", False))

    project_config = OmegaConf.to_container(config)
    logger = setup_saving_and_logging(config)
    writer = instantiate(config.writer, logger, project_config)

    device = config.trainer.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        device, free_memories = select_most_suitable_gpu()
        logger.info(f"Using GPU: {device} with {free_memories / 1024 ** 3:.2f} GB free")

    dataloaders, _ = get_dataloaders(config, device)

    train_dataset = dataloaders["train"].dataset
    OmegaConf.set_struct(config, False)
    config.vocab_size = getattr(train_dataset, "vocab_size", config.vocab_size)
    config.pad_id = getattr(train_dataset, "pad_id", config.pad_id)
    OmegaConf.set_struct(config, True)

    model = instantiate(config.model).to(device)
    if config.trainer.parallel:
        model = torch.nn.DataParallel(model)
    logger.info(model)

    loss_function = instantiate(config.loss_function).to(device)

    metrics = {"train": [], "inference": []}
    for metric_type in ["train", "inference"]:
        for metric_config in config.metrics.get(metric_type, []):
            metrics[metric_type].append(instantiate(metric_config))

    optimizer = instantiate(config.optimizers, model=model)

    lr_scheduler = instantiate(config.lr_scheduler, optimizer=optimizer)

    epoch_len = config.trainer.get("epoch_len")
    trainer = Trainer(
        model=model,
        criterion=loss_function,
        metrics=metrics,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        config=config,
        device=device,
        dtype=config.trainer.get("dtype", "float32"),
        dataloaders=dataloaders,
        epoch_len=epoch_len,
        logger=logger,
        writer=writer,
        batch_transforms=None,
        skip_oom=config.trainer.get("skip_oom", True),
    )

    trainer.train()


if __name__ == "__main__":
    main()
