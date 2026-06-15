from itertools import repeat

from hydra.utils import instantiate

from src.datasets.collate import collate_fn
from src.utils.init_utils import set_worker_seed


def inf_loop(dataloader):
    """
    Wrapper function for endless dataloader.
    Used for iteration-based training scheme.
    """
    for loader in repeat(dataloader):
        yield from loader


def get_dataloaders(config, device):
    """
    Create dataloaders for each of the dataset partitions.
    """
    dataloaders = {}
    pad_id = None

    for dataset_partition in config.datasets.keys():
        if config.datasets[dataset_partition] is None:
            continue

        dataset = instantiate(config.datasets[dataset_partition])
        pad_id = getattr(dataset, "pad_id", config.get("pad_id", 0))

        assert config.batch_size <= len(dataset), (
            f"The batch size ({config.batch_size}) cannot "
            f"be larger than the dataset length ({len(dataset)})"
        )

        partition_dataloader = instantiate(
            config=config.dataloader[f"{dataset_partition}_dataloader"],
            dataset=dataset,
            collate_fn=collate_fn,
            drop_last=(dataset_partition == "train"),
            shuffle=(dataset_partition == "train"),
            worker_init_fn=set_worker_seed,
        )
        dataloaders[dataset_partition] = partition_dataloader

    return dataloaders, None
