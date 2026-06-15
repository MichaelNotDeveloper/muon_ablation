import torch


def collate_fn(dataset_items: list[dict]):
    assert len(dataset_items)
    pad_id = dataset_items[0]["pad_id"]

    input_ids = [item["input_ids"] for item in dataset_items]
    targets = [item["targets"] for item in dataset_items]
    max_len = max(len(x) for x in input_ids)

    input_batch = torch.full((len(input_ids), max_len), pad_id, dtype=torch.long)
    target_batch = torch.full((len(targets), max_len), pad_id, dtype=torch.long)

    for i, (x, y) in enumerate(zip(input_ids, targets)):
        input_batch[i, : len(x)] = x
        target_batch[i, : len(y)] = y

    return {
        "input_ids": input_batch,
        "targets": target_batch,
        "pad_id": pad_id,
    }
