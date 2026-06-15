import logging
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoTokenizer

from src.utils.io_utils import write_json

logger = logging.getLogger(__name__)

DEFAULT_TOKENIZER = "gpt2"


def prepare_tokenized_text_dataset(
    data_dir,
    dataset_name,
    tokenizer_name=DEFAULT_TOKENIZER,
    val_ratio=0.01,
    download_limit=None,
    min_seq_len=8,
    trust_remote_code=False,
):
    """
    Download a HuggingFace text dataset, tokenize, and build train/val index files.
    """
    logger.info(f"Downloading {dataset_name} from HuggingFace...")
    data_dir.mkdir(parents=True, exist_ok=True)
    tokens_dir = data_dir / "tokens"
    tokens_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = load_dataset(
        dataset_name,
        split="train",
        streaming=False,
        trust_remote_code=trust_remote_code,
    )
    if download_limit is not None:
        dataset = dataset.select(range(min(download_limit, len(dataset))))

    train_index = []
    val_index = []
    pad_id = tokenizer.pad_token_id
    val_mod = max(1, int(round(1 / val_ratio)))

    for idx, example in enumerate(dataset):
        text = example.get("text", "").strip()
        if not text:
            continue

        tokens = tokenizer.encode(text, add_special_tokens=True)
        seq_len = len(tokens)
        if seq_len < min_seq_len:
            continue

        token_path = tokens_dir / f"{idx}.pt"
        torch.save(torch.tensor(tokens, dtype=torch.long), token_path)

        entry = {"text_id": str(idx), "seq_len": seq_len}
        if idx % val_mod == 0:
            val_index.append(entry)
        else:
            train_index.append(entry)

        if idx % 1000 == 0 and idx > 0:
            logger.info(f"Processed {idx} examples...")

    write_json(train_index, data_dir / "train_index.json")
    write_json(val_index, data_dir / "val_index.json")
    write_json(
        {
            "vocab_size": len(tokenizer),
            "pad_id": pad_id,
            "tokenizer_name": tokenizer_name,
            "dataset_name": dataset_name,
        },
        data_dir / "meta.json",
    )
    logger.info(
        f"Saved dataset index for {dataset_name}: "
        f"train={len(train_index)}, val={len(val_index)}"
    )
