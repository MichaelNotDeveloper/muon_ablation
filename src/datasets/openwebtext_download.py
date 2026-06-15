import json
import logging
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoTokenizer

from src.datasets.base_text_dataset import BaseTextDataset
from src.utils.io_utils import ROOT_PATH, write_json

logger = logging.getLogger(__name__)

OPENWEBTEXT_DATASET = "Skylion007/openwebtext"
DEFAULT_TOKENIZER = "gpt2"


class OpenWebTextDownload(BaseTextDataset):
    """
    OpenWebText dataset downloaded via HuggingFace datasets.

    Follows the same index + download-on-first-use pattern as the old
    YandexDownload / LibriSpeech-style datasets.
    """

    def __init__(
        self,
        part="train",
        data_dir=None,
        dataset_name=OPENWEBTEXT_DATASET,
        tokenizer_name=DEFAULT_TOKENIZER,
        val_ratio=0.01,
        max_seq_len=512,
        min_seq_len=8,
        limit=None,
        shuffle_index=False,
        download_limit=None,
    ):
        if data_dir is None:
            data_dir = ROOT_PATH / "data" / "datasets" / "openwebtext"
        else:
            data_dir = Path(data_dir)

        self.data_dir = data_dir
        self.part = part
        self.tokenizer_name = tokenizer_name
        self.max_seq_len = max_seq_len
        self.min_seq_len = min_seq_len

        index_path = data_dir / f"{part}_index.json"
        tokens_dir = data_dir / "tokens"

        if not index_path.exists():
            self._download_and_prepare(
                data_dir=data_dir,
                dataset_name=dataset_name,
                tokenizer_name=tokenizer_name,
                val_ratio=val_ratio,
                download_limit=download_limit,
                min_seq_len=min_seq_len,
            )

        index = json.loads(index_path.read_text())
        meta_path = data_dir / "meta.json"
        meta = json.loads(meta_path.read_text())

        self.tokens_dir = tokens_dir
        self.vocab_size = meta["vocab_size"]
        self.pad_id = meta["pad_id"]

        super().__init__(
            index=index,
            pad_id=self.pad_id,
            limit=limit,
            max_seq_len=max_seq_len,
            shuffle_index=shuffle_index,
        )

    def load_tokens(self, data_dict):
        token_path = self.tokens_dir / f"{data_dict['text_id']}.pt"
        return torch.load(token_path, weights_only=True)

    @staticmethod
    def _download_and_prepare(
        data_dir,
        dataset_name,
        tokenizer_name,
        val_ratio,
        download_limit,
        min_seq_len=8,
    ):
        logger.info(f"Downloading {dataset_name} from HuggingFace...")
        data_dir.mkdir(parents=True, exist_ok=True)
        tokens_dir = data_dir / "tokens"
        tokens_dir.mkdir(parents=True, exist_ok=True)

        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        dataset = load_dataset(dataset_name, split="train", streaming=False)
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

            if idx % 10000 == 0 and idx > 0:
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
            f"Saved OpenWebText index: train={len(train_index)}, val={len(val_index)}"
        )
