import json
import logging
from pathlib import Path

import torch
from datasets import load_dataset

from src.datasets.base_text_dataset import BaseTextDataset
from src.datasets.text_download_utils import prepare_tokenized_text_dataset
from src.utils.io_utils import ROOT_PATH

logger = logging.getLogger(__name__)

OPENWEBTEXT_DATASET = "Skylion007/openwebtext"
DEFAULT_TOKENIZER = "gpt2"


class OpenWebTextDownload(BaseTextDataset):
    """
    Full OpenWebText dataset downloaded via HuggingFace datasets.
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
        meta = json.loads((data_dir / "meta.json").read_text())

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
        dataset = load_dataset(dataset_name, split="train", streaming=False)
        if download_limit is not None:
            dataset = dataset.select(range(min(download_limit, len(dataset))))

        texts = [
            example.get("text", "").strip()
            for example in dataset
            if example.get("text", "").strip()
        ]
        prepare_tokenized_text_dataset(
            data_dir=data_dir,
            texts=texts,
            dataset_name=dataset_name,
            tokenizer_name=tokenizer_name,
            val_ratio=val_ratio,
            min_seq_len=min_seq_len,
        )
