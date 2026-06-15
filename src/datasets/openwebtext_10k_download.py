import json
import logging
from pathlib import Path

import torch

from src.datasets.base_text_dataset import BaseTextDataset
from src.datasets.text_download_utils import prepare_openwebtext_10k_dataset
from src.utils.io_utils import ROOT_PATH

logger = logging.getLogger(__name__)

OPENWEBTEXT_10K_DATASET = "stas/openwebtext-10k"


class OpenWebText10kDownload(BaseTextDataset):
    """
    10K subset of OpenWebText.

    Downloads directly from the HuggingFace CDN (no dataset loading script):
    https://huggingface.co/datasets/stas/openwebtext-10k
    """

    def __init__(
        self,
        part="train",
        data_dir=None,
        dataset_name=OPENWEBTEXT_10K_DATASET,
        tokenizer_name="gpt2",
        val_ratio=0.01,
        max_seq_len=512,
        min_seq_len=8,
        limit=None,
        shuffle_index=False,
        download_limit=None,
    ):
        if data_dir is None:
            data_dir = ROOT_PATH / "data" / "datasets" / "openwebtext_10k"
        else:
            data_dir = Path(data_dir)

        self.data_dir = data_dir
        self.part = part

        index_path = data_dir / f"{part}_index.json"
        tokens_dir = data_dir / "tokens"

        if not index_path.exists():
            prepare_openwebtext_10k_dataset(
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
