import logging
import random
from copy import deepcopy

import numpy as np
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class BaseTextDataset(Dataset):
    """
    Base class for text language-modeling datasets.

    Uses the same index-based structure as the audio BaseDataset:
    each index entry is a dict with metadata (text_id, seq_len, etc.).
    """

    def __init__(
        self,
        index,
        pad_id=0,
        limit=None,
        max_seq_len=None,
        shuffle_index=False,
    ):
        self._assert_index_is_valid(index)
        index = self._filter_records_from_dataset(index, max_seq_len)
        index = self._shuffle_and_limit_index(index, limit, shuffle_index)
        if not shuffle_index:
            index = self._sort_index(index)

        self._index = index
        self.pad_id = pad_id

    def __getitem__(self, ind):
        data_dict = deepcopy(self._index[ind])
        tokens = self.load_tokens(data_dict)
        input_ids = tokens[:-1]
        targets = tokens[1:]
        return {
            "input_ids": input_ids,
            "targets": targets,
            "pad_id": self.pad_id,
        }

    def __len__(self):
        return len(self._index)

    def load_tokens(self, data_dict):
        raise NotImplementedError()

    @staticmethod
    def _filter_records_from_dataset(index, max_seq_len):
        initial_size = len(index)
        if max_seq_len is not None:
            exceeds_seq_len = np.array([el["seq_len"] for el in index]) > max_seq_len
            _total = exceeds_seq_len.sum()
            logger.info(
                f"{_total} ({_total / initial_size:.1%}) records are longer than "
                f"{max_seq_len} tokens. Excluding them."
            )
        else:
            exceeds_seq_len = False

        if exceeds_seq_len is not False and exceeds_seq_len.any():
            _total = exceeds_seq_len.sum()
            index = [el for el, exclude in zip(index, exceeds_seq_len) if not exclude]
            logger.info(
                f"Filtered {_total} ({_total / initial_size:.1%}) records from dataset"
            )
        return index

    @staticmethod
    def _assert_index_is_valid(index):
        for entry in index:
            assert "text_id" in entry, "Each dataset item should include field 'text_id'."
            assert "seq_len" in entry, "Each dataset item should include field 'seq_len'."

    @staticmethod
    def _sort_index(index):
        return sorted(index, key=lambda x: x["seq_len"])

    @staticmethod
    def _shuffle_and_limit_index(index, limit, shuffle_index):
        if shuffle_index:
            random.seed(42)
            random.shuffle(index)
        if limit is not None:
            index = index[:limit]
        return index
