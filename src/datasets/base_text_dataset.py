import logging
import random
from copy import deepcopy

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
        index = self._chunk_records_to_max_seq_len(index, max_seq_len)
        index = self._shuffle_and_limit_index(index, limit, shuffle_index)
        if not shuffle_index:
            index = self._sort_index(index)

        self._index = index
        self.pad_id = pad_id

    def __getitem__(self, ind):
        data_dict = deepcopy(self._index[ind])
        tokens = self.load_tokens(data_dict)
        chunk_start = data_dict.get("chunk_start", 0)
        chunk_len = data_dict["seq_len"]
        tokens = tokens[chunk_start : chunk_start + chunk_len]
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
    def _chunk_records_to_max_seq_len(index, max_seq_len, min_seq_len=8):
        """
        Fit each example to max_seq_len input tokens.

        Short sequences are kept as-is. Longer ones are split into contiguous,
        non-overlapping chunks (standard GPT-style block packing).
        """
        if max_seq_len is None:
            return index

        expanded = []
        long_records = 0
        for entry in index:
            input_len = entry["seq_len"] - 1
            if input_len <= max_seq_len:
                expanded.append(entry)
                continue

            long_records += 1
            for chunk_start in range(0, input_len, max_seq_len):
                chunk_input_len = min(max_seq_len, input_len - chunk_start)
                if chunk_input_len < min_seq_len:
                    continue
                expanded.append(
                    {
                        "text_id": entry["text_id"],
                        "seq_len": chunk_input_len + 1,
                        "chunk_start": chunk_start,
                    }
                )

        if long_records:
            logger.info(
                f"Chunked {long_records} long records into {len(expanded)} examples "
                f"of up to {max_seq_len} tokens"
            )
        return expanded

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
