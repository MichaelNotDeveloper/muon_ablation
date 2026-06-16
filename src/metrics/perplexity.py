import math

import numpy as np
import torch
import torch.nn.functional as F

from src.metrics.base_metric import BaseMetric


class Perplexity(BaseMetric):
    """
    Token-weighted perplexity metric.
    """

    def __init__(self, name="perplexity", pad_id=0):
        super().__init__(name=name)
        self.pad_id = pad_id
        self.reset()

    def reset(self):
        self.total_loss = 0.0
        self.total_tokens = 0

    def __call__(self, logits, targets, pad_id=None, **batch):
        pad_id = self.pad_id if pad_id is None else pad_id
        loss_sum = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1),
            ignore_index=pad_id,
            reduction="sum",
        )
        n_tokens = (targets != pad_id).sum().item()
        self.total_loss += loss_sum.item()
        self.total_tokens += n_tokens
        if n_tokens == 0:
            return float("nan")
        return math.exp(loss_sum.item() / n_tokens)

    def value(self):
        if self.total_tokens == 0:
            return float("nan")
        return math.exp(self.total_loss / self.total_tokens)
