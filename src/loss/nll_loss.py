import torch
import torch.nn as nn
import torch.nn.functional as F


class NLLLoss(nn.Module):
    """
    Negative log-likelihood (cross-entropy) loss for language modeling.
    """

    def forward(self, logits, targets, pad_id=0, **batch):
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1),
            ignore_index=pad_id,
        )
        return {"loss": loss}
