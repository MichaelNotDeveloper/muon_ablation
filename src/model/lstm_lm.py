import torch
import torch.nn as nn


class LSTMLanguageModel(nn.Module):
    def __init__(
        self,
        vocab_size,
        emb_dim=128,
        hidden_dim=256,
        num_layers=2,
        dropout=0.2,
        pad_id=0,
    ):
        super().__init__()
        self.pad_id = pad_id

        self.embedding = nn.Embedding(
            vocab_size,
            emb_dim,
            padding_idx=pad_id,
        )
        self.lstm = nn.LSTM(
            input_size=emb_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_dim, vocab_size)

    def forward(self, input_ids, targets=None, **batch):
        x = self.embedding(input_ids)
        out, _ = self.lstm(x)
        out = self.dropout(out)
        logits = self.head(out)
        return {"logits": logits}
