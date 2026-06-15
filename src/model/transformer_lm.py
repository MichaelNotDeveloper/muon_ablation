import math

import torch
import torch.nn as nn


class TransformerLanguageModel(nn.Module):
    def __init__(
        self,
        vocab_size,
        d_model=192,
        nhead=4,
        num_layers=3,
        dim_feedforward=384,
        dropout=0.2,
        max_len=512,
        pad_id=0,
    ):
        super().__init__()

        self.pad_id = pad_id
        self.d_model = d_model
        self.max_len = max_len

        self.embedding = nn.Embedding(
            vocab_size,
            d_model,
            padding_idx=pad_id,
        )
        self.pos_embedding = nn.Embedding(max_len, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids, targets=None, **batch):
        batch_size, seq_len = input_ids.shape
        if seq_len > self.max_len:
            raise ValueError(f"Sequence length {seq_len} > max_len {self.max_len}")

        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        token_emb = self.embedding(input_ids)
        pos_emb = self.pos_embedding(positions)

        hidden = token_emb * math.sqrt(self.d_model) + pos_emb
        hidden = self.dropout(hidden)

        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=input_ids.device, dtype=torch.bool),
            diagonal=1,
        )
        padding_mask = input_ids.eq(self.pad_id)

        hidden = self.transformer(
            hidden,
            mask=causal_mask,
            src_key_padding_mask=padding_mask,
        )
        logits = self.head(hidden)
        return {"logits": logits}
