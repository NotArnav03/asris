"""Bi-LSTM char-level model for SSA name-gender classification.

Architecture (tiny by design -- this is a name classifier, not a
language model):

  char embedding (32-d)
    -> bi-LSTM (hidden 64, 1 layer, dropout 0.2)
    -> max-pool + last-state concat (256-d)
    -> linear -> sigmoid (1-d, P(female))

Vocabulary: lowercase a-z plus a small set of common diacritic-stripped
letters + special tokens.  Names are truncated/padded to MAX_LEN.

Total parameters: ~80k.  Trains in ~10-30 min on CPU.
"""

from __future__ import annotations

import string

import torch
import torch.nn as nn

MAX_LEN = 24
PAD_IDX = 0
UNK_IDX = 1
SOS_IDX = 2
EOS_IDX = 3
SPECIAL_TOKENS = 4

VOCAB_CHARS = string.ascii_lowercase  # 26
VOCAB_SIZE = SPECIAL_TOKENS + len(VOCAB_CHARS)  # 30

CHAR_TO_IDX = {c: i + SPECIAL_TOKENS for i, c in enumerate(VOCAB_CHARS)}


def encode_name(name: str) -> list[int]:
    """Lowercase, ASCII-only, truncate to MAX_LEN-2, prepend SOS, append EOS,
    pad to MAX_LEN."""
    name = name.lower()
    ids = [SOS_IDX]
    for ch in name:
        if len(ids) >= MAX_LEN - 1:
            break
        ids.append(CHAR_TO_IDX.get(ch, UNK_IDX))
    ids.append(EOS_IDX)
    while len(ids) < MAX_LEN:
        ids.append(PAD_IDX)
    return ids


class CharBiLSTMGenderClassifier(nn.Module):
    def __init__(
        self,
        vocab_size: int = VOCAB_SIZE,
        embed_dim: int = 48,
        hidden_dim: int = 96,
        num_layers: int = 1,
        dropout: float = 0.35,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(
            vocab_size, embed_dim, padding_idx=PAD_IDX,
        )
        self.lstm = nn.LSTM(
            embed_dim, hidden_dim,
            num_layers=num_layers, batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        # 2 * hidden_dim (bi) for max-pool, plus 2 * hidden_dim for
        # last-state concat -> 4 * hidden_dim
        self.fc1 = nn.Linear(4 * hidden_dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L)
        mask = (x != PAD_IDX).unsqueeze(-1).float()  # (B, L, 1)
        emb = self.embedding(x)                       # (B, L, E)
        out, (h, _) = self.lstm(emb)                  # out: (B, L, 2H)

        # Max-pool over time, ignoring PAD.
        out_masked = out.masked_fill(mask == 0, float("-inf"))
        pooled, _ = out_masked.max(dim=1)             # (B, 2H)

        # Concatenate last hidden states from each direction.
        h_fwd = h[0]                                  # (B, H)
        h_bwd = h[1]                                  # (B, H)
        last = torch.cat([h_fwd, h_bwd], dim=-1)      # (B, 2H)

        rep = torch.cat([pooled, last], dim=-1)       # (B, 4H)
        rep = self.dropout(rep)
        rep = self.act(self.fc1(rep))
        rep = self.dropout(rep)
        logits = self.fc2(rep).squeeze(-1)            # (B,)
        return logits
