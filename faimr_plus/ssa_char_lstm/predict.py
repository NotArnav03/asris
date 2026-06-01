"""Inference path for the SSA char-LSTM plugin.

Lazy-loads weights.pt once per process.  predict_p_female accepts a
list of names and returns a numpy array of P(female).
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from faimr_plus.ssa_char_lstm.model import (
    CharBiLSTMGenderClassifier, encode_name,
)

WEIGHTS_PATH = Path(__file__).resolve().parent / "weights.pt"

_model: CharBiLSTMGenderClassifier | None = None
_load_lock = threading.Lock()


def _load() -> CharBiLSTMGenderClassifier:
    global _model
    if _model is not None:
        return _model
    with _load_lock:
        if _model is None:
            if not WEIGHTS_PATH.exists():
                raise FileNotFoundError(
                    f"SSA char-LSTM weights not found at {WEIGHTS_PATH}.  "
                    f"Run `python -m faimr_plus.ssa_char_lstm.train` first."
                )
            m = CharBiLSTMGenderClassifier()
            m.load_state_dict(
                torch.load(WEIGHTS_PATH, map_location="cpu"),
            )
            m.eval()
            _model = m
    return _model


@torch.no_grad()
def predict_p_female(names: Sequence[str]) -> np.ndarray:
    """Return P(female) for each name in the batch.

    Names are lowercased and ASCII-truncated by encode_name; non-ASCII
    characters become UNK tokens.  Empty names return p=0.5.
    """
    model = _load()
    if not names:
        return np.array([], dtype=np.float32)
    encoded = np.array(
        [encode_name(n if n else "") for n in names],
        dtype=np.int64,
    )
    x = torch.from_numpy(encoded)
    logits = model(x)
    return torch.sigmoid(logits).cpu().numpy().astype(np.float32)
