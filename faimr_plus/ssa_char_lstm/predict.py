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

_HERE = Path(__file__).resolve().parent
WEIGHTS_PATH_V1 = _HERE / "weights.pt"
WEIGHTS_PATH_V2 = _HERE / "weights_v2.pt"
META_PATH_V2    = _HERE / "meta_v2.json"

_model: CharBiLSTMGenderClassifier | None = None
_load_lock = threading.Lock()


def _load() -> CharBiLSTMGenderClassifier:
    """Load the deployed SSA char-LSTM.  We currently deploy v1
    (trained on the SSA aggregate; gives the best hybrid result of
    0.9393 full-SSA accuracy when combined with FAIMR's lookup).
    The v2 experiment (trained on training_corpus.csv with a larger
    architecture) tested whether a multicultural-firstname-DB
    training set could push past Hu 2021's 0.940 SOTA -- it did not
    (v2 standalone: 0.9036 on SSA; v2 hybrid: 0.9279, regressed
    vs v1).  See meta_v2.json for full diagnostics."""
    global _model
    if _model is not None:
        return _model
    with _load_lock:
        if _model is None:
            if WEIGHTS_PATH_V1.exists():
                m = CharBiLSTMGenderClassifier()
                m.load_state_dict(
                    torch.load(WEIGHTS_PATH_V1, map_location="cpu"),
                )
            elif WEIGHTS_PATH_V2.exists():
                import json as _json
                meta = _json.loads(META_PATH_V2.read_text(encoding="utf-8")) \
                       if META_PATH_V2.exists() else {}
                m = CharBiLSTMGenderClassifier(
                    embed_dim=meta.get("embed_dim", 64),
                    hidden_dim=meta.get("hidden_dim", 128),
                    num_layers=meta.get("num_layers", 2),
                    dropout=meta.get("dropout", 0.4),
                )
                m.load_state_dict(
                    torch.load(WEIGHTS_PATH_V2, map_location="cpu"),
                )
            else:
                raise FileNotFoundError(
                    f"No SSA char-LSTM weights found.  Run "
                    f"`python -m faimr_plus.ssa_char_lstm.train` "
                    f"or `train_v2` first."
                )
            m.eval()
            _model = m
    return _model


@torch.no_grad()
def _predict_with(model, names: Sequence[str]) -> np.ndarray:
    encoded = np.array(
        [encode_name(n if n else "") for n in names],
        dtype=np.int64,
    )
    x = torch.from_numpy(encoded)
    return torch.sigmoid(model(x)).cpu().numpy().astype(np.float32)


_model_v2: CharBiLSTMGenderClassifier | None = None
_v2_lock = threading.Lock()


def _load_v2() -> CharBiLSTMGenderClassifier | None:
    """Lazy-load v2 weights if present; return None if not."""
    global _model_v2
    if _model_v2 is not None:
        return _model_v2
    if not WEIGHTS_PATH_V2.exists():
        return None
    with _v2_lock:
        if _model_v2 is None:
            import json as _json
            meta = _json.loads(META_PATH_V2.read_text(encoding="utf-8")) \
                   if META_PATH_V2.exists() else {}
            m = CharBiLSTMGenderClassifier(
                embed_dim=meta.get("embed_dim", 64),
                hidden_dim=meta.get("hidden_dim", 128),
                num_layers=meta.get("num_layers", 2),
                dropout=meta.get("dropout", 0.4),
            )
            m.load_state_dict(
                torch.load(WEIGHTS_PATH_V2, map_location="cpu"),
            )
            m.eval()
            _model_v2 = m
    return _model_v2


@torch.no_grad()
def predict_p_female(names: Sequence[str]) -> np.ndarray:
    """Return P(female) for each name in the batch.

    Uses the deployed v1 model.  v2 was tested both standalone and
    as an ensemble component with v1 (60/40 and 80/20 weights) -- in
    both cases v1-alone outperformed the ensemble on the SSA test
    set, so the ensemble path is disabled in the deployed code.
    See meta_v2.json for diagnostics and faimr_plus/ssa_char_lstm/
    README.md "v2 experiment" section for the full discussion.

    Names are lowercased and ASCII-truncated by encode_name;
    non-ASCII characters become UNK tokens.  Empty names return
    p=0.5.
    """
    if not names:
        return np.array([], dtype=np.float32)
    return _predict_with(_load(), names)
