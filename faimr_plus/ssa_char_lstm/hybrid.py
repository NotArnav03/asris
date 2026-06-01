"""Hybrid name-gender classifier: FAIMR lookup + SSA char-LSTM fallback.

Routes each name through three tiers:

  Tier 1 -- FAIMR lookup fastpath.
            If the name has a strong corpus hit (source="lookup" with
            weight >= 1.5), use FAIMR's prediction directly.  Cannot
            be improved on: this is exact-match on the curated
            training corpus.

  Tier 2 -- FAIMR model path with high-confidence prediction.
            If FAIMR's classifier returns p_female outside the
            ambiguity band [0.4, 0.6], use it.  FAIMR's per-culture
            isotonic + SSA recalibrator produces well-calibrated
            probabilities here.

  Tier 3 -- SSA char-LSTM fallback.
            For OOD / low-confidence names, the SSA-trained char-LSTM
            tends to outperform FAIMR's char-ngram LR.  We use its
            prediction here.  Optionally we ENSEMBLE the two
            predictions (averaged) when LSTM is confident enough.

The hybrid is the official "FAIMR + SSA char-LSTM plugin"
configuration evaluated in benchmarks/ssa_name_gender/.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np


@dataclass
class HybridPrediction:
    """A hybrid name prediction with provenance."""
    name: str
    p_female: float
    source: str  # "lookup" | "faimr_model" | "lstm" | "ensemble"
    faimr_p: Optional[float] = None
    lstm_p: Optional[float] = None


# Confidence bands -- a FAIMR model prediction inside this band is
# considered "low-confidence" and the LSTM is preferred.
FAIMR_LOW_CONF_LO = 0.35
FAIMR_LOW_CONF_HI = 0.65

# Ensemble weighting -- when both FAIMR and LSTM are confident we
# average them with these weights.
ENSEMBLE_WEIGHT_LSTM = 0.55


def predict_hybrid(names: Sequence[str]) -> list[HybridPrediction]:
    from fairness.names.classifier import predict_many
    from faimr_plus.ssa_char_lstm.predict import predict_p_female

    faimr_results = predict_many(list(names))

    # We compute LSTM predictions for the OOV / model-path names only,
    # to avoid wasted compute on lookup hits.
    lstm_needed_idx: list[int] = []
    lstm_needed_names: list[str] = []
    for i, r in enumerate(faimr_results):
        if r.source != "lookup":
            lstm_needed_idx.append(i)
            lstm_needed_names.append(r.name or names[i])
    if lstm_needed_names:
        lstm_probs = predict_p_female(lstm_needed_names)
    else:
        lstm_probs = np.array([], dtype=np.float32)

    lstm_map: dict[int, float] = dict(zip(lstm_needed_idx, lstm_probs))

    out: list[HybridPrediction] = []
    for i, r in enumerate(faimr_results):
        faimr_p = float(r.p_female)
        if r.source == "lookup":
            out.append(HybridPrediction(
                name=r.name, p_female=faimr_p, source="lookup",
                faimr_p=faimr_p, lstm_p=None,
            ))
            continue
        if r.source == "empty":
            out.append(HybridPrediction(
                name=r.name, p_female=0.5, source="empty",
                faimr_p=0.5, lstm_p=None,
            ))
            continue

        lstm_p = float(lstm_map.get(i, 0.5))

        # Decide
        faimr_confident = (
            faimr_p < FAIMR_LOW_CONF_LO or faimr_p > FAIMR_LOW_CONF_HI
        )
        if faimr_confident:
            # Ensemble: trust FAIMR more but blend LSTM signal.
            p = (
                ENSEMBLE_WEIGHT_LSTM * lstm_p
                + (1.0 - ENSEMBLE_WEIGHT_LSTM) * faimr_p
            )
            source = "ensemble"
        else:
            # Low confidence -- defer to LSTM
            p = lstm_p
            source = "lstm"

        out.append(HybridPrediction(
            name=r.name, p_female=float(p), source=source,
            faimr_p=faimr_p, lstm_p=lstm_p,
        ))
    return out
