"""
FAIMR — Per-culture calibrated classifier wrapper.

The runtime pickle (fairness/names/model.pkl) is an instance of
``CulturalCalibratedClassifier`` defined here.  This module is
intentionally lightweight — it imports only what's needed for
inference so the audit runtime doesn't pull in
sklearn.model_selection / metrics / etc.

The training script (fairness/names/train_classifier.py) constructs
the instance and pickles it; the runtime classifier
(fairness/names/classifier.py) unpickles and calls predict_proba.
Pickle records the qualified class path, so the class lives here
permanently — moving it would break loading of existing model.pkl.

## Design rationale

CalibratedClassifierCV (sklearn) fits ONE isotonic curve on the full
training distribution.  Per-culture ECE measurements show this is
not uniformly correct:

  arab            ECE 0.089
  south_asian     ECE 0.080
  western         ECE 0.066
  east_asian      ECE 0.060
  slavic          ECE 0.084
  european_other  ECE 0.024

The Arab and Slavic clusters have ~3x the European miscalibration.
Fitting one isotonic PER cluster lets each cluster's residual error
be corrected independently.  At inference time we need to know which
cluster an OOV name belongs to — that's what the culture_lr field
does (a multi-class char-ngram classifier trained on the same
TF-IDF features).
"""

from __future__ import annotations

import numpy as np


class CulturalCalibratedClassifier:
    """Per-culture isotonic calibration of a base gender LR.

    Inference flow:
      1. Featurise via shared TF-IDF vectorizer.
      2. Get raw P(female) from gender_lr.
      3. Predict culture cluster from culture_lr.
      4. Apply per_culture_calibrators[predicted_culture] if present;
         else fall back to global_calibrator.
      5. **(Optional second stage, added 2026-05)** Apply
         ssa_recalibrators[predicted_culture] if present.  Fit
         on the SSA national name dataset to correct the residual
         miscalibration that the upstream-firstname-DB-trained
         per-culture isotonic carries when generalising to
         English-dominant rare US names.  Only configured for the
         culture clusters where SSA is the natural distribution
         (typically western / european_other / slavic).  See
         ``fairness.names.fit_ssa_recalibrator``.

    Returns the same (N, 2) array shape as sklearn's predict_proba
    so existing downstream code (``predict_proba(...)[:, 1]``) is
    unaffected.

    Backwards compatibility: ``ssa_recalibrators`` is optional and
    defaults to None.  Existing pickled models load with no behavior
    change; the field is created lazily by ``__setstate__`` if
    missing.
    """

    def __init__(
        self,
        vectorizer,
        gender_lr,
        culture_lr,
        global_calibrator,
        per_culture_calibrators: dict,
        ssa_recalibrators: dict | None = None,
    ) -> None:
        self.vectorizer = vectorizer
        self.gender_lr = gender_lr
        self.culture_lr = culture_lr
        self.global_calibrator = global_calibrator
        self.per_culture_calibrators = per_culture_calibrators
        self.ssa_recalibrators = ssa_recalibrators or {}

    def __setstate__(self, state):
        # Older pickled models do not have ssa_recalibrators.  Treat
        # missing attribute as empty dict so loading older artefacts
        # keeps working.
        if "ssa_recalibrators" not in state:
            state["ssa_recalibrators"] = {}
        self.__dict__.update(state)

    def predict_proba(self, names) -> np.ndarray:
        X = self.vectorizer.transform(list(names))
        raw = self.gender_lr.predict_proba(X)[:, 1]
        cultures = self.culture_lr.predict(X)
        p_female = np.empty(len(cultures), dtype=float)
        ssa_recals = getattr(self, "ssa_recalibrators", {}) or {}
        for i, (r, c) in enumerate(zip(raw, cultures)):
            cal = self.per_culture_calibrators.get(c, self.global_calibrator)
            v = float(cal.predict([r])[0])
            ssa = ssa_recals.get(c)
            if ssa is not None:
                v = float(ssa.predict([v])[0])
            p_female[i] = v
        return np.column_stack([1.0 - p_female, p_female])

    def predict_culture(self, names) -> np.ndarray:
        """Expose the culture classifier's prediction directly so
        downstream consumers can also route audit logic by culture."""
        X = self.vectorizer.transform(list(names))
        return self.culture_lr.predict(X)
