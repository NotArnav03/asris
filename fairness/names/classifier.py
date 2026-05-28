"""
FAIMR — Calibrated name -> gender probability inference.

This module is the *only* public entry point for the name-based gender
proxy.  Everything else in the audit pipeline must go through
``NameGenderClassifier`` so that:

  1. The model file is loaded exactly once per process.
  2. Predictions for the same name within a process are cached.
  3. The lookup-fast-path and the model-fallback path are applied
     consistently across the codebase.

## Why a hybrid lookup + model design?

The trained char n-gram LR generalises well on unseen names but, by
construction, smooths over substring-level patterns.  For *short* and
*ambiguous* names — `wei`, `lee`, `kim`, `taylor` — the n-gram majority
vote can override the per-name empirical label.  Concretely: "wei" is
labelled p_female=0.50 in the corpus (unisex), but the model emits
~0.98 because its substring n-grams overlap heavily with strongly-female
names ending in "-ei".

A hybrid solves this without sacrificing OOV recall:

  * **In-corpus, high-confidence lookup.**  If the lower-cased name is
    in the training corpus AND its row weight is >= LOOKUP_WEIGHT_FLOOR,
    return the empirical p_female from the corpus directly.  This is
    the right answer by construction: those rows are precisely the
    ones the model would be asked to refute.

  * **Model fallback.**  For names absent from the corpus (or with
    weight below the floor — i.e. very weakly attested), invoke the
    trained classifier.  ECE on the holdout is 0.012, so the
    probability is well-calibrated for the OOV regime.

Source of every prediction is surfaced in the result (``"lookup"`` vs
``"model"``) so audit reports can disclose the provenance.

## Backward compatibility

The legacy ``BiasDetector`` dict-of-sets API exposed only three buckets
(male / female / unknown).  The new probabilistic API exposes the full
probability.  ``hard_label(threshold=0.85)`` collapses the probability
back to the old categorical form for code paths that still need it.
"""

from __future__ import annotations

import pickle
import threading
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Optional

import pandas as pd


# --- Public configuration --------------------------------------------------

MODEL_PATH = Path(__file__).resolve().parent / "model.pkl"
CORPUS_PATH = (
    Path(__file__).resolve().parent.parent.parent
    / "data" / "names" / "training_corpus.csv"
)
MODEL_CARD_PATH = Path(__file__).resolve().parent / "model_card.json"

# A corpus row is only used as a lookup fast-path if its training weight
# is at least this floor.  We default to 0.5 — the minimum weight that
# any corpus row carries — because the entire point of the fast-path is
# that *any* per-name empirical evidence in the corpus is, by
# construction, a better predictor for that exact token than the
# substring-generalising n-gram model.  Raising the floor only helps
# if a particular weak-evidence row is *misleading*, which the
# multi-source merge in build_corpus.py already mitigates.
LOOKUP_WEIGHT_FLOOR: float = 0.5

# Default hard-decision threshold for hard_label().  At 0.85 confidence
# the false-positive rate against the holdout is < 5% in every culture
# cluster except 'arab' (where it is ~9%; see model_card.json).
DEFAULT_HARD_THRESHOLD: float = 0.85


# --- Result types ----------------------------------------------------------

@dataclass(frozen=True)
class NameGenderResult:
    """A single name -> gender probability prediction.

    Attributes:
        name:      The normalised query (lower-cased, alpha only).
        p_female:  Probability the name is used by female individuals,
                   in [0, 1].  ``p_male`` is implicit (1 - p_female).
        source:    "lookup" if the value came from the training corpus
                   high-weight lookup table, "model" if from the
                   calibrated classifier, "empty" if the input did not
                   contain any alphabetic characters.
        weight:    For lookup hits, the corpus row weight (a measure
                   of the empirical attestation).  For model hits,
                   None.
        culture:   Best-guess culture cluster for lookup hits, None
                   otherwise.  Sourced from training_corpus.csv.
    """
    name: str
    p_female: float
    source: str
    weight: Optional[float] = None
    culture: Optional[str] = None

    @property
    def p_male(self) -> float:
        return 1.0 - self.p_female

    @property
    def confidence(self) -> float:
        """Distance from the unisex centre, scaled to [0, 1].

        confidence = |p_female - 0.5| * 2.  A confidence of 1.0 means
        the prediction is fully on one side (p=0 or p=1); 0.0 means
        the prediction is exactly unisex.
        """
        return abs(self.p_female - 0.5) * 2.0

    def hard_label(self, threshold: float = DEFAULT_HARD_THRESHOLD) -> str:
        """Collapse to the legacy three-bucket categorical form.

        Returns 'female' if p_female >= threshold,
                'male'   if p_female <= 1 - threshold,
                'unknown' otherwise.
        """
        if self.p_female >= threshold:
            return "female"
        if self.p_female <= 1.0 - threshold:
            return "male"
        return "unknown"


# --- Helpers ---------------------------------------------------------------

def _normalise(name: str) -> str:
    """Lower-case and strip non-letter chars.  Returns '' for inputs
    with no alphabetic content."""
    if not name:
        return ""
    return "".join(ch for ch in name.lower() if ch.isalpha())


# --- The classifier --------------------------------------------------------

class NameGenderClassifier:
    """Process-wide singleton wrapper around the trained model + corpus
    lookup table.

    Use :func:`get_classifier` to obtain the shared instance.  Direct
    instantiation is supported for tests that need an isolated copy.
    """

    def __init__(
        self,
        model_path: Path = MODEL_PATH,
        corpus_path: Path = CORPUS_PATH,
        lookup_weight_floor: float = LOOKUP_WEIGHT_FLOOR,
    ) -> None:
        self._model_path = Path(model_path)
        self._corpus_path = Path(corpus_path)
        self._lookup_weight_floor = lookup_weight_floor
        self._model = None
        self._lookup: dict[str, tuple[float, float, str]] = {}
        self._load_lock = threading.Lock()
        self._loaded = False

    # ----- loading ---------------------------------------------------

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        with self._load_lock:
            if self._loaded:
                return
            self._load_lookup()
            self._load_model()
            self._loaded = True

    def _load_lookup(self) -> None:
        if not self._corpus_path.exists():
            raise FileNotFoundError(
                f"Training corpus not found at {self._corpus_path}.  "
                f"Run data/names/build_corpus.py to regenerate it."
            )
        df = pd.read_csv(
            self._corpus_path, keep_default_na=False, na_values=[""],
        )
        # Only retain rows whose weight clears the lookup floor; the
        # rest will fall through to the model.
        for _, row in df.iterrows():
            name = _normalise(str(row["name"]))
            if not name:
                continue
            weight = float(row["weight"])
            if weight < self._lookup_weight_floor:
                continue
            self._lookup[name] = (
                float(row["p_female"]),
                weight,
                str(row["culture"]),
            )

    def _load_model(self) -> None:
        if not self._model_path.exists():
            raise FileNotFoundError(
                f"Trained model not found at {self._model_path}.  "
                f"Run fairness/names/train_classifier.py to regenerate it."
            )
        with self._model_path.open("rb") as fh:
            self._model = pickle.load(fh)

    # ----- prediction ------------------------------------------------

    def predict(self, name: str) -> NameGenderResult:
        """Predict P(female | name) for a single token."""
        norm = _normalise(name)
        if not norm:
            return NameGenderResult(name="", p_female=0.5, source="empty")
        return self._predict_normalised(norm)

    def predict_many(self, names: list[str]) -> list[NameGenderResult]:
        """Batch interface.  Faster than calling ``predict`` in a loop
        because the model's predict_proba is amortised over the OOV
        batch."""
        self._ensure_loaded()
        results: list[Optional[NameGenderResult]] = [None] * len(names)
        oov_indices: list[int] = []
        oov_names: list[str] = []
        for i, raw in enumerate(names):
            norm = _normalise(raw)
            if not norm:
                results[i] = NameGenderResult(name="", p_female=0.5, source="empty")
                continue
            hit = self._lookup.get(norm)
            if hit is not None:
                p_f, w, culture = hit
                results[i] = NameGenderResult(
                    name=norm, p_female=p_f, source="lookup",
                    weight=w, culture=culture,
                )
            else:
                oov_indices.append(i)
                oov_names.append(norm)
        if oov_names:
            probs = self._model.predict_proba(oov_names)[:, 1]
            for idx, norm, p in zip(oov_indices, oov_names, probs):
                results[idx] = NameGenderResult(
                    name=norm, p_female=float(p), source="model",
                )
        # mypy: every slot was filled above.
        return [r for r in results if r is not None]

    def _predict_normalised(self, norm: str) -> NameGenderResult:
        self._ensure_loaded()
        hit = self._lookup.get(norm)
        if hit is not None:
            p_f, w, culture = hit
            return NameGenderResult(
                name=norm, p_female=p_f, source="lookup",
                weight=w, culture=culture,
            )
        # OOV — fall back to the model.
        prob = self._model.predict_proba([norm])[0, 1]
        return NameGenderResult(
            name=norm, p_female=float(prob), source="model",
        )

    # ----- diagnostics -----------------------------------------------

    def lookup_size(self) -> int:
        """Number of names in the lookup table (after the weight floor)."""
        self._ensure_loaded()
        return len(self._lookup)


# --- Singleton + caching ---------------------------------------------------

_singleton_lock = threading.Lock()
_singleton: Optional[NameGenderClassifier] = None


def get_classifier() -> NameGenderClassifier:
    """Return the process-wide classifier instance (load on first call)."""
    global _singleton
    if _singleton is not None:
        return _singleton
    with _singleton_lock:
        if _singleton is None:
            _singleton = NameGenderClassifier()
    return _singleton


@lru_cache(maxsize=4096)
def predict_cached(name: str) -> NameGenderResult:
    """Cached single-name predictor.

    The cache is keyed on the *raw* input (not the normalised form)
    so that callers paying the normalisation cost twice see the same
    cache hit.  Cache size is bounded to limit memory in long-running
    processes (e.g. the API server).
    """
    return get_classifier().predict(name)


def predict(name: str) -> NameGenderResult:
    """Convenience: predict on a single name via the cached path."""
    return predict_cached(name)


def predict_many(names: list[str]) -> list[NameGenderResult]:
    """Convenience: batch predict via the shared classifier."""
    return get_classifier().predict_many(names)
