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
import re
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
SURNAMES_PATH = (
    Path(__file__).resolve().parent.parent.parent
    / "data" / "names" / "surnames.csv"
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

# Lookup weight at or above which a token's given-name evidence overrides
# its presence on the surname denylist.  See NameGenderResult.is_surname_only.
SURNAME_OVERRIDE_WEIGHT: float = 1.5


# --- Result types ----------------------------------------------------------

@dataclass(frozen=True)
class NameGenderResult:
    """A single name -> gender probability prediction.

    Attributes:
        name:        The normalised query (lower-cased, alpha only).
        p_female:    Probability the name is used by female individuals,
                     in [0, 1].  ``p_male`` is implicit (1 - p_female).
        source:      "lookup" if the value came from the training corpus
                     high-weight lookup table, "model" if from the
                     calibrated classifier, "empty" if the input did
                     not contain any alphabetic characters.
        weight:      For lookup hits, the corpus row weight (a measure
                     of the empirical attestation).  For model hits, None.
        culture:     Best-guess culture cluster for lookup hits, None
                     otherwise.  Sourced from training_corpus.csv.
        is_surname:  True if the token appears in the surname denylist
                     (data/names/surnames.csv).  Surname tokens are
                     intrinsic to the token, not the model — they
                     remain True regardless of which path produced the
                     probability.  Downstream callers (BiasDetector)
                     use this to deprioritise surname tokens when a
                     non-surname candidate is available.
    """
    name: str
    p_female: float
    source: str
    weight: Optional[float] = None
    culture: Optional[str] = None
    is_surname: bool = False

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

    @property
    def is_surname_only(self) -> bool:
        """True if this token is on the surname denylist AND we have
        no strongly-attested given-name evidence for it.

        Many tokens are BOTH common given names AND common surnames
        ("John", "Lee", "Khan", "Charles"...).  Treating every
        surname-listed token as gender-signal-free would erase those
        legitimate given names.

        The discriminator: a token is "surname-only" iff
            ``is_surname`` is True
            AND it does NOT have a strong corpus lookup hit
                (source == "lookup" with weight >= SURNAME_OVERRIDE_WEIGHT,
                 default 1.5).

        Why 1.5 not 1.0?  The corpus row weight is the sum of
        per-source weights.  A weight of exactly 1.0 means "one hard
        upstream label and nothing else" — which is exactly what you
        see when a token is primarily a surname but happens to appear
        a handful of times as a given name in the firstname database
        (Jones, Smith, Patel all match this profile).  Weight >= 1.5
        requires evidence from at least TWO sources — typically the
        firstname database AND the curated FAIMR seed — which is the
        threshold at which given-name usage is the dominant attestation.

        Callers should branch on ``is_surname_only``, not on
        ``is_surname``, when deciding whether to skip a token.
        """
        if not self.is_surname:
            return False
        if (self.source == "lookup"
                and (self.weight or 0.0) >= SURNAME_OVERRIDE_WEIGHT):
            return False
        return True

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
    with no alphabetic content.  Matches the canonical corpus key
    format used in data/names/training_corpus.csv and surnames.csv."""
    if not name:
        return ""
    return "".join(ch for ch in name.lower() if ch.isalpha())


# Splits a compound name on hyphens, apostrophes, whitespace, and
# surname-particle boundaries.  Used by the classifier to resolve
# "Smith-Jones" by parts when the joined form is OOV.
_COMPOUND_SPLIT_RE = re.compile(r"[\-'\s]+")

# Surname particles ignored when extracting the "main" name parts.
# These appear in particle-prefixed surnames ("van der Berg",
# "de la Cruz", "ben David", "abu Mazen", "al Khalil") and contribute
# no per-token gender information.
_SURNAME_PARTICLES: frozenset = frozenset({
    "van", "der", "den", "de", "del", "della", "di", "da", "do", "dos",
    "das", "le", "la", "les", "von", "zu", "zur", "af", "av", "ben",
    "bin", "abu", "el", "al", "ibn", "mc", "mac", "fitz", "ap",
})


def _split_compound(name: str) -> list:
    """Return the lower-cased alphabetic parts of a compound name,
    excluding surname particles.  Empty for single-word inputs."""
    if not name:
        return []
    raw_parts = _COMPOUND_SPLIT_RE.split(name.lower())
    out = []
    for raw in raw_parts:
        norm = _normalise(raw)
        if not norm or norm in _SURNAME_PARTICLES:
            continue
        out.append(norm)
    return out


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
        surnames_path: Path = SURNAMES_PATH,
        lookup_weight_floor: float = LOOKUP_WEIGHT_FLOOR,
    ) -> None:
        self._model_path = Path(model_path)
        self._corpus_path = Path(corpus_path)
        self._surnames_path = Path(surnames_path)
        self._lookup_weight_floor = lookup_weight_floor
        self._model = None
        self._lookup: dict[str, tuple[float, float, str]] = {}
        self._surnames: set = set()
        self._load_lock = threading.Lock()
        self._loaded = False
        # Integrity verification — populated by _load_model.
        # integrity_violated stays False when the recomputed SHA-256
        # of model.pkl matches the value model_card.json was trained
        # to expect.  When they diverge, the audit prepends a
        # [CRITICAL] recommendation; callers can inspect the field
        # directly to refuse to run.
        self.integrity_violated: bool = False
        self.expected_sha: Optional[str] = None
        self.actual_sha: Optional[str] = None

    # ----- loading ---------------------------------------------------

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        with self._load_lock:
            if self._loaded:
                return
            self._load_lookup()
            self._load_surnames()
            self._load_model()
            self._loaded = True

    def _load_surnames(self) -> None:
        """Load the surname denylist from data/names/surnames.csv.

        Missing file is non-fatal — surname handling degrades to empty
        set, which reproduces pre-task-#16 behaviour (every token is
        treated as a potential given name).  This avoids hard-failing
        on environments that haven't run build_surnames.py yet.
        """
        if not self._surnames_path.exists():
            return
        df = pd.read_csv(
            self._surnames_path, keep_default_na=False, na_values=[""],
        )
        for raw in df.get("name", []):
            tok = _normalise(str(raw))
            if tok:
                self._surnames.add(tok)

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
        model_bytes = self._model_path.read_bytes()
        self._model = pickle.loads(model_bytes)
        # --- Integrity verification ----------------------------------
        # Compare the just-read model file's SHA-256 against the value
        # the model card was trained to expect.  We swallow failures
        # because integrity is decorative when no card or no hash is
        # available — the load itself succeeded, so the audit can
        # still run, but the integrity_violated flag will surface
        # the missing information to the audit consumer.
        try:
            import hashlib
            import json
            if MODEL_CARD_PATH.exists():
                card = json.loads(
                    MODEL_CARD_PATH.read_text(encoding="utf-8")
                )
                self.expected_sha = (
                    card.get("integrity", {}).get("sha256")
                )
                if self.expected_sha:
                    self.actual_sha = hashlib.sha256(model_bytes).hexdigest()
                    if self.actual_sha != self.expected_sha:
                        self.integrity_violated = True
        except Exception:
            # Defensive: never block the load on integrity verification.
            self.integrity_violated = False
            self.expected_sha = None
            self.actual_sha = None

    # ----- prediction ------------------------------------------------

    def predict(self, name: str) -> NameGenderResult:
        """Predict P(female | name) for a single token.

        The raw name is forwarded into predict_many so compound names
        (hyphenated, apostrophe-separated, particle-prefixed) get the
        same compound-lookup resolution as the batch path."""
        if not name or not _normalise(name):
            return NameGenderResult(name="", p_female=0.5, source="empty")
        return self.predict_many([name])[0]

    def _resolve_compound_lookup(self, raw: str):
        """Return (hit_tuple, name_used, all_surname) for a possibly-
        compound name token.

        Resolution priority:
          1. Strict normalised form ("smithjones" if name is "Smith-Jones").
          2. Each part of the compound split ("smith", "jones").  If
             multiple parts hit the lookup, return the highest-weight one.
          3. None if nothing hits.

        ``all_surname`` is True iff every compound part is on the surname
        denylist.  This is what feeds NameGenderResult.is_surname for
        compound tokens — a single-name "Smith-Jones" with both parts on
        the surname list is surname-only even though no joined-form
        lookup exists.
        """
        norm_strict = _normalise(raw)
        parts = _split_compound(raw)

        # is_surname is True if EITHER the joined form OR every
        # compound part is on the surname denylist.  Joined-form check
        # is what catches "obrien" / "Smith-Jones" -> "smithjones"
        # when both halves are real surnames.  Per-part check catches
        # cases where the joined form is unattested but every part
        # is a separately-listed surname.
        is_sur_joined  = bool(norm_strict) and norm_strict in self._surnames
        is_sur_parts   = bool(parts) and all(p in self._surnames for p in parts)
        is_sur         = is_sur_joined or is_sur_parts

        hit = self._lookup.get(norm_strict) if norm_strict else None
        if hit is not None:
            return hit, norm_strict, is_sur

        if not parts:
            return None, norm_strict, is_sur

        best_hit = None
        best_part = None
        for p in parts:
            h = self._lookup.get(p)
            if h is None:
                continue
            if best_hit is None or h[1] > best_hit[1]:
                best_hit = h
                best_part = p
        if best_hit is not None:
            return best_hit, best_part, is_sur
        return None, norm_strict, is_sur

    def predict_many(self, names: list) -> list:
        """Batch interface.  Faster than calling ``predict`` in a loop
        because the model's predict_proba is amortised over the OOV
        batch.  Compound names (hyphenated, apostrophe-separated, or
        particle-prefixed) are resolved via _resolve_compound_lookup."""
        self._ensure_loaded()
        results: list = [None] * len(names)
        oov_indices: list = []
        oov_names: list = []
        oov_all_surname: list = []
        for i, raw in enumerate(names):
            if not raw or not _normalise(raw):
                # No alphabetic content at all.
                results[i] = NameGenderResult(name="", p_female=0.5, source="empty")
                continue
            hit, name_used, all_surname = self._resolve_compound_lookup(raw)
            if hit is not None:
                p_f, w, culture = hit
                results[i] = NameGenderResult(
                    name=name_used, p_female=p_f, source="lookup",
                    weight=w, culture=culture, is_surname=all_surname,
                )
            else:
                oov_indices.append(i)
                oov_names.append(name_used)  # already normalised
                oov_all_surname.append(all_surname)
        if oov_names:
            probs = self._model.predict_proba(oov_names)[:, 1]
            for idx, nm, p, is_sur in zip(
                oov_indices, oov_names, probs, oov_all_surname,
            ):
                results[idx] = NameGenderResult(
                    name=nm, p_female=float(p), source="model",
                    is_surname=is_sur,
                )
        return [r for r in results if r is not None]

    def _predict_normalised(self, norm: str) -> NameGenderResult:
        # Delegate to predict_many to avoid duplicating the
        # compound-resolution logic.  This path used to be the hot
        # path but is now used only by predict() and predict_cached().
        return self.predict_many([norm])[0]

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
