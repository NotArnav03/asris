"""
FAIMR — Ranking Utilities
Shared base utilities for all ranking evaluators to eliminate code duplication.
"""

import os
import re
import pandas as pd
from pathlib import Path
from typing import Optional

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (
    BASE_DIR, PROCESSED_RESUME_DIR, RAW_JD_DIR, LABELED_DIR,
    META_DIR, SBERT_MODEL_NAME, get_logger,
)
from embeddings.embedding_manager import EmbeddingManager
from evaluation.metrics import RankingEvaluator, quick_evaluate

logger = get_logger("ranking.utils")


# --- Shared skill-extraction primitives ----------------------------------
# The same look-around-anchored matcher used in
# explainability/counterfactual.py and explainability/explainer.py.
# Plain `skill in text_lower` substring matching produced two false
# positives that this fixes:
#   - "java" matched inside "javascript"
#   - "c++" never matched because \b can't anchor against `+`
# The fix is structurally identical across modules; centralise it
# here so future skill extractors get it for free.
_SKILL_PATTERN_CACHE: dict = {}


def _skill_pattern(skill: str) -> "re.Pattern":
    """Look-around-anchored matcher for a single skill token."""
    if skill not in _SKILL_PATTERN_CACHE:
        _SKILL_PATTERN_CACHE[skill] = re.compile(
            rf"(?<!\w){re.escape(skill)}(?!\w)"
        )
    return _SKILL_PATTERN_CACHE[skill]


def classify_at_percentile(
    scores: list, percentile: float = 50.0,
) -> tuple:
    """Threshold ``scores`` at the given percentile and return
    (threshold, predictions) where predictions are binary labels.

    Centralised so every baseline uses the SAME semantics:

      threshold = np.percentile(scores, percentile)
      predictions = [1 if s >= threshold else 0 for s in scores]

    The `>=` (not `>`) is intentional — a candidate scoring EXACTLY
    at the percentile cutoff should be SELECTED (it ties the
    boundary case rather than falling on the negative side).  The
    prior baselines used strict `>`, which silently rejected
    boundary candidates and gave slightly different counts than the
    threshold computation implied.

    Returns the threshold value alongside the predictions so
    callers can report what cutoff was actually used.
    """
    if not scores:
        return 0.0, []
    import numpy as np
    threshold = float(np.percentile(scores, percentile))
    predictions = [1 if s >= threshold else 0 for s in scores]
    return threshold, predictions


def extract_skills_in_text(skills: list, text: str) -> set:
    """Return the subset of ``skills`` that appear in ``text``.

    Skills are matched with look-around-anchored boundaries so
    "java" inside "javascript" does NOT match, and tokens with
    trailing non-word characters ("c++", "c#") do match correctly.
    Cache-keyed by skill string so each token compiles once.
    """
    text_lower = text.lower()
    return {s for s in skills if _skill_pattern(s).search(text_lower)}


class RankingPipeline:
    """
    Shared pipeline for loading data, computing scores, and evaluating.
    All ranking scripts inherit from this to avoid code duplication.
    """

    def __init__(
        self,
        pairs_file: str,
        name: str = "Ranker",
        model_name: str = SBERT_MODEL_NAME,
    ):
        self.name = name
        self.model_name = model_name
        self.embedding_manager = EmbeddingManager(model_name=model_name)

        # Load labeled pairs
        logger.info(f"[{name}] Loading pairs: {pairs_file}")
        self.pairs = pd.read_csv(LABELED_DIR / pairs_file)
        logger.info(f"[{name}] Loaded {len(self.pairs)} pairs")

        # Load JD data
        self.jds = pd.read_csv(RAW_JD_DIR / "postings_balanced.csv")
        self.jd_dict = dict(zip(self.jds["job_id"], self.jds["description"].astype(str)))

        # Load resume texts
        self.resume_texts = self._load_resume_texts()

        # Skill data (loaded lazily)
        self._skills_loaded = False
        self._jd_skill_map = {}
        self._resume_skill_map = {}
        self._all_skill_names = []

    def _load_resume_texts(self) -> dict[str, str]:
        """Load all cleaned resume texts from disk."""
        resume_texts = {}
        for file in PROCESSED_RESUME_DIR.glob("*.txt"):
            with open(file, "r", encoding="utf-8") as f:
                resume_texts[file.name] = f.read()
        logger.info(f"[{self.name}] Loaded {len(resume_texts)} resume texts")
        return resume_texts

    def load_skills(self):
        """Load skill mappings and compute resume skill maps."""
        if self._skills_loaded:
            return

        job_skills = pd.read_csv(RAW_JD_DIR / "jobs" / "job_skills.csv")
        skills_map = pd.read_csv(RAW_JD_DIR / "mappings" / "skills.csv")

        skill_dict = dict(zip(skills_map["skill_abr"], skills_map["skill_name"]))

        # JD → skills
        for _, row in job_skills.iterrows():
            job_id = row["job_id"]
            skill_abr = row["skill_abr"]
            if skill_abr in skill_dict:
                self._jd_skill_map.setdefault(job_id, set()).add(
                    skill_dict[skill_abr].lower()
                )

        # Resume -> skills.  Uses the look-around-anchored matcher so
        # "java" inside "javascript" does NOT match (this was a real
        # false-positive bug in the prior substring-based code).
        self._all_skill_names = [s.lower() for s in skills_map["skill_name"].tolist()]
        for filename, text in self.resume_texts.items():
            self._resume_skill_map[filename] = extract_skills_in_text(
                self._all_skill_names, text,
            )

        self._skills_loaded = True
        logger.info(f"[{self.name}] Loaded skills for {len(self._jd_skill_map)} JDs, "
                     f"{len(self._resume_skill_map)} resumes")

    def get_jd_skills(self, job_id) -> set:
        self.load_skills()
        return self._jd_skill_map.get(job_id, set())

    def get_resume_skills(self, resume_file: str) -> set:
        self.load_skills()
        return self._resume_skill_map.get(resume_file, set())

    def skill_coverage(self, job_id, resume_file: str) -> float:
        """Compute skill coverage: |JD_skills ∩ resume_skills| / |JD_skills|."""
        jd_skills = self.get_jd_skills(job_id)
        if not jd_skills:
            return 0.0
        resume_skills = self.get_resume_skills(resume_file)
        return len(jd_skills.intersection(resume_skills)) / len(jd_skills)

    def encode_jds_sbert(self) -> dict:
        """Encode all JDs with SBERT (cached)."""
        return self.embedding_manager.encode_sbert(
            self.jd_dict, cache_prefix="sbert_jds"
        )

    def encode_resumes_sbert(self) -> dict:
        """Encode all resumes with SBERT (cached)."""
        return self.embedding_manager.encode_sbert(
            self.resume_texts, cache_prefix="sbert_resumes"
        )

    def evaluate(
        self,
        scores: list[float],
        labels: list[int],
        job_ids: Optional[list] = None,
        threshold_percentile: float = 0.5,
    ) -> dict:
        """
        Run full evaluation: flat metrics + per-query metrics.

        Args:
            scores: Predicted relevance scores
            labels: Ground truth labels (0/1)
            job_ids: Optional query IDs for per-query evaluation
            threshold_percentile: Percentile for binary classification threshold
        """
        import numpy as np
        from sklearn.metrics import classification_report

        # Flat metrics
        flat_results = quick_evaluate(labels, scores)

        # Classification report with threshold.  Uses >= so candidates
        # whose score equals the percentile cutoff are SELECTED (the
        # boundary case for a percentile threshold is to include the
        # boundary candidate, not exclude them).
        threshold = np.percentile(scores, threshold_percentile * 100)
        predictions = [1 if s >= threshold else 0 for s in scores]

        print(f"\n{'═' * 55}")
        print(f"  {self.name} — EVALUATION RESULTS")
        print(f"{'═' * 55}")

        print(f"\n  Classification Report (threshold={threshold:.4f}):")
        print(classification_report(labels, predictions))

        print(f"  Ranking Metrics:")
        for metric, value in flat_results.items():
            print(f"    {metric}: {value}")

        # Per-query evaluation
        if job_ids is not None:
            evaluator = RankingEvaluator()
            query_data = {}
            for jid, score, label in zip(job_ids, scores, labels):
                query_data.setdefault(str(jid), {"scores": [], "labels": []})
                query_data[str(jid)]["scores"].append(score)
                query_data[str(jid)]["labels"].append(label)

            for qid, data in query_data.items():
                evaluator.add_query(qid, data["labels"], data["scores"])

            query_results = evaluator.compute_all()
            evaluator.print_report(query_results)
            return {
                "flat": flat_results,
                "per_query": query_results,
                "threshold": threshold,
            }

        return {
            "flat": flat_results,
            "threshold": threshold,
        }
