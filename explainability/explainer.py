"""
ASRIS — Explainability Module
Generates human-readable explanations for resume-JD match decisions:
skill match breakdown, section-level scoring, and keyword heatmaps.
"""

import re
from typing import Optional
from collections import Counter

import numpy as np
import pandas as pd
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import RAW_JD_DIR, PROCESSED_RESUME_DIR, get_logger

logger = get_logger("explainability")


class MatchExplainer:
    """
    Generates detailed explanations for why a resume was ranked
    high or low for a given job description.
    """

    # ── Comprehensive skill vocabulary for text-based extraction ──
    KNOWN_SKILLS = [
        # AI / ML / DL
        "machine learning", "deep learning", "reinforcement learning",
        "natural language processing", "nlp", "computer vision",
        "neural networks", "convolutional neural networks", "cnns",
        "recurrent neural networks", "rnns", "transformers",
        "attention mechanism", "generative ai", "gen ai",
        "large language models", "llm", "llms",
        "prompt engineering", "fine-tuning", "fine tuning",
        "retrieval augmented generation", "rag",
        "transfer learning", "model optimization", "model deployment",
        "feature engineering", "hyperparameter tuning",
        "ensemble methods", "gradient boosting", "xgboost", "lightgbm",
        "random forest", "logistic regression", "linear regression",
        "classification", "regression", "clustering", "dimensionality reduction",
        "anomaly detection", "fraud detection", "recommendation systems",
        "speech recognition", "text to speech", "tts",
        "object detection", "image classification", "semantic segmentation",
        "time series", "forecasting",
        # Frameworks
        "pytorch", "tensorflow", "keras", "scikit-learn", "sklearn",
        "hugging face", "huggingface", "langchain", "llamaindex",
        "spacy", "nltk", "opencv", "fastai",
        "pandas", "numpy", "scipy", "matplotlib", "seaborn",
        "tabnet", "catboost",
        # LLM platforms
        "openai", "gpt", "chatgpt", "gpt-4", "gpt-3",
        "gemini", "claude", "anthropic", "llama", "mistral",
        "groq", "ollama", "hugging face",
        # Cloud & MLOps
        "aws", "azure", "gcp", "google cloud",
        "docker", "kubernetes", "mlflow", "mlops",
        "ci/cd", "git", "github",
        "sagemaker", "vertex ai", "databricks",
        # Data
        "sql", "nosql", "mongodb", "postgresql", "mysql",
        "data analysis", "data engineering", "data pipelines",
        "etl", "data visualization", "tableau", "power bi",
        "apache spark", "hadoop", "kafka", "airflow",
        # Web / Backend
        "python", "java", "javascript", "typescript", "c++", "rust", "go",
        "flask", "fastapi", "django", "react", "node.js",
        "rest api", "graphql", "microservices",
        # Other
        "statistics", "probability", "linear algebra",
        "a/b testing", "experiment design",
        "agile", "scrum", "jira",
        "communication", "leadership", "teamwork",
    ]

    def __init__(self):
        self._skills_loaded = False
        self._jd_skill_map = {}
        self._all_skill_names = []
        self._skill_dict = {}

    def _load_skills(self):
        """Load skill data once."""
        if self._skills_loaded:
            return

        try:
            job_skills = pd.read_csv(RAW_JD_DIR / "jobs" / "job_skills.csv")
            skills_map = pd.read_csv(RAW_JD_DIR / "mappings" / "skills.csv")

            self._skill_dict = dict(zip(skills_map["skill_abr"], skills_map["skill_name"]))
            csv_skills = [s.lower() for s in skills_map["skill_name"].tolist()]
        except Exception:
            csv_skills = []
            logger.warning("Could not load skill CSVs, using built-in vocabulary only")

        # Merge CSV skills with built-in vocabulary (deduplicated)
        all_skills = set(csv_skills) | set(self.KNOWN_SKILLS)
        self._all_skill_names = sorted(all_skills)

        for _, row in (pd.DataFrame() if not self._skill_dict else
                       pd.read_csv(RAW_JD_DIR / "jobs" / "job_skills.csv")).iterrows():
            job_id = row["job_id"]
            skill_abr = row["skill_abr"]
            if skill_abr in self._skill_dict:
                self._jd_skill_map.setdefault(job_id, set()).add(
                    self._skill_dict[skill_abr].lower()
                )

        self._skills_loaded = True

    def _extract_skills_from_text(self, text: str) -> set:
        """Extract skills mentioned in a text using the full vocabulary."""
        text_lower = text.lower()
        found = set()
        for skill in self._all_skill_names:
            # Use word boundary matching for short skills to avoid false positives
            if len(skill) <= 3:
                if re.search(r'\b' + re.escape(skill) + r'\b', text_lower):
                    found.add(skill)
            else:
                if skill in text_lower:
                    found.add(skill)
        return found

    def explain_match(
        self,
        job_id,
        jd_text: str,
        resume_text: str,
        sbert_score: Optional[float] = None,
        overall_score: Optional[float] = None,
    ) -> dict:
        """
        Generate a comprehensive match explanation.

        Returns a dict with:
        - skill_analysis: matched/missing/extra skills
        - keyword_overlap: shared important terms
        - section_scores: per-section relevance (if parse available)
        - overall_verdict: human-readable summary
        """
        self._load_skills()

        result = {
            "job_id": job_id,
            "scores": {},
            "skill_analysis": {},
            "keyword_overlap": {},
            "verdict": "",
        }

        # ── Scores ──
        if sbert_score is not None:
            result["scores"]["sbert_similarity"] = round(sbert_score, 4)
        if overall_score is not None:
            result["scores"]["overall_score"] = round(overall_score, 4)

        # ── Skill Analysis ──
        # Try pre-mapped skills first; fall back to text extraction
        jd_skills = self._jd_skill_map.get(job_id, set())
        if not jd_skills:
            jd_skills = self._extract_skills_from_text(jd_text)

        resume_skills = self._extract_skills_from_text(resume_text)

        matched_skills = jd_skills & resume_skills
        missing_skills = jd_skills - resume_skills
        extra_skills = resume_skills - jd_skills

        coverage = len(matched_skills) / len(jd_skills) if jd_skills else 0

        result["skill_analysis"] = {
            "required_skills": sorted(jd_skills),
            "matched_skills": sorted(matched_skills),
            "missing_skills": sorted(missing_skills),
            "extra_skills": sorted(list(extra_skills)[:20]),
            "coverage": round(coverage, 3),
            "match_ratio": f"{len(matched_skills)}/{len(jd_skills)}",
        }
        result["scores"]["skill_coverage"] = round(coverage, 4)

        # ── Keyword Overlap ──
        jd_words = set(re.findall(r"\b\w{3,}\b", jd_text.lower()))
        resume_words = set(re.findall(r"\b\w{3,}\b", resume_text.lower()))

        stopwords = {
            "the", "and", "for", "are", "with", "you", "this", "that", "have",
            "from", "will", "can", "your", "our", "not", "has", "was", "all",
            "but", "been", "also", "they", "their", "more", "about", "which",
            "when", "would", "make", "like", "than", "other", "into", "its",
            "over", "such", "any", "only", "who", "what", "how", "most",
            "should", "could", "does", "did", "just", "very", "use", "using",
            "work", "working", "based", "well", "need", "must", "able",
            "including", "across", "ensure", "strong", "new", "best",
        }
        jd_words -= stopwords
        resume_words -= stopwords

        shared = jd_words & resume_words
        jd_only = jd_words - resume_words

        result["keyword_overlap"] = {
            "shared_keywords": sorted(list(shared)[:30]),
            "jd_only_keywords": sorted(list(jd_only)[:20]),
            "overlap_ratio": round(len(shared) / len(jd_words), 3) if jd_words else 0,
        }

        # ── Verdict ──
        result["verdict"] = self._generate_verdict(result)

        return result

    def _generate_verdict(self, explanation: dict) -> str:
        """Generate a human-readable match verdict."""
        skill_cov = explanation["skill_analysis"]["coverage"]
        keyword_overlap = explanation["keyword_overlap"]["overlap_ratio"]
        sbert = explanation["scores"].get("sbert_similarity", 0)

        lines = []

        # Combined signal for verdict
        avg_signal = (skill_cov * 0.4 + keyword_overlap * 0.2 + sbert * 0.4)

        if avg_signal >= 0.45 or (skill_cov >= 0.6 and sbert >= 0.35):
            lines.append("✅ STRONG MATCH — This resume is highly relevant to the job.")
        elif avg_signal >= 0.25 or skill_cov >= 0.3 or sbert >= 0.35:
            lines.append("⚠️ MODERATE MATCH — Partial alignment with job requirements.")
        else:
            lines.append("❌ WEAK MATCH — Limited overlap with job requirements.")

        # Skill details
        matched = explanation["skill_analysis"]["matched_skills"]
        missing = explanation["skill_analysis"]["missing_skills"]

        if matched:
            lines.append(f"  ✓ Matched skills: {', '.join(matched[:8])}")
        if missing:
            lines.append(f"  ✗ Missing skills: {', '.join(missing[:8])}")

        lines.append(f"  Skill coverage: {skill_cov:.0%} | "
                     f"Keyword overlap: {keyword_overlap:.0%}")

        if sbert is not None and sbert > 0:
            lines.append(f"  Semantic similarity: {sbert:.2%}")

        return "\n".join(lines)

    def explain_batch(
        self,
        pairs_df: pd.DataFrame,
        jd_dict: dict,
        resume_texts: dict,
        scores: Optional[list[float]] = None,
    ) -> list[dict]:
        """
        Explain a batch of resume-JD pairs.

        Args:
            pairs_df: DataFrame with job_id, resume_filename, label columns
            jd_dict: {job_id: jd_text}
            resume_texts: {resume_filename: text}
            scores: Optional list of scores matching pairs_df rows
        """
        explanations = []
        for idx, row in pairs_df.iterrows():
            job_id = row["job_id"]
            resume_file = row["resume_filename"]

            jd_text = str(jd_dict.get(job_id, ""))
            resume_text = resume_texts.get(resume_file, "")

            score = scores[idx] if scores else None

            explanation = self.explain_match(
                job_id=job_id,
                jd_text=jd_text,
                resume_text=resume_text,
                overall_score=score,
            )
            explanation["true_label"] = row.get("label")
            explanations.append(explanation)

        return explanations

    def format_explanation(self, explanation: dict) -> str:
        """Format a single explanation as a readable string."""
        lines = [
            f"{'─' * 50}",
            f"Job ID: {explanation['job_id']}",
            "",
            explanation["verdict"],
            "",
            f"Skill Analysis:",
            f"  Required: {', '.join(explanation['skill_analysis']['required_skills'][:10])}",
            f"  Matched:  {explanation['skill_analysis']['match_ratio']}",
            f"  Coverage: {explanation['skill_analysis']['coverage']:.0%}",
            "",
            f"Top Shared Keywords: {', '.join(explanation['keyword_overlap']['shared_keywords'][:10])}",
            f"{'─' * 50}",
        ]
        return "\n".join(lines)


if __name__ == "__main__":
    explainer = MatchExplainer()

    # Synthetic demo
    explanation = explainer.explain_match(
        job_id="demo_001",
        jd_text="Looking for a Python developer experienced in machine learning, TensorFlow, and cloud computing. Must have strong SQL skills.",
        resume_text="Senior developer with 5 years in Python, machine learning, and data analysis. Proficient in TensorFlow, PyTorch, and Docker. Experience with PostgreSQL and cloud deployment on AWS.",
        sbert_score=0.72,
    )
    print(explainer.format_explanation(explanation))
