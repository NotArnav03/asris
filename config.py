"""
ASRIS — Centralized Configuration
All paths, model parameters, and hyperparameters in one place.
"""

import yaml
from pathlib import Path

# ─── Project Root ────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent

# ─── Data Paths ──────────────────────────────────────────────────
DATA_DIR = BASE_DIR / "data"
RAW_RESUME_DIR = DATA_DIR / "raw" / "resumes"
RAW_JD_DIR = DATA_DIR / "raw" / "job_descriptions"
PROCESSED_RESUME_DIR = DATA_DIR / "processed" / "resumes_cleaned"
LABELED_DIR = DATA_DIR / "labeled"
META_DIR = DATA_DIR / "metadata"
EMBEDDING_CACHE_DIR = DATA_DIR / "embeddings_cache"
EXPERIMENT_DIR = BASE_DIR / "experiments" / "results"

# Ensure dirs exist
for d in [PROCESSED_RESUME_DIR, LABELED_DIR, META_DIR, EMBEDDING_CACHE_DIR, EXPERIMENT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ─── Model Configuration ────────────────────────────────────────
SBERT_MODEL_NAME = "all-MiniLM-L6-v2"
CROSS_ENCODER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
SPACY_MODEL = "en_core_web_sm"

# ─── TF-IDF Parameters ──────────────────────────────────────────
TFIDF_MAX_FEATURES = 5000
TFIDF_STOP_WORDS = "english"

# ─── Ranking Hyperparameters ────────────────────────────────────
HYBRID_ALPHA = 0.7         # Weight for SBERT semantic score
HYBRID_BETA = 0.3          # Weight for skill coverage score
CLASSIFICATION_THRESHOLD_PERCENTILE = 0.5  # Median threshold

# ─── Pair Generation ────────────────────────────────────────────
POSITIVE_PAIRS_PER_JD = 10
NEGATIVE_PAIRS_PER_JD = 10
SKILL_POSITIVE_K = 5
SKILL_NEGATIVE_K = 5

# ─── Balancer ────────────────────────────────────────────────────
SAMPLES_PER_DOMAIN = 110

# ─── Evaluation ──────────────────────────────────────────────────
EVAL_TOP_K_VALUES = [1, 3, 5, 10]

# ─── Cross-Encoder ───────────────────────────────────────────────
CROSS_ENCODER_TOP_K = 20       # Re-rank top K candidates
CROSS_ENCODER_BATCH_SIZE = 32

# ─── Learning-to-Rank ────────────────────────────────────────────
LTR_N_ESTIMATORS = 200
LTR_MAX_DEPTH = 6
LTR_LEARNING_RATE = 0.1
LTR_TEST_SIZE = 0.2
LTR_RANDOM_STATE = 42

# ─── Fairness ────────────────────────────────────────────────────
FAIRNESS_ADVERSE_IMPACT_THRESHOLD = 0.8  # 4/5 rule

# ─── API ─────────────────────────────────────────────────────────
API_HOST = "0.0.0.0"
API_PORT = 8000

# ─── Resume Section Keywords ────────────────────────────────────
SECTION_HEADERS = {
    "education": [
        "education", "academic", "qualification", "university",
        "degree", "school", "college", "coursework"
    ],
    "experience": [
        "experience", "employment", "work history", "professional experience",
        "career", "internship", "job history"
    ],
    "skills": [
        "skills", "technical skills", "competencies", "proficiency",
        "technologies", "tools", "programming", "expertise"
    ],
    "projects": [
        "projects", "portfolio", "academic projects", "personal projects",
        "key projects"
    ],
    "certifications": [
        "certifications", "certificates", "licenses", "credentials",
        "professional development", "training"
    ],
    "summary": [
        "summary", "objective", "profile", "about me", "introduction",
        "professional summary", "career objective"
    ],
    "achievements": [
        "achievements", "awards", "honors", "accomplishments",
        "recognition"
    ],
}

# ─── Logging ─────────────────────────────────────────────────────
import logging

def get_logger(name: str) -> logging.Logger:
    """Create a configured logger instance."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(
            "%(asctime)s │ %(name)-24s │ %(levelname)-7s │ %(message)s",
            datefmt="%H:%M:%S"
        ))
        logger.addHandler(handler)
    return logger


# ─── Config Override from YAML ───────────────────────────────────
_config_path = BASE_DIR / "config.yaml"
if _config_path.exists():
    with open(_config_path, "r") as f:
        _overrides = yaml.safe_load(f) or {}

    # Apply overrides dynamically
    _globals = globals()
    for key, value in _overrides.items():
        upper_key = key.upper()
        if upper_key in _globals:
            _globals[upper_key] = value
