"""
ASRIS — Embedding Manager
Centralized embedding generation with disk caching for SBERT and TF-IDF.
Avoids re-encoding on every run.
"""

import hashlib
import json
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (
    SBERT_MODEL_NAME, TFIDF_MAX_FEATURES, TFIDF_STOP_WORDS,
    EMBEDDING_CACHE_DIR, get_logger
)

logger = get_logger("embeddings.manager")


class EmbeddingManager:
    """
    Manages SBERT and TF-IDF embeddings with automatic disk caching.

    Embeddings are cached based on a hash of the input text, so changes
    in the source data automatically invalidate the cache.
    """

    def __init__(
        self,
        model_name: str = SBERT_MODEL_NAME,
        cache_dir: Optional[Path] = None,
        device: Optional[str] = None,
    ):
        self.model_name = model_name
        self.cache_dir = cache_dir or EMBEDDING_CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self._sbert_model = None
        self._tfidf_vectorizer = None
        self.device = device

    # ─── Lazy Loading ────────────────────────────────────────────

    @property
    def sbert_model(self):
        if self._sbert_model is None:
            from sentence_transformers import SentenceTransformer
            logger.info(f"Loading SBERT model: {self.model_name}")
            self._sbert_model = SentenceTransformer(
                self.model_name, device=self.device
            )
        return self._sbert_model

    @property
    def tfidf_vectorizer(self):
        if self._tfidf_vectorizer is None:
            from sklearn.feature_extraction.text import TfidfVectorizer
            self._tfidf_vectorizer = TfidfVectorizer(
                stop_words=TFIDF_STOP_WORDS,
                max_features=TFIDF_MAX_FEATURES,
            )
        return self._tfidf_vectorizer

    # ─── Cache Utilities ─────────────────────────────────────────

    def _cache_key(self, texts: dict[str, str], prefix: str) -> str:
        """Generate a deterministic cache key from input texts."""
        content_hash = hashlib.md5(
            json.dumps(sorted(texts.items()), ensure_ascii=False).encode()
        ).hexdigest()[:12]
        model_tag = self.model_name.replace("/", "_").replace("-", "_")
        return f"{prefix}_{model_tag}_{content_hash}"

    def _cache_path(self, cache_key: str) -> Path:
        return self.cache_dir / f"{cache_key}.pkl"

    def _load_cache(self, cache_key: str) -> Optional[dict]:
        path = self._cache_path(cache_key)
        if path.exists():
            logger.info(f"Loading cached embeddings: {path.name}")
            with open(path, "rb") as f:
                return pickle.load(f)
        return None

    def _save_cache(self, cache_key: str, data: dict):
        path = self._cache_path(cache_key)
        with open(path, "wb") as f:
            pickle.dump(data, f)
        logger.info(f"Saved embeddings cache: {path.name} ({len(data)} entries)")

    # ─── SBERT Embeddings ────────────────────────────────────────

    def encode_sbert(
        self,
        texts: dict[str, str],
        batch_size: int = 64,
        use_cache: bool = True,
        cache_prefix: str = "sbert",
    ) -> dict[str, np.ndarray]:
        """
        Generate SBERT embeddings for a dict of {id: text}.
        Results are cached to disk.

        Returns:
            dict mapping id -> embedding vector (np.ndarray)
        """
        cache_key = self._cache_key(texts, cache_prefix)

        if use_cache:
            cached = self._load_cache(cache_key)
            if cached is not None:
                return cached

        logger.info(f"Encoding {len(texts)} texts with SBERT ({self.model_name})...")

        ids = list(texts.keys())
        text_list = [str(texts[k]) for k in ids]

        # Batch encode for efficiency
        embeddings = self.sbert_model.encode(
            text_list,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
        )

        result = {ids[i]: embeddings[i] for i in range(len(ids))}

        if use_cache:
            self._save_cache(cache_key, result)

        return result

    # ─── TF-IDF Embeddings ───────────────────────────────────────

    def encode_tfidf(
        self,
        texts: dict[str, str],
        fit_corpus: Optional[list[str]] = None,
        use_cache: bool = True,
        cache_prefix: str = "tfidf",
    ) -> dict[str, np.ndarray]:
        """
        Generate TF-IDF vectors for a dict of {id: text}.

        Args:
            texts: Dict of {id: text} to encode
            fit_corpus: Optional larger corpus to fit the vectorizer on.
                        If None, fits on the texts themselves.
            use_cache: Whether to use disk cache

        Returns:
            dict mapping id -> sparse TF-IDF vector
        """
        cache_key = self._cache_key(texts, cache_prefix)

        if use_cache:
            cached = self._load_cache(cache_key)
            if cached is not None:
                return cached

        logger.info(f"Encoding {len(texts)} texts with TF-IDF...")

        corpus = fit_corpus if fit_corpus else list(texts.values())
        self.tfidf_vectorizer.fit(corpus)

        ids = list(texts.keys())
        text_list = [str(texts[k]) for k in ids]
        matrix = self.tfidf_vectorizer.transform(text_list)

        result = {ids[i]: matrix[i] for i in range(len(ids))}

        if use_cache:
            self._save_cache(cache_key, result)

        return result

    # ─── Similarity Computation ──────────────────────────────────

    @staticmethod
    def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        from sklearn.metrics.pairwise import cosine_similarity as sklearn_cos_sim

        if hasattr(vec_a, "toarray"):
            vec_a = vec_a.toarray()
        if hasattr(vec_b, "toarray"):
            vec_b = vec_b.toarray()

        vec_a = np.array(vec_a).reshape(1, -1)
        vec_b = np.array(vec_b).reshape(1, -1)

        return float(sklearn_cos_sim(vec_a, vec_b)[0][0])

    def clear_cache(self):
        """Remove all cached embeddings."""
        count = 0
        for path in self.cache_dir.glob("*.pkl"):
            path.unlink()
            count += 1
        logger.info(f"Cleared {count} cached embedding files")

    def cache_stats(self) -> dict:
        """Get cache statistics."""
        files = list(self.cache_dir.glob("*.pkl"))
        total_size = sum(f.stat().st_size for f in files)
        return {
            "num_files": len(files),
            "total_size_mb": round(total_size / (1024 * 1024), 2),
            "files": [f.name for f in files],
        }


# ─── Convenience Functions ───────────────────────────────────────

_default_manager = None

def get_manager(**kwargs) -> EmbeddingManager:
    """Get or create a default EmbeddingManager instance."""
    global _default_manager
    if _default_manager is None:
        _default_manager = EmbeddingManager(**kwargs)
    return _default_manager


if __name__ == "__main__":
    # Demo
    manager = EmbeddingManager()

    sample_texts = {
        "resume_1": "Experienced Python developer with 5 years in machine learning and data science.",
        "resume_2": "Marketing manager with expertise in digital advertising and brand strategy.",
        "jd_1": "Looking for a senior ML engineer proficient in Python, TensorFlow, and cloud deployment.",
    }

    print("=== SBERT Embeddings ===")
    sbert_embeddings = manager.encode_sbert(sample_texts, use_cache=False)
    for key, emb in sbert_embeddings.items():
        print(f"  {key}: shape={emb.shape}")

    print("\n=== Similarities ===")
    sim_r1_jd = manager.cosine_similarity(sbert_embeddings["resume_1"], sbert_embeddings["jd_1"])
    sim_r2_jd = manager.cosine_similarity(sbert_embeddings["resume_2"], sbert_embeddings["jd_1"])
    print(f"  resume_1 ↔ jd_1 (ML dev vs ML job): {sim_r1_jd:.4f}")
    print(f"  resume_2 ↔ jd_1 (Marketing vs ML job): {sim_r2_jd:.4f}")

    print(f"\n=== Cache Stats ===")
    print(f"  {manager.cache_stats()}")
