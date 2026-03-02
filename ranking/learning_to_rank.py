"""
ASRIS — Learning-to-Rank Model
Trains an XGBoost/LightGBM ranker on multi-signal features:
SBERT similarity, TF-IDF similarity, skill coverage, section scores,
keyword overlap — learns optimal weights automatically.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (
    LTR_N_ESTIMATORS, LTR_MAX_DEPTH, LTR_LEARNING_RATE,
    LTR_TEST_SIZE, LTR_RANDOM_STATE, PROCESSED_RESUME_DIR,
    RAW_JD_DIR, LABELED_DIR, get_logger,
)
from ranking.ranking_utils import RankingPipeline
from evaluation.metrics import RankingEvaluator

logger = get_logger("ranking.learning_to_rank")


class LearningToRank:
    """
    Learning-to-Rank model that combines multiple ranking signals
    into a single optimal score using gradient-boosted trees.

    Features:
    1. SBERT cosine similarity
    2. TF-IDF cosine similarity
    3. Skill coverage ratio
    4. Number of matched skills
    5. Keyword overlap ratio
    6. Resume length (word count)
    """

    def __init__(self, model_type: str = "xgboost"):
        """
        Args:
            model_type: 'xgboost' or 'lightgbm'
        """
        self.model_type = model_type
        self.model = None
        self.feature_names = [
            "sbert_similarity",
            "tfidf_similarity",
            "skill_coverage",
            "num_matched_skills",
            "num_jd_skills",
            "keyword_overlap_ratio",
            "resume_word_count",
            "jd_word_count",
        ]

    def _extract_features(
        self,
        pipeline: RankingPipeline,
        jd_embeddings: dict,
        resume_sbert_embeddings: dict,
        resume_tfidf_embeddings: dict,
        jd_tfidf_embeddings: dict,
    ) -> tuple[np.ndarray, np.ndarray, list]:
        """
        Extract feature vectors for all pairs.

        Returns:
            X: feature matrix (n_pairs, n_features)
            y: labels (n_pairs,)
            job_ids: list of job IDs for per-query grouping
        """
        import re

        features = []
        labels = []
        job_ids = []

        logger.info("Extracting features for all pairs...")

        for _, row in tqdm(pipeline.pairs.iterrows(), total=len(pipeline.pairs)):
            job_id = row["job_id"]
            resume_file = row["resume_filename"]

            jd_emb = jd_embeddings.get(job_id)
            resume_sbert_emb = resume_sbert_embeddings.get(resume_file)
            jd_tfidf = jd_tfidf_embeddings.get(job_id)
            resume_tfidf = resume_tfidf_embeddings.get(resume_file)

            if jd_emb is None or resume_sbert_emb is None:
                continue

            # Feature 1: SBERT similarity
            sbert_sim = pipeline.embedding_manager.cosine_similarity(
                jd_emb, resume_sbert_emb
            )

            # Feature 2: TF-IDF similarity
            tfidf_sim = 0.0
            if jd_tfidf is not None and resume_tfidf is not None:
                tfidf_sim = pipeline.embedding_manager.cosine_similarity(
                    jd_tfidf, resume_tfidf
                )

            # Feature 3-4: Skill coverage
            jd_skills = pipeline.get_jd_skills(job_id)
            resume_skills = pipeline.get_resume_skills(resume_file)
            matched = jd_skills.intersection(resume_skills) if jd_skills else set()
            coverage = len(matched) / len(jd_skills) if jd_skills else 0

            # Feature 5-6: Keyword overlap
            jd_text = str(pipeline.jd_dict.get(job_id, ""))
            resume_text = pipeline.resume_texts.get(resume_file, "")

            jd_words = set(re.findall(r"\b\w{3,}\b", jd_text.lower()))
            resume_words = set(re.findall(r"\b\w{3,}\b", resume_text.lower()))
            kw_overlap = (len(jd_words & resume_words) / len(jd_words)) if jd_words else 0

            # Feature 7-8: Document lengths
            resume_wc = len(resume_text.split())
            jd_wc = len(jd_text.split())

            features.append([
                sbert_sim,
                tfidf_sim,
                coverage,
                len(matched),
                len(jd_skills),
                kw_overlap,
                resume_wc,
                jd_wc,
            ])
            labels.append(row["label"])
            job_ids.append(job_id)

        return np.array(features), np.array(labels), job_ids

    def train_and_evaluate(
        self,
        pairs_file: str = "skill_based_pairs.csv",
    ) -> dict:
        """
        Train the LTR model and evaluate with proper ranking metrics.
        """
        pipeline = RankingPipeline(pairs_file=pairs_file, name="LTR")

        # Compute all embeddings
        logger.info("Computing SBERT embeddings...")
        jd_sbert = pipeline.encode_jds_sbert()
        resume_sbert = pipeline.encode_resumes_sbert()

        logger.info("Computing TF-IDF embeddings...")
        all_texts = {**pipeline.jd_dict, **pipeline.resume_texts}
        fit_corpus = list(all_texts.values())

        jd_tfidf = pipeline.embedding_manager.encode_tfidf(
            pipeline.jd_dict, fit_corpus=fit_corpus, cache_prefix="tfidf_jds"
        )
        resume_tfidf = pipeline.embedding_manager.encode_tfidf(
            pipeline.resume_texts, fit_corpus=fit_corpus, cache_prefix="tfidf_resumes"
        )

        # Extract features
        X, y, job_ids = self._extract_features(
            pipeline, jd_sbert, resume_sbert, resume_tfidf, jd_tfidf
        )

        logger.info(f"Feature matrix: {X.shape}, Labels: {y.shape}")
        logger.info(f"Label distribution: 0={sum(y == 0)}, 1={sum(y == 1)}")

        # Train/test split (by query for proper ranking evaluation)
        unique_jobs = list(set(job_ids))
        np.random.seed(LTR_RANDOM_STATE)
        np.random.shuffle(unique_jobs)
        split_idx = int(len(unique_jobs) * (1 - LTR_TEST_SIZE))
        train_jobs = set(unique_jobs[:split_idx])
        test_jobs = set(unique_jobs[split_idx:])

        train_mask = [jid in train_jobs for jid in job_ids]
        test_mask = [jid in test_jobs for jid in job_ids]

        X_train, y_train = X[train_mask], y[train_mask]
        X_test, y_test = X[test_mask], y[test_mask]
        test_job_ids = [jid for jid, m in zip(job_ids, test_mask) if m]

        logger.info(f"Train: {len(X_train)} pairs | Test: {len(X_test)} pairs")

        # Train model
        if self.model_type == "xgboost":
            import xgboost as xgb
            self.model = xgb.XGBClassifier(
                n_estimators=LTR_N_ESTIMATORS,
                max_depth=LTR_MAX_DEPTH,
                learning_rate=LTR_LEARNING_RATE,
                random_state=LTR_RANDOM_STATE,
                eval_metric="logloss",
                use_label_encoder=False,
            )
        else:
            import lightgbm as lgb
            self.model = lgb.LGBMClassifier(
                n_estimators=LTR_N_ESTIMATORS,
                max_depth=LTR_MAX_DEPTH,
                learning_rate=LTR_LEARNING_RATE,
                random_state=LTR_RANDOM_STATE,
            )

        logger.info(f"Training {self.model_type} model...")
        self.model.fit(X_train, y_train)

        # Predict
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]

        # Feature importance
        importances = self.model.feature_importances_
        print(f"\n{'═' * 55}")
        print(f"  FEATURE IMPORTANCE")
        print(f"{'═' * 55}")
        for name, imp in sorted(
            zip(self.feature_names, importances), key=lambda x: x[1], reverse=True
        ):
            bar = "█" * int(imp * 50)
            print(f"  {name:<25} {imp:.4f}  {bar}")
        print()

        # Evaluate
        results = pipeline.evaluate(
            y_pred_proba.tolist(),
            y_test.tolist(),
            test_job_ids,
        )

        return results

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Score candidates with the trained model."""
        if self.model is None:
            raise ValueError("Model not trained. Call train_and_evaluate() first.")
        return self.model.predict_proba(features)[:, 1]


if __name__ == "__main__":
    ltr = LearningToRank(model_type="xgboost")
    ltr.train_and_evaluate()
