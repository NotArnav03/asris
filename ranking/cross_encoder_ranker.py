"""
ASRIS — Cross-Encoder Re-Ranker
Uses a cross-encoder model to re-rank top-K candidates from the
bi-encoder for higher precision on the final shortlist.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (
    CROSS_ENCODER_MODEL_NAME, CROSS_ENCODER_TOP_K,
    CROSS_ENCODER_BATCH_SIZE, PROCESSED_RESUME_DIR, RAW_JD_DIR,
    LABELED_DIR, get_logger,
)
from ranking.ranking_utils import RankingPipeline

logger = get_logger("ranking.cross_encoder")


class CrossEncoderRanker:
    """
    Two-stage ranking pipeline:
    1. Bi-encoder (SBERT) retrieves top-K candidates
    2. Cross-encoder re-ranks those candidates for higher precision

    Cross-encoders jointly encode the (JD, resume) pair and are more
    accurate than bi-encoders but much slower — hence the two-stage approach.
    """

    def __init__(
        self,
        cross_encoder_model: str = CROSS_ENCODER_MODEL_NAME,
        top_k: int = CROSS_ENCODER_TOP_K,
        batch_size: int = CROSS_ENCODER_BATCH_SIZE,
    ):
        self.cross_encoder_model_name = cross_encoder_model
        self.top_k = top_k
        self.batch_size = batch_size
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder
            logger.info(f"Loading cross-encoder: {self.cross_encoder_model_name}")
            self._model = CrossEncoder(self.cross_encoder_model_name)
        return self._model

    def rerank(
        self,
        jd_text: str,
        candidates: list[tuple[str, float]],
        resume_texts: dict[str, str],
    ) -> list[tuple[str, float]]:
        """
        Re-rank top-K candidates using the cross-encoder.

        Args:
            jd_text: Job description text
            candidates: List of (resume_filename, bi_encoder_score) sorted descending
            resume_texts: {filename: text}

        Returns:
            Re-ranked list of (resume_filename, cross_encoder_score)
        """
        top_candidates = candidates[:self.top_k]

        pairs = []
        filenames = []
        for filename, _ in top_candidates:
            resume_text = resume_texts.get(filename, "")
            if resume_text:
                pairs.append([jd_text, resume_text])
                filenames.append(filename)

        if not pairs:
            return candidates

        # Cross-encoder scoring
        ce_scores = self.model.predict(
            pairs,
            batch_size=self.batch_size,
            show_progress_bar=False,
        )

        # Build re-ranked list
        reranked = list(zip(filenames, ce_scores.tolist()))
        reranked.sort(key=lambda x: x[1], reverse=True)

        # Append remaining candidates (not re-ranked)
        reranked_set = set(filenames)
        for filename, score in candidates[self.top_k:]:
            if filename not in reranked_set:
                reranked.append((filename, score))

        return reranked

    def evaluate(
        self,
        pairs_file: str = "skill_based_pairs.csv",
    ):
        """
        Full cross-encoder evaluation pipeline.
        Stage 1: SBERT bi-encoder retrieves candidates
        Stage 2: Cross-encoder re-ranks top-K
        """
        pipeline = RankingPipeline(pairs_file=pairs_file, name="CrossEncoder")

        # Stage 1: Bi-encoder embeddings
        logger.info("Stage 1: Computing bi-encoder scores...")
        jd_embeddings = pipeline.encode_jds_sbert()
        resume_embeddings = pipeline.encode_resumes_sbert()

        # Group pairs by job_id
        from collections import defaultdict
        job_groups = defaultdict(list)
        for _, row in pipeline.pairs.iterrows():
            job_groups[row["job_id"]].append({
                "resume_filename": row["resume_filename"],
                "label": row["label"],
            })

        all_scores = []
        all_labels = []
        all_job_ids = []

        logger.info(f"Stage 2: Cross-encoder re-ranking top-{self.top_k} per JD...")

        for job_id, group in tqdm(job_groups.items()):
            jd_emb = jd_embeddings.get(job_id)
            jd_text = str(pipeline.jd_dict.get(job_id, ""))

            if jd_emb is None or not jd_text:
                continue

            # Bi-encoder scores for all candidates
            bi_scores = []
            for item in group:
                resume_emb = resume_embeddings.get(item["resume_filename"])
                if resume_emb is not None:
                    score = pipeline.embedding_manager.cosine_similarity(jd_emb, resume_emb)
                    bi_scores.append((item["resume_filename"], score, item["label"]))

            if not bi_scores:
                continue

            # Sort by bi-encoder score
            bi_scores.sort(key=lambda x: x[1], reverse=True)

            candidates = [(f, s) for f, s, _ in bi_scores]
            label_map = {f: l for f, _, l in bi_scores}

            # Re-rank with cross-encoder
            reranked = self.rerank(jd_text, candidates, pipeline.resume_texts)

            for filename, score in reranked:
                if filename in label_map:
                    all_scores.append(score)
                    all_labels.append(label_map[filename])
                    all_job_ids.append(job_id)

        # Evaluate
        pipeline.evaluate(all_scores, all_labels, all_job_ids)


if __name__ == "__main__":
    ranker = CrossEncoderRanker()
    ranker.evaluate()
