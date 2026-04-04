"""
FAIMR — Baseline Comparison with Established Retrieval Methods
Implements 5 literature baselines for resume-JD matching:
1. BM25 (Okapi BM25) — Robertson & Zaragoza 2009
2. Doc2Vec — Le & Mikolov 2014
3. Jaccard Skill Matching — Rule-based skill overlap
4. MPNet-SBERT (all-mpnet-base-v2) — Reimers & Gurevych 2019
5. BERT Cross-Encoder — Nogueira & Cho 2019

Run: python experiments/baseline_comparison.py
"""

import os
import re
import sys
import pickle
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_recall_curve
from tqdm import tqdm

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import get_logger, PROCESSED_RESUME_DIR, RAW_JD_DIR, LABELED_DIR
from ranking.ranking_utils import RankingPipeline
from evaluation.metrics import ndcg_at_k, precision_at_k

logger = get_logger("experiments.baselines")


def per_query_ndcg_p(y_true, scores, job_ids, k=5):
    query_groups = defaultdict(lambda: {"y": [], "s": []})
    for yt, sc, jid in zip(y_true, scores, job_ids):
        query_groups[jid]["y"].append(yt)
        query_groups[jid]["s"].append(sc)
    ndcg_vals, p_vals = [], []
    for qid, data in query_groups.items():
        if len(data["y"]) < 2 or sum(data["y"]) == 0 or sum(data["y"]) == len(data["y"]):
            continue
        ndcg_vals.append(ndcg_at_k(data["y"], data["s"], k))
        p_vals.append(precision_at_k(data["y"], data["s"], k))
    return np.mean(ndcg_vals) if ndcg_vals else 0.0, np.mean(p_vals) if p_vals else 0.0


def optimal_f1_threshold(y_true, y_scores):
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    f1s = 2 * (precision * recall) / (precision + recall + 1e-8)
    best_idx = np.argmax(f1s)
    return thresholds[best_idx] if best_idx < len(thresholds) else 0.5


def evaluate_scores(y_true, scores, job_ids, y_train=None, train_scores=None):
    if y_train is not None and train_scores is not None:
        thresh = optimal_f1_threshold(y_train, train_scores)
    else:
        thresh = optimal_f1_threshold(y_true, scores)
    y_pred = (scores >= thresh).astype(int)
    ndcg_q, p_q = per_query_ndcg_p(y_true, scores, job_ids, k=5)
    try:
        auc = roc_auc_score(y_true, scores)
    except ValueError:
        auc = 0.0
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "auc": auc,
        "ndcg@5": ndcg_q,
        "p@5": p_q,
    }


# ========================================================================
# BASELINE 1: BM25
# ========================================================================
def compute_bm25_scores(pipeline):
    logger.info("Computing BM25 scores...")
    try:
        from rank_bm25 import BM25Okapi
    except ImportError:
        logger.error("rank_bm25 not installed. Run: pip install rank-bm25")
        return None

    all_resume_names = list(pipeline.resume_texts.keys())
    tokenized_resumes = [pipeline.resume_texts[r].lower().split() for r in all_resume_names]
    bm25 = BM25Okapi(tokenized_resumes)
    resume_idx = {name: i for i, name in enumerate(all_resume_names)}

    # Cache: compute BM25 scores once per unique JD (not per pair)
    unique_jds = pipeline.pairs["job_id"].unique()
    logger.info(f"Scoring {len(unique_jds)} unique JDs against {len(all_resume_names)} resumes...")
    jd_score_cache = {}
    for jid in tqdm(unique_jds, desc="BM25 (per JD)"):
        jd_text = str(pipeline.jd_dict.get(jid, ""))
        jd_tokens = jd_text.lower().split()
        doc_scores = bm25.get_scores(jd_tokens)
        max_score = doc_scores.max() if doc_scores.max() > 0 else 1.0
        jd_score_cache[jid] = doc_scores / max_score

    # Look up cached scores for each pair
    scores = []
    for _, row in pipeline.pairs.iterrows():
        ridx = resume_idx[row["resume_filename"]]
        scores.append(jd_score_cache[row["job_id"]][ridx])

    return np.array(scores)


# ========================================================================
# BASELINE 2: Doc2Vec
# ========================================================================
def compute_doc2vec_scores(pipeline):
    logger.info("Computing Doc2Vec scores...")
    try:
        from gensim.models.doc2vec import Doc2Vec, TaggedDocument
    except ImportError:
        logger.error("gensim not installed. Run: pip install gensim")
        return None

    all_docs = []
    jd_tags = {}
    resume_tags = {}

    for jid, text in pipeline.jd_dict.items():
        tag = f"jd_{jid}"
        all_docs.append(TaggedDocument(str(text).lower().split(), [tag]))
        jd_tags[jid] = tag

    for rfile, text in pipeline.resume_texts.items():
        tag = f"res_{rfile}"
        all_docs.append(TaggedDocument(text.lower().split(), [tag]))
        resume_tags[rfile] = tag

    model = Doc2Vec(all_docs, vector_size=100, window=5, min_count=2,
                    workers=4, epochs=20, seed=42)

    scores = []
    for _, row in tqdm(pipeline.pairs.iterrows(), total=len(pipeline.pairs), desc="Doc2Vec"):
        jd_vec = model.dv[jd_tags[row["job_id"]]]
        res_vec = model.dv[resume_tags[row["resume_filename"]]]
        cos_sim = np.dot(jd_vec, res_vec) / (np.linalg.norm(jd_vec) * np.linalg.norm(res_vec) + 1e-8)
        scores.append((cos_sim + 1) / 2)  # Normalize to [0, 1]

    return np.array(scores)


# ========================================================================
# BASELINE 3: Jaccard Skill Matching
# ========================================================================
def compute_jaccard_scores(pipeline):
    logger.info("Computing Jaccard Skill Matching scores...")
    pipeline.load_skills()
    scores = []
    for _, row in tqdm(pipeline.pairs.iterrows(), total=len(pipeline.pairs), desc="Jaccard"):
        jd_skills = pipeline.get_jd_skills(row["job_id"])
        res_skills = pipeline.get_resume_skills(row["resume_filename"])
        union = jd_skills | res_skills
        if len(union) == 0:
            scores.append(0.0)
        else:
            scores.append(len(jd_skills & res_skills) / len(union))
    return np.array(scores)


# ========================================================================
# BASELINE 4: MPNet-SBERT (all-mpnet-base-v2)
# ========================================================================
def compute_mpnet_scores(pipeline):
    logger.info("Computing MPNet-SBERT (all-mpnet-base-v2) scores...")
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        logger.error("sentence-transformers not installed.")
        return None

    model = SentenceTransformer("all-mpnet-base-v2")

    jd_list = list(set(pipeline.pairs["job_id"].tolist()))
    jd_texts_list = [str(pipeline.jd_dict.get(jid, ""))[:512] for jid in jd_list]
    jd_embs = model.encode(jd_texts_list, show_progress_bar=True, batch_size=64)
    jd_emb_map = dict(zip(jd_list, jd_embs))

    res_list = list(set(pipeline.pairs["resume_filename"].tolist()))
    res_texts_list = [pipeline.resume_texts.get(r, "")[:512] for r in res_list]
    res_embs = model.encode(res_texts_list, show_progress_bar=True, batch_size=64)
    res_emb_map = dict(zip(res_list, res_embs))

    scores = []
    for _, row in tqdm(pipeline.pairs.iterrows(), total=len(pipeline.pairs), desc="MPNet"):
        jd_emb = jd_emb_map[row["job_id"]]
        res_emb = res_emb_map[row["resume_filename"]]
        cos_sim = np.dot(jd_emb, res_emb) / (np.linalg.norm(jd_emb) * np.linalg.norm(res_emb) + 1e-8)
        scores.append((cos_sim + 1) / 2)

    return np.array(scores)


# ========================================================================
# BASELINE 5: BERT Cross-Encoder
# ========================================================================
def compute_crossencoder_scores(pipeline):
    logger.info("Computing BERT Cross-Encoder scores...")
    try:
        from sentence_transformers import CrossEncoder
    except ImportError:
        logger.error("sentence-transformers not installed.")
        return None

    model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", max_length=512)

    pairs_text = []
    for _, row in pipeline.pairs.iterrows():
        jd_text = str(pipeline.jd_dict.get(row["job_id"], ""))[:256]
        res_text = pipeline.resume_texts.get(row["resume_filename"], "")[:256]
        pairs_text.append([jd_text, res_text])

    logger.info(f"Scoring {len(pairs_text)} pairs with Cross-Encoder...")
    raw_scores = model.predict(pairs_text, show_progress_bar=True, batch_size=64)

    # Normalize to [0, 1]
    min_s, max_s = raw_scores.min(), raw_scores.max()
    if max_s > min_s:
        scores = (raw_scores - min_s) / (max_s - min_s)
    else:
        scores = np.full_like(raw_scores, 0.5)

    return scores


def run_baselines():
    print("=" * 70)
    print("  FAIMR — Baseline Comparison with Established Methods")
    print("=" * 70)

    pipeline = RankingPipeline(pairs_file="domain_match_pairs.csv", name="Baselines")

    labels = pipeline.pairs["label"].values
    job_ids = pipeline.pairs["job_id"].values

    baselines = {
        "BM25":              compute_bm25_scores,
        "Doc2Vec":           compute_doc2vec_scores,
        "Jaccard Skill":     compute_jaccard_scores,
        "MPNet-SBERT":       compute_mpnet_scores,
        "BERT Cross-Enc":    compute_crossencoder_scores,
    }

    gkf = GroupKFold(n_splits=5)
    all_results = {}

    cache_dir = Path(__file__).resolve().parent.parent / "data" / "baseline_cache"
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
    except (FileExistsError, OSError):
        pass

    for name, score_fn in baselines.items():
        print(f"\n--- {name} ---")
        cache_file = cache_dir / f"{name.replace(' ', '_').lower()}_scores.pkl"

        if cache_file.exists():
            logger.info(f"Loading cached scores from {cache_file.name}")
            with open(cache_file, "rb") as f:
                scores = pickle.load(f)
        else:
            scores = score_fn(pipeline)
            if scores is None:
                print(f"  SKIPPED (missing dependency)")
                continue
            with open(cache_file, "wb") as f:
                pickle.dump(scores, f)
            logger.info(f"Cached scores to {cache_file.name}")

        fold_metrics = defaultdict(list)
        for fold, (train_idx, test_idx) in enumerate(gkf.split(scores, labels, groups=job_ids)):
            result = evaluate_scores(
                labels[test_idx], scores[test_idx],
                [job_ids[i] for i in test_idx],
                y_train=labels[train_idx],
                train_scores=scores[train_idx],
            )
            for metric in ["accuracy", "f1", "auc", "ndcg@5", "p@5"]:
                fold_metrics[metric].append(result[metric])

        config_result = {}
        for metric, values in fold_metrics.items():
            config_result[f"{metric}_mean"] = round(np.mean(values), 4)
            config_result[f"{metric}_std"] = round(np.std(values, ddof=1), 4)
        all_results[name] = config_result

    print(f"\n{'=' * 70}")
    print("  BASELINE COMPARISON RESULTS (5-Fold GroupKFold CV)")
    print(f"{'=' * 70}")
    header = f"{'Method':<20} {'Accuracy':>16} {'F1':>16} {'AUC':>16} {'NDCG@5':>16} {'P@5':>16}"
    print(header)
    print("-" * len(header))
    for name, metrics in all_results.items():
        row = f"{name:<20}"
        for m in ["accuracy", "f1", "auc", "ndcg@5", "p@5"]:
            mean = metrics.get(f"{m}_mean", 0)
            std = metrics.get(f"{m}_std", 0)
            row += f" {mean:.4f}±{std:.4f}"
        print(row)

    print(f"\n{'=' * 70}")
    print("  LATEX TABLE")
    print(f"{'=' * 70}")
    print("\\begin{table}[t]")
    print("\\centering")
    print("\\caption{Comparison with established retrieval methods}")
    print("\\label{tab:baselines}")
    print("\\begin{tabular}{lccccc}")
    print("\\toprule")
    print("Method & Accuracy & F1 & AUC & NDCG@5 & P@5 \\\\")
    print("\\midrule")
    for name, metrics in all_results.items():
        cells = [name]
        for m in ["accuracy", "f1", "auc", "ndcg@5", "p@5"]:
            mean = metrics.get(f"{m}_mean", 0)
            std = metrics.get(f"{m}_std", 0)
            cells.append(f"${mean:.4f} \\pm {std:.4f}$")
        print(" & ".join(cells) + " \\\\")
    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{table}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    run_baselines()
