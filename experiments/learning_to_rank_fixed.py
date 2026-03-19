"""
FAIMR — Fixed Learning-to-Rank Evaluation Pipeline
Master script that runs the complete corrected experimental suite:
- 5-Fold GroupKFold cross-validation
- Optimal-threshold F1 for baselines
- 5 ablation configurations
- Fairness evaluation (AIR before/after FCR)
- Counterfactual explainer statistics

Run from project root:
    python -m experiments.learning_to_rank_fixed
"""

import re
import sys
import time
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
from sklearn.model_selection import GroupKFold
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    precision_recall_curve
)
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (
    LTR_N_ESTIMATORS, LTR_MAX_DEPTH, LTR_LEARNING_RATE,
    LTR_RANDOM_STATE, get_logger,
)
from ranking.ranking_utils import RankingPipeline
from evaluation.metrics import ndcg_at_k, precision_at_k
from experiments.fcr import fcr_rerank, compute_air
from experiments.counterfactual_explainer import evaluate_counterfactuals

logger = get_logger("experiments.ltr_fixed")

FEATURE_NAMES = [
    "sbert_similarity",
    "tfidf_similarity",
    "skill_coverage",
    "num_matched_skills",
    "keyword_overlap_ratio",
]

ABLATION_CONFIGS = {
    "SBERT-only": [0],
    "TF-IDF-only": [1],
    "SBERT+TF-IDF": [0, 1],
    "Full-LTR": [0, 1, 2, 3, 4],
    "FAIMR-Full": [0, 1, 2, 3, 4],
}


def extract_features(pipeline, jd_sbert, resume_sbert, jd_tfidf, resume_tfidf):
    features = []
    labels = []
    job_ids = []

    logger.info("Extracting features for all pairs...")
    for _, row in tqdm(pipeline.pairs.iterrows(), total=len(pipeline.pairs)):
        job_id = row["job_id"]
        resume_file = row["resume_filename"]

        jd_emb = jd_sbert.get(job_id)
        resume_sbert_emb = resume_sbert.get(resume_file)
        jd_tf = jd_tfidf.get(job_id)
        resume_tf = resume_tfidf.get(resume_file)

        if jd_emb is None or resume_sbert_emb is None:
            continue

        sbert_sim = pipeline.embedding_manager.cosine_similarity(jd_emb, resume_sbert_emb)

        tfidf_sim = 0.0
        if jd_tf is not None and resume_tf is not None:
            tfidf_sim = pipeline.embedding_manager.cosine_similarity(jd_tf, resume_tf)

        jd_skills = pipeline.get_jd_skills(job_id)
        resume_skills = pipeline.get_resume_skills(resume_file)
        matched = jd_skills.intersection(resume_skills) if jd_skills else set()
        coverage = len(matched) / len(jd_skills) if jd_skills else 0

        jd_text = str(pipeline.jd_dict.get(job_id, ""))
        resume_text = pipeline.resume_texts.get(resume_file, "")
        jd_words = set(re.findall(r"\b\w{3,}\b", jd_text.lower()))
        resume_words = set(re.findall(r"\b\w{3,}\b", resume_text.lower()))
        kw_overlap = len(jd_words & resume_words) / len(jd_words) if jd_words else 0

        features.append([sbert_sim, tfidf_sim, coverage, len(matched), kw_overlap])
        labels.append(row["label"])
        job_ids.append(job_id)

    return np.array(features), np.array(labels), job_ids


def optimal_f1_threshold(y_true, y_scores):
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
    best_idx = np.argmax(f1_scores)
    return thresholds[best_idx] if best_idx < len(thresholds) else 0.5


def infer_gender(resume_filenames):
    try:
        import gender_guesser.detector as gender
        d = gender.Detector()
    except ImportError:
        logger.warning("gender_guesser not installed, using random proxy demographics")
        rng = np.random.RandomState(42)
        return [rng.choice(["male", "female"], p=[0.6, 0.4]) for _ in resume_filenames]

    genders = []
    for fname in resume_filenames:
        name_part = fname.split("_")[-1].replace(".txt", "")
        guess = d.get_gender(name_part.capitalize())
        if guess in ("male", "mostly_male"):
            genders.append("male")
        elif guess in ("female", "mostly_female"):
            genders.append("female")
        else:
            genders.append(np.random.choice(["male", "female"], p=[0.55, 0.45]))
    return genders


def evaluate_fold(X_train, y_train, X_test, y_test, feature_indices, config_name):
    import xgboost as xgb

    X_tr = X_train[:, feature_indices]
    X_te = X_test[:, feature_indices]

    if len(feature_indices) <= 2:
        scores = X_te.mean(axis=1) if X_te.shape[1] > 1 else X_te[:, 0]
        threshold = optimal_f1_threshold(y_train, X_tr.mean(axis=1) if X_tr.shape[1] > 1 else X_tr[:, 0])
        y_pred_bin = (scores >= threshold).astype(int)

        return {
            "accuracy": accuracy_score(y_test, y_pred_bin),
            "f1": f1_score(y_test, y_pred_bin, zero_division=0),
            "auc": roc_auc_score(y_test, scores) if len(set(y_test)) > 1 else 0.0,
            "ndcg@5": ndcg_at_k(list(y_test), list(scores), 5),
            "p@5": precision_at_k(list(y_test), list(scores), 5),
            "threshold": threshold,
            "scores": scores,
        }

    model = xgb.XGBClassifier(
        n_estimators=LTR_N_ESTIMATORS,
        max_depth=LTR_MAX_DEPTH,
        learning_rate=LTR_LEARNING_RATE,
        random_state=LTR_RANDOM_STATE,
        eval_metric="logloss",
        use_label_encoder=False,
    )
    model.fit(X_tr, y_train)
    scores = model.predict_proba(X_te)[:, 1]
    y_pred_bin = model.predict(X_te)

    return {
        "accuracy": accuracy_score(y_test, y_pred_bin),
        "f1": f1_score(y_test, y_pred_bin, zero_division=0),
        "auc": roc_auc_score(y_test, scores) if len(set(y_test)) > 1 else 0.0,
        "ndcg@5": ndcg_at_k(list(y_test), list(scores), 5),
        "p@5": precision_at_k(list(y_test), list(scores), 5),
        "threshold": 0.5,
        "scores": scores,
        "model": model,
    }


def run_pipeline():
    print("=" * 70)
    print("  FAIMR — Corrected Experimental Pipeline")
    print("=" * 70)

    pipeline = RankingPipeline(pairs_file="domain_match_pairs.csv", name="FAIMR-Fixed")

    logger.info("Computing embeddings...")
    jd_sbert = pipeline.encode_jds_sbert()
    resume_sbert = pipeline.encode_resumes_sbert()
    all_texts = {**pipeline.jd_dict, **pipeline.resume_texts}
    fit_corpus = list(all_texts.values())
    jd_tfidf = pipeline.embedding_manager.encode_tfidf(
        pipeline.jd_dict, fit_corpus=fit_corpus, cache_prefix="tfidf_jds_v2"
    )
    resume_tfidf = pipeline.embedding_manager.encode_tfidf(
        pipeline.resume_texts, fit_corpus=fit_corpus, cache_prefix="tfidf_resumes_v2"
    )

    X, y, job_ids = extract_features(pipeline, jd_sbert, resume_sbert, jd_tfidf, resume_tfidf)
    logger.info(f"Feature matrix: {X.shape}, Labels: {y.shape}")
    logger.info(f"Label distribution: 0={sum(y == 0)}, 1={sum(y == 1)}")

    gkf = GroupKFold(n_splits=5)
    job_id_array = np.array(job_ids)

    all_results = {}

    for config_name, feature_indices in ABLATION_CONFIGS.items():
        logger.info(f"Running {config_name}...")
        fold_metrics = defaultdict(list)

        for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups=job_id_array)):
            X_train, y_train = X[train_idx], y[train_idx]
            X_test, y_test = X[test_idx], y[test_idx]

            result = evaluate_fold(X_train, y_train, X_test, y_test, feature_indices, config_name)

            for metric in ["accuracy", "f1", "auc", "ndcg@5", "p@5"]:
                fold_metrics[metric].append(result[metric])

            if config_name == "FAIMR-Full" and fold == 0:
                scores = result["scores"]
                test_resumes = [pipeline.pairs.iloc[i]["resume_filename"] for i in test_idx]
                test_groups = infer_gender(test_resumes)
                k_select = len(test_idx) // 2

                top_k_groups = [test_groups[np.argsort(scores)[::-1][i]] for i in range(k_select)]
                all_groups_list = test_groups
                air_before, _ = compute_air(None, top_k_groups, None, all_groups_list)

                fcr_result = fcr_rerank(scores, test_groups, k_select, threshold=0.8)

                fold_metrics["air_before"].append(fcr_result["air_before"])
                fold_metrics["air_after"].append(fcr_result["air_after"])
                fold_metrics["mean_displacement"].append(fcr_result["mean_displacement"])

                if "model" in result:
                    cf_pred = result["scores"]
                    cf_stats = evaluate_counterfactuals(
                        result["model"], X[test_idx], y[test_idx], cf_pred,
                        [job_ids[i] for i in test_idx], FEATURE_NAMES,
                        pipeline, threshold=0.5, max_samples=100
                    )
                    fold_metrics["cf_mean_e_star"].append(cf_stats.get("mean_e_star", 0))
                    fold_metrics["cf_latency_ms"].append(cf_stats.get("mean_latency_ms", 0))
                    fold_metrics["cf_pct_optimal"].append(cf_stats.get("pct_greedy_optimal", 0))

        config_result = {}
        for metric, values in fold_metrics.items():
            config_result[f"{metric}_mean"] = round(np.mean(values), 4)
            config_result[f"{metric}_std"] = round(np.std(values, ddof=1), 4) if len(values) > 1 else 0.0

        all_results[config_name] = config_result

    print("\n" + "=" * 70)
    print("  ABLATION STUDY RESULTS (5-Fold GroupKFold CV)")
    print("=" * 70)

    header = f"{'Config':<20} {'Accuracy':>16} {'F1':>16} {'AUC':>16} {'NDCG@5':>16} {'P@5':>16}"
    print(header)
    print("-" * len(header))

    for config_name, metrics in all_results.items():
        row = f"{config_name:<20}"
        for m in ["accuracy", "f1", "auc", "ndcg@5", "p@5"]:
            mean = metrics.get(f"{m}_mean", 0)
            std = metrics.get(f"{m}_std", 0)
            row += f" {mean:.4f}±{std:.4f}"
        print(row)

    faimr = all_results.get("FAIMR-Full", {})
    if "air_before_mean" in faimr:
        print(f"\n{'=' * 70}")
        print("  FAIRNESS EVALUATION (FAIMR-Full)")
        print(f"{'=' * 70}")
        print(f"  AIR before FCR:     {faimr.get('air_before_mean', 'N/A')}")
        print(f"  AIR after FCR:      {faimr.get('air_after_mean', 'N/A')}")
        print(f"  Mean displacement:  {faimr.get('mean_displacement_mean', 'N/A')}")

    if "cf_mean_e_star_mean" in faimr:
        print(f"\n{'=' * 70}")
        print("  COUNTERFACTUAL EXPLAINER STATISTICS")
        print(f"{'=' * 70}")
        print(f"  Mean |E*|:          {faimr.get('cf_mean_e_star_mean', 'N/A')}")
        print(f"  Mean latency (ms):  {faimr.get('cf_latency_ms_mean', 'N/A')}")
        print(f"  % greedy==optimal:  {faimr.get('cf_pct_optimal_mean', 'N/A')}")

    print(f"\n{'=' * 70}")
    print("  LATEX TABLE")
    print(f"{'=' * 70}")
    print("\\begin{table}[ht]")
    print("\\centering")
    print("\\caption{Ablation study results (5-fold GroupKFold CV, domain-match labels)}")
    print("\\begin{tabular}{lccccc}")
    print("\\toprule")
    print("Model & Accuracy & F1 & AUC & NDCG@5 & P@5 \\\\")
    print("\\midrule")
    for config_name, metrics in all_results.items():
        cells = [config_name.replace("_", "\\_")]
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
    run_pipeline()
