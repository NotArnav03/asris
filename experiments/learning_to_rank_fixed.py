"""
FAIMR — Fixed Learning-to-Rank Evaluation Pipeline (v3)
- 5-Fold GroupKFold cross-validation
- XGBoost for ALL configs (including single-feature baselines)
- 5 additional baseline classifiers (LR, RF, SVM, kNN, NB)
- Per-query NDCG@5 and P@5
- Fairness evaluation (AIR before/after FCR)
- Counterfactual explainer statistics

Run: python experiments/learning_to_rank_fixed.py
"""

import re
import sys
import time
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
from sklearn.model_selection import GroupKFold
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    precision_recall_curve
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

warnings.filterwarnings("ignore")

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

# --- Ablation configs (feature subsets, all use XGBoost) ---
ABLATION_CONFIGS = {
    "SBERT-only": [0],
    "TF-IDF-only": [1],
    "SBERT+TF-IDF": [0, 1],
    "Full-LTR": [0, 1, 2, 3, 4],
    "FAIMR-Full": [0, 1, 2, 3, 4],
}

# --- Baseline models (all use full 5 features, different classifiers) ---
BASELINE_MODELS = {
    "Logistic Regression":  lambda: LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest":        lambda: RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42),
    "SVM (RBF)":            lambda: SVC(probability=True, kernel="rbf", random_state=42),
    "k-NN (k=5)":           lambda: KNeighborsClassifier(n_neighbors=5),
    "Naive Bayes":          lambda: GaussianNB(),
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


def per_query_ndcg_p(y_true, scores, job_ids, k=5):
    query_groups = defaultdict(lambda: {"y": [], "s": []})
    for yt, sc, jid in zip(y_true, scores, job_ids):
        query_groups[jid]["y"].append(yt)
        query_groups[jid]["s"].append(sc)

    ndcg_vals = []
    p_vals = []
    for qid, data in query_groups.items():
        if len(data["y"]) < 2:
            continue
        if sum(data["y"]) == 0 or sum(data["y"]) == len(data["y"]):
            continue
        ndcg_vals.append(ndcg_at_k(data["y"], data["s"], k))
        p_vals.append(precision_at_k(data["y"], data["s"], k))

    return (
        np.mean(ndcg_vals) if ndcg_vals else 0.0,
        np.mean(p_vals) if p_vals else 0.0,
    )


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


def evaluate_fold_xgb(X_train, y_train, X_test, y_test, test_job_ids, feature_indices):
    import xgboost as xgb

    X_tr = X_train[:, feature_indices]
    X_te = X_test[:, feature_indices]

    model = xgb.XGBClassifier(
        n_estimators=LTR_N_ESTIMATORS,
        max_depth=LTR_MAX_DEPTH,
        learning_rate=LTR_LEARNING_RATE,
        random_state=LTR_RANDOM_STATE,
        eval_metric="logloss",
        verbosity=0,
    )
    model.fit(X_tr, y_train)
    scores = model.predict_proba(X_te)[:, 1]

    opt_threshold = optimal_f1_threshold(y_train, model.predict_proba(X_tr)[:, 1])
    y_pred_bin = (scores >= opt_threshold).astype(int)

    ndcg_q, p_q = per_query_ndcg_p(y_test, scores, test_job_ids, k=5)

    try:
        auc = roc_auc_score(y_test, scores)
    except ValueError:
        auc = 0.0

    return {
        "accuracy": accuracy_score(y_test, y_pred_bin),
        "f1": f1_score(y_test, y_pred_bin, zero_division=0),
        "auc": auc,
        "ndcg@5": ndcg_q,
        "p@5": p_q,
        "scores": scores,
        "model": model,
    }


def evaluate_fold_baseline(X_train, y_train, X_test, y_test, test_job_ids, model_factory):
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train)
    X_te = scaler.transform(X_test)

    model = model_factory()
    model.fit(X_tr, y_train)

    if hasattr(model, "predict_proba"):
        scores = model.predict_proba(X_te)[:, 1]
    else:
        scores = model.decision_function(X_te)

    opt_threshold = optimal_f1_threshold(y_train,
        model.predict_proba(X_tr)[:, 1] if hasattr(model, "predict_proba")
        else model.decision_function(X_tr))
    y_pred_bin = (scores >= opt_threshold).astype(int)

    ndcg_q, p_q = per_query_ndcg_p(y_test, scores, test_job_ids, k=5)

    try:
        auc = roc_auc_score(y_test, scores)
    except ValueError:
        auc = 0.0

    return {
        "accuracy": accuracy_score(y_test, y_pred_bin),
        "f1": f1_score(y_test, y_pred_bin, zero_division=0),
        "auc": auc,
        "ndcg@5": ndcg_q,
        "p@5": p_q,
        "scores": scores,
    }


def run_pipeline():
    print("=" * 70)
    print("  FAIMR — Corrected Experimental Pipeline (v3)")
    print("  Now with 5 baseline classifiers + ablation study")
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

    # ========== PART 1: BASELINE MODEL COMPARISON ==========
    print(f"\n{'=' * 70}")
    print("  PART 1: BASELINE MODEL COMPARISON (all use 5 features)")
    print(f"{'=' * 70}")

    baseline_results = {}
    for model_name, model_factory in BASELINE_MODELS.items():
        logger.info(f"Running 5-fold CV: {model_name}...")
        fold_metrics = defaultdict(list)

        for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups=job_id_array)):
            X_train, y_train = X[train_idx], y[train_idx]
            X_test, y_test = X[test_idx], y[test_idx]
            test_jids = [job_ids[i] for i in test_idx]

            result = evaluate_fold_baseline(X_train, y_train, X_test, y_test,
                                            test_jids, model_factory)

            for metric in ["accuracy", "f1", "auc", "ndcg@5", "p@5"]:
                fold_metrics[metric].append(result[metric])

        config_result = {}
        for metric, values in fold_metrics.items():
            config_result[f"{metric}_mean"] = round(np.mean(values), 4)
            config_result[f"{metric}_std"] = round(np.std(values, ddof=1), 4) if len(values) > 1 else 0.0

        baseline_results[model_name] = config_result

    header = f"{'Model':<25} {'Accuracy':>16} {'F1':>16} {'AUC':>16} {'NDCG@5':>16} {'P@5':>16}"
    print(header)
    print("-" * len(header))
    for model_name, metrics in baseline_results.items():
        row = f"{model_name:<25}"
        for m in ["accuracy", "f1", "auc", "ndcg@5", "p@5"]:
            mean = metrics.get(f"{m}_mean", 0)
            std = metrics.get(f"{m}_std", 0)
            row += f" {mean:.4f}±{std:.4f}"
        print(row)

    # ========== PART 2: ABLATION STUDY ==========
    print(f"\n{'=' * 70}")
    print("  PART 2: ABLATION STUDY (XGBoost with feature subsets)")
    print(f"{'=' * 70}")

    ablation_results = {}
    for config_name, feature_indices in ABLATION_CONFIGS.items():
        logger.info(f"Running 5-fold CV: {config_name}...")
        fold_metrics = defaultdict(list)

        for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups=job_id_array)):
            X_train, y_train = X[train_idx], y[train_idx]
            X_test, y_test = X[test_idx], y[test_idx]
            test_jids = [job_ids[i] for i in test_idx]

            result = evaluate_fold_xgb(X_train, y_train, X_test, y_test,
                                       test_jids, feature_indices)

            for metric in ["accuracy", "f1", "auc", "ndcg@5", "p@5"]:
                fold_metrics[metric].append(result[metric])

            # Fairness + Counterfactuals on FAIMR-Full fold 0
            if config_name == "FAIMR-Full" and fold == 0:
                scores = result["scores"]
                test_resumes = [pipeline.pairs.iloc[i]["resume_filename"] for i in test_idx]
                test_groups = infer_gender(test_resumes)
                k_select = max(len(test_idx) // 2, 10)

                fcr_result = fcr_rerank(scores, test_groups, k_select, threshold=0.8)
                fold_metrics["air_before"].append(fcr_result["air_before"])
                fold_metrics["air_after"].append(fcr_result["air_after"])
                fold_metrics["mean_displacement"].append(fcr_result["mean_displacement"])

                if "model" in result:
                    cf_stats = evaluate_counterfactuals(
                        result["model"], X[test_idx], y[test_idx], result["scores"],
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

        ablation_results[config_name] = config_result

    header = f"{'Config':<25} {'Accuracy':>16} {'F1':>16} {'AUC':>16} {'NDCG@5':>16} {'P@5':>16}"
    print(header)
    print("-" * len(header))
    for config_name, metrics in ablation_results.items():
        row = f"{config_name:<25}"
        for m in ["accuracy", "f1", "auc", "ndcg@5", "p@5"]:
            mean = metrics.get(f"{m}_mean", 0)
            std = metrics.get(f"{m}_std", 0)
            row += f" {mean:.4f}±{std:.4f}"
        print(row)

    # ========== PART 3: FAIRNESS + COUNTERFACTUALS ==========
    faimr = ablation_results.get("FAIMR-Full", {})
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

    # ========== LATEX TABLES ==========
    print(f"\n{'=' * 70}")
    print("  LATEX — BASELINE COMPARISON TABLE")
    print(f"{'=' * 70}")
    print("\\begin{table}[t]")
    print("\\centering")
    print("\\caption{Comparison with baseline classifiers (5-fold GroupKFold CV, all features)}")
    print("\\label{tab:baselines}")
    print("\\begin{tabular}{lccccc}")
    print("\\toprule")
    print("Model & Accuracy & F1 & AUC & NDCG@5 & P@5 \\\\")
    print("\\midrule")
    for model_name, metrics in baseline_results.items():
        cells = [model_name]
        for m in ["accuracy", "f1", "auc", "ndcg@5", "p@5"]:
            mean = metrics.get(f"{m}_mean", 0)
            std = metrics.get(f"{m}_std", 0)
            cells.append(f"${mean:.4f} \\pm {std:.4f}$")
        print(" & ".join(cells) + " \\\\")
    print("\\midrule")
    faimr_m = ablation_results.get("FAIMR-Full", {})
    cells = ["\\textbf{FAIMR (XGBoost)}"]
    for m in ["accuracy", "f1", "auc", "ndcg@5", "p@5"]:
        mean = faimr_m.get(f"{m}_mean", 0)
        std = faimr_m.get(f"{m}_std", 0)
        cells.append(f"$\\mathbf{{{mean:.4f} \\pm {std:.4f}}}$")
    print(" & ".join(cells) + " \\\\")
    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{table}")

    print(f"\n{'=' * 70}")
    print("  LATEX — ABLATION TABLE")
    print(f"{'=' * 70}")
    print("\\begin{table}[t]")
    print("\\centering")
    print("\\caption{Ablation study — feature subset analysis (5-fold GroupKFold CV)}")
    print("\\label{tab:ablation}")
    print("\\begin{tabular}{lccccc}")
    print("\\toprule")
    print("Configuration & Accuracy & F1 & AUC & NDCG@5 & P@5 \\\\")
    print("\\midrule")
    for config_name, metrics in ablation_results.items():
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
