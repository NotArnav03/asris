"""
FAIMR — W4 Fix: Paired t-test on Ablation Increments
      + W5 Fix: Full Counterfactual Evaluation on All Rejected Candidates

Run: python experiments/ablation_stats.py
"""

import sys
import time
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
from scipy import stats
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, precision_recall_curve
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm
from itertools import combinations

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import get_logger
from ranking.ranking_utils import RankingPipeline

logger = get_logger("experiments.ablation_stats")


def run_ablation_with_per_fold():
    print("=" * 70)
    print("  FAIMR — Ablation with Per-Fold AUC + Full Counterfactual Eval")
    print("=" * 70)

    pipeline = RankingPipeline(pairs_file="domain_match_pairs.csv", name="Ablation-Stats")
    labels = pipeline.pairs["label"].values
    job_ids = pipeline.pairs["job_id"].values

    # === FEATURE COMPUTATION ===
    logger.info("Computing SBERT embeddings...")
    from sentence_transformers import SentenceTransformer
    sbert = SentenceTransformer("all-MiniLM-L6-v2")

    jd_ids = pipeline.pairs["job_id"].unique().tolist()
    jd_texts = [str(pipeline.jd_dict.get(jid, ""))[:512] for jid in jd_ids]
    jd_embs = sbert.encode(jd_texts, show_progress_bar=True, batch_size=64)
    jd_emb_map = dict(zip(jd_ids, jd_embs))

    res_files = list(pipeline.resume_texts.keys())
    res_texts = [pipeline.resume_texts[f][:512] for f in res_files]
    res_embs = sbert.encode(res_texts, show_progress_bar=True, batch_size=64)
    res_emb_map = dict(zip(res_files, res_embs))

    logger.info("Computing TF-IDF embeddings...")
    all_texts = jd_texts + res_texts
    tfidf = TfidfVectorizer(max_features=5000)
    tfidf_matrix = tfidf.fit_transform(all_texts)
    jd_tfidf = {jid: tfidf_matrix[i].toarray().flatten() for i, jid in enumerate(jd_ids)}
    res_tfidf = {f: tfidf_matrix[len(jd_ids) + i].toarray().flatten() for i, f in enumerate(res_files)}

    logger.info("Computing skill features...")
    pipeline.load_skills()

    logger.info("Building feature matrix...")
    features = []
    for _, row in tqdm(pipeline.pairs.iterrows(), total=len(pipeline.pairs), desc="Features"):
        jid = row["job_id"]
        rfile = row["resume_filename"]

        # SBERT cosine
        je = jd_emb_map.get(jid)
        re_ = res_emb_map.get(rfile)
        if je is not None and re_ is not None:
            sbert_sim = float(np.dot(je, re_) / (np.linalg.norm(je) * np.linalg.norm(re_) + 1e-8))
        else:
            sbert_sim = 0.0

        # TF-IDF cosine
        jt = jd_tfidf.get(jid)
        rt = res_tfidf.get(rfile)
        if jt is not None and rt is not None:
            tfidf_sim = float(np.dot(jt, rt) / (np.linalg.norm(jt) * np.linalg.norm(rt) + 1e-8))
        else:
            tfidf_sim = 0.0

        # Skills
        jd_skills = pipeline.get_jd_skills(jid)
        res_skills = pipeline.get_resume_skills(rfile)
        n_match = len(jd_skills & res_skills)
        scr = n_match / len(jd_skills) if len(jd_skills) > 0 else 0.0

        # Keyword overlap
        jd_tokens = set(str(pipeline.jd_dict.get(jid, "")).lower().split())
        res_tokens = set(pipeline.resume_texts.get(rfile, "").lower().split())
        kw_overlap = len(jd_tokens & res_tokens) / len(jd_tokens) if len(jd_tokens) > 0 else 0.0

        features.append([sbert_sim, tfidf_sim, scr, n_match, kw_overlap])

    X = np.array(features)
    y = labels

    # === ABLATION WITH PER-FOLD AUC (W4) ===
    configs = {
        "SBERT-only":    [0],
        "TF-IDF-only":   [1],
        "SBERT+TF-IDF":  [0, 1],
        "Full-LTR":      [0, 1, 2, 3, 4],
    }

    from xgboost import XGBClassifier

    gkf = GroupKFold(n_splits=5)
    per_fold_auc = {name: [] for name in configs}

    for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups=job_ids)):
        print(f"\n--- Fold {fold+1}/5 ---")
        for name, feat_cols in configs.items():
            X_train = X[train_idx][:, feat_cols]
            X_test = X[test_idx][:, feat_cols]

            model = XGBClassifier(
                n_estimators=200, max_depth=6, learning_rate=0.1,
                random_state=42, use_label_encoder=False,
                eval_metric="logloss", verbosity=0
            )
            model.fit(X_train, y[train_idx])
            y_prob = model.predict_proba(X_test)[:, 1]

            auc = roc_auc_score(y[test_idx], y_prob)
            per_fold_auc[name].append(auc)
            print(f"  {name:>16}: AUC = {auc:.4f}")

    # Paired t-tests
    print(f"\n{'='*70}")
    print("  PAIRED T-TESTS (W4)")
    print(f"{'='*70}")

    auc_dual = np.array(per_fold_auc["SBERT+TF-IDF"])
    auc_full = np.array(per_fold_auc["Full-LTR"])
    auc_sbert = np.array(per_fold_auc["SBERT-only"])

    print(f"\n  Per-fold AUC:")
    for name in configs:
        vals = per_fold_auc[name]
        print(f"    {name:>16}: {[f'{v:.4f}' for v in vals]}")

    print(f"\n  SBERT+TF-IDF vs Full-LTR:")
    print(f"    Differences: {auc_full - auc_dual}")
    t1, p1 = stats.ttest_rel(auc_full, auc_dual)
    print(f"    t = {t1:.4f}, p = {p1:.4f}")
    print(f"    Significant at 5%? {'YES' if p1 < 0.05 else 'NO'}")
    print(f"    Significant at 10%? {'YES' if p1 < 0.10 else 'NO'}")

    print(f"\n  SBERT-only vs Full-LTR:")
    t2, p2 = stats.ttest_rel(auc_full, auc_sbert)
    print(f"    t = {t2:.4f}, p = {p2:.4f}")
    print(f"    Significant at 5%? {'YES' if p2 < 0.05 else 'NO'}")

    print(f"\n  Summary:")
    for name in configs:
        vals = per_fold_auc[name]
        print(f"    {name:>16}: {np.mean(vals):.4f} +/- {np.std(vals, ddof=1):.4f}")

    # === FULL COUNTERFACTUAL EVALUATION (W5) ===
    print(f"\n{'='*70}")
    print("  FULL COUNTERFACTUAL EVALUATION (W5)")
    print(f"{'='*70}")

    # Train full model
    model_full = XGBClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.1,
        random_state=42, use_label_encoder=False,
        eval_metric="logloss", verbosity=0
    )
    model_full.fit(X, y)
    all_probs = model_full.predict_proba(X)[:, 1]

    # Threshold
    prec, rec, thresholds = precision_recall_curve(y, all_probs)
    f1s = 2 * (prec * rec) / (prec + rec + 1e-8)
    threshold = thresholds[np.argmax(f1s)]

    rejected_mask = all_probs < threshold
    rejected_indices = np.where(rejected_mask)[0]
    print(f"\n  Threshold: {threshold:.4f}")
    print(f"  Total rejected: {len(rejected_indices)}")

    explanation_sizes = []
    latencies = []
    greedy_optimal = 0
    total_verified = 0
    skipped = 0

    for idx in tqdm(rejected_indices, desc="Counterfactual (full set)"):
        row = pipeline.pairs.iloc[idx]
        jid = row["job_id"]
        rfile = row["resume_filename"]

        jd_skills = pipeline.get_jd_skills(jid)
        res_skills = pipeline.get_resume_skills(rfile)
        deficiency = jd_skills - res_skills

        if len(deficiency) == 0:
            skipped += 1
            continue

        x_curr = X[idx].copy()
        start_time = time.time()

        # Greedy search
        added = set()
        best_score = all_probs[idx]

        for step in range(len(deficiency)):
            best_skill = None
            best_new_score = -1

            for skill in deficiency - added:
                test_skills = res_skills | added | {skill}
                n_match_new = len(jd_skills & test_skills)
                scr_new = n_match_new / len(jd_skills) if len(jd_skills) > 0 else 0.0

                x_test = x_curr.copy()
                x_test[2] = scr_new
                x_test[3] = n_match_new

                prob = model_full.predict_proba(x_test.reshape(1, -1))[0, 1]
                if prob > best_new_score:
                    best_new_score = prob
                    best_skill = skill

            if best_skill is None:
                break

            added.add(best_skill)
            n_m = len(jd_skills & (res_skills | added))
            x_curr[2] = n_m / len(jd_skills) if len(jd_skills) > 0 else 0.0
            x_curr[3] = n_m
            best_score = best_new_score

            if best_score >= threshold:
                break

        elapsed = (time.time() - start_time) * 1000
        latencies.append(elapsed)

        if best_score >= threshold:
            explanation_sizes.append(len(added))
        else:
            explanation_sizes.append(len(deficiency))

        # Brute-force verify for small sets
        if len(deficiency) <= 8 and best_score >= threshold:
            total_verified += 1
            bf_min = len(deficiency)
            for size in range(1, len(added) + 1):
                found = False
                for combo in combinations(deficiency, size):
                    test_skills = res_skills | set(combo)
                    n_m = len(jd_skills & test_skills)
                    scr_t = n_m / len(jd_skills) if len(jd_skills) > 0 else 0.0
                    x_t = X[idx].copy()
                    x_t[2] = scr_t
                    x_t[3] = n_m
                    p = model_full.predict_proba(x_t.reshape(1, -1))[0, 1]
                    if p >= threshold:
                        bf_min = size
                        found = True
                        break
                if found:
                    break
            if bf_min == len(added):
                greedy_optimal += 1

    flipped = [s for s in explanation_sizes if s < 50]  # exclude unflippable
    print(f"\n  Results:")
    print(f"  Candidates evaluated:    {len(explanation_sizes)}")
    print(f"  Skipped (no deficiency): {skipped}")
    print(f"  Successfully flipped:    {len([s for s, l in zip(explanation_sizes, latencies) if s < 50])}")
    print(f"  Mean |delta*|:           {np.mean(explanation_sizes):.2f}")
    print(f"  Median |delta*|:         {np.median(explanation_sizes):.1f}")
    print(f"  Std |delta*|:            {np.std(explanation_sizes):.2f}")
    print(f"  Max |delta*|:            {np.max(explanation_sizes)}")
    print(f"  Mean latency:            {np.mean(latencies):.2f} ms")
    print(f"  Median latency:          {np.median(latencies):.2f} ms")
    if total_verified > 0:
        print(f"  Greedy=Optimal:          {greedy_optimal}/{total_verified} ({100*greedy_optimal/total_verified:.1f}%)")
    else:
        print(f"  Greedy=Optimal:          N/A (no verifiable instances)")


if __name__ == "__main__":
    run_ablation_with_per_fold()
