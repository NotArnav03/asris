"""
FAIMR -- Counterfactual Re-evaluation Script

Re-runs the model-based greedy counterfactual algorithm on all
rejected candidates to produce corrected paper statistics.

The algorithm uses the trained XGBoost model's predict_proba as the
score function g(S), making it the canonical evaluation version that
matches the paper's theory section (g(S) = F(x_R union S)).

This REPLACES the numbers in the paper:
  - Mean |delta*|
  - Median |delta*|
  - Std |delta*|, Max |delta*|
  - Greedy-optimal rate (brute-force verified for |Delta| <= 8)
  - Mean and median latency

Run on Colab: python experiments/counterfactual_reeval.py
"""

import sys
import time
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from itertools import combinations
from collections import defaultdict
from scipy.stats import pointbiserialr
from sklearn.metrics import precision_recall_curve
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import get_logger
from ranking.ranking_utils import RankingPipeline

logger = get_logger("experiments.cf_reeval")

BRUTE_FORCE_MAX_DELTA = 8   # enumerate all subsets for |Delta| <= this


# --- Feature computation (same as ablation_stats.py) ----------------------

def compute_features(pipeline: "RankingPipeline") -> np.ndarray:
    """
    Compute the 5 ranking features for all pairs.
    Returns X: np.ndarray of shape (n_pairs, 5).
    Feature order: [sbert_sim, tfidf_sim, skill_cov, n_matched, kw_overlap]
    """
    logger.info("Computing SBERT embeddings...")
    from sentence_transformers import SentenceTransformer
    sbert = SentenceTransformer("all-MiniLM-L6-v2")

    jd_ids = pipeline.pairs["job_id"].unique().tolist()
    jd_texts = [str(pipeline.jd_dict.get(jid, ""))[:512] for jid in jd_ids]
    jd_embs = sbert.encode(jd_texts, show_progress_bar=True, batch_size=64)
    jd_emb_map = dict(zip(jd_ids, jd_embs))

    res_files = list(pipeline.resume_texts.keys())
    res_texts_list = [pipeline.resume_texts[f][:512] for f in res_files]
    res_embs = sbert.encode(res_texts_list, show_progress_bar=True, batch_size=64)
    res_emb_map = dict(zip(res_files, res_embs))

    logger.info("Computing TF-IDF embeddings...")
    all_texts = jd_texts + res_texts_list
    tfidf = TfidfVectorizer(max_features=5000)
    tfidf_matrix = tfidf.fit_transform(all_texts)
    jd_tfidf = {
        jid: tfidf_matrix[i].toarray().flatten()
        for i, jid in enumerate(jd_ids)
    }
    res_tfidf = {
        f: tfidf_matrix[len(jd_ids) + i].toarray().flatten()
        for i, f in enumerate(res_files)
    }

    logger.info("Computing skill features...")
    pipeline.load_skills()

    logger.info("Building feature matrix...")
    features = []
    for _, row in tqdm(pipeline.pairs.iterrows(),
                       total=len(pipeline.pairs), desc="Features"):
        jid = row["job_id"]
        rfile = row["resume_filename"]

        je = jd_emb_map.get(jid)
        re_ = res_emb_map.get(rfile)
        sbert_sim = (
            float(np.dot(je, re_) / (np.linalg.norm(je) * np.linalg.norm(re_) + 1e-8))
            if je is not None and re_ is not None else 0.0
        )

        jt = jd_tfidf.get(jid)
        rt = res_tfidf.get(rfile)
        tfidf_sim = (
            float(np.dot(jt, rt) / (np.linalg.norm(jt) * np.linalg.norm(rt) + 1e-8))
            if jt is not None and rt is not None else 0.0
        )

        jd_skills = pipeline.get_jd_skills(jid)
        res_skills = pipeline.get_resume_skills(rfile)
        n_matched = len(jd_skills & res_skills)
        skill_cov = n_matched / len(jd_skills) if jd_skills else 0.0

        jd_tokens = set(str(pipeline.jd_dict.get(jid, "")).lower().split())
        res_tokens = set(pipeline.resume_texts.get(rfile, "").lower().split())
        kw_overlap = len(jd_tokens & res_tokens) / len(jd_tokens) if jd_tokens else 0.0

        features.append([sbert_sim, tfidf_sim, skill_cov, n_matched, kw_overlap])

    return np.array(features, dtype=np.float32)


# --- Greedy counterfactual (model-based) ----------------------------------

def greedy_counterfactual(
    idx: int,
    pipeline: "RankingPipeline",
    model,
    X: np.ndarray,
    all_probs: np.ndarray,
    threshold: float,
) -> dict:
    """
    Greedy submodular counterfactual for one rejected candidate.

    At each step, greedily adds the skill from the JD deficiency set
    that maximises the model's predicted probability.  Stops when
    prob >= threshold (flip achieved) or no more skills remain.

    Also runs brute-force verification for |Delta| <= BRUTE_FORCE_MAX_DELTA.

    Returns:
        {
            "delta_size":      int,   # |S*| selected skills
            "flip_achieved":   bool,
            "greedy_optimal":  bool | None,  # None if |Delta| > threshold
            "latency_ms":      float,
        }
    """
    row = pipeline.pairs.iloc[idx]
    jid = row["job_id"]
    rfile = row["resume_filename"]

    jd_skills = pipeline.get_jd_skills(jid)
    res_skills = pipeline.get_resume_skills(rfile)
    deficiency = jd_skills - res_skills

    if not deficiency:
        return {
            "delta_size": 0,
            "flip_achieved": all_probs[idx] >= threshold,
            "greedy_optimal": True,
            "latency_ms": 0.0,
        }

    x_curr = X[idx].copy()
    start = time.perf_counter()

    added: set = set()

    for _ in range(len(deficiency)):
        best_skill = None
        best_prob = -1.0

        for skill in deficiency - added:
            test_skills = res_skills | added | {skill}
            n_m = len(jd_skills & test_skills)
            scr = n_m / len(jd_skills) if jd_skills else 0.0

            x_test = x_curr.copy()
            x_test[2] = scr          # skill_cov
            x_test[3] = float(n_m)   # n_matched

            p = float(model.predict_proba(x_test.reshape(1, -1))[0, 1])
            if p > best_prob:
                best_prob = p
                best_skill = skill

        if best_skill is None:
            break

        added.add(best_skill)
        n_m = len(jd_skills & (res_skills | added))
        x_curr[2] = n_m / len(jd_skills) if jd_skills else 0.0
        x_curr[3] = float(n_m)

        if best_prob >= threshold:
            break

    latency_ms = (time.perf_counter() - start) * 1000.0
    flip_achieved = best_prob >= threshold
    delta_size = len(added) if flip_achieved else len(deficiency)

    # Brute-force verification
    greedy_optimal: bool | None = None
    if len(deficiency) <= BRUTE_FORCE_MAX_DELTA and flip_achieved:
        bf_min = len(deficiency) + 1
        for size in range(1, len(added) + 1):
            found = False
            for combo in combinations(deficiency, size):
                test_skills = res_skills | set(combo)
                n_m = len(jd_skills & test_skills)
                scr = n_m / len(jd_skills) if jd_skills else 0.0
                x_t = X[idx].copy()
                x_t[2] = scr
                x_t[3] = float(n_m)
                p = float(model.predict_proba(x_t.reshape(1, -1))[0, 1])
                if p >= threshold:
                    bf_min = size
                    found = True
                    break
            if found:
                break
        greedy_optimal = (bf_min == len(added))

    return {
        "delta_size": delta_size,
        "flip_achieved": flip_achieved,
        "greedy_optimal": greedy_optimal,
        "latency_ms": latency_ms,
    }


# --- Summary statistics ----------------------------------------------------

def compute_summary(results: list[dict]) -> dict:
    """Aggregate statistics over all evaluated candidates."""
    flipped = [r for r in results if r["flip_achieved"] and r["delta_size"] < 50]
    delta_sizes = [r["delta_size"] for r in flipped]
    latencies = [r["latency_ms"] for r in results]

    verified = [r for r in results if r["greedy_optimal"] is not None]
    optimal = [r for r in verified if r["greedy_optimal"]]

    return {
        "n_evaluated": len(results),
        "n_flipped": len(flipped),
        "n_skipped": sum(1 for r in results if r["delta_size"] == 0),
        "mean_delta": float(np.mean(delta_sizes)) if delta_sizes else float("nan"),
        "median_delta": float(np.median(delta_sizes)) if delta_sizes else float("nan"),
        "std_delta": float(np.std(delta_sizes)) if delta_sizes else float("nan"),
        "max_delta": int(max(delta_sizes)) if delta_sizes else 0,
        "n_verified": len(verified),
        "n_optimal": len(optimal),
        "greedy_optimal_rate": (
            len(optimal) / len(verified) if verified else float("nan")
        ),
        "mean_latency_ms": float(np.mean(latencies)),
        "median_latency_ms": float(np.median(latencies)),
        "max_latency_ms": float(np.max(latencies)) if latencies else 0.0,
    }


# --- LaTeX output ----------------------------------------------------------

def print_latex_table(summary: dict) -> None:
    sep = "=" * 70
    print(f"\n{sep}")
    print("  LATEX TABLE: Counterfactual Statistics (update paper)")
    print(sep)
    opt_str = (
        f"{summary['greedy_optimal_rate']:.1%} "
        f"({summary['n_optimal']}/{summary['n_verified']} verified)"
        if summary["n_verified"] > 0 else "N/A"
    )
    print("\\begin{table}[t]")
    print("\\centering")
    print("\\caption{Counterfactual explanation statistics over all "
          f"{summary['n_evaluated']} rejected candidates "
          f"({summary['n_flipped']} with non-empty deficiencies). "
          "Greedy-optimal rate verified by brute-force enumeration "
          f"for $|\\Delta| \\leq {BRUTE_FORCE_MAX_DELTA}$.}}")
    print("\\label{tab:counterfactual}")
    print("\\begin{tabular}{lc}")
    print("\\toprule")
    print("Metric & Value \\\\")
    print("\\midrule")
    print(f"Candidates evaluated & {summary['n_evaluated']} \\\\")
    print(f"Successfully flipped & {summary['n_flipped']} \\\\")
    print(f"Skipped (no deficiency) & {summary['n_skipped']} \\\\")
    print(f"Mean $|\\delta^*|$ & {summary['mean_delta']:.2f} \\\\")
    print(f"Median $|\\delta^*|$ & {summary['median_delta']:.1f} \\\\")
    print(f"Std $|\\delta^*|$ & {summary['std_delta']:.2f} \\\\")
    print(f"Max $|\\delta^*|$ & {summary['max_delta']} \\\\")
    print(f"Greedy-optimal rate & {opt_str} \\\\")
    print(f"Mean latency & {summary['mean_latency_ms']:.2f}\\,ms \\\\")
    print(f"Median latency & {summary['median_latency_ms']:.2f}\\,ms \\\\")
    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{table}")
    print(sep)


# --- Entry point -----------------------------------------------------------

def main() -> None:
    print("=" * 70)
    print("  FAIMR -- Counterfactual Re-evaluation (model-based greedy)")
    print("=" * 70)

    pipeline = RankingPipeline(
        pairs_file="domain_match_pairs.csv", name="CF-Reeval"
    )
    labels = pipeline.pairs["label"].values
    print(f"\n  Pairs: {len(pipeline.pairs)}  "
          f"(pos={int(labels.sum())}, neg={int(len(labels)-labels.sum())})")

    # Feature matrix
    X = compute_features(pipeline)

    # Train full XGBoost model (same config as ablation_stats.py)
    from xgboost import XGBClassifier
    logger.info("Training XGBoost model on full dataset...")
    model = XGBClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.1,
        random_state=42, eval_metric="logloss", verbosity=0,
    )
    model.fit(X, labels)
    all_probs = model.predict_proba(X)[:, 1]

    # Optimal threshold (F1-maximising)
    prec, rec, thresholds = precision_recall_curve(labels, all_probs)
    f1s = 2 * (prec * rec) / (prec + rec + 1e-8)
    threshold = float(thresholds[np.argmax(f1s)])
    logger.info(f"F1-optimal threshold: {threshold:.4f}")

    rejected_mask = all_probs < threshold
    rejected_indices = np.where(rejected_mask)[0]
    print(f"\n  Threshold:       {threshold:.4f}")
    print(f"  Total rejected:  {len(rejected_indices)}")
    print(f"\n  Running counterfactual on all rejected candidates...")

    results = []
    for idx in tqdm(rejected_indices, desc="Counterfactual"):
        r = greedy_counterfactual(idx, pipeline, model, X, all_probs, threshold)
        results.append(r)

    summary = compute_summary(results)

    # Human-readable summary
    print(f"\n{'='*70}")
    print("  RESULTS")
    print(f"{'='*70}")
    print(f"  Candidates evaluated:    {summary['n_evaluated']}")
    print(f"  Skipped (no deficiency): {summary['n_skipped']}")
    print(f"  Successfully flipped:    {summary['n_flipped']}")
    print(f"  Mean |delta*|:           {summary['mean_delta']:.2f}")
    print(f"  Median |delta*|:         {summary['median_delta']:.1f}")
    print(f"  Std |delta*|:            {summary['std_delta']:.2f}")
    print(f"  Max |delta*|:            {summary['max_delta']}")
    if summary["n_verified"] > 0:
        print(f"  Greedy-optimal:          "
              f"{summary['n_optimal']}/{summary['n_verified']} "
              f"({summary['greedy_optimal_rate']:.1%})")
    else:
        print("  Greedy-optimal:          N/A (no verifiable instances)")
    print(f"  Mean latency:            {summary['mean_latency_ms']:.2f} ms")
    print(f"  Median latency:          {summary['median_latency_ms']:.2f} ms")

    print_latex_table(summary)

    print("\n  [ACTION REQUIRED] Update the following values in the paper:")
    print(f"    Abstract/Table: Mean |delta*| = {summary['mean_delta']:.2f}")
    print(f"    Abstract/Table: Median latency = {summary['median_latency_ms']:.2f} ms")
    if summary["n_verified"] > 0:
        print(f"    Abstract/Table: Greedy-optimal = "
              f"{summary['greedy_optimal_rate']:.1%}")


if __name__ == "__main__":
    main()
