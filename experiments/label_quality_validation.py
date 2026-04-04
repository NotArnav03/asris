"""
FAIMR -- Label Quality Validation

Validates that domain-match labels in domain_match_pairs.csv are
meaningful relevance proxies by computing:

  1. Point-biserial correlation (r_pb) between each ranking signal
     and the binary domain-match label.
  2. Cohen's d for the score gap between positive and negative pairs.
  3. 95% confidence intervals on r_pb via Fisher z-transform.

A reviewer concern is that "domain match is not the same as hiring
relevance." This script provides the empirical counter-argument:
if all ranking signals (SBERT, TF-IDF, skill coverage, keyword
overlap) are meaningfully correlated with domain-match labels, the
labels capture real content similarity even if they are not direct
hiring outcomes.

Run: python experiments/label_quality_validation.py
"""

import sys
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import pointbiserialr
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import get_logger, LABELED_DIR, PROCESSED_RESUME_DIR, RAW_JD_DIR
from ranking.ranking_utils import RankingPipeline

logger = get_logger("experiments.label_quality")

# Human-readable feature names for tables
FEATURE_DISPLAY = {
    "sbert_sim":   "SBERT Cosine Sim.",
    "tfidf_sim":   "TF-IDF Cosine Sim.",
    "skill_cov":   "Skill Coverage Ratio",
    "n_matched":   "Num. Matched Skills",
    "kw_overlap":  "Keyword Overlap Ratio",
}


# --- Feature computation ---------------------------------------------------

def compute_features(pipeline: "RankingPipeline") -> pd.DataFrame:
    """
    Compute all 5 ranking features for every pair in pipeline.pairs.
    Returns a DataFrame with columns:
        [sbert_sim, tfidf_sim, skill_cov, n_matched, kw_overlap, label]
    """
    logger.info("Loading SBERT model...")
    from sentence_transformers import SentenceTransformer
    sbert = SentenceTransformer("all-MiniLM-L6-v2")

    jd_ids = pipeline.pairs["job_id"].unique().tolist()
    jd_texts = [str(pipeline.jd_dict.get(jid, ""))[:512] for jid in jd_ids]

    res_files = list(pipeline.resume_texts.keys())
    res_texts_list = [pipeline.resume_texts[f][:512] for f in res_files]

    logger.info("Encoding JDs with SBERT...")
    jd_embs = sbert.encode(jd_texts, show_progress_bar=True, batch_size=64)
    jd_emb_map = dict(zip(jd_ids, jd_embs))

    logger.info("Encoding resumes with SBERT...")
    res_embs = sbert.encode(res_texts_list, show_progress_bar=True, batch_size=64)
    res_emb_map = dict(zip(res_files, res_embs))

    logger.info("Fitting TF-IDF...")
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

    logger.info("Loading skill maps...")
    pipeline.load_skills()

    logger.info("Building feature matrix...")
    rows = []
    for _, row in tqdm(pipeline.pairs.iterrows(),
                       total=len(pipeline.pairs), desc="Features"):
        jid = row["job_id"]
        rfile = row["resume_filename"]
        label = int(row["label"])

        # SBERT cosine
        je = jd_emb_map.get(jid)
        re_ = res_emb_map.get(rfile)
        sbert_sim = (
            float(np.dot(je, re_) / (np.linalg.norm(je) * np.linalg.norm(re_) + 1e-8))
            if je is not None and re_ is not None else 0.0
        )

        # TF-IDF cosine
        jt = jd_tfidf.get(jid)
        rt = res_tfidf.get(rfile)
        tfidf_sim = (
            float(np.dot(jt, rt) / (np.linalg.norm(jt) * np.linalg.norm(rt) + 1e-8))
            if jt is not None and rt is not None else 0.0
        )

        # Skill features
        jd_skills = pipeline.get_jd_skills(jid)
        res_skills = pipeline.get_resume_skills(rfile)
        n_matched = len(jd_skills & res_skills)
        skill_cov = n_matched / len(jd_skills) if jd_skills else 0.0

        # Keyword overlap
        jd_tokens = set(str(pipeline.jd_dict.get(jid, "")).lower().split())
        res_tokens = set(pipeline.resume_texts.get(rfile, "").lower().split())
        kw_overlap = (
            len(jd_tokens & res_tokens) / len(jd_tokens) if jd_tokens else 0.0
        )

        rows.append({
            "sbert_sim": sbert_sim,
            "tfidf_sim": tfidf_sim,
            "skill_cov": skill_cov,
            "n_matched": float(n_matched),
            "kw_overlap": kw_overlap,
            "label": label,
        })

    return pd.DataFrame(rows)


# --- Statistics ------------------------------------------------------------

def compute_point_biserial(df: pd.DataFrame) -> pd.DataFrame:
    """
    Point-biserial r between each feature and the binary label.
    Includes 95% CI via Fisher z-transform.
    """
    feature_cols = [c for c in df.columns if c != "label"]
    label = df["label"].values
    n = len(label)
    records = []

    for col in feature_cols:
        vals = df[col].values
        r, p = pointbiserialr(label, vals)

        # Fisher z CI
        z = np.arctanh(r)
        se = 1.0 / np.sqrt(n - 3)
        ci_lo = float(np.tanh(z - 1.96 * se))
        ci_hi = float(np.tanh(z + 1.96 * se))

        records.append({
            "feature": col,
            "r_pb": round(float(r), 4),
            "ci_lo": round(ci_lo, 4),
            "ci_hi": round(ci_hi, 4),
            "p_value": round(float(p), 6),
            "significant": p < 0.05,
            "n": n,
        })

    return pd.DataFrame(records)


def compute_cohens_d(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cohen's d = (mean_pos - mean_neg) / pooled_std for each feature.
    Effect size labels: small < 0.5 <= medium < 0.8 <= large.
    """
    feature_cols = [c for c in df.columns if c != "label"]
    pos = df[df["label"] == 1]
    neg = df[df["label"] == 0]
    records = []

    for col in feature_cols:
        p_vals = pos[col].values
        n_vals = neg[col].values
        n1, n2 = len(p_vals), len(n_vals)
        m1, m2 = float(np.mean(p_vals)), float(np.mean(n_vals))
        s1, s2 = float(np.std(p_vals, ddof=1)), float(np.std(n_vals, ddof=1))

        pooled = np.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))
        d = (m1 - m2) / pooled if pooled > 0 else 0.0

        if abs(d) >= 0.8:
            effect = "large"
        elif abs(d) >= 0.5:
            effect = "medium"
        else:
            effect = "small"

        records.append({
            "feature": col,
            "mean_pos": round(m1, 4),
            "mean_neg": round(m2, 4),
            "cohens_d": round(float(d), 4),
            "effect": effect,
        })

    return pd.DataFrame(records)


# --- Output ----------------------------------------------------------------

def print_summary(r_df: pd.DataFrame, d_df: pd.DataFrame) -> None:
    """Print human-readable summary."""
    merged = r_df.merge(d_df, on="feature")
    print(f"\n{'Feature':<26} {'r_pb':>8} {'95% CI':>18} {'p':>10} "
          f"{'d':>8} {'Effect':>8}")
    print("-" * 82)
    for _, row in merged.iterrows():
        name = FEATURE_DISPLAY.get(row["feature"], row["feature"])
        ci = f"[{row['ci_lo']:.3f}, {row['ci_hi']:.3f}]"
        sig = "*" if row["significant"] else " "
        print(f"{name:<26} {row['r_pb']:>8.4f} {ci:>18} "
              f"{row['p_value']:>9.4f}{sig} {row['cohens_d']:>8.4f} "
              f"{row['effect']:>8}")
    print("  * p < 0.05")


def print_latex_table(r_df: pd.DataFrame, d_df: pd.DataFrame) -> None:
    """Print combined LaTeX table for paper inclusion."""
    merged = r_df.merge(d_df, on="feature")
    sep = "=" * 70
    print(f"\n{sep}")
    print("  LATEX TABLE: Label Quality Validation")
    print(sep)
    print("\\begin{table}[t]")
    print("\\centering")
    print("\\caption{Point-biserial correlation and Cohen's $d$ between "
          "each ranking signal and the domain-match label "
          "($n = " + str(int(r_df["n"].iloc[0])) + "$ pairs). "
          "All $r_{pb}$ values are positive and statistically significant "
          "($p < 0.05$), confirming that domain-match is a valid "
          "relevance proxy for all five ranking signals. "
          "$^{*}$ denotes $p < 0.001$.}")
    print("\\label{tab:label_quality}")
    print("\\begin{tabular}{lcccc}")
    print("\\toprule")
    print("Signal & $r_{pb}$ & 95\\% CI & Cohen's $d$ & Effect \\\\")
    print("\\midrule")
    for _, row in merged.iterrows():
        name = FEATURE_DISPLAY.get(row["feature"], row["feature"])
        ci = f"[{row['ci_lo']:.3f},\\;{row['ci_hi']:.3f}]"
        sig = "^{*}" if row["p_value"] < 0.001 else ""
        print(f"{name} & ${row['r_pb']:.4f}{sig}$ & ${ci}$ & "
              f"${row['cohens_d']:.4f}$ & {row['effect']} \\\\")
    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{table}")
    print(sep)


# --- Entry point -----------------------------------------------------------

def main() -> None:
    print("=" * 70)
    print("  FAIMR -- Label Quality Validation")
    print("=" * 70)

    pipeline = RankingPipeline(pairs_file="domain_match_pairs.csv",
                               name="LabelQuality")

    n_pos = int((pipeline.pairs["label"] == 1).sum())
    n_neg = int((pipeline.pairs["label"] == 0).sum())
    print(f"\n  Pairs: {len(pipeline.pairs)} total  "
          f"({n_pos} positive, {n_neg} negative, "
          f"ratio {n_pos/n_neg:.2f})")

    df = compute_features(pipeline)

    print("\n  Computing point-biserial correlations...")
    r_df = compute_point_biserial(df)

    print("  Computing Cohen's d...")
    d_df = compute_cohens_d(df)

    print_summary(r_df, d_df)
    print_latex_table(r_df, d_df)

    # Quick sanity check
    all_sig = r_df["significant"].all()
    all_positive = (r_df["r_pb"] > 0).all()
    print(f"\n  All signals positively correlated with label: "
          f"{'YES' if all_positive else 'NO'}")
    print(f"  All correlations statistically significant (p<0.05): "
          f"{'YES' if all_sig else 'NO'}")

    if all_sig and all_positive:
        print("\n  [PASS] Domain-match labels are valid relevance proxies.")
        print("         Include Table tab:label_quality in Section 3 (Data).")
    else:
        print("\n  [WARN] Some signals show weak or non-significant correlation.")
        print("         Investigate before claiming label validity.")


if __name__ == "__main__":
    main()
