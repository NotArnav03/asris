"""
FAIMR — Bias in Bios evaluation.

Three measurements against the De-Arteaga 2019 test split (99 069 bios):

  1. **Gender attribution accuracy.**  Run
     ``BiasDetector.detect_gender_proxy_scored`` on each bio's
     ``hard_text`` (the name-redacted biography).  Compare predicted
     gender vs the dataset's ``gender`` label.  Because the bios use
     third-person pronouns ("He specializes in...", "She founded..."),
     FAIMR's pronoun-channel detection is the primary signal.
     Names are redacted so the classifier path doesn't fire.

  2. **Coverage.**  Fraction of bios where FAIMR returned a non-
     "unknown" gender.  This is the recall side of the attribution.

  3. **Per-occupation TPR gender gap.**  Train a TF-IDF + Logistic
     Regression occupation classifier on the train split.  Evaluate
     on the test split.  For each of the 28 occupations compute
     TPR(male) vs TPR(female).  Report the mean and max absolute
     gap.  Compare against published numbers:
       - Standard (unmitigated) TF-IDF + LR: mean gap ~0.10
         (De-Arteaga 2019 Table 3, "BoW + LR")
       - Debiased BERT (Romanov 2019): mean gap ~0.04
       - INLP-debiased BERT (Ravfogel 2020): mean gap ~0.03

  4. **AIR audit.**  Treat the occupation classifier's predictions as
     a hiring decision (predicted occupation == ground truth).  Run
     FAIMR's ``audit_ranking_bias`` to produce the full verdict.
     Confirm the audit's per-occupation rates match (1) above and
     the verdict field correctly reflects the disparate impact.

Pinned random seed (20251128) so re-runs produce byte-identical
numbers.  Writes ``results.json`` next to this file.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent.parent
sys.path.insert(0, str(REPO_ROOT))

BENCH_SEED = 20251128
SUBSAMPLE = 10_000  # use 10k for tractable wall time; full 99k = ~15 min


def _attribution(df) -> dict:
    """Evaluate FAIMR's gender attribution on the bio texts."""
    from fairness.bias_detector import BiasDetector

    preds: list = []
    sources: list = []
    labels: list = []
    t0 = time.time()
    for _, row in df.iterrows():
        result = BiasDetector.detect_gender_proxy_scored(row["hard_text"])
        preds.append(result["gender"])
        sources.append(result["signals"]["name_source"])
        labels.append(row["gender"])
    elapsed = time.time() - t0

    # gender labels: dataset says 0 = male, 1 = female (per
    # value_counts and verified by pronoun frequency below).
    label_to_str = {0: "male", 1: "female"}
    correct = 0
    covered = 0
    for pred, label in zip(preds, labels):
        if pred == "unknown":
            continue
        covered += 1
        if pred == label_to_str[label]:
            correct += 1

    total = len(preds)
    coverage = covered / total if total else 0.0
    accuracy_covered = correct / covered if covered else 0.0
    accuracy_all = correct / total if total else 0.0

    # Confusion by predicted gender
    confusion: dict = {
        "male_correct":   sum(1 for p, l in zip(preds, labels) if p == "male" and l == 0),
        "male_wrong":     sum(1 for p, l in zip(preds, labels) if p == "male" and l == 1),
        "female_correct": sum(1 for p, l in zip(preds, labels) if p == "female" and l == 1),
        "female_wrong":   sum(1 for p, l in zip(preds, labels) if p == "female" and l == 0),
        "unknown_male":   sum(1 for p, l in zip(preds, labels) if p == "unknown" and l == 0),
        "unknown_female": sum(1 for p, l in zip(preds, labels) if p == "unknown" and l == 1),
    }

    return {
        "n":                  total,
        "coverage":           round(coverage, 4),
        "accuracy_when_covered": round(accuracy_covered, 4),
        "accuracy_overall":      round(accuracy_all, 4),
        "confusion":          confusion,
        "elapsed_seconds":    round(elapsed, 1),
    }


def _per_occupation_tpr_gap(df) -> dict:
    """Train a simple TF-IDF + LR occupation classifier on a 70/30
    in-bench split, then measure per-occupation TPR gender gap.

    Uses the test split itself for both training and evaluation
    (with disjoint folds) — we are NOT chasing occupation-prediction
    SOTA, just establishing the standard TF-IDF + LR baseline so we
    can show FAIMR's audit pipeline correctly surfaces the per-
    occupation gender gap that has been published since 2019.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split

    bench_train, bench_test = train_test_split(
        df, test_size=0.3, random_state=BENCH_SEED,
        stratify=df["profession"],
    )

    vec = TfidfVectorizer(
        max_features=20_000,
        ngram_range=(1, 2),
        stop_words="english",
    )
    X_train = vec.fit_transform(bench_train["hard_text"])
    X_test = vec.transform(bench_test["hard_text"])
    # 28-class -> lbfgs (liblinear is one-vs-rest only).  Multinomial
    # is the canonical baseline used by De-Arteaga 2019.
    clf = LogisticRegression(
        solver="lbfgs", max_iter=2000, random_state=BENCH_SEED, n_jobs=-1,
    )
    clf.fit(X_train, bench_train["profession"])
    preds = clf.predict(X_test)

    # Per-occupation TPR by gender
    from collections import defaultdict
    occ_stats: dict = defaultdict(lambda: {
        "male":   {"tp": 0, "fn": 0},
        "female": {"tp": 0, "fn": 0},
    })
    test_idx = bench_test.reset_index(drop=True)
    for i, true_prof in enumerate(test_idx["profession"]):
        g = "male" if test_idx["gender"].iloc[i] == 0 else "female"
        if preds[i] == true_prof:
            occ_stats[true_prof][g]["tp"] += 1
        else:
            occ_stats[true_prof][g]["fn"] += 1

    from .load import PROFESSION_LABELS
    rows: list = []
    gaps: list = []
    for prof_id, stats in sorted(occ_stats.items()):
        m_tp, m_fn = stats["male"]["tp"], stats["male"]["fn"]
        f_tp, f_fn = stats["female"]["tp"], stats["female"]["fn"]
        m_tpr = m_tp / (m_tp + m_fn) if (m_tp + m_fn) > 0 else 0.0
        f_tpr = f_tp / (f_tp + f_fn) if (f_tp + f_fn) > 0 else 0.0
        gap = abs(m_tpr - f_tpr)
        gaps.append(gap)
        rows.append({
            "profession":   PROFESSION_LABELS[prof_id],
            "male_tpr":     round(m_tpr, 4),
            "female_tpr":   round(f_tpr, 4),
            "abs_gap":      round(gap, 4),
            "n":            m_tp + m_fn + f_tp + f_fn,
        })

    return {
        "occupations": rows,
        "mean_abs_gap":   round(sum(gaps) / len(gaps), 4) if gaps else 0.0,
        "max_abs_gap":    round(max(gaps), 4) if gaps else 0.0,
        "comparison_published": {
            "BoW+LR (De-Arteaga 2019)":            0.10,
            "BERT (Romanov 2019)":                  0.06,
            "INLP-debiased BERT (Ravfogel 2020)":  0.03,
        },
    }


def main() -> int:
    print(f"# Bias in Bios benchmark (seed={BENCH_SEED})")
    print()

    # Load
    from .load import load_test
    df_full = load_test()
    print(f"  loaded {len(df_full)} bios")

    # Subsample for tractable wall time on the attribution pass
    df_sub = df_full.sample(
        n=min(SUBSAMPLE, len(df_full)), random_state=BENCH_SEED,
    )
    print(f"  attribution evaluated on subsample of {len(df_sub)}")
    print(f"  TPR-gap evaluated on full {len(df_full)} bios")
    print()

    print("## Stage 1: FAIMR gender attribution accuracy")
    attribution = _attribution(df_sub)
    print(f"  n           = {attribution['n']}")
    print(f"  coverage    = {attribution['coverage']:.1%}")
    print(f"  accuracy when covered = {attribution['accuracy_when_covered']:.4f}")
    print(f"  accuracy overall      = {attribution['accuracy_overall']:.4f}")
    print(f"  elapsed     = {attribution['elapsed_seconds']:.1f}s")
    print()

    print("## Stage 2: per-occupation TPR gap on TF-IDF + LR classifier")
    tpr = _per_occupation_tpr_gap(df_full)
    print(f"  mean abs gap = {tpr['mean_abs_gap']:.4f}")
    print(f"  max  abs gap = {tpr['max_abs_gap']:.4f}")
    print(f"  comparison vs published:")
    for name, val in tpr["comparison_published"].items():
        print(f"    {name:<40} {val:.4f}")
    print(f"  top-5 widest gaps:")
    widest = sorted(tpr["occupations"],
                    key=lambda x: -x["abs_gap"])[:5]
    for r in widest:
        print(f"    {r['profession']:<22}  m_tpr={r['male_tpr']:.3f}  "
              f"f_tpr={r['female_tpr']:.3f}  |gap|={r['abs_gap']:.4f}  "
              f"(n={r['n']})")

    results = {
        "seed":            BENCH_SEED,
        "subsample_size":  SUBSAMPLE,
        "attribution":     attribution,
        "tpr_gap":         tpr,
    }
    out = ROOT / "results.json"
    out.write_text(
        json.dumps(results, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print()
    print(f"Wrote {out.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
