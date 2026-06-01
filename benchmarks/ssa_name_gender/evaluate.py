"""
FAIMR -- SSA name-gender benchmark: evaluation.

Measures FAIMR's calibrated name classifier on the public US SSA
baby-names corpus.  Three stages:

  Stage A -- Full-SSA evaluation.  FAIMR runs against every SSA
             name; lookup hits use the training-corpus fastpath,
             misses fall back to the char-ngram classifier.  This
             is the deployment-scenario number and is what compares
             head-to-head with the published char-LSTM band.

  Stage B -- Out-of-distribution holdout.  Names that do NOT appear
             in FAIMR's committed training_corpus.csv.  This is by
             construction the rare-tail of SSA and is harder than
             the popular-name distribution the literature reports.

  Stage C -- Apples-to-apples baseline.  An inline TF-IDF + LR
             char-ngram model trained on the SAME training corpus
             FAIMR uses, evaluated on the same holdout.  Isolates
             the incremental value of FAIMR's lookup, nickname
             mapping, and per-culture isotonic calibration.

All stages report accuracy + ROC-AUC + Brier + ECE.  Stage A also
reports per-attestation-bucket accuracy so reviewers can see where
the FAIMR classifier sits per name-popularity tier.

Determinism: BENCH_SEED = 20251128 throughout.

Citations for the published comparison band (see README.md):
  - Char-LSTM, English-only:   ~0.95--0.97 accuracy
  - Char-CNN,  English-only:   ~0.94--0.96 accuracy
  - TF-IDF + LR char-ngrams:   ~0.88--0.91 accuracy
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


def _expected_calibration_error(y_true, y_prob, n_bins: int = 15) -> float:
    import numpy as np
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for lo, hi in zip(bins[:-1], bins[1:]):
        in_bin = (y_prob >= lo) & (y_prob < hi if hi < 1.0 else y_prob <= hi)
        n_in = int(in_bin.sum())
        if n_in == 0:
            continue
        conf = float(y_prob[in_bin].mean())
        acc = float(y_true[in_bin].mean())
        ece += (n_in / n) * abs(acc - conf)
    return float(ece)


def _metrics(y_true, p_pred) -> dict:
    import numpy as np
    from sklearn.metrics import roc_auc_score, brier_score_loss
    y_pred = (p_pred >= 0.5).astype(int)
    try:
        auc = float(roc_auc_score(y_true, p_pred))
    except ValueError:
        auc = None
    return {
        "n":         int(len(y_true)),
        "accuracy":  round(float((y_pred == y_true).mean()), 4),
        "roc_auc":   None if auc is None else round(auc, 4),
        "brier":     round(float(brier_score_loss(y_true, p_pred)), 4),
        "ece":       round(_expected_calibration_error(y_true, p_pred), 4),
    }


def _holdout_split(ssa_df, training_corpus_path: Path):
    import pandas as pd
    training = pd.read_csv(
        training_corpus_path, keep_default_na=False, na_values=[""],
    )
    training_names = set(
        n.lower() for n in training["name"].astype(str)
    )
    is_in_training = ssa_df["name"].str.lower().isin(training_names)
    return (
        ssa_df[is_in_training].copy(),
        ssa_df[~is_in_training].copy(),
        training,
    )


def _run_faimr(df) -> tuple[dict, "np.ndarray"]:
    import numpy as np
    from fairness.names.classifier import predict_many

    names = df["name"].tolist()
    t0 = time.time()
    results = predict_many(names)
    elapsed = time.time() - t0

    p_pred = np.array([r.p_female for r in results])
    sources = [r.source for r in results]
    y_true = (df["p_female"].to_numpy() >= 0.5).astype(int)

    m = _metrics(y_true, p_pred)
    m["elapsed_s"] = round(elapsed, 1)
    m["sources"] = {
        "lookup": sum(1 for s in sources if s == "lookup"),
        "model":  sum(1 for s in sources if s == "model"),
        "empty":  sum(1 for s in sources if s == "empty"),
    }
    return m, p_pred


def _run_hybrid(df) -> tuple[dict, "np.ndarray"]:
    """FAIMR + SSA char-LSTM hybrid (faimr_plus plugin).  Returns None
    metrics if plugin weights are not present, so the benchmark can
    run without the plugin installed."""
    import numpy as np
    from pathlib import Path
    weights = (
        Path(__file__).resolve().parent.parent.parent
        / "faimr_plus" / "ssa_char_lstm" / "weights.pt"
    )
    if not weights.exists():
        return None, None

    from faimr_plus.ssa_char_lstm.hybrid import predict_hybrid

    names = df["name"].tolist()
    t0 = time.time()
    results = predict_hybrid(names)
    elapsed = time.time() - t0

    p_pred = np.array([r.p_female for r in results])
    sources = [r.source for r in results]
    y_true = (df["p_female"].to_numpy() >= 0.5).astype(int)

    m = _metrics(y_true, p_pred)
    m["elapsed_s"] = round(elapsed, 1)
    m["sources"] = {
        s: sum(1 for x in sources if x == s)
        for s in ("lookup", "ensemble", "lstm", "empty")
    }
    return m, p_pred


def _attestation_buckets(df, p_pred) -> dict:
    """Stratify accuracy by name-attestation strength.

    SSA names with more years of attestation are intrinsically easier
    (more signal, less noise).  Showing this stratification answers
    the natural reviewer question: "is FAIMR good at the names that
    actually appear in production resumes, or only at the popular
    tail?"
    """
    import numpy as np
    y_true = (df["p_female"].to_numpy() >= 0.5).astype(int)
    n_years = df["n_years"].to_numpy()

    buckets = [
        ("1-4 years (rare tail)",       n_years <= 4),
        ("5-19 years",                  (n_years >= 5) & (n_years <= 19)),
        ("20-49 years",                 (n_years >= 20) & (n_years <= 49)),
        ("50+ years (canonical)",       n_years >= 50),
    ]
    out = []
    for label, mask in buckets:
        if not mask.any():
            continue
        out.append({
            "bucket": label,
            **_metrics(y_true[mask], p_pred[mask]),
        })
    return out


def _train_corpus_baseline(training_df, eval_df) -> dict:
    """Train an inline TF-IDF + LR char-ngram baseline on the SAME
    training corpus FAIMR uses (data/names/training_corpus.csv),
    evaluate on `eval_df`.  This is the apples-to-apples comparison.
    """
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression

    # Restrict to rows with a definite gender label in the corpus.
    train = training_df.copy()
    train["name"] = train["name"].astype(str)
    train = train[
        (train["p_female"] >= 0.7) | (train["p_female"] <= 0.3)
    ].copy()
    train["y"] = (train["p_female"] >= 0.5).astype(int)
    weights = train["weight"].astype(float).to_numpy() \
        if "weight" in train.columns else None

    vec = TfidfVectorizer(
        analyzer="char_wb", ngram_range=(2, 5),
        min_df=2, max_df=0.95, sublinear_tf=True, lowercase=True,
    )
    X_train = vec.fit_transform(train["name"])
    X_test = vec.transform(eval_df["name"])
    y_train = train["y"].to_numpy()
    y_test = (eval_df["p_female"].to_numpy() >= 0.5).astype(int)

    clf = LogisticRegression(
        solver="liblinear", C=10.0, max_iter=2000,
        random_state=BENCH_SEED,
    )
    clf.fit(X_train, y_train, sample_weight=weights)
    p_pred = clf.predict_proba(X_test)[:, 1]

    m = _metrics(y_test, p_pred)
    m["n_train"] = int(len(train))
    return m


def main() -> int:
    print(f"# SSA name-gender benchmark (seed={BENCH_SEED})")
    print()

    from benchmarks.ssa_name_gender.load import load_per_name_aggregate
    ssa = load_per_name_aggregate()
    print(f"SSA aggregate: {len(ssa)} unique names "
          f"(female={int((ssa['p_female']>=0.5).sum())}, "
          f"male={int((ssa['p_female']<0.5).sum())})")
    in_train, holdout, training_df = _holdout_split(
        ssa, REPO_ROOT / "data" / "names" / "training_corpus.csv",
    )
    print(f"  overlap with FAIMR training corpus: {len(in_train)}")
    print(f"  out-of-distribution holdout:        {len(holdout)}")
    print()

    # ------------------------------------------------------------
    # Stage A: full-SSA evaluation (deployment scenario)
    # ------------------------------------------------------------
    print("## Stage A: FAIMR on the FULL SSA corpus "
          "(lookup + classifier fallback)")
    full_metrics, full_p_pred = _run_faimr(ssa)
    print(f"  n         = {full_metrics['n']}")
    print(f"  accuracy  = {full_metrics['accuracy']:.4f}")
    print(f"  roc_auc   = {full_metrics['roc_auc']}")
    print(f"  brier     = {full_metrics['brier']:.4f}")
    print(f"  ECE       = {full_metrics['ece']:.4f}")
    print(f"  elapsed   = {full_metrics['elapsed_s']:.1f}s")
    print(f"  sources   = {full_metrics['sources']}")
    print()

    print("### Stage A.1: per-attestation-bucket breakdown")
    buckets = _attestation_buckets(ssa, full_p_pred)
    for b in buckets:
        print(f"  {b['bucket']:<28}  n={b['n']:>5}  "
              f"acc={b['accuracy']:.4f}  auc={b['roc_auc']}  "
              f"ECE={b['ece']:.4f}")
    print()

    # ------------------------------------------------------------
    # Stage B: out-of-distribution holdout
    # ------------------------------------------------------------
    print("## Stage B: FAIMR on the OOD holdout "
          "(names NOT in training corpus)")
    holdout_attested = holdout[holdout["n_years"] >= 5].copy()
    print(f"  (filtered to n_years >= 5: {len(holdout_attested)} names)")
    holdout_metrics, holdout_p_pred = _run_faimr(holdout_attested)
    print(f"  n         = {holdout_metrics['n']}")
    print(f"  accuracy  = {holdout_metrics['accuracy']:.4f}")
    print(f"  roc_auc   = {holdout_metrics['roc_auc']}")
    print(f"  brier     = {holdout_metrics['brier']:.4f}")
    print(f"  ECE       = {holdout_metrics['ece']:.4f}")
    print(f"  sources   = {holdout_metrics['sources']}")
    print()

    # ------------------------------------------------------------
    # Stage C: apples-to-apples baseline
    # ------------------------------------------------------------
    print("## Stage C: inline TF-IDF + LR trained on FAIMR's SAME "
          "training corpus, evaluated on the OOD holdout")
    baseline_metrics = _train_corpus_baseline(training_df, holdout_attested)
    print(f"  train n   = {baseline_metrics['n_train']}")
    print(f"  test n    = {baseline_metrics['n']}")
    print(f"  accuracy  = {baseline_metrics['accuracy']:.4f}")
    print(f"  roc_auc   = {baseline_metrics['roc_auc']}")
    print(f"  brier     = {baseline_metrics['brier']:.4f}")
    print(f"  ECE       = {baseline_metrics['ece']:.4f}")
    print()

    # ------------------------------------------------------------
    # Stage D: hybrid (FAIMR + SSA char-LSTM plugin)
    # ------------------------------------------------------------
    print("## Stage D: FAIMR + SSA char-LSTM hybrid plugin "
          "(faimr_plus.ssa_char_lstm)")
    hybrid_full, _ = _run_hybrid(ssa)
    if hybrid_full is None:
        print("  plugin not installed (weights.pt missing) -- SKIPPED")
        print()
    else:
        print(f"  full-SSA n={hybrid_full['n']}  "
              f"acc={hybrid_full['accuracy']:.4f}  "
              f"auc={hybrid_full['roc_auc']}  "
              f"ECE={hybrid_full['ece']:.4f}")
        print(f"  sources    = {hybrid_full['sources']}")
        hybrid_ood, _ = _run_hybrid(holdout_attested)
        print(f"  OOD-holdout n={hybrid_ood['n']}  "
              f"acc={hybrid_ood['accuracy']:.4f}  "
              f"auc={hybrid_ood['roc_auc']}  "
              f"ECE={hybrid_ood['ece']:.4f}")
        print(f"  sources    = {hybrid_ood['sources']}")
        print()

    print("## Headline numbers")
    print(f"  Published char-LSTM (English-only):    ~0.95-0.97")
    print(f"  Published char-CNN  (English-only):    ~0.94-0.96")
    print(f"  Published TF-IDF + LR char-ngram band: ~0.88-0.91")
    print(f"  FAIMR full-SSA accuracy (Stage A):     "
          f"{full_metrics['accuracy']:.4f}")
    print(f"  FAIMR OOD-holdout accuracy (Stage B):  "
          f"{holdout_metrics['accuracy']:.4f}")
    print(f"  Baseline OOD-holdout (Stage C):        "
          f"{baseline_metrics['accuracy']:.4f}")
    if hybrid_full is not None:
        print(f"  Hybrid full-SSA accuracy (Stage D):    "
              f"{hybrid_full['accuracy']:.4f}")
        print(f"  Hybrid OOD-holdout accuracy (Stage D): "
              f"{hybrid_ood['accuracy']:.4f}")
        print(f"  -- delta (hybrid vs baseline on OOD):  "
              f"{hybrid_ood['accuracy'] - baseline_metrics['accuracy']:+.4f}")
    print(f"  -- delta (FAIMR vs baseline on OOD):   "
          f"{holdout_metrics['accuracy'] - baseline_metrics['accuracy']:+.4f}")

    results = {
        "seed":               BENCH_SEED,
        "n_ssa_total":        int(len(ssa)),
        "n_overlap":          int(len(in_train)),
        "n_holdout":          int(len(holdout)),
        "n_holdout_attested": int(len(holdout_attested)),
        "stage_a_full_ssa":   full_metrics,
        "stage_a_attestation_buckets": buckets,
        "stage_b_ood_holdout":         holdout_metrics,
        "stage_c_apples_to_apples_baseline": baseline_metrics,
        "stage_d_hybrid_plugin": {
            "full_ssa":    hybrid_full,
            "ood_holdout": hybrid_ood if hybrid_full is not None else None,
        },
        "comparison_published": {
            "char_lstm_english_only_range":  [0.95, 0.97],
            "char_cnn_english_only_range":   [0.94, 0.96],
            "tfidf_lr_char_ngram_range":     [0.88, 0.91],
        },
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
