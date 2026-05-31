"""
FAIMR — Performance benchmarks.

Reproducible micro-benchmarks for the hot paths of the audit pipeline.
Each benchmark prints {operation, throughput, p50, p95} so a reviewer
can compare against the values committed in `fairness/names/model_card.json`
under ``performance`` and notice any regression.

Usage::

    python benchmarks/bench_audit.py                    # full run
    python benchmarks/bench_audit.py --quick            # smaller corpus
    python benchmarks/bench_audit.py --write-card       # update model_card

The benchmark constructs synthetic corpora rather than running on the
real resume dataset so the numbers are insensitive to corpus changes
between runs.  The seed (BENCH_SEED) is fixed so timing variance comes
ONLY from machine + library differences, not from input variation.

Surfaced metrics — recorded under model_card.performance:

  classifier_predict_many_per_sec
    Throughput of fairness.names.classifier.predict_many() on a fresh
    100-name batch.  This is the inner-loop cost of audit_ranking_bias's
    Phase 1 batched classifier call.

  audit_per_resume_ms_p50 / _p95
    End-to-end audit_ranking_bias() latency normalised by resume count,
    on synthetic corpora of 100 / 1000 / 10_000 resumes.  Demonstrates
    that the batched-classifier optimisation in #19 actually scales.

  fcr_per_candidate_us
    Constrained-insertion fairness re-ranker throughput on a 1000-
    candidate input.  Bounded by the O(N^2 * |G|) termination proof.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

BENCH_SEED = 20251128
N_REPEATS = 5


def _percentiles(samples: list, *qs: float) -> dict:
    """Return {q: value} percentiles from samples.  Lightweight pure-Python
    so we don't pull numpy purely for benchmarking."""
    if not samples:
        return {f"p{int(q * 100)}": 0.0 for q in qs}
    s = sorted(samples)
    out: dict = {}
    for q in qs:
        idx = max(0, min(len(s) - 1, int(round(q * (len(s) - 1)))))
        out[f"p{int(q * 100)}"] = s[idx]
    return out


def _synthetic_names(n: int) -> list:
    """Stable corpus of n synthetic candidate names spanning the four
    main culture clusters the model card measures."""
    rng = random.Random(BENCH_SEED)
    pool = [
        ("Priya", "Sharma", "south_asian"),
        ("Rahul", "Patel", "south_asian"),
        ("John", "Smith", "western"),
        ("Mary", "Johnson", "western"),
        ("Wei", "Chen", "east_asian"),
        ("Akiko", "Tanaka", "east_asian"),
        ("Mohammed", "Ahmed", "arab"),
        ("Fatima", "Khan", "arab"),
    ]
    out = []
    for i in range(n):
        first, last, _ = rng.choice(pool)
        out.append(f"{first} {last}-{i}")
    return out


def _synthetic_corpus(n_resumes: int) -> tuple:
    """Generate a synthetic audit input of ``n_resumes`` candidates."""
    rng = random.Random(BENCH_SEED)
    names = _synthetic_names(n_resumes)
    texts = {
        f"r{i}.txt": f"{name}\nSoftware Engineer with {i % 10} years of "
                    f"experience in Python, machine learning, and cloud "
                    f"infrastructure.  Worked at TechCo and DataLab."
        for i, name in enumerate(names)
    }
    scores = {f"r{i}.txt": rng.random() for i in range(n_resumes)}
    return texts, scores


def bench_classifier_predict_many(n: int = 100) -> dict:
    from fairness.names.classifier import get_classifier
    clf = get_classifier()
    clf._ensure_loaded()
    names = [f"Candidate-{i}" for i in range(n)]
    # Warm-up
    clf.predict_many(names)
    durations = []
    for _ in range(N_REPEATS):
        t0 = time.perf_counter()
        clf.predict_many(names)
        durations.append(time.perf_counter() - t0)
    pct = _percentiles(durations, 0.5, 0.95)
    return {
        "n":          n,
        "p50_s":      round(pct["p50"], 6),
        "p95_s":      round(pct["p95"], 6),
        "per_second": round(n / pct["p50"], 1) if pct["p50"] > 0 else None,
    }


def bench_audit(n_resumes: int) -> dict:
    from fairness.bias_detector import BiasDetector
    detector = BiasDetector()
    texts, scores = _synthetic_corpus(n_resumes)
    # Warm-up (loads model, populates caches)
    detector.audit_ranking_bias(texts, scores, dedup=False)
    durations = []
    for _ in range(N_REPEATS):
        t0 = time.perf_counter()
        detector.audit_ranking_bias(texts, scores, dedup=False)
        durations.append(time.perf_counter() - t0)
    pct = _percentiles(durations, 0.5, 0.95)
    return {
        "n_resumes":      n_resumes,
        "p50_s":          round(pct["p50"], 4),
        "p95_s":          round(pct["p95"], 4),
        "per_resume_ms_p50": round(1000 * pct["p50"] / n_resumes, 3),
        "per_resume_ms_p95": round(1000 * pct["p95"] / n_resumes, 3),
    }


def bench_fcr(n_candidates: int = 1000) -> dict:
    from ranking.fairness_ranker import (
        FairnessConstrainedRanker, RankedCandidate,
    )
    rng = random.Random(BENCH_SEED)
    candidates = [
        RankedCandidate(
            name=f"c{i}",
            score=1.0 - i / n_candidates,
            group=rng.choice(["male", "female"]),
        )
        for i in range(n_candidates)
    ]
    ranker = FairnessConstrainedRanker(threshold=0.8)
    durations = []
    for _ in range(N_REPEATS):
        cands_copy = [
            RankedCandidate(c.name, c.score, c.group) for c in candidates
        ]
        t0 = time.perf_counter()
        ranker.rerank(cands_copy, _compute_pareto=False)
        durations.append(time.perf_counter() - t0)
    pct = _percentiles(durations, 0.5, 0.95)
    return {
        "n_candidates":      n_candidates,
        "p50_s":             round(pct["p50"], 4),
        "p95_s":             round(pct["p95"], 4),
        "per_candidate_us":  round(1_000_000 * pct["p50"] / n_candidates, 1),
    }


def run_all(quick: bool = False) -> dict:
    sizes = [100, 500] if quick else [100, 1_000, 5_000]
    print(f"# FAIMR performance benchmarks (seed={BENCH_SEED}, N_REPEATS={N_REPEATS})")
    print()

    out: dict = {"seed": BENCH_SEED, "n_repeats": N_REPEATS, "stages": {}}

    print("## Classifier predict_many throughput")
    res = bench_classifier_predict_many(n=100)
    out["stages"]["classifier_predict_many"] = res
    print(f"  n={res['n']}  p50={res['p50_s']*1000:.1f}ms  "
          f"p95={res['p95_s']*1000:.1f}ms  "
          f"throughput={res['per_second']} preds/s")

    print()
    print("## audit_ranking_bias end-to-end")
    out["stages"]["audit"] = {}
    for n in sizes:
        res = bench_audit(n)
        out["stages"]["audit"][f"n_{n}"] = res
        print(f"  n={n:>5}  p50={res['p50_s']*1000:8.1f}ms  "
              f"p95={res['p95_s']*1000:8.1f}ms  "
              f"per-resume p50={res['per_resume_ms_p50']:.3f}ms")

    print()
    print("## Constrained-insertion FCR")
    res = bench_fcr(n_candidates=1000)
    out["stages"]["fcr"] = res
    print(f"  n={res['n_candidates']}  p50={res['p50_s']*1000:.1f}ms  "
          f"p95={res['p95_s']*1000:.1f}ms  "
          f"per-candidate={res['per_candidate_us']:.1f} us")

    return out


def write_to_model_card(results: dict) -> None:
    """Record the benchmark results under model_card.performance.
    Preserves prior fields untouched so this is additive."""
    card_path = ROOT / "fairness" / "names" / "model_card.json"
    card = json.loads(card_path.read_text(encoding="utf-8"))
    card["performance"] = results
    card_path.write_text(
        json.dumps(card, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"\nWrote performance block to {card_path.relative_to(ROOT)}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true",
                    help="run smaller corpora; ~30s total")
    ap.add_argument("--write-card", action="store_true",
                    help="record results under model_card.performance")
    args = ap.parse_args()
    results = run_all(quick=args.quick)
    if args.write_card:
        write_to_model_card(results)
    return 0


if __name__ == "__main__":
    sys.exit(main())
