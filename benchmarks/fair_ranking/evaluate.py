"""
FAIMR -- FA*IR vs constrained-insertion FCR head-to-head.

Honest head-to-head: both algorithms are evaluated by the SAME
fairness metric (minimum prefix-AIR), so the NDCG comparison is
apples-to-apples.

Protocol:
  * Two groups: protected vs non-protected.
  * Scores drawn from Normal(mu_g, 1.0).  Non-protected has mu=0.5,
    protected has mu in {0.0, -0.5} (small / large baseline gap).
  * Protected proportion p in {0.2, 0.3}.
  * List sizes N in {100, 500, 1000}.

Per condition we:
  1. Sweep FA*IR alpha in {0.01, 0.05, 0.1, 0.15, 0.2}.
  2. Sweep FAIMR FCR threshold in {0.6, 0.7, 0.8, 0.9, 0.95}.
  3. For each (algorithm, parameter) pair record:
       - min-prefix-AIR (the fairness metric)
       - NDCG@N (the utility metric)
       - max promotion gap (indefensibility proxy)
       - within-group order preservation (bool)
  4. Build the Pareto frontier per algorithm.
  5. Report NDCG@equal-min-prefix-AIR for both.

Prefix-AIR semantics: at every prefix k >= k_min (default 10),
compute the 4/5-Rule AIR.  Report the MINIMUM across all valid
prefixes.  This is the standard fair-ranking benchmark metric.

Determinism: seed 20251128.
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent.parent
sys.path.insert(0, str(REPO_ROOT))

BENCH_SEED = 20251128
PREFIX_AIR_KMIN = 10  # ignore prefixes too small for AIR to be stable


def _ndcg(scores_in_order: list[float]) -> float:
    dcg = sum(s / math.log2(i + 2) for i, s in enumerate(scores_in_order))
    ideal = sorted(scores_in_order, reverse=True)
    idcg = sum(s / math.log2(i + 2) for i, s in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0


def _min_prefix_air(groups: list[str], k_min: int = PREFIX_AIR_KMIN) -> float:
    """Minimum 4/5-Rule AIR across all prefixes k >= k_min.

    AIR at prefix k:
        SR_g = (count of group g in prefix k) / (count of group g overall)
        AIR  = min_g SR_g / max_g SR_g
    """
    n = len(groups)
    n_p_total = sum(1 for g in groups if g == "P")
    n_n_total = n - n_p_total
    if n_p_total == 0 or n_n_total == 0:
        return 1.0
    cum_p = 0
    min_air = 1.0
    for k, g in enumerate(groups, start=1):
        if g == "P":
            cum_p += 1
        if k < k_min:
            continue
        sr_p = cum_p / n_p_total
        sr_n = (k - cum_p) / n_n_total
        if max(sr_p, sr_n) == 0:
            continue
        air = min(sr_p, sr_n) / max(sr_p, sr_n)
        if air < min_air:
            min_air = air
    return min_air


def _max_promotion_gap(
    reranked_ids: list[str],
    id_to_score: dict[str, float],
    score_order_ids: list[str],
) -> float:
    """Worst-case promotion gap: the largest (displaced_score - inserted_score)
    where a candidate is moved up past a higher-scoring candidate.
    """
    max_gap = 0.0
    score_at_position = [id_to_score[i] for i in score_order_ids]
    for new_pos, item_id in enumerate(reranked_ids):
        s_here = id_to_score[item_id]
        s_natural = score_at_position[new_pos]
        if s_natural - s_here > max_gap:
            max_gap = s_natural - s_here
    return float(max_gap)


def _gen_population(n, p_protected, mu_protected, rng):
    items = []
    for i in range(n):
        is_p = rng.random() < p_protected
        mu = mu_protected if is_p else 0.5
        score = rng.gauss(mu, 1.0)
        items.append((f"i{i:05d}", "P" if is_p else "N", score))
    items.sort(key=lambda t: -t[2])
    return items


def _run_fair(items, p_protected, alpha):
    from benchmarks.fair_ranking.fair_baseline import FairItem, fair_rerank
    fitems = [
        FairItem(id=i, score=s, is_protected=(g == "P"))
        for (i, g, s) in items
    ]
    t0 = time.time()
    ranked, audit = fair_rerank(fitems, p=p_protected, alpha=alpha)
    elapsed = time.time() - t0
    return ranked, audit, elapsed


def _run_fcr(items, threshold):
    from ranking.fairness_ranker import (
        FairnessConstrainedRanker, RankedCandidate,
    )
    cands = [
        RankedCandidate(name=i, score=s, group=g, original_rank=r)
        for r, (i, g, s) in enumerate(items)
    ]
    t0 = time.time()
    ranker = FairnessConstrainedRanker(
        threshold=threshold, quality_threshold=0.0,
    )
    report = ranker.rerank(cands, min_group_size=2, _compute_pareto=False)
    elapsed = time.time() - t0
    return report, elapsed


def _evaluate_ranking(reranked_ids, id_to_group, id_to_score, original_ids):
    groups = [id_to_group[i] for i in reranked_ids]
    scores = [id_to_score[i] for i in reranked_ids]
    return {
        "min_prefix_air":   round(_min_prefix_air(groups), 4),
        "ndcg":             round(_ndcg(scores), 4),
        "max_promotion_gap":round(_max_promotion_gap(
            reranked_ids, id_to_score, original_ids), 4),
    }


def _sweep_one_condition(n, p_protected, mu_protected, seed):
    import random
    rng = random.Random(seed)
    items = _gen_population(n, p_protected, mu_protected, rng)
    id_to_group = {i: g for (i, g, _) in items}
    id_to_score = {i: s for (i, _, s) in items}
    original_ids = [i for (i, _, _) in items]

    baseline_groups = [id_to_group[i] for i in original_ids]
    baseline = {
        "min_prefix_air":   round(_min_prefix_air(baseline_groups), 4),
        "ndcg":             1.0,
        "max_promotion_gap":0.0,
    }

    fair_pareto = []
    for alpha in (0.01, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50):
        ranked, audit, t = _run_fair(items, p_protected, alpha)
        ids = [it.id for it in ranked]
        m = _evaluate_ranking(ids, id_to_group, id_to_score, original_ids)
        m["alpha"] = alpha
        m["within_group"] = bool(audit["within_group_order_preserved"])
        m["wall_s"] = round(t, 4)
        fair_pareto.append(m)

    fcr_pareto = []
    for threshold in (0.6, 0.7, 0.8, 0.9, 0.95):
        report, t = _run_fcr(items, threshold)
        ids = list(report.fair_ranking)
        m = _evaluate_ranking(ids, id_to_group, id_to_score, original_ids)
        m["threshold"] = threshold
        m["within_group"] = bool(report.within_group_order_preserved)
        m["wall_s"] = round(t, 4)
        fcr_pareto.append(m)

    return {
        "condition": {
            "n": n, "p_protected": p_protected,
            "mu_protected": mu_protected, "seed": seed,
        },
        "baseline_unfair":  baseline,
        "fair_star_ir":     fair_pareto,
        "faimr_fcr":        fcr_pareto,
    }


def _matched_air_comparison(
    fair_points: list[dict], fcr_points: list[dict],
    air_targets: list[float] = (0.50, 0.60, 0.70, 0.80, 0.90),
) -> list[dict]:
    """For each target min-prefix-AIR T, find the best-NDCG point on
    each algorithm's Pareto curve with min_prefix_air >= T.  Returns
    a per-target comparison: (T, fair_ndcg_at_T, fcr_ndcg_at_T, delta).
    """
    out = []
    for T in air_targets:
        eligible_fair = [p for p in fair_points if p["min_prefix_air"] >= T]
        eligible_fcr  = [p for p in fcr_points if p["min_prefix_air"] >= T]
        fair_ndcg = max((p["ndcg"] for p in eligible_fair), default=None)
        fcr_ndcg  = max((p["ndcg"] for p in eligible_fcr),  default=None)
        out.append({
            "air_target": T,
            "fair_star_ir_ndcg": fair_ndcg,
            "faimr_fcr_ndcg":    fcr_ndcg,
            "delta":             None if (fair_ndcg is None or fcr_ndcg is None)
                                  else round(fcr_ndcg - fair_ndcg, 4),
            "fair_eligible":     len(eligible_fair),
            "fcr_eligible":      len(eligible_fcr),
        })
    return out


def main() -> int:
    print(f"# Fair-ranking head-to-head: FA*IR vs FAIMR FCR "
          f"(seed={BENCH_SEED})")
    print(f"# Honest comparison: best-NDCG at matched min-prefix-AIR target.")
    print()

    conditions = [
        (100,  0.3, 0.0),
        (100,  0.3, -0.5),
        (100,  0.2, -0.5),
        (500,  0.3, 0.0),
        (500,  0.3, -0.5),
        (500,  0.2, -0.5),
        (1000, 0.3, -0.5),
        (1000, 0.2, -0.5),
    ]

    all_results = []
    matched_aggregate = {0.50: [], 0.60: [], 0.70: [], 0.80: [], 0.90: []}
    fcr_wins_total = {T: 0 for T in matched_aggregate}
    fair_wins_total = {T: 0 for T in matched_aggregate}
    tie_total = {T: 0 for T in matched_aggregate}

    for cond in conditions:
        n, p, mu = cond
        r = _sweep_one_condition(n, p, mu, BENCH_SEED)
        matched = _matched_air_comparison(
            r["fair_star_ir"], r["faimr_fcr"],
        )
        r["matched_air_comparison"] = matched
        all_results.append(r)
        print(f"### n={n}  p={p}  mu_protected={mu:+.1f}  "
              f"baseline-AIR={r['baseline_unfair']['min_prefix_air']:.3f}")
        for row in matched:
            T = row["air_target"]
            f = row["fair_star_ir_ndcg"]
            c = row["faimr_fcr_ndcg"]
            d = row["delta"]
            f_str = f"{f:.4f}" if f is not None else "-----"
            c_str = f"{c:.4f}" if c is not None else "-----"
            d_str = f"{d:+.4f}" if d is not None else "-----"
            print(f"  AIR>={T:.2f}: FA*IR-NDCG={f_str}  "
                  f"FCR-NDCG={c_str}  delta={d_str}")
            if d is not None:
                matched_aggregate[T].append(d)
                if d > 0:
                    fcr_wins_total[T] += 1
                elif d < 0:
                    fair_wins_total[T] += 1
                else:
                    tie_total[T] += 1
        print()

    print("## Headline -- averaged NDCG delta (FCR - FA*IR) per AIR target")
    for T, deltas in matched_aggregate.items():
        if deltas:
            mean_d = sum(deltas) / len(deltas)
            print(f"  AIR>={T:.2f}: mean delta = {mean_d:+.4f}  "
                  f"FCR wins: {fcr_wins_total[T]}, "
                  f"FA*IR wins: {fair_wins_total[T]}, "
                  f"tie: {tie_total[T]}  (n={len(deltas)})")

    summary = {
        "seed":           BENCH_SEED,
        "prefix_kmin":    PREFIX_AIR_KMIN,
        "conditions":     all_results,
        "matched_air_aggregate": {
            f"air_target_{int(T*100):03d}": {
                "mean_delta_fcr_minus_fair":
                    round(sum(ds) / len(ds), 4) if ds else None,
                "fcr_wins": fcr_wins_total[T],
                "fair_wins": fair_wins_total[T],
                "tie":      tie_total[T],
                "n":        len(ds),
            }
            for T, ds in matched_aggregate.items()
        },
    }
    out = ROOT / "results.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print()
    print(f"Wrote {out.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
