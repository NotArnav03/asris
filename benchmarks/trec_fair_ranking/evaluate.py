"""
FAIMR -- TREC Fair Ranking 2022 head-to-head evaluator.

Compares FAIMR's constrained-insertion FCR against FA*IR (Zehlike
2017) on the TREC 2022 Fair Ranking training set (50 Wikimedia
WikiProject queries, ~597k relevant docs with gender attribute).

Protocol:
  1. For each of the 50 queries:
     a. Retrieve the query's relevant documents (per qrels).
     b. Filter to docs with a clean binary gender label (~25% of pool).
     c. Sort by pred_qual descending -- this is the "unfair" baseline.
     d. Sweep FA*IR alpha and FAIMR FCR threshold, re-rank each.
  2. For every (algorithm, parameter) point, compute:
       * NDCG vs the score-optimal baseline
       * AWRF (attention-weighted rank fairness) with decay=0.85
       * Combined TREC metric M = NDCG * AWRF
       * Min-prefix-AIR (auxiliary fairness metric)
  3. Build the (NDCG, AWRF) Pareto front per algorithm and report
     the algorithm dominance across queries.

AWRF formulation (Diaz et al. 2020-style):
   For each rank i in [1, N]:
       p_i = 0.15 * 0.85^(i-1)        (geometric attention)
   For each group g in {"female", "male"}:
       exposure_g = sum(p_i for positions where doc_i in g)
       norm_g     = exposure_g / sum(p_i)
   target_g = 0.5 (binary parity target)
   AWRF = 1 - 0.5 * sum_g |norm_g - target_g|     in [0, 1]

The combined TREC track metric is M = NDCG * AWRF (Ekstrand 2022).
Higher is better on both axes.

Determinism: random sampling and ranking are deterministic given
the fixed pred_qual order; no RNG is used.
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


def _binary_gender(gender_list) -> str | None:
    """Reduce a Wikidata gender list to {"female", "male", None}."""
    if not gender_list:
        return None
    have_f = any("female" in str(g).lower() for g in gender_list)
    have_m = any(
        "male" in str(g).lower() and "female" not in str(g).lower()
        for g in gender_list
    )
    if have_f and not have_m:
        return "female"
    if have_m and not have_f:
        return "male"
    if have_f and have_m:
        return None  # ambiguous, skip
    return None


def _ndcg(scores_in_emit_order: list[float]) -> float:
    dcg = sum(s / math.log2(i + 2) for i, s in enumerate(scores_in_emit_order))
    ideal = sorted(scores_in_emit_order, reverse=True)
    idcg = sum(s / math.log2(i + 2) for i, s in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0


def _awrf(
    groups: list[str], target: dict[str, float], decay: float = 0.85,
) -> float:
    """Attention-Weighted Rank Fairness in [0, 1].  Higher = fairer."""
    n = len(groups)
    if n == 0:
        return 1.0
    weights = [(1 - decay) * (decay ** i) for i in range(n)]
    z = sum(weights)
    if z == 0:
        return 1.0
    actual = {g: 0.0 for g in target}
    for g, w in zip(groups, weights):
        if g in actual:
            actual[g] += w
    actual = {g: v / z for g, v in actual.items()}
    l1 = sum(abs(actual.get(g, 0.0) - target[g]) for g in target)
    return max(0.0, 1.0 - 0.5 * l1)


def _min_prefix_air(
    groups: list[str], protected: str, k_min: int = 10,
) -> float:
    """Same prefix-AIR formulation as benchmarks/fair_ranking/."""
    n = len(groups)
    n_p_total = sum(1 for g in groups if g == protected)
    n_n_total = n - n_p_total
    if n_p_total == 0 or n_n_total == 0:
        return 1.0
    cum_p = 0
    min_air = 1.0
    for k, g in enumerate(groups, start=1):
        if g == protected:
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


def _run_fair(items, p_protected, alpha):
    from benchmarks.fair_ranking.fair_baseline import FairItem, fair_rerank
    fitems = [
        FairItem(id=str(i["doc_id"]), score=float(i["score"]),
                 is_protected=(i["gender"] == "female"))
        for i in items
    ]
    ranked, audit = fair_rerank(fitems, p=p_protected, alpha=alpha)
    return [it.id for it in ranked]


def _run_fcr(items, threshold):
    from ranking.fairness_ranker import (
        FairnessConstrainedRanker, RankedCandidate,
    )
    cands = [
        RankedCandidate(
            name=str(it["doc_id"]), score=float(it["score"]),
            group=str(it["gender"]), original_rank=r,
        )
        for r, it in enumerate(items)
    ]
    ranker = FairnessConstrainedRanker(
        threshold=threshold, quality_threshold=0.0,
    )
    report = ranker.rerank(cands, min_group_size=2, _compute_pareto=False)
    return list(report.fair_ranking)


def evaluate_query(
    rel_docs: list[int], docs_meta: dict, top_k: int | None = None,
) -> dict | None:
    """Run all algorithm * parameter sweeps for one query.  Returns
    None if the query has too few docs with binary gender."""
    items = []
    for d_id in rel_docs:
        meta = docs_meta.get(d_id)
        if meta is None:
            continue
        gender = _binary_gender(meta.get("gender", []))
        if gender is None:
            continue
        score = meta.get("pred_qual", 0.0)
        if score is None:
            continue
        items.append({
            "doc_id": d_id,
            "score":  float(score),
            "gender": gender,
        })
    # Need both groups represented and a minimum list size.
    n_f = sum(1 for x in items if x["gender"] == "female")
    n_m = sum(1 for x in items if x["gender"] == "male")
    if n_f < 5 or n_m < 5 or len(items) < 30:
        return None

    items.sort(key=lambda x: -x["score"])
    if top_k:
        items = items[:top_k]
    id_to_score = {it["doc_id"]: it["score"] for it in items}
    id_to_gender = {it["doc_id"]: it["gender"] for it in items}
    score_order_ids = [it["doc_id"] for it in items]
    p_protected = n_f / (n_f + n_m)

    target = {"female": 0.5, "male": 0.5}

    def metrics_for_ranking(ranked_ids: list[str | int]) -> dict:
        ranked_ids = [int(x) for x in ranked_ids]
        groups = [id_to_gender[i] for i in ranked_ids]
        scores = [id_to_score[i] for i in ranked_ids]
        return {
            "ndcg":            round(_ndcg(scores), 4),
            "awrf":            round(_awrf(groups, target), 4),
            "min_prefix_air":  round(_min_prefix_air(groups, "female"), 4),
        }

    baseline_metrics = metrics_for_ranking(score_order_ids)
    baseline_metrics["m1_ndcg_x_awrf"] = round(
        baseline_metrics["ndcg"] * baseline_metrics["awrf"], 4,
    )

    fair_pareto = []
    for alpha in (0.05, 0.10, 0.20, 0.30, 0.40, 0.50):
        try:
            r = _run_fair(items, p_protected, alpha)
            m = metrics_for_ranking(r)
            m["m1_ndcg_x_awrf"] = round(m["ndcg"] * m["awrf"], 4)
            m["alpha"] = alpha
            fair_pareto.append(m)
        except Exception as e:
            fair_pareto.append({"alpha": alpha, "error": str(e)})

    fcr_pareto = []
    for threshold in (0.6, 0.7, 0.8, 0.9, 0.95, 0.99):
        try:
            r = _run_fcr(items, threshold)
            m = metrics_for_ranking(r)
            m["m1_ndcg_x_awrf"] = round(m["ndcg"] * m["awrf"], 4)
            m["threshold"] = threshold
            fcr_pareto.append(m)
        except Exception as e:
            fcr_pareto.append({"threshold": threshold, "error": str(e)})

    return {
        "n_docs":         len(items),
        "n_female":       n_f,
        "n_male":         n_m,
        "p_protected":    round(p_protected, 4),
        "baseline":       baseline_metrics,
        "fair_star_ir":   fair_pareto,
        "faimr_fcr":      fcr_pareto,
    }


def main() -> int:
    from benchmarks.trec_fair_ranking.load import load_queries, build_doc_lookup

    print("# TREC Fair Ranking 2022 head-to-head: FA*IR vs FAIMR FCR")
    print()
    print("Loading queries + relevant doc IDs ...")
    queries = load_queries()
    print(f"  {len(queries)} queries")
    needed = {d for q in queries for d in q["rel_docs"]}
    print(f"  {len(needed)} unique relevant docs across all queries")

    t0 = time.time()
    print("Loading per-doc metadata ...")
    docs_meta = build_doc_lookup(needed_ids=needed)
    print(f"  loaded {len(docs_meta)} doc metadata records "
          f"in {time.time() - t0:.1f}s")
    print()

    # Cap each query's working set so the benchmark is tractable.
    # 500 docs per query is plenty for the AWRF / NDCG comparison and
    # avoids letting any single huge query dominate the aggregate.
    TOP_K = 500
    print(f"Per-query top-K cap: {TOP_K} (by pred_qual)")
    print()

    results = []
    for q in queries:
        rep = evaluate_query(q["rel_docs"], docs_meta, top_k=TOP_K)
        if rep is None:
            print(f"  skip query {q['query_id']:>4} ({q['title'][:30]}): "
                  f"too few clean-gender docs")
            continue
        rep["query_id"] = q["query_id"]
        rep["title"]    = q["title"]
        results.append(rep)

    print()
    print(f"Evaluated {len(results)} / {len(queries)} queries")
    print()

    # ----------------------------------------------------------
    # Per-query head-to-head: best FA*IR vs best FCR at matched AWRF
    # ----------------------------------------------------------
    def matched_awrf(fair_pts, fcr_pts, targets=(0.70, 0.80, 0.85, 0.90, 0.95)):
        out = []
        for T in targets:
            f_eligible = [p for p in fair_pts
                          if "error" not in p and p.get("awrf", 0) >= T]
            c_eligible = [p for p in fcr_pts
                          if "error" not in p and p.get("awrf", 0) >= T]
            f_best = max((p["m1_ndcg_x_awrf"] for p in f_eligible), default=None)
            c_best = max((p["m1_ndcg_x_awrf"] for p in c_eligible), default=None)
            out.append({
                "awrf_target": T,
                "fair_star_ir_m1": f_best,
                "faimr_fcr_m1":    c_best,
                "delta":           None if (f_best is None or c_best is None)
                                    else round(c_best - f_best, 4),
            })
        return out

    aggregates = {T: [] for T in (0.70, 0.80, 0.85, 0.90, 0.95)}
    fcr_wins  = {T: 0 for T in aggregates}
    fair_wins = {T: 0 for T in aggregates}
    eligible_fcr = {T: 0 for T in aggregates}
    eligible_fair = {T: 0 for T in aggregates}

    # Also track auxiliary metric: min-prefix-AIR (legal 4/5 rule).
    # FCR optimises for this directly; FA*IR does not.
    air_targets = (0.50, 0.60, 0.70, 0.80)
    air_aggregates = {T: {"fcr": [], "fair": []} for T in air_targets}

    for r in results:
        match = matched_awrf(r["fair_star_ir"], r["faimr_fcr"])
        r["matched_awrf"] = match
        for row in match:
            T = row["awrf_target"]
            if row["fair_star_ir_m1"] is not None:
                eligible_fair[T] += 1
            if row["faimr_fcr_m1"] is not None:
                eligible_fcr[T] += 1
            if row["delta"] is not None:
                aggregates[T].append(row["delta"])
                if row["delta"] > 0:
                    fcr_wins[T] += 1
                elif row["delta"] < 0:
                    fair_wins[T] += 1

        # Auxiliary: best achievable min-prefix-AIR per algorithm
        fair_best_air = max(
            (p.get("min_prefix_air", 0) for p in r["fair_star_ir"]
             if "error" not in p), default=0,
        )
        fcr_best_air = max(
            (p.get("min_prefix_air", 0) for p in r["faimr_fcr"]
             if "error" not in p), default=0,
        )
        for T in air_targets:
            air_aggregates[T]["fcr"].append(int(fcr_best_air >= T))
            air_aggregates[T]["fair"].append(int(fair_best_air >= T))

    print("## Aggregate -- M1 = NDCG * AWRF, paired across queries")
    print(f"{'AWRF target':<14}  {'FCR-elig':>9}  {'FA*IR-elig':>11}  "
          f"{'mean dM1':>9}  {'FCR wins':>9}  {'FA*IR wins':>11}")
    for T, deltas in aggregates.items():
        if deltas:
            mean_d = sum(deltas) / len(deltas)
            print(f"  >= {T:.2f}      {eligible_fcr[T]:>9}  "
                  f"{eligible_fair[T]:>11}  {mean_d:>+9.4f}  "
                  f"{fcr_wins[T]:>9}  {fair_wins[T]:>11}")
        else:
            print(f"  >= {T:.2f}      {eligible_fcr[T]:>9}  "
                  f"{eligible_fair[T]:>11}  {'n/a':>9}  "
                  f"{fcr_wins[T]:>9}  {fair_wins[T]:>11}")

    print()
    print("## Auxiliary -- min-prefix-AIR (legal 4/5-Rule metric)")
    print(f"{'AIR target':<14}  {'FCR reaches':>13}  {'FA*IR reaches':>15}")
    air_capability = {}
    for T in air_targets:
        fcr_hits = sum(air_aggregates[T]["fcr"])
        fair_hits = sum(air_aggregates[T]["fair"])
        n = len(results)
        print(f"  >= {T:.2f}      {fcr_hits:>3}/{n}             "
              f"{fair_hits:>3}/{n}")
        air_capability[f"air_target_{int(T*100):03d}"] = {
            "fcr_reaches":  fcr_hits,
            "fair_reaches": fair_hits,
            "n_queries":    n,
        }

    summary = {
        "track":             "TREC Fair Ranking 2022 (train split)",
        "n_queries":         len(queries),
        "n_evaluated":       len(results),
        "top_k_per_query":   TOP_K,
        "metric":            "M1 = NDCG * AWRF (Ekstrand 2022)",
        "awrf_decay":        0.85,
        "per_query":         results,
        "auxiliary_min_prefix_air_capability": air_capability,
        "aggregate_matched_awrf": {
            f"awrf_target_{int(T*100):03d}": {
                "mean_delta_fcr_minus_fair":
                    round(sum(ds)/len(ds), 4) if ds else None,
                "fcr_wins":      fcr_wins[T],
                "fair_wins":     fair_wins[T],
                "fcr_eligible":  eligible_fcr[T],
                "fair_eligible": eligible_fair[T],
                "n":             len(ds),
            }
            for T, ds in aggregates.items()
        },
    }
    out_path = ROOT / "results.json"
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print()
    print(f"Wrote {out_path.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
