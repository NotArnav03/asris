"""
FAIMR — Standalone Fairness-Constrained Re-ranking (FCR)
Implements the greedy swap algorithm that enforces AIR >= 0.8
while minimizing rank displacement. Returns all metrics needed
for the paper: AIR before/after, mean displacement, NDCG before/after.
"""

import numpy as np
from evaluation.metrics import ndcg_at_k


def compute_air(labels_selected, groups_selected, labels_all, groups_all):
    unique_groups = set(g for g in groups_all if g != "unknown")
    if len(unique_groups) < 2:
        return 1.0, {}

    total_per_group = {}
    selected_per_group = {}
    for g in unique_groups:
        total_per_group[g] = sum(1 for gi in groups_all if gi == g)
        selected_per_group[g] = sum(1 for gi in groups_selected if gi == g)

    rates = {}
    for g in unique_groups:
        rates[g] = selected_per_group[g] / total_per_group[g] if total_per_group[g] > 0 else 0.0

    if max(rates.values()) == 0:
        return 1.0, rates

    air = min(rates.values()) / max(rates.values())
    return air, rates


def fcr_rerank(scores, groups, k, threshold=0.8):
    n = len(scores)
    indices = np.argsort(scores)[::-1].tolist()
    original_order = list(indices)

    max_iterations = n * 2
    for _ in range(max_iterations):
        top_k_groups = [groups[indices[i]] for i in range(k)]
        all_groups = [groups[i] for i in indices]
        air, _ = compute_air(None, top_k_groups, None, all_groups)

        if air >= threshold:
            break

        unique_groups = set(g for g in all_groups if g != "unknown")
        total_per_group = {}
        selected_per_group = {}
        for g in unique_groups:
            total_per_group[g] = sum(1 for gi in all_groups if gi == g)
            selected_per_group[g] = sum(1 for gi in top_k_groups if gi == g)

        rates = {}
        for g in unique_groups:
            rates[g] = selected_per_group[g] / total_per_group[g] if total_per_group[g] > 0 else 0.0

        underrep = min(rates, key=rates.get)
        overrep = max(rates, key=rates.get)
        if underrep == overrep:
            break

        best_underrep_pos = None
        best_underrep_score = -np.inf
        for pos in range(k, n):
            idx = indices[pos]
            if groups[idx] == underrep and scores[idx] > best_underrep_score:
                best_underrep_score = scores[idx]
                best_underrep_pos = pos

        worst_overrep_pos = None
        worst_overrep_score = np.inf
        for pos in range(k - 1, -1, -1):
            idx = indices[pos]
            if groups[idx] == overrep and scores[idx] < worst_overrep_score:
                worst_overrep_score = scores[idx]
                worst_overrep_pos = pos

        if best_underrep_pos is None or worst_overrep_pos is None:
            break

        indices[best_underrep_pos], indices[worst_overrep_pos] = (
            indices[worst_overrep_pos], indices[best_underrep_pos]
        )

    fair_order = list(indices)

    orig_rank = {idx: pos for pos, idx in enumerate(original_order)}
    fair_rank = {idx: pos for pos, idx in enumerate(fair_order)}
    displacements = [abs(orig_rank[idx] - fair_rank[idx]) for idx in range(n)]
    mean_displacement = np.mean(displacements)

    top_k_groups_after = [groups[fair_order[i]] for i in range(k)]
    all_groups = [groups[i] for i in fair_order]
    air_after, rates_after = compute_air(None, top_k_groups_after, None, all_groups)

    top_k_groups_before = [groups[original_order[i]] for i in range(k)]
    air_before, rates_before = compute_air(None, top_k_groups_before, None, all_groups)

    return {
        "fair_order": fair_order,
        "original_order": original_order,
        "air_before": round(air_before, 4),
        "air_after": round(air_after, 4),
        "mean_displacement": round(mean_displacement, 4),
        "max_displacement": max(displacements),
        "rates_before": rates_before,
        "rates_after": rates_after,
    }


def evaluate_fcr_ndcg(y_true, scores, groups, k=5, threshold=0.8):
    y_true = np.array(y_true)
    scores = np.array(scores)

    ndcg_before = ndcg_at_k(list(y_true), list(scores), k)

    result = fcr_rerank(scores, groups, k, threshold)
    fair_order = result["fair_order"]

    fair_scores = np.zeros_like(scores)
    for new_pos, orig_idx in enumerate(fair_order):
        fair_scores[orig_idx] = len(fair_order) - new_pos

    ndcg_after = ndcg_at_k(list(y_true), list(fair_scores), k)

    result["ndcg_before"] = round(ndcg_before, 4)
    result["ndcg_after"] = round(ndcg_after, 4)
    result["ndcg_delta"] = round(ndcg_after - ndcg_before, 4)

    return result


if __name__ == "__main__":
    np.random.seed(42)
    n = 50
    scores = np.random.rand(n)
    groups = np.random.choice(["male", "female"], size=n, p=[0.7, 0.3])
    y_true = (scores > 0.5).astype(int)

    result = evaluate_fcr_ndcg(y_true, scores, groups, k=10)

    print("=" * 55)
    print("  FCR EVALUATION DEMO")
    print("=" * 55)
    print(f"  AIR before: {result['air_before']}")
    print(f"  AIR after:  {result['air_after']}")
    print(f"  Mean displacement: {result['mean_displacement']}")
    print(f"  Max displacement:  {result['max_displacement']}")
    print(f"  NDCG@5 before: {result['ndcg_before']}")
    print(f"  NDCG@5 after:  {result['ndcg_after']}")
    print(f"  NDCG delta:    {result['ndcg_delta']}")
    print("=" * 55)
