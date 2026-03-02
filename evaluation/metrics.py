"""
ASRIS — Evaluation Metrics
Comprehensive ranking evaluation: P@K, R@K, NDCG, MRR, MAP, ROC-AUC,
with per-query and aggregate reporting.
"""

import numpy as np
from typing import Optional
from collections import defaultdict

import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))
from config import EVAL_TOP_K_VALUES, get_logger

logger = get_logger("evaluation.metrics")


# ─── Core Ranking Metrics ────────────────────────────────────────

def precision_at_k(y_true: list[int], y_scores: list[float], k: int) -> float:
    """
    Precision@K: fraction of top-K results that are relevant.
    """
    if k <= 0 or len(y_true) == 0:
        return 0.0

    sorted_indices = np.argsort(y_scores)[::-1][:k]
    relevant = sum(y_true[i] for i in sorted_indices)
    return relevant / k


def recall_at_k(y_true: list[int], y_scores: list[float], k: int) -> float:
    """
    Recall@K: fraction of all relevant items found in top-K.
    """
    total_relevant = sum(y_true)
    if total_relevant == 0 or k <= 0:
        return 0.0

    sorted_indices = np.argsort(y_scores)[::-1][:k]
    found_relevant = sum(y_true[i] for i in sorted_indices)
    return found_relevant / total_relevant


def ndcg_at_k(y_true: list[int], y_scores: list[float], k: int) -> float:
    """
    Normalized Discounted Cumulative Gain @ K.
    Measures ranking quality — rewards relevant items ranked higher.
    """
    if k <= 0 or len(y_true) == 0:
        return 0.0

    sorted_indices = np.argsort(y_scores)[::-1][:k]

    # DCG
    dcg = 0.0
    for rank, idx in enumerate(sorted_indices):
        rel = y_true[idx]
        dcg += rel / np.log2(rank + 2)  # rank+2 because log2(1)=0

    # Ideal DCG
    ideal_sorted = sorted(y_true, reverse=True)[:k]
    idcg = 0.0
    for rank, rel in enumerate(ideal_sorted):
        idcg += rel / np.log2(rank + 2)

    if idcg == 0:
        return 0.0

    return dcg / idcg


def mean_reciprocal_rank(y_true: list[int], y_scores: list[float]) -> float:
    """
    MRR: 1/rank of the first relevant result.
    """
    sorted_indices = np.argsort(y_scores)[::-1]

    for rank, idx in enumerate(sorted_indices):
        if y_true[idx] == 1:
            return 1.0 / (rank + 1)

    return 0.0


def average_precision(y_true: list[int], y_scores: list[float]) -> float:
    """
    Average Precision: area under the precision-recall curve.
    """
    sorted_indices = np.argsort(y_scores)[::-1]
    relevant_count = 0
    precision_sum = 0.0

    for rank, idx in enumerate(sorted_indices):
        if y_true[idx] == 1:
            relevant_count += 1
            precision_sum += relevant_count / (rank + 1)

    total_relevant = sum(y_true)
    if total_relevant == 0:
        return 0.0

    return precision_sum / total_relevant


# ─── Classification Metrics ──────────────────────────────────────

def compute_roc_auc(y_true: list[int], y_scores: list[float]) -> float:
    """Compute ROC-AUC score."""
    from sklearn.metrics import roc_auc_score
    try:
        return roc_auc_score(y_true, y_scores)
    except ValueError:
        return 0.0


def compute_classification_report(
    y_true: list[int],
    y_pred: list[int],
) -> dict:
    """Generate classification report as a dictionary."""
    from sklearn.metrics import classification_report
    return classification_report(y_true, y_pred, output_dict=True, zero_division=0)


# ─── Per-Query Evaluation ────────────────────────────────────────

class RankingEvaluator:
    """
    Evaluates ranking quality on a per-query basis.

    Expects data grouped by query (e.g., job_id), with each query
    having a list of candidate scores and relevance labels.
    """

    def __init__(self, k_values: Optional[list[int]] = None):
        self.k_values = k_values or EVAL_TOP_K_VALUES
        self.query_results = {}

    def add_query(
        self,
        query_id: str,
        y_true: list[int],
        y_scores: list[float],
    ):
        """Add results for a single query."""
        self.query_results[query_id] = {
            "y_true": list(y_true),
            "y_scores": list(y_scores),
        }

    def compute_all(self) -> dict:
        """
        Compute all metrics across all queries.

        Returns:
            Dict with per-metric averages and a detailed per-query breakdown.
        """
        if not self.query_results:
            logger.warning("No query results to evaluate")
            return {}

        per_query = {}
        aggregated = defaultdict(list)

        for query_id, data in self.query_results.items():
            y_true = data["y_true"]
            y_scores = data["y_scores"]

            query_metrics = {}

            # Ranking metrics at different K
            for k in self.k_values:
                query_metrics[f"P@{k}"] = precision_at_k(y_true, y_scores, k)
                query_metrics[f"R@{k}"] = recall_at_k(y_true, y_scores, k)
                query_metrics[f"NDCG@{k}"] = ndcg_at_k(y_true, y_scores, k)

            query_metrics["MRR"] = mean_reciprocal_rank(y_true, y_scores)
            query_metrics["AP"] = average_precision(y_true, y_scores)

            per_query[query_id] = query_metrics

            for metric, value in query_metrics.items():
                aggregated[metric].append(value)

        # Compute means
        mean_metrics = {
            metric: round(np.mean(values), 4)
            for metric, values in aggregated.items()
        }

        # Compute MAP
        mean_metrics["MAP"] = mean_metrics.pop("AP", 0.0)

        # Overall ROC-AUC (flat)
        all_y_true = []
        all_y_scores = []
        for data in self.query_results.values():
            all_y_true.extend(data["y_true"])
            all_y_scores.extend(data["y_scores"])
        mean_metrics["ROC-AUC"] = round(compute_roc_auc(all_y_true, all_y_scores), 4)

        return {
            "aggregate": mean_metrics,
            "num_queries": len(self.query_results),
            "per_query": per_query,
        }

    def print_report(self, results: Optional[dict] = None):
        """Print a formatted evaluation report."""
        if results is None:
            results = self.compute_all()

        if not results:
            print("No results to report.")
            return

        agg = results["aggregate"]
        n = results["num_queries"]

        print(f"\n{'═' * 55}")
        print(f"  RANKING EVALUATION REPORT  ({n} queries)")
        print(f"{'═' * 55}")

        # Group by metric type
        print(f"\n  {'Metric':<15} {'Score':>10}")
        print(f"  {'─' * 30}")

        for k in self.k_values:
            print(f"  Precision@{k:<4} {agg.get(f'P@{k}', 0):.4f}")

        print()
        for k in self.k_values:
            print(f"  Recall@{k:<7} {agg.get(f'R@{k}', 0):.4f}")

        print()
        for k in self.k_values:
            print(f"  NDCG@{k:<9} {agg.get(f'NDCG@{k}', 0):.4f}")

        print()
        print(f"  {'MRR':<15} {agg.get('MRR', 0):.4f}")
        print(f"  {'MAP':<15} {agg.get('MAP', 0):.4f}")
        print(f"  {'ROC-AUC':<15} {agg.get('ROC-AUC', 0):.4f}")
        print(f"\n{'═' * 55}\n")

    def to_dataframe(self, results: Optional[dict] = None):
        """Convert per-query results to a pandas DataFrame."""
        import pandas as pd
        if results is None:
            results = self.compute_all()
        return pd.DataFrame.from_dict(results["per_query"], orient="index")


# ─── Convenience Function ────────────────────────────────────────

def quick_evaluate(
    y_true: list[int],
    y_scores: list[float],
    k_values: Optional[list[int]] = None,
) -> dict:
    """
    Quick flat evaluation (not per-query).
    Useful for evaluating a single ranked list.
    """
    k_values = k_values or EVAL_TOP_K_VALUES
    results = {}

    for k in k_values:
        results[f"P@{k}"] = round(precision_at_k(y_true, y_scores, k), 4)
        results[f"R@{k}"] = round(recall_at_k(y_true, y_scores, k), 4)
        results[f"NDCG@{k}"] = round(ndcg_at_k(y_true, y_scores, k), 4)

    results["MRR"] = round(mean_reciprocal_rank(y_true, y_scores), 4)
    results["AP"] = round(average_precision(y_true, y_scores), 4)
    results["ROC-AUC"] = round(compute_roc_auc(y_true, y_scores), 4)

    return results


if __name__ == "__main__":
    # Demo with synthetic data
    print("=== Quick Evaluate Demo ===")
    y_true = [1, 0, 1, 0, 1, 0, 0, 1, 0, 0]
    y_scores = [0.9, 0.8, 0.7, 0.65, 0.6, 0.5, 0.45, 0.3, 0.2, 0.1]

    results = quick_evaluate(y_true, y_scores)
    for metric, value in results.items():
        print(f"  {metric}: {value}")

    print("\n=== Per-Query Evaluator Demo ===")
    evaluator = RankingEvaluator()

    evaluator.add_query("job_1", [1, 0, 1, 0, 0], [0.9, 0.7, 0.8, 0.3, 0.1])
    evaluator.add_query("job_2", [0, 1, 0, 1, 0], [0.5, 0.9, 0.4, 0.8, 0.2])
    evaluator.add_query("job_3", [1, 1, 0, 0, 0], [0.95, 0.85, 0.6, 0.3, 0.1])

    results = evaluator.compute_all()
    evaluator.print_report(results)
