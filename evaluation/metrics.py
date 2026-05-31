"""
FAIMR — Evaluation Metrics
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

def _check_aligned(y_true, y_scores) -> None:
    """Common preconditions for all ranking metrics.

    Raises ValueError when lengths differ — silently truncating to the
    shorter list, as numpy would do, hides off-by-one bugs in caller
    code.  Audit-critical metric functions must be strict.
    """
    if len(y_true) != len(y_scores):
        raise ValueError(
            f"y_true (len={len(y_true)}) and y_scores "
            f"(len={len(y_scores)}) must be the same length"
        )


def precision_at_k(y_true: list, y_scores: list, k: int) -> float:
    """Precision@K: fraction of top-K results that are relevant.

    Assumes BINARY relevance (y_true entries 0 or 1).  Non-binary input
    is summed as graded weight, which gives meaningful values for
    graded relevance but the metric is no longer strictly "precision";
    callers requesting graded support should use NDCG instead.
    """
    _check_aligned(y_true, y_scores)
    if k <= 0 or len(y_true) == 0:
        return 0.0

    sorted_indices = np.argsort(y_scores)[::-1][:k]
    relevant = sum(y_true[i] for i in sorted_indices)
    return relevant / k


def recall_at_k(y_true: list, y_scores: list, k: int) -> float:
    """Recall@K: fraction of all relevant items found in top-K."""
    _check_aligned(y_true, y_scores)
    total_relevant = sum(y_true)
    if total_relevant == 0 or k <= 0:
        return 0.0

    sorted_indices = np.argsort(y_scores)[::-1][:k]
    found_relevant = sum(y_true[i] for i in sorted_indices)
    return found_relevant / total_relevant


def ndcg_at_k(
    y_true: list, y_scores: list, k: int,
    gain: str = "linear",
) -> float:
    """Normalized Discounted Cumulative Gain @ K.

    Args:
        y_true: relevance labels (binary or graded integer)
        y_scores: predicted scores
        k: cutoff rank
        gain: "linear" (DCG numerator = rel) or "exponential"
              (DCG numerator = 2^rel - 1).  Linear is fine for binary
              relevance (the two formulas agree on {0,1}); for graded
              relevance the exponential form is the canonical Burges /
              LambdaMART definition and should be used.

    DCG@K = Σ_{rank=0..K-1} g(rel_rank) / log2(rank + 2)
    NDCG@K = DCG@K / IDCG@K

    Returns 0.0 when k <= 0, when y_true is empty, or when the ideal
    DCG is zero (i.e. no relevant items at all).
    """
    _check_aligned(y_true, y_scores)
    if k <= 0 or len(y_true) == 0:
        return 0.0
    if gain not in ("linear", "exponential"):
        raise ValueError(
            f"gain must be 'linear' or 'exponential', got {gain!r}"
        )

    def _g(rel):
        return rel if gain == "linear" else (2 ** rel - 1)

    sorted_indices = np.argsort(y_scores)[::-1][:k]

    dcg = 0.0
    for rank, idx in enumerate(sorted_indices):
        dcg += _g(y_true[idx]) / np.log2(rank + 2)  # rank+2 because log2(1)=0

    ideal_sorted = sorted(y_true, reverse=True)[:k]
    idcg = 0.0
    for rank, rel in enumerate(ideal_sorted):
        idcg += _g(rel) / np.log2(rank + 2)

    if idcg == 0:
        return 0.0
    return dcg / idcg


def mean_reciprocal_rank(y_true: list, y_scores: list) -> float:
    """MRR: 1 / rank of the first relevant result.

    Treats y_true[i] >= 1 as relevant (not strict equality to 1) so
    graded labels work correctly.  Returns 0.0 when no relevant
    items are present.
    """
    _check_aligned(y_true, y_scores)
    sorted_indices = np.argsort(y_scores)[::-1]

    for rank, idx in enumerate(sorted_indices):
        if y_true[idx] >= 1:
            return 1.0 / (rank + 1)
    return 0.0


def average_precision(y_true: list, y_scores: list) -> float:
    """Average Precision: area under the precision-recall curve.

    Assumes BINARY relevance.  Items with y_true[i] >= 1 count as
    positives; the denominator is the count of positives.
    """
    _check_aligned(y_true, y_scores)
    sorted_indices = np.argsort(y_scores)[::-1]
    relevant_count = 0
    precision_sum = 0.0

    for rank, idx in enumerate(sorted_indices):
        if y_true[idx] >= 1:
            relevant_count += 1
            precision_sum += relevant_count / (rank + 1)

    total_relevant = sum(1 for v in y_true if v >= 1)
    if total_relevant == 0:
        return 0.0
    return precision_sum / total_relevant


# ─── Classification Metrics ──────────────────────────────────────

def compute_roc_auc(y_true: list, y_scores: list) -> Optional[float]:
    """Compute ROC-AUC score.

    Returns None when sklearn refuses (typically: only one class
    present in y_true).  The prior version silently returned 0.0
    in that case, which is a real score in the [0, 1] range and
    therefore indistinguishable from "the model is anti-correlated
    with truth".  None is the honest answer.
    """
    _check_aligned(y_true, y_scores)
    from sklearn.metrics import roc_auc_score
    import warnings
    try:
        # sklearn now warns + returns NaN for single-class instead of
        # raising; suppress the warning since we surface None ourselves.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            val = float(roc_auc_score(y_true, y_scores))
        # NaN -> undefined (single class present or other degeneracy).
        if val != val:  # NaN check
            return None
        return val
    except ValueError as e:
        logger.debug("ROC-AUC undefined: %s", e)
        return None


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

    def __init__(self, k_values: Optional[list] = None):
        self.k_values = k_values or EVAL_TOP_K_VALUES
        self.query_results: dict = {}

    def add_query(
        self,
        query_id: str,
        y_true: list,
        y_scores: list,
    ):
        """Add results for a single query.

        Enforces equal-length inputs and a non-empty list — silently
        accepting mismatched lengths would let off-by-one errors in
        caller code propagate as quietly-wrong metrics.
        """
        if len(y_true) != len(y_scores):
            raise ValueError(
                f"add_query({query_id!r}): y_true ({len(y_true)}) "
                f"and y_scores ({len(y_scores)}) must align"
            )
        if not y_true:
            raise ValueError(
                f"add_query({query_id!r}): empty input rejected"
            )
        self.query_results[query_id] = {
            "y_true":   list(y_true),
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

        # Compute means.  Annotated as dict[str, Any] because the
        # ROC-AUC keys downstream may be None when AUC is undefined
        # (only one class present) — mypy otherwise infers
        # dict[str, float] and refuses the None assignment.
        mean_metrics: dict = {
            metric: round(float(np.mean(values)), 4)
            for metric, values in aggregated.items()
        }

        # MAP = mean of per-query Average Precisions.
        if "AP" in mean_metrics:
            mean_metrics["MAP"] = mean_metrics["AP"]

        # ROC-AUC: report BOTH the flat (concatenated) score and the
        # mean of per-query scores.  Flat ROC-AUC is sensitive to
        # cross-query score-scale differences (scores not comparable
        # across queries); per-query mean is the correct "average
        # ranking quality per query" measurement.
        all_y_true: list = []
        all_y_scores: list = []
        per_query_aucs: list = []
        for data in self.query_results.values():
            all_y_true.extend(data["y_true"])
            all_y_scores.extend(data["y_scores"])
            auc_q = compute_roc_auc(data["y_true"], data["y_scores"])
            if auc_q is not None:
                per_query_aucs.append(auc_q)

        flat_auc = compute_roc_auc(all_y_true, all_y_scores)
        mean_metrics["ROC-AUC_flat"] = (
            None if flat_auc is None else round(float(flat_auc), 4)
        )
        mean_metrics["ROC-AUC_mean_per_query"] = (
            round(float(np.mean(per_query_aucs)), 4) if per_query_aucs else None
        )
        # Legacy key retained for backwards compat: defaults to flat.
        mean_metrics["ROC-AUC"] = mean_metrics["ROC-AUC_flat"]

        return {
            "aggregate":   mean_metrics,
            "num_queries": len(self.query_results),
            "per_query":   per_query,
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
    # compute_roc_auc returns Optional[float] (None on undefined input).
    # Preserve None in the report rather than forcing it to a numeric.
    _auc = compute_roc_auc(y_true, y_scores)
    results["ROC-AUC"] = None if _auc is None else round(_auc, 4)

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
