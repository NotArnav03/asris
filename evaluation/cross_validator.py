"""
FAIMR — Cross-Validation & Statistical Testing Module
Implements k-fold cross-validation with per-fold metric tracking,
paired t-tests, confidence intervals, and LaTeX-ready results tables.
"""

import numpy as np
from typing import Optional, Callable
from dataclasses import dataclass, field
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import get_logger

logger = get_logger("evaluation.cross_validator")


@dataclass
class FoldResult:
    """Results from a single CV fold."""
    fold_idx: int
    train_size: int
    test_size: int
    metrics: dict = field(default_factory=dict)


@dataclass
class CVResult:
    """Aggregated cross-validation results."""
    model_name: str
    k_folds: int
    fold_results: list[FoldResult] = field(default_factory=list)
    mean_metrics: dict = field(default_factory=dict)
    std_metrics: dict = field(default_factory=dict)
    ci_metrics: dict = field(default_factory=dict)  # 95% CI

    def __repr__(self):
        lines = [f"CVResult({self.model_name}, {self.k_folds}-fold)"]
        for metric, mean in self.mean_metrics.items():
            std = self.std_metrics.get(metric, 0)
            ci = self.ci_metrics.get(metric, (0, 0))
            lines.append(f"  {metric}: {mean:.4f} ± {std:.4f} "
                         f"[{ci[0]:.4f}, {ci[1]:.4f}]")
        return "\n".join(lines)


class CrossValidator:
    """
    k-Fold cross-validator with statistical testing.

    Provides:
    - Stratified k-fold splitting by query (job_id)
    - Per-fold metric computation
    - Aggregated mean ± std with 95% confidence intervals
    - Paired t-test between two model configurations
    - Effect size (Cohen's d) computation
    """

    def __init__(self, k: int = 5, random_state: int = 42):
        self.k = k
        self.random_state = random_state

    def create_folds(
        self,
        job_ids: list,
        n_folds: Optional[int] = None,
    ) -> list[tuple[list[int], list[int]]]:
        """
        Create k-fold splits grouped by job_id.
        Ensures all pairs for a given job_id are in the same fold.

        Returns:
            List of (train_indices, test_indices) tuples
        """
        k = n_folds or self.k
        rng = np.random.RandomState(self.random_state)

        unique_jobs = list(set(job_ids))
        rng.shuffle(unique_jobs)

        # Assign jobs to folds
        job_to_fold = {}
        for i, job in enumerate(unique_jobs):
            job_to_fold[job] = i % k

        folds = []
        for fold_idx in range(k):
            test_jobs = {j for j, f in job_to_fold.items() if f == fold_idx}
            train_idx = [i for i, j in enumerate(job_ids) if j not in test_jobs]
            test_idx = [i for i, j in enumerate(job_ids) if j in test_jobs]
            folds.append((train_idx, test_idx))

        return folds

    def cross_validate(
        self,
        model_name: str,
        X: np.ndarray,
        y: np.ndarray,
        job_ids: list,
        train_eval_fn: Callable,
    ) -> CVResult:
        """
        Run k-fold cross-validation.

        Args:
            model_name: Identifier for this model config
            X: Feature matrix
            y: Labels
            job_ids: Per-sample job IDs for query-grouped splitting
            train_eval_fn: Function(X_train, y_train, X_test, y_test, test_job_ids) -> dict
                          Returns a dict of metric_name: value

        Returns:
            CVResult with per-fold and aggregate metrics
        """
        folds = self.create_folds(job_ids)
        fold_results = []

        logger.info(f"Running {self.k}-fold CV for '{model_name}'...")

        for fold_idx, (train_idx, test_idx) in enumerate(folds):
            X_train, y_train = X[train_idx], y[train_idx]
            X_test, y_test = X[test_idx], y[test_idx]
            test_jids = [job_ids[i] for i in test_idx]

            metrics = train_eval_fn(X_train, y_train, X_test, y_test, test_jids)

            fold_result = FoldResult(
                fold_idx=fold_idx,
                train_size=len(train_idx),
                test_size=len(test_idx),
                metrics=metrics,
            )
            fold_results.append(fold_result)

            logger.info(f"  Fold {fold_idx + 1}/{self.k}: {metrics}")

        # Aggregate
        all_metric_names = set()
        for fr in fold_results:
            all_metric_names.update(fr.metrics.keys())

        mean_metrics = {}
        std_metrics = {}
        ci_metrics = {}

        for metric in all_metric_names:
            values = [fr.metrics.get(metric, 0.0) for fr in fold_results]
            mean = np.mean(values)
            std = np.std(values, ddof=1) if len(values) > 1 else 0.0
            se = std / np.sqrt(len(values)) if len(values) > 0 else 0.0

            # 95% CI using t-distribution
            from scipy import stats as sp_stats
            if len(values) > 1 and se > 0:
                t_crit = sp_stats.t.ppf(0.975, df=len(values) - 1)
                ci = (mean - t_crit * se, mean + t_crit * se)
            else:
                ci = (mean, mean)

            mean_metrics[metric] = round(float(mean), 4)
            std_metrics[metric] = round(float(std), 4)
            ci_metrics[metric] = (round(float(ci[0]), 4), round(float(ci[1]), 4))

        result = CVResult(
            model_name=model_name,
            k_folds=self.k,
            fold_results=fold_results,
            mean_metrics=mean_metrics,
            std_metrics=std_metrics,
            ci_metrics=ci_metrics,
        )

        logger.info(f"CV complete for '{model_name}': {mean_metrics}")
        return result

    # ─── Statistical Tests ───────────────────────────────────────

    @staticmethod
    def paired_t_test(
        result_a: CVResult,
        result_b: CVResult,
        metric: str,
    ) -> dict:
        """
        Paired t-test between two CV results on a specific metric.

        Tests H0: mean(A) = mean(B) vs H1: mean(A) ≠ mean(B).

        Returns dict with t_statistic, p_value, significant (at α=0.05),
        effect_size (Cohen's d), and winner.
        """
        from scipy import stats

        values_a = [fr.metrics.get(metric, 0.0) for fr in result_a.fold_results]
        values_b = [fr.metrics.get(metric, 0.0) for fr in result_b.fold_results]

        if len(values_a) != len(values_b):
            raise ValueError(
                f"Fold counts differ: {len(values_a)} vs {len(values_b)}"
            )

        t_stat, p_value = stats.ttest_rel(values_a, values_b)

        # Cohen's d for paired samples
        diffs = np.array(values_a) - np.array(values_b)
        d = float(np.mean(diffs) / np.std(diffs, ddof=1)) if np.std(diffs, ddof=1) > 0 else 0.0

        mean_a = np.mean(values_a)
        mean_b = np.mean(values_b)
        winner = result_a.model_name if mean_a > mean_b else result_b.model_name

        return {
            "metric": metric,
            "model_a": result_a.model_name,
            "model_b": result_b.model_name,
            "mean_a": round(float(mean_a), 4),
            "mean_b": round(float(mean_b), 4),
            "t_statistic": round(float(t_stat), 4),
            "p_value": round(float(p_value), 6),
            "significant": p_value < 0.05,
            "effect_size_d": round(d, 4),
            "effect_size_label": (
                "negligible" if abs(d) < 0.2 else
                "small" if abs(d) < 0.5 else
                "medium" if abs(d) < 0.8 else
                "large"
            ),
            "winner": winner,
        }

    # ─── Formatting ──────────────────────────────────────────────

    @staticmethod
    def format_results_table(
        results: list[CVResult],
        metrics: Optional[list[str]] = None,
    ) -> str:
        """Format CV results as a markdown table."""
        if not results:
            return "No results."

        if metrics is None:
            metrics = sorted(results[0].mean_metrics.keys())

        # Header
        header = "| Model | " + " | ".join(metrics) + " |"
        separator = "|---|" + "|".join(["---"] * len(metrics)) + "|"
        rows = [header, separator]

        for r in results:
            cells = [r.model_name]
            for m in metrics:
                mean = r.mean_metrics.get(m, 0)
                std = r.std_metrics.get(m, 0)
                cells.append(f"{mean:.4f}±{std:.4f}")
            rows.append("| " + " | ".join(cells) + " |")

        return "\n".join(rows)

    @staticmethod
    def format_latex_table(
        results: list[CVResult],
        metrics: Optional[list[str]] = None,
        caption: str = "Cross-validation results",
    ) -> str:
        """Format CV results as a LaTeX table."""
        if not results:
            return ""

        if metrics is None:
            metrics = sorted(results[0].mean_metrics.keys())

        lines = [
            "\\begin{table}[ht]",
            "\\centering",
            f"\\caption{{{caption}}}",
            "\\begin{tabular}{l" + "c" * len(metrics) + "}",
            "\\toprule",
            "Model & " + " & ".join(m.replace("_", "\\_") for m in metrics) + " \\\\",
            "\\midrule",
        ]

        # Find best per metric
        best_per_metric = {}
        for m in metrics:
            best_val = max(r.mean_metrics.get(m, 0) for r in results)
            best_per_metric[m] = best_val

        for r in results:
            cells = [r.model_name.replace("_", "\\_")]
            for m in metrics:
                mean = r.mean_metrics.get(m, 0)
                std = r.std_metrics.get(m, 0)
                val_str = f"${mean:.4f} \\pm {std:.4f}$"
                if mean == best_per_metric[m]:
                    val_str = f"\\textbf{{{val_str}}}"
                cells.append(val_str)
            lines.append(" & ".join(cells) + " \\\\")

        lines.extend([
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table}",
        ])

        return "\n".join(lines)

    @staticmethod
    def format_significance_table(
        tests: list[dict],
    ) -> str:
        """Format paired t-test results as markdown."""
        if not tests:
            return "No tests."

        header = "| Metric | Model A | Model B | Δ Mean | p-value | Sig. | Effect |"
        sep = "|---|---|---|---|---|---|---|"
        rows = [header, sep]

        for t in tests:
            delta = t["mean_a"] - t["mean_b"]
            sig = "✅" if t["significant"] else "—"
            rows.append(
                f"| {t['metric']} | {t['model_a']} | {t['model_b']} | "
                f"{delta:+.4f} | {t['p_value']:.4f} | {sig} | {t['effect_size_label']} |"
            )

        return "\n".join(rows)


# ─── Demo ────────────────────────────────────────────────────────

if __name__ == "__main__":
    cv = CrossValidator(k=5)

    # Simulate data
    np.random.seed(42)
    n = 100
    X = np.random.randn(n, 4)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    job_ids = [f"job_{i % 10}" for i in range(n)]

    def dummy_train_eval(X_tr, y_tr, X_te, y_te, test_jids):
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression(max_iter=200)
        model.fit(X_tr, y_tr)
        acc = model.score(X_te, y_te)
        return {"accuracy": acc, "ndcg@5": acc * 0.95}

    result = cv.cross_validate("LogReg_baseline", X, y, job_ids, dummy_train_eval)
    print(result)
    print()
    print(cv.format_results_table([result]))
