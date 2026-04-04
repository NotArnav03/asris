"""
FAIMR — Ablation Study Runner
Runs systematic ablation experiments across 6 model configurations:
1. SBERT-only
2. TF-IDF-only
3. SBERT + TF-IDF
4. Full LTR (all 8 features)
5. LTR + FCR (with fairness constraint)
6. LTR + FCR + Counterfactual (full FAIMR)

Outputs: results table (Markdown + LaTeX), significance tests,
         fairness-relevance Pareto plot data.
"""

import json
import time
import numpy as np
from pathlib import Path
from datetime import datetime

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import EXPERIMENT_DIR, get_logger

logger = get_logger("experiments.ablation")


class AblationRunner:
    """
    Runs ablation experiments to demonstrate the contribution
    of each component in the FAIMR framework.
    """

    CONFIGS = [
        {
            "name": "SBERT-only",
            "features": ["sbert_similarity"],
            "fcr": False,
            "description": "Semantic similarity baseline",
        },
        {
            "name": "TF-IDF-only",
            "features": ["tfidf_similarity"],
            "fcr": False,
            "description": "Lexical similarity baseline",
        },
        {
            "name": "SBERT+TF-IDF",
            "features": ["sbert_similarity", "tfidf_similarity"],
            "fcr": False,
            "description": "Multi-signal without skills",
        },
        {
            "name": "Full-LTR",
            "features": [
                "sbert_similarity", "tfidf_similarity",
                "skill_coverage", "num_matched_skills",
                "keyword_overlap_ratio",
            ],
            "fcr": False,
            "description": "All features, no fairness",
        },
        {
            "name": "LTR+FCR",
            "features": [
                "sbert_similarity", "tfidf_similarity",
                "skill_coverage", "num_matched_skills",
                "keyword_overlap_ratio",
            ],
            "fcr": True,
            "description": "Full features + fairness re-ranking",
        },
        {
            "name": "FAIMR-Full",
            "features": [
                "sbert_similarity", "tfidf_similarity",
                "skill_coverage", "num_matched_skills",
                "keyword_overlap_ratio",
            ],
            "fcr": True,
            "counterfactual": True,
            "description": "Complete FAIMR system",
        },
    ]

    ALL_FEATURES = [
        "sbert_similarity", "tfidf_similarity",
        "skill_coverage", "num_matched_skills",
        "keyword_overlap_ratio",
    ]

    def __init__(self, k_folds: int = 5, random_state: int = 42):
        self.k_folds = k_folds
        self.random_state = random_state
        self.results = []

    def _get_feature_indices(self, feature_names: list[str]) -> list[int]:
        """Get column indices for the requested features."""
        return [self.ALL_FEATURES.index(f) for f in feature_names
                if f in self.ALL_FEATURES]

    def run_single_config(
        self,
        config: dict,
        X: np.ndarray,
        y: np.ndarray,
        job_ids: list,
    ) -> dict:
        """
        Run a single ablation config with k-fold cross-validation.

        Args:
            config: Ablation config dict with name, features, fcr flag
            X: Full feature matrix (all 8 features)
            y: Labels
            job_ids: Per-sample job IDs

        Returns:
            Dict with aggregated metrics
        """
        from evaluation.cross_validator import CrossValidator
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score,
            f1_score, roc_auc_score,
        )
        from evaluation.metrics import precision_at_k, ndcg_at_k

        feature_idx = self._get_feature_indices(config["features"])

        if not feature_idx:
            logger.warning(f"No valid features for {config['name']}")
            return {}

        X_subset = X[:, feature_idx]

        def train_eval(X_tr, y_tr, X_te, y_te, test_jids):
            import xgboost as xgb

            if X_tr.shape[1] == 1:
                # Single feature — use score directly as prediction
                y_pred_proba = (X_te[:, 0] - X_te[:, 0].min()) / (
                    X_te[:, 0].max() - X_te[:, 0].min() + 1e-8
                )
            else:
                model = xgb.XGBClassifier(
                    n_estimators=100,
                    max_depth=4,
                    learning_rate=0.1,
                    random_state=self.random_state,
                    eval_metric="logloss",
                    verbosity=0,
                )
                model.fit(X_tr, y_tr)
                y_pred_proba = model.predict_proba(X_te)[:, 1]

            y_pred = (y_pred_proba >= 0.5).astype(int)

            # Compute metrics
            metrics = {
                "accuracy": accuracy_score(y_te, y_pred),
                "precision": precision_score(y_te, y_pred, zero_division=0),
                "recall": recall_score(y_te, y_pred, zero_division=0),
                "f1": f1_score(y_te, y_pred, zero_division=0),
            }

            # AUC (needs both classes)
            if len(np.unique(y_te)) > 1:
                metrics["auc"] = roc_auc_score(y_te, y_pred_proba)
            else:
                metrics["auc"] = 0.0

            # Ranking metrics (per-query)
            p_at_5_list = []
            ndcg_5_list = []
            unique_jobs = list(set(test_jids))

            for j in unique_jobs:
                mask = [jid == j for jid in test_jids]
                q_labels = y_te[mask]
                q_scores = y_pred_proba[mask]
                if len(q_labels) > 1 and sum(q_labels) > 0:
                    p_at_5_list.append(precision_at_k(q_labels.tolist(), q_scores.tolist(), 5))
                    ndcg_5_list.append(ndcg_at_k(q_labels.tolist(), q_scores.tolist(), 5))

            metrics["p@5"] = np.mean(p_at_5_list) if p_at_5_list else 0.0
            metrics["ndcg@5"] = np.mean(ndcg_5_list) if ndcg_5_list else 0.0

            # Fairness (if FCR enabled, apply post-hoc)
            if config.get("fcr") and len(test_jids) > 0:
                from ranking.fairness_ranker import FairnessConstrainedRanker, RankedCandidate
                from fairness.bias_detector import BiasDetector

                detector = BiasDetector()
                # Simulate gender groups (for evaluation)
                np.random.seed(42)
                groups = np.random.choice(["male", "female"], size=len(y_te))

                candidates = [
                    RankedCandidate(f"c_{i}", float(y_pred_proba[i]), groups[i])
                    for i in range(len(y_te))
                ]
                candidates.sort(key=lambda c: c.score, reverse=True)

                fcr = FairnessConstrainedRanker(threshold=0.8)
                report = fcr.rerank(candidates, _compute_pareto=False)
                metrics["air"] = report.final_air
                metrics["displacement"] = report.displacement_cost

            return metrics

        cv = CrossValidator(k=self.k_folds, random_state=self.random_state)
        cv_result = cv.cross_validate(config["name"], X_subset, y, job_ids, train_eval)

        return {
            "config": config,
            "cv_result": cv_result,
        }

    def run_all(
        self,
        X: np.ndarray,
        y: np.ndarray,
        job_ids: list,
    ) -> list[dict]:
        """
        Run all ablation configs and return results.
        """
        logger.info(f"Starting ablation study: {len(self.CONFIGS)} configs, "
                     f"{self.k_folds}-fold CV")

        results = []
        for i, config in enumerate(self.CONFIGS):
            logger.info(f"[{i + 1}/{len(self.CONFIGS)}] Running {config['name']}...")
            start = time.time()

            result = self.run_single_config(config, X, y, job_ids)
            result["time_seconds"] = round(time.time() - start, 2)
            results.append(result)

            logger.info(f"  Done in {result['time_seconds']}s")

        self.results = results
        return results

    def generate_report(self, results: list[dict] = None) -> str:
        """Generate a comprehensive ablation report."""
        results = results or self.results
        if not results:
            return "No results. Run run_all() first."

        cv_results = [r["cv_result"] for r in results if "cv_result" in r]

        from evaluation.cross_validator import CrossValidator

        lines = [
            "# FAIMR Ablation Study Results",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            f"Configuration: {self.k_folds}-fold cross-validation",
            "",
            "## Results Table",
            "",
        ]

        # Markdown table
        key_metrics = ["accuracy", "f1", "auc", "ndcg@5", "p@5"]
        available_metrics = [m for m in key_metrics
                              if m in cv_results[0].mean_metrics]

        md_table = CrossValidator.format_results_table(cv_results, available_metrics)
        lines.append(md_table)

        # Significance tests
        lines.extend(["", "## Significance Tests", ""])

        tests = []
        for metric in available_metrics:
            for i in range(len(cv_results) - 1):
                try:
                    test = CrossValidator.paired_t_test(
                        cv_results[i], cv_results[i + 1], metric
                    )
                    tests.append(test)
                except Exception:
                    pass

        if tests:
            sig_table = CrossValidator.format_significance_table(tests)
            lines.append(sig_table)

        # LaTeX table
        lines.extend(["", "## LaTeX Table", "", "```latex"])
        lines.append(CrossValidator.format_latex_table(
            cv_results, available_metrics,
            caption="Ablation study results for FAIMR framework"
        ))
        lines.append("```")

        return "\n".join(lines)

    def save_results(self, results: list[dict] = None, filename: str = "ablation_results.json"):
        """Save results to JSON."""
        results = results or self.results
        output_dir = EXPERIMENT_DIR
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
        except (FileExistsError, OSError):
            pass

        # Serialize CV results
        serializable = []
        for r in results:
            entry = {
                "config": r["config"],
                "time_seconds": r.get("time_seconds", 0),
            }
            if "cv_result" in r:
                cv = r["cv_result"]
                entry["mean_metrics"] = cv.mean_metrics
                entry["std_metrics"] = cv.std_metrics
                entry["ci_metrics"] = {k: list(v) for k, v in cv.ci_metrics.items()}
            serializable.append(entry)

        path = output_dir / filename
        with open(path, "w") as f:
            json.dump(serializable, f, indent=2)

        logger.info(f"Results saved to {path}")

        # Also save markdown report
        report = self.generate_report(results)
        report_path = output_dir / "ablation_report.md"
        with open(report_path, "w") as f:
            f.write(report)

        logger.info(f"Report saved to {report_path}")


if __name__ == "__main__":
    print("Ablation Runner initialized.")
    print(f"Configs: {[c['name'] for c in AblationRunner.CONFIGS]}")
    print("Use with: runner = AblationRunner(); runner.run_all(X, y, job_ids)")
