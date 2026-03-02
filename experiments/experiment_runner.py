"""
ASRIS — Experiment Runner
MLflow-based experiment tracking with automatic logging of
parameters, metrics, artifacts, and model comparison.
"""

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Callable

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import EXPERIMENT_DIR, get_logger

logger = get_logger("experiments.runner")


class ExperimentTracker:
    """
    Lightweight experiment tracking system.
    Logs parameters, metrics, and artifacts to disk in structured JSON format.
    Optionally integrates with MLflow if available.
    """

    def __init__(
        self,
        experiment_name: str = "asris_ranking",
        output_dir: Optional[Path] = None,
        use_mlflow: bool = False,
    ):
        self.experiment_name = experiment_name
        self.output_dir = output_dir or EXPERIMENT_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.use_mlflow = use_mlflow
        self._mlflow = None

        if use_mlflow:
            try:
                import mlflow
                self._mlflow = mlflow
                mlflow.set_experiment(experiment_name)
                logger.info(f"MLflow tracking enabled: {experiment_name}")
            except ImportError:
                logger.warning("MLflow not installed, falling back to file-based tracking")
                self.use_mlflow = False

    def run_experiment(
        self,
        run_name: str,
        run_fn: Callable,
        params: Optional[dict] = None,
        tags: Optional[dict] = None,
    ) -> dict:
        """
        Execute an experiment run with automatic tracking.

        Args:
            run_name: Name of this run (e.g., "sbert_baseline_v2")
            run_fn: Callable that returns a dict of metrics
            params: Hyperparameters to log
            tags: Additional tags

        Returns:
            Dict with run results including metrics, params, and timestamps
        """
        params = params or {}
        tags = tags or {}

        logger.info(f"Starting experiment run: {run_name}")
        start_time = time.time()
        timestamp = datetime.now().isoformat()

        # Run the experiment
        try:
            metrics = run_fn()
            status = "COMPLETED"
            error = None
        except Exception as e:
            metrics = {}
            status = "FAILED"
            error = str(e)
            logger.error(f"Experiment {run_name} failed: {e}")

        elapsed = round(time.time() - start_time, 2)

        # Build result
        result = {
            "run_name": run_name,
            "experiment": self.experiment_name,
            "timestamp": timestamp,
            "elapsed_seconds": elapsed,
            "status": status,
            "params": params,
            "metrics": metrics,
            "tags": tags,
        }
        if error:
            result["error"] = error

        # Save to disk
        run_file = self.output_dir / f"{run_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(run_file, "w") as f:
            json.dump(result, f, indent=2, default=str)
        logger.info(f"Results saved: {run_file.name}")

        # Log to MLflow
        if self.use_mlflow and self._mlflow:
            with self._mlflow.start_run(run_name=run_name):
                self._mlflow.log_params(params)
                for key, value in self._flatten_metrics(metrics).items():
                    if isinstance(value, (int, float)):
                        self._mlflow.log_metric(key, value)
                self._mlflow.set_tags(tags)

        self._print_run_summary(result)
        return result

    @staticmethod
    def _flatten_metrics(metrics: dict, prefix: str = "") -> dict:
        """Flatten nested metric dicts."""
        flat = {}
        for key, value in metrics.items():
            full_key = f"{prefix}{key}" if prefix else key
            if isinstance(value, dict):
                flat.update(ExperimentTracker._flatten_metrics(value, f"{full_key}."))
            else:
                flat[full_key] = value
        return flat

    def _print_run_summary(self, result: dict):
        """Print a formatted run summary."""
        print(f"\n{'═' * 55}")
        print(f"  EXPERIMENT: {result['run_name']}")
        print(f"{'═' * 55}")
        print(f"  Status: {result['status']}")
        print(f"  Duration: {result['elapsed_seconds']}s")
        print(f"  Timestamp: {result['timestamp']}")

        if result["params"]:
            print(f"\n  Parameters:")
            for key, value in result["params"].items():
                print(f"    {key}: {value}")

        if result["metrics"]:
            flat = self._flatten_metrics(result["metrics"])
            print(f"\n  Metrics:")
            for key, value in sorted(flat.items()):
                if isinstance(value, float):
                    print(f"    {key}: {value:.4f}")
                else:
                    print(f"    {key}: {value}")

        print(f"\n{'═' * 55}\n")

    def list_runs(self) -> list[dict]:
        """List all experiment runs from disk."""
        runs = []
        for f in sorted(self.output_dir.glob("*.json")):
            with open(f, "r") as fp:
                runs.append(json.load(fp))
        return runs

    def compare_runs(self, metric_key: str = "flat.ROC-AUC") -> None:
        """
        Compare all runs on a specific metric.
        Prints a ranked leaderboard.
        """
        runs = self.list_runs()
        if not runs:
            print("No experiment runs found.")
            return

        scored_runs = []
        for run in runs:
            flat = self._flatten_metrics(run.get("metrics", {}))
            score = flat.get(metric_key)
            if score is not None:
                scored_runs.append((run["run_name"], score, run["timestamp"]))

        scored_runs.sort(key=lambda x: x[1], reverse=True)

        print(f"\n{'═' * 60}")
        print(f"  EXPERIMENT LEADERBOARD — {metric_key}")
        print(f"{'═' * 60}")
        print(f"  {'Rank':<6} {'Run':<30} {'Score':>8}  {'Timestamp'}")
        print(f"  {'─' * 56}")

        for rank, (name, score, ts) in enumerate(scored_runs, 1):
            medal = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else "  "
            print(f"  {medal} {rank:<4} {name:<30} {score:>8.4f}  {ts[:19]}")

        print(f"\n{'═' * 60}\n")


# ─── Pre-built Experiment Configs ────────────────────────────────

def run_all_experiments():
    """Run all ranking models and compare."""
    tracker = ExperimentTracker()

    experiments = [
        {
            "name": "tfidf_baseline",
            "params": {"model": "TF-IDF", "max_features": 5000},
            "fn": lambda: _run_tfidf(),
        },
        {
            "name": "sbert_baseline",
            "params": {"model": "SBERT", "variant": "all-MiniLM-L6-v2"},
            "fn": lambda: _run_sbert(),
        },
        {
            "name": "hybrid_sbert_skills",
            "params": {"model": "Hybrid", "alpha": 0.7, "beta": 0.3},
            "fn": lambda: _run_hybrid(),
        },
    ]

    for exp in experiments:
        tracker.run_experiment(
            run_name=exp["name"],
            run_fn=exp["fn"],
            params=exp["params"],
        )

    tracker.compare_runs()


def _run_tfidf():
    """Stub — import and run TF-IDF baseline."""
    logger.info("Running TF-IDF baseline...")
    return {"note": "Run ranking/tfidf_baseline.py directly"}


def _run_sbert():
    """Stub — import and run SBERT baseline."""
    logger.info("Running SBERT baseline...")
    return {"note": "Run ranking/sbert_baseline.py directly"}


def _run_hybrid():
    """Stub — import and run Hybrid model."""
    logger.info("Running Hybrid model...")
    return {"note": "Run ranking/hybrid_eval.py directly"}


if __name__ == "__main__":
    # Demo with synthetic experiment
    tracker = ExperimentTracker()

    result = tracker.run_experiment(
        run_name="demo_experiment",
        run_fn=lambda: {
            "flat": {"ROC-AUC": 0.823, "P@5": 0.75, "NDCG@10": 0.68},
            "model": "demo",
        },
        params={"model_type": "demo", "max_features": 5000},
        tags={"dataset": "skill_based_pairs"},
    )
