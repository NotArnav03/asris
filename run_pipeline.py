"""
ASRIS — Pipeline Orchestration
CLI tool to run the full ASRIS pipeline or individual stages.

Usage:
    python run_pipeline.py --all              # Run everything
    python run_pipeline.py --stage ingestion  # Run one stage
    python run_pipeline.py --stage evaluate   # Evaluate all models
    python run_pipeline.py --list             # List available stages
"""

import argparse
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import get_logger

logger = get_logger("pipeline")


# ─── Stage Definitions ───────────────────────────────────────────

STAGES = {
    "ingestion": {
        "description": "Parse raw resumes (PDF/CSV) and build metadata",
        "scripts": [
            ("ingestion.resume_ingestion", "Resume text extraction"),
            ("ingestion.resume_metadata_builder", "Resume metadata builder"),
            ("ingestion.jd_domain_labeler", "JD domain labeling"),
            ("ingestion.jd_balancer", "JD domain balancing"),
        ],
    },
    "pairs": {
        "description": "Generate training pairs (domain, semantic, skill-based)",
        "scripts": [
            ("ingestion.pair_generator", "Domain-based pair generation"),
            ("ingestion.semantic_pair_generator", "Semantic pair generation"),
            ("ingestion.skill_based_pair_generator", "Skill-based pair generation"),
        ],
    },
    "preprocessing": {
        "description": "Normalize and parse resume sections",
        "scripts": [
            ("preprocessing.text_normalizer", "Text normalization"),
            ("preprocessing.section_parser", "Section parsing"),
        ],
    },
    "evaluate": {
        "description": "Run all ranking models and evaluate",
        "scripts": [
            ("ranking.tfidf_baseline", "TF-IDF baseline"),
            ("ranking.sbert_baseline", "SBERT baseline"),
            ("ranking.sbert_semantic_eval", "SBERT semantic evaluation"),
            ("ranking.sbert_skill_eval", "SBERT skill-based evaluation"),
            ("ranking.hybrid_eval", "Hybrid SBERT + skills"),
        ],
    },
    "advanced": {
        "description": "Run advanced ranking models",
        "scripts": [
            ("ranking.cross_encoder_ranker", "Cross-encoder re-ranking"),
            ("ranking.learning_to_rank", "Learning-to-Rank (XGBoost)"),
        ],
    },
    "explain": {
        "description": "Generate match explanations",
        "scripts": [
            ("explainability.explainer", "Match explanations"),
        ],
    },
    "fairness": {
        "description": "Run bias detection audit",
        "scripts": [
            ("fairness.bias_detector", "Bias detection"),
        ],
    },
}


def run_stage(stage_name: str):
    """Run all scripts in a pipeline stage."""
    if stage_name not in STAGES:
        logger.error(f"Unknown stage: {stage_name}")
        logger.info(f"Available stages: {', '.join(STAGES.keys())}")
        return

    stage = STAGES[stage_name]
    logger.info(f"\n{'═' * 50}")
    logger.info(f"  STAGE: {stage_name.upper()} — {stage['description']}")
    logger.info(f"{'═' * 50}")

    for module_path, description in stage["scripts"]:
        logger.info(f"\n  Running: {description} ({module_path})")
        start = time.time()

        try:
            import importlib
            module = importlib.import_module(module_path)
            elapsed = round(time.time() - start, 2)
            logger.info(f"  ✅ {description} completed in {elapsed}s")
        except Exception as e:
            elapsed = round(time.time() - start, 2)
            logger.error(f"  ❌ {description} failed after {elapsed}s: {e}")

    logger.info(f"\n  Stage '{stage_name}' complete.\n")


def run_all():
    """Run all pipeline stages in order."""
    total_start = time.time()
    logger.info("Starting full ASRIS pipeline...")

    for stage_name in ["ingestion", "pairs", "preprocessing", "evaluate"]:
        run_stage(stage_name)

    total_elapsed = round(time.time() - total_start, 2)
    logger.info(f"\n{'═' * 50}")
    logger.info(f"  FULL PIPELINE COMPLETE — {total_elapsed}s total")
    logger.info(f"{'═' * 50}\n")


def list_stages():
    """Print available pipeline stages."""
    print(f"\n{'═' * 55}")
    print(f"  ASRIS Pipeline Stages")
    print(f"{'═' * 55}")

    for name, stage in STAGES.items():
        print(f"\n  {name.upper()}")
        print(f"  {stage['description']}")
        for module, desc in stage["scripts"]:
            print(f"    → {desc}")

    print(f"\n{'═' * 55}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="ASRIS Pipeline Orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python run_pipeline.py --all
    python run_pipeline.py --stage ingestion
    python run_pipeline.py --stage evaluate
    python run_pipeline.py --stage advanced
    python run_pipeline.py --list
        """,
    )
    parser.add_argument("--stage", type=str, help="Run a specific pipeline stage")
    parser.add_argument("--all", action="store_true", help="Run the full pipeline")
    parser.add_argument("--list", action="store_true", help="List all stages")

    args = parser.parse_args()

    if args.list:
        list_stages()
    elif args.all:
        run_all()
    elif args.stage:
        run_stage(args.stage)
    else:
        parser.print_help()
