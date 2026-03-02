# ASRIS — AI-Powered Resume Screening & Intelligent Shortlisting

An end-to-end AI system that matches job descriptions (JDs) to resumes using multi-signal ranking — combining semantic embeddings, skill-based matching, and learned relevance models to produce explainable, fair candidate shortlists.

## Architecture

```
asris/
├── config.py                    # Centralized configuration
├── run_pipeline.py              # Pipeline orchestration CLI
├── ingestion/                   # Data ingestion & pair generation
│   ├── resume_ingestion.py      # PDF/CSV resume parsing
│   ├── jd_domain_labeler.py     # Domain label assignment
│   ├── jd_balancer.py           # Balanced domain sampling
│   ├── resume_metadata_builder.py
│   ├── pair_generator.py        # Domain-based pairs
│   ├── semantic_pair_generator.py
│   └── skill_based_pair_generator.py
├── preprocessing/               # Text normalization & section parsing
│   ├── text_normalizer.py       # Clean, normalize, lemmatize text
│   └── section_parser.py        # Extract resume sections
├── embeddings/                  # Embedding generation & caching
│   └── embedding_manager.py     # SBERT/TF-IDF with disk caching
├── ranking/                     # Ranking models & evaluators
│   ├── ranking_utils.py         # Shared evaluation utilities
│   ├── tfidf_baseline.py
│   ├── sbert_baseline.py
│   ├── sbert_semantic_eval.py
│   ├── sbert_skill_eval.py
│   ├── hybrid_eval.py
│   ├── cross_encoder_ranker.py  # Cross-encoder re-ranking
│   └── learning_to_rank.py      # LambdaMART / XGBoost ranker
├── evaluation/                  # Ranking metrics framework
│   └── metrics.py               # P@K, R@K, NDCG, MRR, MAP
├── explainability/              # Match explanation engine
│   └── explainer.py             # Skill match, section scores, heatmaps
├── fairness/                    # Bias detection & mitigation
│   └── bias_detector.py         # Demographic parity, adverse impact
├── experiments/                 # Experiment tracking
│   └── experiment_runner.py     # MLflow-based experiment logging
├── api/                         # REST API
│   └── server.py                # FastAPI endpoints
├── tests/                       # Unit & integration tests
│   └── test_preprocessing.py
├── notebooks/                   # Data inspection scripts
└── data/
    ├── raw/                     # Original resumes & JDs
    ├── processed/               # Cleaned text files
    ├── labeled/                 # Training pairs
    └── metadata/                # Domain mappings & resume metadata
```

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download spaCy model
python -m spacy download en_core_web_sm

# 3. Run the full pipeline
python run_pipeline.py --all

# 4. Or run individual stages
python run_pipeline.py --stage ingestion
python run_pipeline.py --stage preprocessing
python run_pipeline.py --stage embeddings
python run_pipeline.py --stage evaluate
python run_pipeline.py --stage explain

# 5. Start the API server
python -m api.server
```

## Ranking Approaches

| Model | Method | Signals |
|---|---|---|
| TF-IDF Baseline | Sparse vector cosine similarity | Keyword overlap |
| SBERT Baseline | Dense embedding cosine similarity | Semantic meaning |
| Skill-Based SBERT | SBERT on skill-matched pairs | Skills + semantics |
| Hybrid | Weighted SBERT + skill coverage | Multi-signal |
| Cross-Encoder | Pairwise relevance scoring | Deep contextual |
| Learning-to-Rank | XGBoost on feature vectors | All signals combined |

## Evaluation Metrics

- **Precision@K** / **Recall@K** — relevance in top-K results
- **NDCG** — normalized discounted cumulative gain
- **MRR** — mean reciprocal rank
- **MAP** — mean average precision
- **ROC-AUC** — area under ROC curve

## Configuration

All settings are centralized in `config.py`. Override defaults via `config.yaml` in the project root.

## License

MIT
