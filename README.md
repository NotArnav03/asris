# FAIMR — Fairness-Aware Interpretable Multi-Signal Ranking

[![tests](https://github.com/NotArnav03/FAIMR/actions/workflows/test.yml/badge.svg)](https://github.com/NotArnav03/FAIMR/actions/workflows/test.yml)
[![python](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue)](https://github.com/NotArnav03/FAIMR)
[![coverage](https://img.shields.io/badge/fairness%20coverage-60%25%2B-brightgreen)](#testing)
[![reproducible](https://img.shields.io/badge/artefacts-hash--pinned-success)](#reproducing-the-paper-artefacts)

An end-to-end resume → JD matching pipeline with a calibrated fairness audit:
character n-gram classifier with per-culture isotonic calibration, dual
soft/hard adverse-impact ratio, EEOC-style directional AIR with Wilson
confidence intervals, constrained-insertion fairness-aware re-ranking
with a written termination proof, deduplication, drift detection, and a
production-grade API with GCRA rate limiting.

## Why FAIMR

Most resume-ranking systems publish a single AIR number and call it
"fair."  FAIMR's audit publishes:

| Field | Meaning |
|---|---|
| `verdict` | publish-ready answer: `pass` / `fail` / `pass_with_drift_warning` / 6 inconclusive variants |
| `adverse_impact_ratio_hard` + `_soft` | hard categorical AIR + probability-mass AIR; `verdict` uses min |
| `directional_air` + `air_lower_ci` + `air_upper_ci` | one-directional EEOC AIR with Wilson 95% CI |
| `parity_statistics` | size-weighted DPD, chi-squared p-value, Theil-T inequality |
| `calibration_drift` | corpus-weighted ECE + 3-tier gate (`ok` / `warn` / `inconclusive_high_drift`) |
| `detection_coverage` | refuse to publish below 50% gender-signal coverage |
| `culture_distribution` | per-cluster mean P(female), lookup share, model-card ECE |
| `dedup` | exact + SimHash near-duplicate counts, ballot-stuffing alert |
| `counterfactual_robustness` | name-swap test embedded in the audit when a scorer is provided |
| `drift_since_baseline` | optional historical-baseline comparison via audit log |
| `integrity` | SHA-256 round-trip from training to audit; CRITICAL on mismatch |
| `model_card_validation` | JSON schema check |
| `per_resume` | full per-candidate signal trail for forensic review |

Every field is regression-tested.  See `tests/test_core.py` and the
audit-driven fields rendered in `audit_ranking_bias`.

## Architecture

```
faimr/
├── reproduce.py                 # One-shot rebuild + manifest (see below)
├── config.py
├── run_pipeline.py
├── requirements.txt
├── ingestion/                   # Data ingestion + pair generation
├── preprocessing/               # Text normaliser + section parser
│   ├── text_normalizer.py       # Multi-script-safe normalisation
│   └── section_parser.py
├── embeddings/
│   └── embedding_manager.py     # Per-call TF-IDF vectorizer; SHA-256 cache
├── ranking/
│   ├── ranking_utils.py         # extract_skills_in_text, classify_at_percentile
│   ├── fairness_ranker.py       # Constrained-insertion FCR with proof
│   ├── cross_encoder_ranker.py
│   ├── learning_to_rank.py
│   ├── hybrid_eval.py
│   ├── tfidf_baseline.py
│   ├── sbert_baseline.py
│   ├── sbert_semantic_eval.py
│   └── sbert_skill_eval.py
├── evaluation/
│   ├── metrics.py               # P@K, R@K, NDCG (linear+exp), MRR, MAP, AUC
│   ├── counterfactual_robustness.py
│   └── cross_validator.py       # Significance-aware paired t-tests
├── explainability/
│   ├── counterfactual.py        # Greedy submodular minimum-flip-set
│   └── explainer.py             # Skill / keyword / verdict explanation
├── fairness/
│   ├── bias_detector.py         # The audit pipeline (verdict + 13 fields)
│   └── names/
│       ├── classifier.py        # Hybrid lookup + char-ngram OOV
│       ├── cultural_classifier.py  # Per-culture isotonic calibration
│       ├── seed_lists.py        # GENDERED_NAMES + _UNISEX_NAMES (build-time)
│       ├── train_classifier.py
│       ├── model.pkl            # Pinned; SHA-256 in model_card.json
│       └── model_card.json      # Full provenance + per-culture ECE
├── api/
│   └── server.py                # FastAPI + GCRA rate limit + auth + caps
├── frontend/
├── data/names/                  # Whitelisted training corpora
│   ├── firstnames_raw.csv       # GFDL upstream (Michael 2007 / Winkelmann 2016)
│   ├── us_surnames_raw.csv      # US Census 2010
│   ├── nicknames.csv            # Curated 80-pair canonical map
│   ├── surname_holdout.csv      # 95 names for coverage validation
│   ├── training_corpus.csv      # Built from above by build_corpus.py
│   ├── surnames.csv             # Built from above by build_surnames.py
│   └── ATTRIBUTION.md
├── tests/                       # 228+ regression tests
│   └── test_core.py
└── .github/workflows/test.yml   # CI pipeline (pytest + coverage gate)
```

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. (Optional) Download spaCy model for lemmatisation
python -m spacy download en_core_web_sm

# 3. Run the full pipeline
python run_pipeline.py --all

# 4. Start the API server
python -m api.server
# or: uvicorn api.server:app --host 0.0.0.0 --port 8000 --reload
```

## Reproducing the paper artefacts

A single command rebuilds every derived artefact from committed inputs:

```bash
python reproduce.py             # full rebuild: corpus -> surnames -> model -> tests
python reproduce.py --skip-train  # reuse committed model.pkl
python reproduce.py --grid-search # re-search hyperparameters
python reproduce.py --check       # verify hashes match without rebuilding
```

The runner emits `reproducibility_manifest.json` with SHA-256 hashes of
every input + output, Python and library versions, and per-stage timing.
`--check` lets a reviewer confirm a fresh checkout produces byte-identical
artefacts on their machine.

## Fairness audit usage

```python
from fairness.bias_detector import BiasDetector

detector = BiasDetector()
audit = detector.audit_ranking_bias(
    resume_texts={"r1.txt": "Priya Sharma\n...", "r2.txt": "John Smith\n..."},
    scores={"r1.txt": 0.92, "r2.txt": 0.78},
    cutoff_method="top_k", top_k=10,       # operational cutoff disclosure
    scorer=my_scorer, jd_text=my_jd,       # optional counterfactual check
    audit_log_path="audit_log.jsonl",      # optional drift baseline
    write_baseline=True,
)
print(audit["gender_bias_analysis"]["verdict"])
print(audit["calibration_drift"]["status"])
print(audit["counterfactual_robustness"]["all_robust"])
```

`audit_ranking_bias` accepts these kwargs:

| kwarg | default | purpose |
|---|---|---|
| `selection_threshold` | `None` | explicit score floor |
| `cutoff_method` | `"median"` | one of `median` / `top_k` / `percentile` / `explicit` |
| `top_k`, `percentile` | `None` | required for the matching cutoff method |
| `dedup` | `True` | enable exact + SimHash dedup |
| `near_dup_hamming` | `3` | SimHash Hamming distance threshold |
| `scorer`, `jd_text` | `None` | enable embedded counterfactual robustness |
| `counterfactual_sample_size` | `10` | candidates sampled for swap audit |
| `audit_log_path`, `write_baseline` | `None`, `False` | historical drift detection |

## API Endpoints

All endpoints honour the configured rate limit, optional API-key auth, and
input-size caps (per-resume 100 KB, request 50 MB, 5000 resumes max).

| Method | Endpoint | Quota |
|---|---|---|
| GET | `/` | default |
| GET | `/health` | 1000/min |
| GET | `/stats` / `/cache/stats` | 200/min |
| POST | `/rank` | 60/min, burst 10 |
| POST | `/rank-pdfs` | 10/min, burst 5 |
| POST | `/audit` | 10/min, burst 5 |
| POST | `/explain` | 60/min, burst 10 |
| POST | `/counterfactual` | 30/min, burst 10 |
| POST | `/upload-pdf` | 60/min, burst 10 |

Production deployments configure via env vars:

```bash
FAIMR_API_KEY=<secret>           # require X-API-Key header
FAIMR_CORS_ORIGINS=https://hr.example.com,https://ats.example.com
FAIMR_TRUSTED_PROXIES=10.0.0.1
FAIMR_RATE_LIMIT_ALLOWLIST=10.0.0.5   # bypass IPs (health-check monitors)
FAIMR_MAX_RESUME_BYTES=100000
FAIMR_MAX_REQUEST_RESUMES=5000
```

## Ranking Approaches

| Model | Method | Signals |
|---|---|---|
| TF-IDF Baseline | Sparse vector cosine | Keyword overlap |
| SBERT Baseline | Dense embedding cosine | Semantic meaning |
| Skill-Based SBERT | SBERT on skill-matched pairs | Skills + semantics |
| Hybrid | Weighted SBERT + skill coverage | Multi-signal |
| Cross-Encoder | Pairwise relevance scoring | Deep contextual |
| Learning-to-Rank | XGBoost on 7 features | All signals combined |
| Fairness Re-ranker | Constrained insertion | Relevance + group parity |

## Evaluation Metrics

`evaluation/metrics.py` provides:

- **Precision@K / Recall@K** with input length validation.
- **NDCG@K** with both linear and exponential gain (Burges/LambdaMART form).
- **MRR / Average Precision** with graded-relevance support.
- **ROC-AUC** flat AND per-query mean (reported side-by-side).
- All metric functions raise `ValueError` on length mismatch; `compute_roc_auc` returns `None` on undefined (not a misleading `0.0`).

## Testing

```bash
pytest tests/test_core.py -v
# CI runs the suite + a coverage gate of 60% over actually-tested modules
```

228+ regression tests organised into:

- `TestBiasDetector` — audit pipeline + dual AIR + drift gate
- `TestNameClassifier` — calibrated classifier behaviour
- `TestSurnameCoverage` — denylist holdout validation
- `TestCounterfactualRobustness` — name-swap harness
- `TestConstrainedInsertionFCR` — FCR rewrite invariants
- `TestApiHardening` — rate limit + auth + input caps
- `TestEvaluationMetrics` — ranking metric correctness
- `TestCounterfactual` — submodular flip-set explanation
- `TestTextNormalizer` — multi-script normalisation

## For paper reviewers

Code locations for every claim:

| Claim | File:Function |
|---|---|
| Per-culture isotonic calibration | `fairness/names/cultural_classifier.py:CulturalCalibratedClassifier` |
| Hybrid lookup + char-ngram OOV | `fairness/names/classifier.py:_resolve_compound_lookup` |
| Surname-aware first-token rule | `fairness/bias_detector.py:_pick_name_signal` |
| Dual soft/hard AIR with conservative gate | `fairness/bias_detector.py:_air_soft` + `audit_ranking_bias` |
| Directional AIR + Wilson CIs | `fairness/bias_detector.py:adverse_impact_ratio` |
| Constrained-insertion FCR with termination proof | `ranking/fairness_ranker.py:FairnessConstrainedRanker.rerank` |
| Counterfactual name-swap robustness | `evaluation/counterfactual_robustness.py:name_swap_robustness` |
| Cyrillic / Greek confusables defence | `fairness/bias_detector.py:_sanitise_for_detection` |
| RTL honorifics (Arabic, Hebrew) | `fairness/bias_detector.py:_rtl_honorific_fires` |
| Calibration-drift gate (3 tiers) | `fairness/bias_detector.py:_CALIBRATION_DRIFT_*` |
| Model integrity (SHA-256 round-trip) | `fairness/names/classifier.py:_load_model` |

## Configuration

`config.py` exposes defaults; `config.yaml` at project root overrides.

## License

MIT.  Training corpus `data/names/training_corpus.csv` is a derivative
work of the firstname-database project and inherits its GFDL-1.2+ licence
(see `data/names/ATTRIBUTION.md`).
