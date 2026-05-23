# FAIMR — Team Roles & Responsibilities

**Project:** FAIMR (Fairness-Aware Interpretable Multi-Signal Ranking)  
**Course:** [Course Name]  
**Team:** Arnav, Ameya, Ayush, Aditya, Bakshi

---

## Overview

The project is split into 5 components of roughly equal scope. Each member owns one component end-to-end: design, implementation, testing, and the corresponding section(s) of the paper/report.

---

## Part 1 — Project Lead & Learning-to-Rank Core
**Owner: Arnav** *(Main Role)*

**Scope:**
- Overall system architecture, integration of all five components, and final pipeline orchestration
- Learning-to-Rank (LTR) models: feature engineering, XGBoost and LightGBM rankers (`ranking/ltr_model.py`, `ranking/hybrid_fusion.py`)
- Feature set design: combining TF-IDF, SBERT, and skill-graph signals into a unified feature vector
- Hyperparameter tuning, model serialization, and the end-to-end `rank` API endpoint
- Final paper writing coordination; responsible for the Abstract, Introduction, and System Architecture sections

**Key Files:**
- `ranking/ltr_model.py`, `ranking/hybrid_fusion.py`, `ranking/cross_encoder.py`
- `experiments/learning_to_rank_fixed.py`
- `api/server.py` (rank endpoint)
- `COLAB_RUNBOOK.md`

**Deliverable Weight:** Architecture decisions + highest-complexity ML component + integration glue

---

## Part 2 — Data Ingestion & Preprocessing
**Owner: Ameya**

**Scope:**
- PDF/DOCX resume parsing and text extraction (`ingestion/resume_parser.py`, `ingestion/pdf_parser.py`)
- Job description ingestion, domain labeling, and domain-balanced sampling (`ingestion/jd_ingestion.py`, `ingestion/domain_classifier.py`)
- Training pair generation: domain-based, semantic, and skill-based strategies (`ingestion/pair_generator.py`)
- Text normalization: tokenization, stop-word removal, lemmatization (`preprocessing/text_normalizer.py`)
- Resume section parsing: education, skills, experience detection (`preprocessing/section_parser.py`)
- Dataset construction and data quality validation

**Key Files:**
- `ingestion/` (all 7 modules)
- `preprocessing/text_normalizer.py`, `preprocessing/section_parser.py`
- `data/raw/`, `data/processed/`, `data/pairs/`

**Deliverable Weight:** Entire data layer; quality here directly determines model quality

---

## Part 3 — Embeddings & Baseline Ranking Models
**Owner: Ayush**

**Scope:**
- Sentence-BERT embedding pipeline with disk-based caching (`embeddings/embedding_cache.py`)
- TF-IDF vectorization and cosine-similarity ranker (`ranking/tfidf_ranker.py`)
- SBERT semantic baseline ranker (`ranking/sbert_ranker.py`)
- Skill-graph construction and skill-based matching ranker (`ranking/skill_ranker.py`)
- Semantic evaluation ranker (`ranking/semantic_eval.py`)
- Embedding ablation: impact of embedding choice on downstream ranking

**Key Files:**
- `embeddings/embedding_cache.py`
- `ranking/tfidf_ranker.py`, `ranking/sbert_ranker.py`, `ranking/skill_ranker.py`, `ranking/semantic_eval.py`

**Deliverable Weight:** All baseline models and the embedding layer that feeds the LTR core

---

## Part 4 — Fairness & Explainability
**Owner: Aditya**

**Scope:**
- Algorithmic bias detection: demographic parity, adverse impact ratio computation (`fairness/bias_detector.py`)
- Fairness-Constrained Re-ranking (FCR): enforcing the four-fifths rule post-ranking (`fairness/fair_ranker.py`)
- Counterfactual skill-gap explanations via greedy submodular optimization (`explainability/counterfactual.py`)
- Skill-gap analysis and human-readable explanation generation (`explainability/skill_gap.py`)
- Explain API endpoint integration

**Key Files:**
- `fairness/bias_detector.py`, `fairness/fair_ranker.py`
- `explainability/counterfactual.py`, `explainability/skill_gap.py`
- `api/server.py` (explain endpoint)

**Deliverable Weight:** Two full research contributions (FCR + counterfactuals) that form the paper's novel claims

---

## Part 5 — Evaluation, Experiments & API/Frontend
**Owner: Bakshi**

**Scope:**
- Evaluation metrics: Precision@K, NDCG, MRR, MAP, ROC-AUC (`evaluation/metrics.py`, `evaluation/evaluator.py`)
- Ablation studies and FCR stress tests (`experiments/`)
- MLflow experiment tracking setup and results logging
- FastAPI REST server (non-rank/explain endpoints: upload, stats, health) (`api/server.py`)
- Web frontend: ranking UI, explanation tab, dashboard (`frontend/`)
- Final results tables and figures for the paper

**Key Files:**
- `evaluation/metrics.py`, `evaluation/evaluator.py`
- `experiments/ablation_study.py`, `experiments/fcr_stress_test.py`
- `api/server.py` (upload, stats endpoints), `frontend/`

**Deliverable Weight:** All quantitative validation + user-facing interface

---

## Integration Points & Dependencies

```
Ameya (Data)
    └──> Ayush (Embeddings & Baselines)
              └──> Arnav (LTR Core)
                        ├──> Aditya (Fairness & Explainability)
                        └──> Bakshi (Evaluation & API/Frontend)
```

- Ameya must deliver cleaned pairs before Ayush can generate embeddings
- Ayush must deliver embedding cache and baseline scores before Arnav trains the LTR model
- Arnav's ranked outputs feed both Aditya (re-ranking + explanations) and Bakshi (evaluation)
- Bakshi integrates everyone's outputs into the final API and frontend

---

## Suggested Milestones

| Week | Milestone |
|------|-----------|
| 1–2  | Ameya: data pipeline complete; all members: environment setup |
| 3–4  | Ayush: embeddings + baselines; Arnav: feature engineering design |
| 5–6  | Arnav: LTR model trained; Aditya: fairness + explainability modules |
| 7    | Bakshi: evaluation suite + API + frontend complete |
| 8    | Full integration, paper draft, final testing |
