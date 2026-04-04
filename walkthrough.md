# FAIMR Implementation Walkthrough

## What Changed

Transformed ASRIS into **FAIMR** (Fairness-Aware Interpretable Multi-Signal Ranking) with 4 novel contributions for Q2 journal publication.

---

## New Modules Created

### 1. Fairness-Constrained Re-ranker
[fairness_ranker.py](file:///c:/asris/ranking/fairness_ranker.py) — **Primary novel algorithm**

- Greedy swap-based re-ranking to enforce AIR ≥ 0.8 (4/5 rule) at every prefix k
- Minimizes Kendall-τ displacement from relevance ordering
- Computes Pareto frontier across 9 fairness thresholds
- Reports: AIR before/after, displacement cost, per-group statistics

### 2. Counterfactual Explainer
[counterfactual.py](file:///c:/asris/explainability/counterfactual.py) — **Second novel contribution**

- Skill perturbation analysis: simulates adding each missing skill
- Reports: *"Adding Python would move you from rank #5 to rank #2"*
- Handles batch explanations for all candidates

### 3. Cross-Validator
[cross_validator.py](file:///c:/asris/evaluation/cross_validator.py)

- k-fold CV with query-grouped (job_id) splitting
- Paired t-tests with Cohen's d effect sizes
- 95% confidence intervals via t-distribution
- Auto-generated LaTeX + markdown results tables

### 4. Ablation Runner
[ablation_runner.py](file:///c:/asris/experiments/ablation_runner.py)

6 configs: SBERT-only → TF-IDF-only → SBERT+TF-IDF → Full-LTR → LTR+FCR → FAIMR-Full

---

## Modified Files

| File | Changes |
|---|---|
| [bias_detector.py](file:///c:/asris/fairness/bias_detector.py) | +3 metrics: DPD, equalized odds, SPD |
| [server.py](file:///c:/asris/api/server.py) | +`/audit` and `/counterfactual` endpoints |
| [test_core.py](file:///c:/asris/tests/test_core.py) | +12 new tests (FCR, counterfactual, fairness) |

---

## Test Results

```
39 collected, 31 passed, 8 failed (pre-existing spacy import issue)
```

All 12 new tests pass:
- ✅ 5 FCR tests (fair ranking, biased fix, displacement, Pareto)
- ✅ 4 counterfactual tests (skill extraction, reports, batch)
- ✅ 3 extended fairness tests (DPD, equalized odds, SPD)
