# Novel Research Contributions for Q2 Journal Publication

## Paper Framing

**Title:** *"FAIMR: A Fairness-Aware Interpretable Multi-Signal Ranking Framework for Automated Resume Screening"*

**Core narrative:** Most resume screening systems optimize purely for relevance. FAIMR introduces (1) a **Fairness-Constrained Re-ranking** algorithm that provably satisfies demographic parity without sacrificing ranking quality, and (2) **Counterfactual Explanations** that tell candidates *what would need to change* for a better rank — not just what matched.

---

## Novel Contribution 1: Fairness-Constrained Re-ranking (FCR)

> [!IMPORTANT]
> This is the **primary novel algorithm** — a post-processing re-ranker that balances relevance with demographic parity using constrained optimization.

### What's new
Existing fairness work in hiring AI either: (a) removes bias at the embedding level (pre-processing), or (b) only detects/reports bias (audit-only). FAIMR does **in-ranking fairness enforcement** — it re-ranks candidates to satisfy the 4/5 rule while minimizing displacement from the original relevance order.

### Algorithm (Constrained Rank Optimization)
```
Input: ranked list R, demographic groups G, fairness threshold τ=0.8
Output: re-ranked list R' satisfying AIR ≥ τ with minimum rank displacement

1. Compute selection rates per group at each cutoff k
2. If AIR(k) ≥ τ for all k → return R (no change needed)
3. Else → solve: minimize Σ|rank(i) - rank'(i)| subject to AIR(k) ≥ τ ∀k
4. Use a greedy swap strategy: promote highest-scored underrepresented candidates
5. Report: displacement cost, fairness gain, Pareto frontier
```

### Files to create/modify

#### [NEW] [fairness_ranker.py](file:///c:/asris/ranking/fairness_ranker.py)
- `FairnessConstrainedRanker` class
- `rerank_with_fairness(ranked_list, demographics, threshold)` → returns re-ranked list + displacement report
- `compute_pareto_frontier(scores, demographics)` → fairness-relevance tradeoff curve
- `displacement_cost(original_ranks, new_ranks)` → Kendall tau distance

#### [MODIFY] [bias_detector.py](file:///c:/asris/fairness/bias_detector.py)
- Add `demographic_parity_distance()` metric
- Add `equalized_odds()` metric
- Add `statistical_parity_difference()` metric

---

## Novel Contribution 2: Counterfactual Explanations

> [!IMPORTANT]
> Goes beyond "what matched" to "what would change your rank" — actionable feedback for candidates.

### What's new
Current explainability shows matched/missing skills. Counterfactual explanations answer: *"If this candidate had skill X, their rank would improve by Y positions"* — this is novel in resume screening literature.

### Approach
For each candidate, perturb the feature vector (add each missing skill one at a time) and measure rank change. Report the top-k most impactful skill gaps.

### Files to create/modify

#### [NEW] [counterfactual.py](file:///c:/asris/explainability/counterfactual.py)
- `CounterfactualExplainer` class
- `explain_rank_change(candidate, jd, all_candidates)` → "Adding [Python] would move you from rank #5 to rank #2"
- `skill_impact_analysis(jd, resume)` → ordered list of skills by potential rank improvement
- `generate_improvement_report(candidate)` → actionable text summary

#### [MODIFY] [explainer.py](file:///c:/asris/explainability/explainer.py)
- Integrate counterfactual results into the existing explanation pipeline

---

## Novel Contribution 3: Ablation Study & Formal Evaluation

> [!IMPORTANT]
> Required for any Q2 paper — systematic proof that each component adds value.

### Ablation experiments
| Experiment | What it tests |
|---|---|
| SBERT-only | Semantic baseline |
| TF-IDF-only | Lexical baseline |
| SBERT + TF-IDF | Multi-signal without skills |
| Full LTR (all 8 features) | Complete model |
| LTR + FCR | With fairness constraint |
| LTR + FCR + Counterfactual | Full FAIMR system |

### Statistical rigor to add
- 5-fold cross-validation (not just train/test split)
- Paired t-test between model variants
- Confidence intervals on all metrics
- Effect size (Cohen's d)

### Files to create

#### [NEW] [ablation_runner.py](file:///c:/asris/experiments/ablation_runner.py)
- Runs all 6 ablation configs
- Outputs results table (LaTeX-ready)
- Computes statistical significance between variants
- Generates fairness-relevance Pareto plot

#### [NEW] [cross_validator.py](file:///c:/asris/evaluation/cross_validator.py)
- k-fold cross-validation with per-fold metric tracking
- Paired t-test between two model configs
- Confidence interval computation
- Results formatted for paper tables

---

## Novel Contribution 4: Fairness Metrics Dashboard

#### [MODIFY] [index.html](file:///c:/asris/frontend/index.html)
- Add a **Fairness Audit** section to the frontend

#### [MODIFY] [app.js](file:///c:/asris/frontend/app.js)
- Visualize fairness metrics: AIR gauge, demographic parity chart, Pareto frontier plot

#### [MODIFY] [server.py](file:///c:/asris/api/server.py)
- Add `/audit` endpoint that runs bias detection on ranked results
- Add `/rerank` endpoint that applies FCR and returns both original and fair rankings

---

## Implementation Order

1. **Fairness-Constrained Re-ranker** (`ranking/fairness_ranker.py`) — the core novel algorithm
2. **Extended fairness metrics** (modify [bias_detector.py](file:///c:/asris/fairness/bias_detector.py)) — demographic parity, equalized odds
3. **Counterfactual Explainer** (`explainability/counterfactual.py`) — the second novel contribution
4. **Cross-validator** (`evaluation/cross_validator.py`) — k-fold + significance tests
5. **Ablation runner** (`experiments/ablation_runner.py`) — systematic experiments
6. **API + frontend integration** — fairness audit UI, `/audit` and `/rerank` endpoints

## Verification Plan

### Automated Tests

All tests run from project root with:
```bash
cd C:\asris
py -m pytest tests/ -v
```

New test cases to add in [tests/test_core.py](file:///c:/asris/tests/test_core.py):
- `TestFairnessRanker`: verify FCR satisfies AIR threshold, verify minimum displacement
- `TestCounterfactual`: verify skill perturbation produces valid rank changes
- `TestCrossValidator`: verify k-fold produces correct number of folds, paired t-test logic
- `TestAblation`: verify all 6 configs run without errors

### Manual Verification
1. Run `py -m api.server`, open `http://localhost:8000`
2. Upload multiple PDFs with different names (to trigger gender proxy detection)
3. Check the fairness audit section shows AIR, demographic parity metrics
4. Verify counterfactual explanations appear in the Explain tab
