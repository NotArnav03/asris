# FAIMR Benchmark Suite

Reproducible head-to-head comparisons of FAIMR's audit components
against published fairness baselines.  Three benchmarks, one
positioning: FAIMR as a comprehensive audit *framework* rather than a
single-metric SOTA classifier.

| Benchmark | Domain | What FAIMR is measured on | Reference |
|---|---|---|---|
| `bias_in_bios/` | Resume / biography fairness | Per-occupation TPR gender gap; FAIMR's pronoun-based gender attribution accuracy | [De-Arteaga et al. 2019 (FAccT)](https://arxiv.org/abs/1901.09451) |
| `ssa_name_gender/` | Name → gender classification | Accuracy, ECE, ROC-AUC on a year-stratified SSA holdout | [Hu et al. 2021 (arXiv:2102.03692)](https://arxiv.org/abs/2102.03692) -- verified char-LSTM SOTA = 0.940 |
| `fair_ranking/` | Fairness-aware re-ranking algorithm | NDCG-displacement Pareto curve vs FA*IR | [Zehlike et al. 2017 (CIKM)](https://arxiv.org/abs/1706.06368) |
| `trec_fair_ranking/` | Fair re-ranking on TREC Wikipedia editor task | M1=NDCG*AWRF vs FA*IR; auxiliary min-prefix-AIR | [TREC 2022 Fair Ranking (Ekstrand)](https://arxiv.org/abs/2302.05558) |

Each subdirectory contains:

- `load.py` — dataset acquisition (fetch from the canonical public
  source; do not redistribute weights or labels in this repo).
- `evaluate.py` — runs FAIMR's audit + the relevant baseline, writes
  `results.json` with numbers a reviewer can grep for.
- `results.json` — committed reference numbers from the last run on
  the maintainer's machine, with Python + library versions in scope.
- `README.md` — citation, prior SOTA numbers, FAIMR's results,
  reproduction instructions.

## Running a benchmark

```bash
python benchmarks/bias_in_bios/load.py        # downloads + caches
python benchmarks/bias_in_bios/evaluate.py    # runs FAIMR + writes results
```

Each evaluate.py is self-contained, deterministic (fixed seed
`20251128`), and prints a comparison table at the end.

## Positioning for paper review

FAIMR's claim is **not** "we beat every SOTA on every metric." The
honest, verified claims are:

1. **Bias in Bios**: FAIMR's pronoun-based gender attribution is
   97.7% accurate on covered bios (98.6% coverage) -- inside the
   debiased-BERT band on the *attribution* sub-task. On the TPR-gap
   sub-task, FAIMR's TF-IDF+LR pipeline reports mean-abs 0.0887,
   which translates to **GAP_RMS ~0.10**, comparable to the published
   INLP-debiased BERT (GAP_RMS 0.095, Ravfogel 2020 Table 2). The
   RoBERTa+LEACE plugin in `faimr_plus/bias_in_bios_roberta_inlp/`
   targets strictly beating INLP-BERT via the closed-form optimal
   linear concept erasure (LEACE, Belrose NeurIPS 2023). Concrete
   numbers: `benchmarks/bias_in_bios/results.json`.
2. **SSA name-gender**: FAIMR's hybrid lookup + char-ngram + per-
   culture calibration + char-LSTM plugin hits **0.9747 accuracy** on
   the canonical-name slice (>=50 years of SSA attestation) -- **+3.5
   points above the verified published char-LSTM SOTA of 0.940**
   (Hu 2021 Table 6). On full-SSA the hybrid is at 0.9393, essentially
   tying Hu's char-LSTM and beating Hu's char-BERT (0.930). The
   per-attestation stratification shows clean degradation toward the
   rare tail, where no architecture recovers full accuracy. Full
   numbers and an honest account of where the per-culture calibration
   mildly underperforms a raw TF-IDF + LR baseline on OOD English
   names: `benchmarks/ssa_name_gender/README.md`. Published-SOTA
   citation has been corrected to Hu et al. 2021
   ([arXiv:2102.03692](https://arxiv.org/abs/2102.03692)).
3. **FA\*IR (synthetic, Zehlike 2017 protocol)**: FAIMR's
   constrained-insertion FCR matches or beats FA\*IR's NDCG at
   min-prefix-AIR ≥ 0.60 in **7 of 8 conditions**, and reaches the
   legal 4/5-Rule standard (AIR ≥ 0.80) in **all 8 conditions**
   where FA\*IR only reaches it in 2 of 8. Plus written termination
   proof, within-group order invariant, and a Pareto-frontier
   trade-off curve. Full numbers:
   `benchmarks/fair_ranking/README.md`.

4. **TREC Fair Ranking 2022 (real Wikipedia editor task)**: On the
   official M1 = NDCG × AWRF metric, FAIMR FCR and FA\*IR are
   **essentially tied across 46 queries** (22 wins vs 23). On the
   auxiliary legal 4/5-Rule metric, **FAIMR FCR reaches AIR ≥ 0.80
   on 4.2× more queries than FA\*IR** (21/46 vs 5/46). The pattern
   replicates the synthetic finding: M1 ties, AIR dominates. Full
   numbers: `benchmarks/trec_fair_ranking/README.md`.

The novel contributions FAIMR has that aren't measured on any
benchmark above (Unicode-confusable defence, RTL honorifics, drift
gate, embedded counterfactual robustness, dual soft/hard AIR with
conservative pass) are **methodology contributions**, not benchmark
SOTA claims.  They make the audit defensible at production scale —
but they don't have a published competitor to "beat."

## Reproducibility

All three benchmarks pin random seeds and library versions.  The
top-level `python reproduce.py` script does NOT run benchmarks (they
require external dataset downloads); use `python -m benchmarks.run`
to drive the full suite once the datasets are cached locally.
