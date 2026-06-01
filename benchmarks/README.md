# FAIMR Benchmark Suite

Reproducible head-to-head comparisons of FAIMR's audit components
against published fairness baselines.  Three benchmarks, one
positioning: FAIMR as a comprehensive audit *framework* rather than a
single-metric SOTA classifier.

| Benchmark | Domain | What FAIMR is measured on | Reference |
|---|---|---|---|
| `bias_in_bios/` | Resume / biography fairness | Per-occupation TPR gender gap; FAIMR's pronoun-based gender attribution accuracy | [De-Arteaga et al. 2019 (FAccT)](https://arxiv.org/abs/1901.09451) |
| `ssa_name_gender/` | Name → gender classification | Accuracy, ECE, ROC-AUC on a year-stratified SSA holdout | [US SSA national baby names](https://www.ssa.gov/oact/babynames/limits.html) |
| `fair_ranking/` | Fairness-aware re-ranking algorithm | NDCG-displacement Pareto curve vs FA*IR | [Zehlike et al. 2017 (CIKM)](https://arxiv.org/abs/1706.06368) |

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

FAIMR's claim is **not** "we beat every SOTA on every metric."
That's not honest for a methodology paper.  The claim is:

1. **Bias in Bios**: FAIMR's pronoun-based attribution matches the
   accuracy of an explicit-gender baseline, AND the audit pipeline
   surfaces per-occupation TPR gaps without requiring access to the
   gender labels (which is the standard published setup).  Concrete
   number: `benchmarks/bias_in_bios/results.json::attribution_accuracy`.
2. **SSA name-gender**: FAIMR's hybrid lookup + char-ngram + per-
   culture calibration hits **0.9747 accuracy / 0.0224 ECE** on the
   canonical-name slice (≥50 years of SSA attestation), inside the
   published char-LSTM band of 0.95--0.97. Stratified per-attestation
   breakdown shows clean degradation toward the rare tail, where no
   architecture recovers full accuracy. Full numbers and an honest
   account of where the per-culture calibration mildly underperforms a
   raw TF-IDF + LR baseline on OOD English names:
   `benchmarks/ssa_name_gender/README.md`.
3. **FA\*IR**: FAIMR's constrained-insertion FCR achieves equal or
   better NDCG-at-equal-AIR than FA*IR, with the additional
   guarantees of within-group order preservation and a written
   termination proof.

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
