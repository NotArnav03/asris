# FAIMR Plus -- SSA char-LSTM plugin

An optional plugin that adds a bi-directional character LSTM trained
on the US SSA national baby-names corpus, used as a higher-capacity
fallback above FAIMR's hybrid lookup + char-ngram LR classifier on
the rare-tail OOD slice.

## Why a plugin and not core FAIMR?

FAIMR's design goal is a **small, dependable audit framework** with
exact-match accuracy on the popular name tail (via the lookup
fastpath) and well-calibrated probabilities everywhere else. A
character LSTM adds 600 KB of weights, requires torch as a runtime
dependency, and only meaningfully improves accuracy on the rare-tail
OOD slice that the audit pipeline already abstains on. We ship it as
an optional plugin so users who need the extra OOD accuracy can opt
in; the core install stays small.

## Architecture

- Char embedding (48-d, vocab 30: SOS/EOS/PAD/UNK + a-z)
- 1-layer bi-LSTM (hidden 96 per direction)
- Max-pool over time + last-state concat (384-d)
- MLP head: Linear(384 → 96) + GELU + Dropout(0.35) + Linear(96 → 1)
- ~150 k parameters, 607 KB pickled

## Training

Run from the repo root:

```bash
python -m faimr_plus.ssa_char_lstm.train
```

- Data: SSA per-name aggregate (5,572 names with n_years ≥ 3)
- Deterministic 80/10/10 train/val/test split (hashed by name)
- Hard labels with 0.05 label smoothing
- Sample-weighted BCE loss (weight = sqrt(n_years))
- AdamW (lr 1.5e-3, weight_decay 5e-4) + cosine schedule
- Early stopping on val accuracy (patience 8)
- ~3 minutes on a laptop CPU

## How the hybrid routes predictions

See `hybrid.py`. Per name:

1. **Lookup hit** -- use FAIMR's exact-match prediction unchanged.
2. **FAIMR high-confidence** (`p_female` outside `[0.35, 0.65]`) --
   ensemble FAIMR + LSTM with weights 0.45/0.55. Both signals are
   reliable; the weighted average reduces variance.
3. **FAIMR low-confidence** (inside the band) -- defer entirely to
   the LSTM. FAIMR's char-ngram LR is exactly where it has least
   information; the LSTM's sequence model carries useful residual
   signal.

## Headline numbers

Run `python -m benchmarks.ssa_name_gender.evaluate` and read the
"Stage D" rows. Latest run (seed 20251128):

| Configuration | Full-SSA Acc | ROC-AUC | ECE | OOD-Holdout Acc |
|---|---:|---:|---:|---:|
| FAIMR alone | 0.9216 | 0.9747 | 0.0244 | 0.8170 |
| Same-data TF-IDF + LR baseline (OOD only) | -- | -- | -- | 0.8497 |
| **FAIMR + char-LSTM hybrid (this plugin)** | **0.9393** | **0.9808** | **0.0197** | **0.9046** |
| Hybrid Δ vs FAIMR alone | **+0.0177** | +0.0061 | -0.0047 | **+0.0876** |
| Hybrid Δ vs same-data baseline | -- | -- | -- | **+0.0549** |

- **+5.5 pts vs the same-data TF-IDF+LR baseline on OOD names** --
  this was the long-standing gap from `benchmarks/ssa_name_gender/`.
  Plugin closes and then beats it.

- **Full-SSA ECE 0.0197** is the best calibration FAIMR has shipped
  on this benchmark.

- The 50+-year-attestation slice (the fair comparison vs published
  char-LSTM 0.95-0.97 band) is **0.9747 via the lookup fastpath alone**;
  the LSTM doesn't touch those names. The plugin's contribution is
  on the rare tail where published char-LSTM numbers don't report.

## Why the full-SSA number isn't above 0.97

The public SSA mirror we use (`hadley/data-baby-names`) covers ~6 k
unique aggregated names. The published 0.95-0.97 char-LSTM numbers
in the literature train on the FULL national SSA data (~100 k unique
names) and evaluate on filtered high-attestation subsets. On the
fair-comparison slice (50+ years of attestation, the same kind of
filtering the literature uses), FAIMR's lookup-fastpath already sits
at 0.9747 -- inside the published band.

## Install

The plugin requires `torch`. The core FAIMR install does not pull
torch; install it separately:

```bash
pip install torch
```

Then verify the plugin is wired up:

```bash
python -c "from faimr_plus.ssa_char_lstm.hybrid import predict_hybrid; \
           print(predict_hybrid(['Aisha', 'Liam', 'Riley']))"
```
