# TREC Fair Ranking 2022 head-to-head benchmark

Compares FAIMR's constrained-insertion FCR against FA*IR
(Zehlike CIKM 2017) on the TREC Fair Ranking 2022 train split
(50 Wikimedia WikiProject queries, ~597 k relevant docs with a
clean binary gender attribute).

## Setup

- **Dataset:** TREC Fair Ranking 2022 train split, accessed via
  `ir_datasets` (loads `trec_2022_train_reldocs.jsonl` and
  `trec_2022_articles_discrete.json.gz` to `~/.ir_datasets/`).
- **Citation:** Ekstrand et al., "Overview of the TREC 2022 Fair
  Ranking Track", [arXiv:2302.05558](https://arxiv.org/abs/2302.05558).
- **Protected attribute:** binary gender (`female` vs `male`),
  derived from each article's Wikidata gender list. Articles with
  empty or mixed gender lists are excluded.
- **Score signal:** `pred_qual` (the model-predicted article quality
  score, used as the relevance/utility for ranking).
- **Per-query cap:** top 500 docs by `pred_qual` (avoids single
  large queries dominating the aggregate).

## How to run

```bash
python -m benchmarks.trec_fair_ranking.load        # downloads + caches
python -m benchmarks.trec_fair_ranking.evaluate    # runs head-to-head
```

First run takes ~3 minutes for the metadata download. Subsequent
runs take ~3 minutes for the in-memory load and sweep across all
46 evaluable queries. Results land in `results.json`.

## Metrics

- **M1 = NDCG × AWRF** (the official TREC 2022 track metric).
  - NDCG: standard discount, ideal-DCG normalised.
  - AWRF: Attention-Weighted Rank Fairness with geometric attention
    decay = 0.85. AWRF = 1 − 0.5 × L1(actual_exposure, target);
    target is binary parity (0.5 / 0.5).
- **min-prefix-AIR** (auxiliary, the legal 4/5-Rule metric).
  - At every prefix k ≥ 10, compute the 4/5-Rule AIR.
  - Report the minimum across all valid prefixes.

We sweep FA*IR alpha in {0.05, 0.10, 0.20, 0.30, 0.40, 0.50} and
FCR threshold in {0.6, 0.7, 0.8, 0.9, 0.95, 0.99} per query, then
extract each algorithm's best M1 at each matched AWRF target.

## Headline result

### M1 = NDCG × AWRF (TREC's official metric)

| Min AWRF target | FCR eligible | FA*IR eligible | Mean ΔM1 (FCR − FA*IR) | FCR wins | FA*IR wins |
|---|---:|---:|---:|---:|---:|
| ≥ 0.70 | 28/46 | 25/46 | **+0.003** | **13** | 10 |
| ≥ 0.80 | 12/46 | 12/46 | -0.014 | 4 | **7** |
| ≥ 0.85 | 7/46 | 8/46 | -0.013 | 2 | **4** |
| ≥ 0.90 | 4/46 | 6/46 | -0.009 | 2 | 2 |
| ≥ 0.95 | 2/46 | 3/46 | **+0.036** | **1** | 0 |

**Across all AWRF targets, total head-to-head wins are essentially
tied: FCR 22, FA\*IR 23.** Neither algorithm dominates the M1
metric on this benchmark. FA\*IR has a small edge at the mid-band
(0.80-0.85 AWRF); FCR has the edge at very low (0.70) and very
high (0.95) AWRF bands.

### Auxiliary metric: min-prefix-AIR (legal 4/5-Rule)

This is where FAIMR FCR's design pays off. The 4/5 Rule is the
US EEOC anti-discrimination threshold; reaching it at every prefix
is what production audit deployments care about.

| AIR target | FCR reaches | FA*IR reaches | Ratio |
|---|---:|---:|---:|
| ≥ 0.50 | **42 / 46** (91%) | 32 / 46 (70%) | 1.3× |
| ≥ 0.60 | **42 / 46** (91%) | 27 / 46 (59%) | 1.6× |
| ≥ 0.70 | **32 / 46** (70%) | 9 / 46 (20%) | 3.6× |
| **≥ 0.80 (4/5 Rule -- legal standard)** | **21 / 46** (46%) | **5 / 46** (11%) | **4.2×** |

**FAIMR FCR reaches the legal 4/5-Rule fairness standard on 4×
more queries than FA*IR**, replicating the capability finding from
the synthetic benchmark (`benchmarks/fair_ranking/README.md`).

## Why FA*IR wins (modestly) on AWRF

AWRF measures **soft proportional exposure with geometric position
weighting**. FA*IR's statistical floor (m_alpha table) is a
natural fit for this metric: it permits the protected count to dip
below proportion as long as it stays statistically plausible, and
this looseness lets FA*IR retain more of the top-of-rank utility.

FAIMR FCR's per-prefix AIR floor is a STRICTER constraint -- it
forbids the protected count from dipping below the 4/5 threshold
at any prefix. That strictness costs a small amount of NDCG and
hence M1, but is what makes it reach AIR ≥ 0.80 on 4× more
queries.

**This is the canonical fairness-utility trade-off, and the
TREC benchmark surfaces it cleanly: M1 (which rewards soft
exposure) and AIR (which rewards hard proportional floors)
measure different things, and the two algorithms optimise for
different sides of the trade-off.**

## Honest framing for the paper

FAIMR's claim on this benchmark is NOT "we beat FA\*IR on M1".
The claim is:

1. **FAIMR FCR matches FA\*IR on the TREC M1 metric** (22-23 wins
   essentially tied across all AWRF targets and 46 queries).
2. **FAIMR FCR clearly beats FA\*IR on the legal 4/5-Rule metric**
   (21/46 queries vs 5/46 -- a 4.2× capability advantage).
3. Both algorithms preserve within-group score order
   (a property of their respective queue-based constructions).
4. FAIMR FCR additionally provides a Pareto frontier of
   (AIR, displacement) trade-offs, a termination proof in code,
   and an algorithm-provenance field in the report.

The "M1 ties, AIR dominates" pattern matches the synthetic FA*IR
benchmark verbatim -- two head-to-heads, same finding, replicated
across a synthetic protocol and a real Wikimedia editor task.

## Limitations

- We use the **train split** (publicly accessible via ir_datasets).
  The TREC evaluation split's qrels are only available to track
  participants. Future work: register with NIST for the 2024 track.
- Binary gender simplification (`female` vs `male`) excludes
  non-binary and mixed-gender articles (~3.5% of the rel pool).
- Per-query top-500 cap is a practical concession; for the very
  largest queries (76k+ rel docs) the cap drops them to a more
  tractable size that still preserves the algorithmic comparison.

## Citation

```bibtex
@inproceedings{ekstrand2022trec,
  title  = {Overview of the TREC 2022 Fair Ranking Track},
  author = {Ekstrand, Michael D. and McDonald, Graham and
            Raj, Amifa and Johnson, Isaac},
  booktitle = {Proceedings of the Thirty-First Text REtrieval
               Conference (TREC 2022)},
  year   = {2022},
  url    = {https://arxiv.org/abs/2302.05558},
}
```
