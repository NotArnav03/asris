# FA*IR vs FAIMR Constrained-Insertion FCR

Head-to-head fair-ranking benchmark comparing FAIMR's
constrained-insertion FCR against the published FA*IR algorithm
(Zehlike et al., CIKM 2017).

## What's being compared

**FA\*IR** (Zehlike 2017) enforces a *statistical* minimum: at every
prefix k, the protected count must be ≥ the binomial-CDF-inverse
floor `m_α(k) = F⁻¹_{Binom(k,p)}(α)`. The default α=0.1 is permissive
-- it allows the AIR to drop substantially at small k as long as the
drop is consistent with random sampling.

**FAIMR FCR** (this paper) enforces a *deterministic* minimum: at
every prefix k, the 4/5-Rule AIR ≥ τ (default 0.8). The algorithm
uses a constrained-insertion construction that:

  - Preserves within-group score order by popping queue heads
  - Has a written termination proof (single pass, O(n²·|G|))
  - Produces a Pareto frontier of (AIR, displacement) trade-offs

## Protocol

  - Two groups: protected ("P") vs non-protected ("N")
  - Scores: Normal(μ_g, 1.0). μ_N = 0.5, μ_P ∈ {0.0, -0.5}
  - List sizes N ∈ {100, 500, 1000}, protected proportion p ∈ {0.2, 0.3}
  - 8 conditions total
  - FA\*IR α swept over {0.01, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50}
  - FAIMR FCR threshold τ swept over {0.6, 0.7, 0.8, 0.9, 0.95}
  - For each algorithm, take the **best-NDCG point** at each
    min-prefix-AIR target

## How to reproduce

```bash
python -m benchmarks.fair_ranking.evaluate
```

Pinned seed `20251128`, ~5 s wall-clock. Results in `results.json`.

## Headline result -- NDCG @ matched min-prefix-AIR target

| Min-prefix-AIR target | FCR wins | FA*IR wins | Mean NDCG delta (FCR − FA*IR) | Conditions |
|---|---:|---:|---:|---:|
| ≥ 0.50 | 5 | 3 | +0.0040 | 8 |
| ≥ 0.60 | **7** | 1 | **+0.0068** | 8 |
| ≥ 0.70 | **6** | 2 | **+0.0043** | 8 |
| **≥ 0.80 (4/5 Rule -- legal standard)** | **2** | **0** | **+0.0054** | 2 of 8\* |
| ≥ 0.90 | 0 | 0 | -- | 0 of 8 |

\* In 6 of 8 conditions, FA\*IR **cannot reach AIR ≥ 0.80 at any α**
in the swept range. FAIMR FCR reaches AIR ≥ 0.80 in **all 8** conditions.
This is not a small NDCG win -- it's a **capability win**.

## The capability story

The 4/5 Rule (AIR ≥ 0.80) is the legal anti-discrimination threshold
enforced by the US EEOC. FA\*IR's per-prefix statistical floor is too
loose to guarantee this even at α=0.5 (where the floor is the median
protected count under the null). FAIMR FCR is built on the AIR target
directly, so it satisfies the 4/5 Rule by construction or returns a
proof that it cannot.

The 2-of-8 FA\*IR conditions where AIR ≥ 0.80 IS reachable are
the ones where the baseline distribution is closest to proportional
already (large N, μ_protected close to non-protected). Both
algorithms perform similarly there. Where the baseline is biased
(small N, large μ-gap), only FAIMR FCR can reach the legal standard.

## At more permissive targets

At AIR ≥ 0.60 -- a reasonable "demographic parity but with room"
target -- FAIMR FCR matches or beats FA\*IR in **7 of 8 conditions**
with mean NDCG advantage **+0.68 percentage points**. The one win
for FA\*IR is the n=100, p=0.2, μ_P=-0.5 small-list case, where
FA\*IR's lookup-table approach is slightly more efficient than
FCR's greedy insertion. We document this as a known small-N regime.

## What both algorithms preserve

Both implementations preserve within-group score order in 8 of 8
conditions. FA\*IR achieves this by construction (pop queue heads).
FAIMR FCR achieves this by an explicit invariant verified after
re-ranking (`within_group_order_preserved` field in the
`FairnessReport`). Both pass the audit.

## Beyond NDCG: what FAIMR FCR adds

- **Pareto frontier**: a single FCR call returns the
  (AIR, displacement) trade-off curve. FA\*IR is a single-point
  algorithm; reproducing the curve requires re-running with each α.

- **Termination proof in code**: the FCR `FairnessReport.termination_proof`
  field is a short prose explanation of the bounded-iteration
  guarantee, useful for auditor-facing reports.

- **Algorithm provenance**: the `algorithm` field is part of the
  output so downstream tooling can verify which variant ran.

- **Quality-threshold variant** (legacy swap-based path): FCR has a
  documented `quality_threshold` knob that blocks indefensible
  promotions (e.g., displacing a 0.9-score candidate with a 0.3-score
  one). Currently active in the swap-based variant; porting to
  constrained-insertion is on the roadmap.

## Citation

```bibtex
@inproceedings{zehlike2017fair,
  title  = {FA*IR: A Fair Top-k Ranking Algorithm},
  author = {Zehlike, Meike and Bonchi, Francesco and Castillo, Carlos
            and Hajian, Sara and Megahed, Mohamed and Baeza-Yates, Ricardo},
  booktitle = {Proceedings of the 2017 ACM Conference on Information
               and Knowledge Management (CIKM)},
  year   = {2017},
  doi    = {10.1145/3132847.3132938},
}
```
