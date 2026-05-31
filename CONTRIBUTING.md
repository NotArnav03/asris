# Contributing to FAIMR

Welcome.  This project sits at the intersection of "fairness research"
and "production audit tooling," which means correctness invariants are
load-bearing — accidentally breaking one can produce subtly wrong
audit numbers without crashing anything.  Before editing anything in
`fairness/`, `ranking/fairness_ranker.py`, or `evaluation/metrics.py`,
please read this file end-to-end.

## Setup

```bash
git clone https://github.com/NotArnav03/FAIMR.git
cd FAIMR
pip install -r requirements.txt
python reproduce.py --check         # verify your checkout matches the manifest
pytest tests/test_core.py -v        # 228+ regression tests
```

## Invariants editors MUST preserve

These are the load-bearing assumptions the audit relies on.  Each has
a regression test; if you find yourself disabling one, you're almost
certainly breaking something downstream.

### 1. Name vocab disjointness

`fairness/names/seed_lists.py` enforces at import time that
`GENDERED_NAMES["male"]`, `GENDERED_NAMES["female"]`, and
`_UNISEX_NAMES` are pairwise disjoint.  A token in two sets silently
cancels — every candidate with that name lands in `"unknown"` and gets
dropped from the AIR denominator.  Past incidents: `chen`, `li`,
`wang` (surnames in male list); `hyun` (in both male and female).

### 2. Surname denylist coverage floor

`tests/test_core.py::TestSurnameCoverage` asserts the denylist achieves
≥ 85% coverage per culture and ≥ 95% overall.  If you remove a token
from `data/names/surnames.csv`, run `python data/names/validate_surnames.py`
first.

### 3. Per-culture calibration improvement

`tests/test_core.py::test_per_culture_calibration_improved_over_global_baseline`
asserts a majority of culture clusters improve ECE over the
global-only baseline.  If you retrain and the per-culture isotonic
fits regress, the test fires.

### 4. Model SHA-256 round-trip

`model_card.json::integrity.sha256` MUST match
`hashlib.sha256(model.pkl).hexdigest()`.  Both `train_classifier.py`
(write side) and `classifier.py::_load_model` (read side) enforce this.
A mismatch becomes a `[CRITICAL]` recommendation in every audit.

### 5. Audit verdict priority order

`audit_ranking_bias` resolves the publish-ready `verdict` field in
this order — EARLIER gates dominate later ones:

  1. `integrity_violated` → `[CRITICAL]` (separate channel)
  2. `coverage_rate < 0.50` → `inconclusive_low_detection_coverage`
  3. `ece_coverage < 0.50` → `inconclusive_low_ece_coverage`
  4. `weighted_ece > 0.10` → `inconclusive_high_drift`
  5. `0.05 < weighted_ece ≤ 0.10` + AIR passes → `pass_with_drift_warning`
  6. AIR fails → `fail`
  7. otherwise → `pass`

The conservative ordering matters: an `inconclusive` due to low
coverage is more informative than a `fail` computed from the same
small denominator.

### 6. Within-group order preservation in the FCR

`ranking/fairness_ranker.py::FairnessConstrainedRanker.rerank`
guarantees that candidates within a single demographic group appear
in the fair ranking in the same order they appeared in the input
(which IS the within-group score order).  Any rearrangement WITHIN a
group is gratuitous displacement the AIR constraint never required.
The report's `within_group_order_preserved` field surfaces this; the
test `test_within_group_order_is_preserved` fires if violated.

### 7. Look-around-anchored skill matching

Skill extractors (`explainability/counterfactual.py`,
`explainability/explainer.py`, `ranking/ranking_utils.py`,
`ranking/hybrid_eval.py`) use
`(?<!\w){re.escape(skill)}(?!\w)` — NOT `\b...\b` (fails on trailing
`+`/`#`) and NOT plain `skill in text` (matches "java" inside
"javascript").  If you add a new skill extractor, use the same
pattern.

### 8. Per-call TF-IDF vectorizer

`embeddings/embedding_manager.py::encode_tfidf` constructs a FRESH
`TfidfVectorizer` per call.  The singleton pattern in the prior code
corrupted call-1's vectors when call-2 re-fit on a different corpus.
If you add a new TF-IDF entry point, do not reuse a long-lived
`TfidfVectorizer` instance.

### 9. Reproducibility — `python reproduce.py --check` must pass

Every PR that touches `data/names/` build scripts, the classifier
trainer, or anything that affects derived artefacts MUST update the
manifest by running `python reproduce.py` and inspecting the diff.
CI runs `--check` to verify hashes line up across machines.

## Commit conventions

- One logical change per commit.  Audit fixes don't pile up with API
  refactors.
- Commit messages explain the WHY, not just the what.  Past commit
  messages document the bug being fixed AND the regression test that
  guards against reintroduction.  Example:
  > Purge surnames and dedupe collisions from name vocab.  Closes the
  > "Sarah Chen silently lands in unknown" cancellation pathology
  > flagged in the security review; the import-time invariant in
  > seed_lists prevents reintroduction.
- For every correctness fix, add a regression test in the same
  commit.  Bug-fix-without-test PRs will be asked to add one.

## Branches & PRs

- Branch off `main`.
- Open a PR; CI must clear (228+ tests + coverage gate ≥ 60% on the
  scoped modules).
- If you're touching anything in `fairness/`, also run
  `python reproduce.py --check` and paste the output (or a "manifest
  hashes unchanged" note) into the PR description.

## Adding a new audit field

If you want `audit_ranking_bias` to surface a new field:

1. Compute it inside `audit_ranking_bias`, ideally in its own clearly-
   marked block with a `# --- Section title ----` banner.
2. Add the field to the top-level `results: dict` definition near the
   start of the function, so the schema is visible up-front.
3. Add at least one regression test asserting the field exists with
   the expected shape.
4. Update the "Why FAIMR" table in `README.md` with a one-line
   semantics description.
5. Update the `audit["model_card_validation"]` schema if the field
   becomes required for downstream consumers.

## Adding a new gate to the verdict

The verdict field is conservative-of-many-gates by design.  To add a
new gate (e.g. a future EEOC compliance check):

1. Decide its priority position (see invariant #5).  Earlier = more
   informative.  Drift gates usually come BEFORE math gates.
2. Add the threshold constants near the top of `bias_detector.py`,
   commented with WHY the chosen threshold is defensible.
3. Compute the gate inside `audit_ranking_bias` BEFORE the existing
   verdict assignment.
4. Add an `inconclusive_*` variant to the verdict string vocabulary
   AND a `[INCONCLUSIVE]` recommendation explaining what tripped it.
5. Write 2 tests: one that proves the gate fires when it should, and
   one that proves it does NOT fire on healthy input.

## What we won't merge

- New skill extractors that use `skill in text` substring matching.
- AIR formulations that compute `min/max` and call it "directional".
- New verdict states without a corresponding regression test.
- Changes to the SHA-256 model integrity that weaken the round-trip.
- Anything that downgrades the dedup + drift + coverage gate
  thresholds without an attached documented rationale.

## Questions

Open a discussion on the GitHub repository's "Discussions" tab.
Architecture questions get priority when they reference a specific
code location and one of the invariants above.
