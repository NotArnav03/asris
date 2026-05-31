# Changelog

All notable architectural changes to FAIMR.  Format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); the project
itself follows semantic versioning at the **classifier model** level
(see `fairness/names/model_card.json`'s `version` field), while the
codebase uses dated milestones.

## [2026-05-31] Fairness Audit Hardening — Module & Infrastructure Pass

Audit-driven correctness fixes to every secondary module, plus CI,
reproducibility, and documentation infrastructure.

### Added

- **GitHub Actions CI** (`.github/workflows/test.yml`) with pytest on Python 3.11 + 3.12, coverage gate, and an artefact-integrity job.
- **`reproduce.py`** — single-command rebuild of every derived artefact (training corpus → surnames → model → tests) with `reproducibility_manifest.json` and a `--check` mode.
- **README** rewritten to surface the 13 audit-output fields, the full `audit_ranking_bias` kwarg table, claim-to-code map for paper reviewers, and reproducibility usage.
- `evaluation/metrics.py::ndcg_at_k` now accepts `gain="linear"|"exponential"` for graded relevance.
- `ranking/ranking_utils.py::extract_skills_in_text` + `::classify_at_percentile` — shared correct helpers reused by every baseline.

### Fixed

- **`explainability/{counterfactual,explainer}.py`**: skill substring match falsely linked "Java" inside "JavaScript" and never matched "C++" (boundary `\b` can't anchor against `+`).  Replaced with look-around-anchored matcher `(?<!\w){escape(skill)}(?!\w)`.
- **`evaluation/metrics.py`**: `compute_roc_auc` returned `0.0` on undefined inputs (indistinguishable from perfectly anti-correlated); now returns `None`.  Length validation added to every metric.  `ROC-AUC` reported as `flat` AND `mean_per_query`.
- **`evaluation/cross_validator.py::paired_t_test`**: declared a "winner" even when `p > 0.05`.  Now returns `"tie"` and exposes `higher_mean_model` as the point-estimate fallback.
- **`ranking/learning_to_rank.py`**: docstring claimed 6 features but only 5 were appended.  `resume_word_count` and `jd_word_count` now correctly added; `use_label_encoder=False` removed (deprecated in xgboost ≥ 1.6); train/test shuffle moved off the global numpy RNG.
- **`ranking/*_baseline.py` + `*_eval.py`**: strict `>` boundary silently rejected candidates at the percentile cutoff; missing-pair handling diverged across the 4 scripts.  Centralised through `classify_at_percentile` (`>=`) and consistent skip-on-missing.
- **`embeddings/embedding_manager.py`**: `encode_tfidf` used a singleton vectorizer, corrupting call-1's vectors when call-2 re-fit on a different corpus.  Now builds a fresh `TfidfVectorizer` per call.  Cache key now folds in the `fit_corpus` SHA-256 so the same `texts` with different fit corpora resolve to different cache entries.
- **`preprocessing/text_normalizer.py`**: `normalize_unicode` silently destroyed non-Latin scripts.  New `ascii_only=False` mode preserves Arabic / Chinese / Devanagari / Hebrew / Korean characters using NFKC.  Bullet normalisation reordered to run BEFORE unicode-stripping (pre-existing ordering bug surfaced once a spaCy import-failure mask was lifted).
- **`api/server.py`**: removed eager `get_embedding_manager()` from the FastAPI startup event (broke CI when sentence-transformers isn't installed).  Lazy loads now raise a typed `_MLDepMissing` (HTTP 503) instead of an uncaught `ImportError`.  TestClient created with `raise_server_exceptions=False`.

### Test count

228 → from 134 at the start of this pass.

---

## [2026-05-30] API Hardening — Token-bucket → GCRA

### Changed

- **`api/server.py`** — replaced the homegrown sliding-window deque with `throttled-py` GCRA limiter.
  - Per-endpoint quotas (`/audit` 10/min burst 5, `/health` 1000/min, etc.).
  - `Retry-After` + `X-RateLimit-Limit/Remaining/Reset` headers on every 429.
  - Trusted-proxy `X-Forwarded-For` resolution.
  - Allowlist for monitoring IPs.
  - Optional `X-API-Key` auth.
  - Tightened CORS to env-configured origins.
- API version bumped 1.0.0 → 1.1.0.

---

## [2026-05-28..30] Tasks 1–10 + Ultimate-Maxxing Hardening

### The 13-field audit ships

- **#1** Strict honorific detection (`Ms.` must precede a Title-cased non-denylist name).
- **#2** Vocab cleanup: Chinese surnames removed from male list; Korean unisex syllables relocated to `_UNISEX_NAMES`; import-time invariant prevents reintroduction.
- **#3** Calibrated char-ngram classifier (45 230-row corpus, isotonic-calibrated, holdout ECE 0.012) with hybrid lookup + model fallback.
- **#4** Detection-coverage gate (50% floor → `inconclusive_low_detection_coverage`).
- **#5** Demographic-parity statistics: size-weighted DPD, chi-squared independence, Theil-T inequality.
- **#6** Directional EEOC AIR with `protected_group` + `reference_group`, Wilson 95% CIs on rates and AIR ratio.
- **#7** Cutoff method decoupling: `median` / `top_k` / `percentile` / `explicit`.
- **#8** SimHash near-duplicate dedup + ballot-stuffing alert.
- **#9** Constrained-insertion FCR with termination proof and within-group order invariant.
- **#10** API input caps (per-resume 100 KB / request 50 MB / 5000 resumes) + rate limit + optional auth.

### Hardening follow-ups (#16–#46)

- Surname denylist (5 134 entries; US Census 2010 + curated multi-cultural) with `is_surname_only` discriminator that handles dual given/surname tokens like "John".
- Unicode confusables defence (NFKC + zero-width strip + Cyrillic/Greek fold).
- RTL honorific scan (Arabic, Hebrew).
- Nickname canonical mapping (80 unambiguous pairs).
- Multi-token name handling (hyphens, apostrophes, particles).
- Adaptive header window (replaces hard 200-char cap).
- Per-culture isotonic calibration (Arab ECE 0.089 → 0.047, East Asian 0.060 → 0.023, European 0.024 → 0.006).
- Language detection + per-locale denylists (en / es / fr / de / pt / it).
- Counterfactual robustness baked into the audit (`scorer` + `jd_text` kwargs).
- Historical drift detection via JSONL audit log.
- Model semver bump + lineage tracking.
- Per-resume audit trail.
- Model card JSON schema validation.

---

## [Pre-2026-05-28] v1.0

Initial release — single-AIR audit, dict-based name detection,
in-process token-bucket limiter, no CI.
