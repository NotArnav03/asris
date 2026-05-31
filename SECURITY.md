# Security Policy

## Threat Model

FAIMR is a fairness auditing pipeline.  Its security properties divide
into two categories:

1. **Audit integrity** — the system's reported AIR / verdict must not
   be exploitable by adversarial input (a candidate's resume, a
   ballot-stuffing attacker, a tampered model file).
2. **API hardening** — the production HTTP surface must withstand
   casual abuse (input flooding, oversized payloads, cross-origin
   requests from untrusted browsers).

### In scope

- **Adversarial classification bypass.**  Attempts to flip a
  candidate's detected gender by editing their resume.  Closed
  attack surface includes: honorific-acronym injection ("MS Office",
  "MR-aware"), Cyrillic/Greek confusables, zero-width insertions,
  fullwidth Latin, surname-only headers, padded-header bypass,
  hyphenated / particle-prefixed name spoofing, and nickname
  ambiguity.
- **Ballot stuffing.**  Submitting the same (or near-identical)
  resume many times to inflate one group's AIR.  Closed via exact
  SHA-256 dedup and SimHash near-duplicate detection (Hamming ≤ 3).
- **Model file tampering.**  Modifying `fairness/names/model.pkl`
  after training.  Detected via SHA-256 round-trip from
  `model_card.json`; audit emits `[CRITICAL]` recommendation.
- **API input flooding / DoS.**  Closed via GCRA rate limiting,
  per-resume and per-request body caps, and trusted-proxy IP
  resolution.
- **Calibration drift exploitation.**  An attacker engineering a
  corpus where the classifier is poorly calibrated (e.g. all Arab
  names where per-cluster ECE is highest) would otherwise extract
  a "pass" verdict from an unreliable AIR.  Closed via the
  calibration-drift gate (`inconclusive_high_drift`).

### Out of scope

- **Self-reported group membership.**  The candidate authors the
  text we infer group from.  An adversary can always disappear from
  the audit denominator by stripping every gender signal (no name,
  no pronouns, unisex first name).  This is a structural limit of
  any name-proxy gender detector; the audit surfaces detection
  coverage so reviewers can see when this is happening (the
  detection-coverage gate refuses to publish a verdict below 50%).
- **Non-binary gender.**  Treated as `unknown`.  See `bias_detector`
  module docstring for the documented limitation.
- **Adversarial training-data poisoning.**  An attacker submitting
  patches to `fairness/names/seed_lists.py` could degrade the
  classifier.  Defence is code review + the import-time vocab
  invariant + the per-culture calibration regression tests.
- **Production deployment hardening beyond the API surface.**
  Network policy, log scrubbing, secret rotation, container
  hardening — out of scope for the audit code itself.

## Reporting a Vulnerability

If you discover a way to:

- Flip a candidate's detected gender via resume editing in ways NOT
  covered by the existing regression suite, OR
- Inflate / suppress an AIR verdict beyond what dedup + drift gate
  + coverage gate already catch, OR
- Cause the audit to publish a `pass` verdict on a corpus where
  EEOC-defined adverse impact is real, OR
- Bypass the rate limiter, API key, or input-size caps,

please open a **private security advisory** on the GitHub repository
("Security" tab → "Report a vulnerability") rather than a public
issue.  Include:

- A minimal reproducing input (resume text, audit kwargs, expected
  vs actual `verdict`).
- The commit SHA the issue was found on.
- Your assessment of severity (informational / low / medium / high
  / critical).

We aim to acknowledge within a week and patch high/critical issues
in a follow-up release.  For paper-cited claims that turn out to be
exploitable, we'll also update the `verdict` field's documentation
and the `Why FAIMR` table in `README.md`.

## Verified Threats Closed

The regression suite in `tests/test_core.py` contains a named test
for every attack we've explicitly demonstrated and closed.  Grep for
`test_` lines beginning with one of:

- `test_ms_office_does_not_fire`
- `test_ms_in_cs_does_not_fire`
- `test_*_acronym_does_not_fire`
- `test_*_confusable_*`
- `test_zero_width_*`
- `test_fullwidth_*`
- `test_cyrillic_*`
- `test_padded_header_does_not_evade`
- `test_*_does_not_match_inside_*`  (substring false positives)
- `test_*_surname_only_*`
- `test_audit_drops_exact_duplicate_resumes`
- `test_audit_flags_ballot_stuffing_*`
- `test_audit_detects_near_duplicates_via_simhash`
- `test_drift_gate_overrides_pass_when_corpus_is_high_drift`
- `test_drift_low_coverage_forces_inconclusive`
- `test_low_coverage_forces_inconclusive_verdict`
- `test_integrity_violation_surfaces_critical_recommendation`
- `test_rate_limit_429_*`
- `test_audit_rejects_oversized_*`
- `test_audit_rejects_too_many_resumes`
- `test_api_key_required_when_set`

If your reported vulnerability is already covered by one of these
tests passing in CI, your reproduction must either invalidate the
existing test or surface a NEW exploit path that the test doesn't
cover.

## Cryptographic Choices

- **Model integrity:** SHA-256.  Stored under
  `model_card.json::integrity.sha256`, recomputed on every classifier
  load, mismatch surfaces as `[CRITICAL]` in the audit recommendation
  chain.
- **Embedding cache keys:** SHA-256, truncated to 16 hex chars
  (64 bits).  No security implication — this is a cache, not an
  authentication primitive — but auditors should never see MD5 in
  a hash chain, so we use SHA-256 anyway.
- **API authentication:** constant-time string comparison via direct
  `==` (Python strings; not constant-time, but the API key is a
  long random string and timing attacks are not in our threat model
  for an internal audit service).
