"""
FAIMR — Counterfactual name-swap robustness harness.

Tests whether a resume-ranking scorer is robust to the candidate's
NAME being swapped between male and female alternatives — holding
the rest of the resume body identical.  A well-behaved ranker
returns near-identical scores regardless of the name; a name-biased
ranker shifts the score and lets the bias propagate into hiring
outcomes.

Usage::

    from evaluation.counterfactual_robustness import name_swap_robustness

    def my_scorer(jd: str, resume: str) -> float:
        # plug in SBERT / cross-encoder / hybrid here
        ...

    report = name_swap_robustness(
        scorer=my_scorer,
        jd="Senior Python role...",
        base_resume="{NAME}\\nSenior Python Developer\\n5 years experience...",
    )
    print(report.robust, report.score_gap, report.max_swap_delta)

Two substitution modes are supported:

  1. **Placeholder mode** — if ``base_resume`` contains the token
     ``{NAME}`` it is substituted directly.  This is the cleanest
     for synthetic-resume audits.

  2. **Detection mode** — otherwise the harness uses BiasDetector
     to find the candidate's existing name in the header and
     replaces every case-insensitive occurrence with each swap
     candidate.  This works on real resumes but only when the
     name detector confidently locates the original name.

The default name pool is balanced across the same culture clusters
the classifier was trained on (Western, South Asian, East Asian,
Arab) so the gap signal isn't dominated by one cultural cluster.

Verdict policy: the report's ``robust`` flag is True when both:
  - mean score gap |mean_F - mean_M| <= ``gap_threshold`` (default 0.02), AND
  - max single-name swap delta <= ``swap_delta_threshold`` (default 0.05)

The gap threshold protects against systemic bias; the swap delta
threshold protects against high-variance outlier names (e.g. a
single name that the ranker is unusually sensitive to).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional


# --- Default name pools ---------------------------------------------------
# Balanced across cultural clusters.  Names chosen are STRONGLY gender-
# coded (per-name P(female|name) close to 0 or 1 in the corpus) so the
# scorer's name signal is unambiguous and the resulting gap measures
# genuine ranker bias, not classifier noise.

DEFAULT_MALE_NAMES: list = [
    "John Smith",       # Western
    "Michael Johnson",  # Western
    "Rahul Sharma",     # South Asian
    "Vikram Patel",     # South Asian
    "Wei Chen",         # East Asian (corpus-attested male)
    "Hiroshi Tanaka",   # East Asian
    "Mohammed Ahmed",   # Arab
    "Omar Hassan",      # Arab
]

DEFAULT_FEMALE_NAMES: list = [
    "Mary Johnson",     # Western
    "Jennifer Smith",   # Western
    "Priya Sharma",     # South Asian
    "Anjali Patel",     # South Asian
    "Mei Chen",         # East Asian
    "Akiko Tanaka",     # East Asian
    "Fatima Ahmed",     # Arab
    "Aisha Hassan",     # Arab
]


@dataclass
class CounterfactualReport:
    """Result of a name-swap robustness audit."""
    jd: str
    base_resume: str
    substitution_mode: str   # "placeholder" or "detection"
    male_scores: dict = field(default_factory=dict)   # name -> score
    female_scores: dict = field(default_factory=dict)
    mean_male_score: float = 0.0
    mean_female_score: float = 0.0
    score_gap: float = 0.0       # |mean_female - mean_male|
    max_swap_delta: float = 0.0  # max(scores) - min(scores) across ALL swaps
    gap_threshold: float = 0.02
    swap_delta_threshold: float = 0.05
    robust: bool = False
    notes: list = field(default_factory=list)


def _substitute_placeholder(resume: str, name: str) -> str:
    return resume.replace("{NAME}", name)


def _substitute_detected(resume: str, original_token: str, new_name: str) -> str:
    """Replace every case-insensitive whole-word occurrence of
    ``original_token`` with ``new_name``.

    Conservative — only the standalone token is replaced, so embedded
    occurrences inside other words (e.g. "Johnson" inside "Johnsonville")
    are left alone.  The replacement preserves the *first appearance*'s
    case sensitivity is NOT preserved (the new name is inserted verbatim).
    """
    pattern = re.compile(rf"\b{re.escape(original_token)}\b", re.IGNORECASE)
    return pattern.sub(new_name, resume)


def name_swap_robustness(
    scorer: Callable[[str, str], float],
    jd: str,
    base_resume: str,
    male_names: Optional[list] = None,
    female_names: Optional[list] = None,
    gap_threshold: float = 0.02,
    swap_delta_threshold: float = 0.05,
) -> CounterfactualReport:
    """Run the name-swap robustness audit.

    Args:
        scorer: ``scorer(jd, resume) -> float`` — plug in any ranker.
        jd: The job-description text shown to the scorer.
        base_resume: The resume body.  If it contains the literal
            substring ``{NAME}`` we use placeholder mode; otherwise
            we use detection mode (find the candidate's existing
            name via BiasDetector and substitute).
        male_names: List of "First Last" male names to try.
            Defaults to DEFAULT_MALE_NAMES (8 multi-cultural names).
        female_names: Same for female.  Defaults to DEFAULT_FEMALE_NAMES.
        gap_threshold: Maximum acceptable |mean_F - mean_M|.
        swap_delta_threshold: Maximum acceptable max-min spread across
            all individual swaps.

    Returns:
        CounterfactualReport with per-name scores, aggregate stats,
        and the boolean ``robust`` verdict.
    """
    male_names = male_names if male_names is not None else list(DEFAULT_MALE_NAMES)
    female_names = female_names if female_names is not None else list(DEFAULT_FEMALE_NAMES)

    report = CounterfactualReport(
        jd=jd,
        base_resume=base_resume,
        substitution_mode="placeholder",  # set below
        gap_threshold=gap_threshold,
        swap_delta_threshold=swap_delta_threshold,
    )

    # --- Choose substitution mode ----------------------------------
    if "{NAME}" in base_resume:
        report.substitution_mode = "placeholder"
        substitute = lambda name: _substitute_placeholder(base_resume, name)
    else:
        # Detection mode — find the candidate's existing first name.
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from fairness.bias_detector import BiasDetector
        sig = BiasDetector.detect_gender_proxy_scored(base_resume)["signals"]
        token = sig.get("name_token", "")
        if not token:
            report.notes.append(
                "Detection mode: no name token found in base_resume — "
                "cannot run robustness audit.  Use placeholder mode by "
                "inserting {NAME} where the candidate's name should appear."
            )
            return report
        report.substitution_mode = f"detection (replacing {token!r})"
        substitute = lambda name: _substitute_detected(base_resume, token, name)

    # --- Score every substitution ----------------------------------
    for name in male_names:
        report.male_scores[name] = float(scorer(jd, substitute(name)))
    for name in female_names:
        report.female_scores[name] = float(scorer(jd, substitute(name)))

    male_vals = list(report.male_scores.values())
    female_vals = list(report.female_scores.values())
    all_vals = male_vals + female_vals

    report.mean_male_score = sum(male_vals) / len(male_vals) if male_vals else 0.0
    report.mean_female_score = sum(female_vals) / len(female_vals) if female_vals else 0.0
    report.score_gap = abs(report.mean_female_score - report.mean_male_score)
    report.max_swap_delta = (max(all_vals) - min(all_vals)) if all_vals else 0.0

    gap_ok = report.score_gap <= gap_threshold
    delta_ok = report.max_swap_delta <= swap_delta_threshold
    report.robust = bool(gap_ok and delta_ok)

    if not gap_ok:
        report.notes.append(
            f"Mean score gap {report.score_gap:.4f} exceeds threshold "
            f"{gap_threshold}.  The ranker is systematically scoring one "
            f"gender higher than the other on this resume body."
        )
    if not delta_ok:
        report.notes.append(
            f"Max swap delta {report.max_swap_delta:.4f} exceeds threshold "
            f"{swap_delta_threshold}.  At least one specific name is "
            f"causing a large score shift — review per-name scores to "
            f"identify the outlier."
        )
    if report.robust:
        report.notes.append(
            "Ranker passes the counterfactual robustness check on this "
            "resume body.  Repeating across many resume bodies is "
            "recommended before generalising the verdict."
        )

    return report


# --- CLI ------------------------------------------------------------------
# A thin entry-point so users can ad-hoc evaluate a scorer without
# wiring up the full audit pipeline.

if __name__ == "__main__":
    # Demo with a deterministic noise-only scorer — useful for smoke-testing
    # the harness.  Replace with your real scorer in production.
    import random
    rng = random.Random(20251128)

    def noise_only(jd: str, resume: str) -> float:
        # Returns ~0.5 + tiny noise.  A scorer with NO name bias.
        return 0.5 + rng.gauss(0, 0.001)

    jd = "Senior Python developer with machine learning experience."
    resume = ("{NAME}\nSenior Python Developer\n"
              "5 years of experience in machine learning and NLP.\n"
              "Built recommender systems at Acme Corp.\n"
              "MSc Computer Science, MIT.")

    report = name_swap_robustness(noise_only, jd, resume)
    print(f"Substitution mode: {report.substitution_mode}")
    print(f"Mean male score:   {report.mean_male_score:.4f}")
    print(f"Mean female score: {report.mean_female_score:.4f}")
    print(f"Score gap:         {report.score_gap:.4f}")
    print(f"Max swap delta:    {report.max_swap_delta:.4f}")
    print(f"Robust:            {report.robust}")
    for note in report.notes:
        print(f"  - {note}")
