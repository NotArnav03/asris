"""
FAIMR — Fairness & Bias Detection Module
Detects potential bias in resume ranking across demographic groups.
Implements the 4/5 Rule (Adverse Impact Ratio), demographic parity,
and statistical significance testing.

LIMITATIONS (gender proxy detection):
  - Name-to-gender mapping is probabilistic and culturally incomplete.
    Coverage spans Western, South Asian, East Asian, and Arab naming
    conventions, but many names worldwide are absent from the vocabulary.
  - Gender is treated as binary (male/female) for aggregate auditing only.
    This does not reflect the full spectrum of gender identity.
  - Pronoun detection is error-prone in third-person-heavy resumes.
  - Title detection (Mr./Ms.) is strong evidence but not infallible
    (e.g., "Dr." is gender-neutral; "Ms." may appear in quotes).
  - Confidence scores are heuristic and uncalibrated against ground truth.
  - This module MUST NOT be used for individual candidate decision-making.
    It is intended solely for aggregate fairness auditing.
"""

import re
import unicodedata
from typing import Optional
from collections import defaultdict

import numpy as np
import pandas as pd
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import FAIRNESS_ADVERSE_IMPACT_THRESHOLD, get_logger

logger = get_logger("fairness.bias_detector")


# --- Gender Proxy Detection -----------------------------------------------

# Pronoun and kin-term patterns (apply to lowercased full text)
GENDER_INDICATORS = {
    "male": [
        r"\bhe\b", r"\bhis\b", r"\bhim\b", r"\bfather\b",
        r"\bson\b", r"\bbrother\b", r"\bgentleman\b",
    ],
    "female": [
        r"\bshe\b", r"\bher\b", r"\bhers\b", r"\bmother\b",
        r"\bdaughter\b", r"\bsister\b", r"\blady\b",
    ],
}

# --- Honorifics ------------------------------------------------------------
# Title detection is intentionally STRICT: an honorific only fires when it
# is immediately followed by a plausible proper-name token.  The previous
# implementation used `\bms\.?\b` against lowercased text, which fired on
# "MS Office", "MS in CS", "MS Excel" — making the female title the most
# common false positive in the dataset and a single-line attack vector
# (any candidate could flip their detected gender by writing "MS Office"
# in the header summary).
#
# The new pipeline uses two layers of defence:
#
#   1. Pattern: the honorific match is case-insensitive (scoped (?i:...))
#      but the follow-on capture group is *case-sensitive* and must begin
#      with a capital letter.  This blocks "MS in CS" because "in" is
#      lowercase — the all-caps "MS" alone is no longer enough.
#
#   2. Denylist: the captured follow-on token is checked (case-insensitive)
#      against a list of degree, product, and connector words.  This blocks
#      the residual false positives where the follow-on happens to be
#      Title-cased — "MS Office" matches the pattern but "Office" is on
#      the denylist, so the honorific does not fire.
#
# The honorific scan operates on the ORIGINAL-CASE first 200 chars of the
# resume.  Pronoun and name scans continue to use the lowercased text.

_HONORIFIC_DENYLIST: frozenset = frozenset({
    # Connectors that can appear after a degree abbreviation
    "in", "of", "from", "the", "a", "an", "by", "for", "and", "or", "at",
    "with", "as",
    # Degree fields commonly written after "MS" / "MA" / "MSc"
    "computer", "science", "engineering", "mathematics", "math", "maths",
    "business", "administration", "arts", "education", "statistics",
    "economics", "finance", "marketing", "psychology", "biology",
    "chemistry", "physics", "data", "information", "technology",
    "management", "operations", "systems", "analytics", "accounting",
    "communications", "humanities", "law", "medicine", "nursing",
    "degree", "thesis", "research", "studies", "program", "programme",
    "candidate", "graduate", "honors", "honours", "minor", "major",
    # Microsoft / common tech product names that collide with "MS"
    "office", "word", "excel", "powerpoint", "outlook", "project",
    "access", "teams", "visio", "sharepoint", "onenote", "dynamics",
    "sql", "server", "windows", "azure", "visual", "studio", "code",
    "exchange", "edge", "store", "bing", "copilot", "fabric", "graph",
    # Doctor false positives — "Dr" preceding non-name terms
    "drive", "driver", "driving",
    # Other common falsely-Title-cased follow-ons
    "level", "certified", "certification", "license", "diploma",
})


def _build_honorific_pattern(tokens: list) -> re.Pattern:
    """Compile a strict honorific pattern.

    The honorific itself is matched case-insensitively (scoped flag).
    The follow-on capture group matches any Unicode word characters
    plus apostrophe and hyphen — `\\w[\\w'\\-]+` — which gives us at
    least two characters total and admits accented letters needed for
    names like "María", "Søren", "Müller", "Łukasz".  The follow-on's
    *first character* is then checked for ``isupper()`` in Python
    (Unicode-aware), and the lower-cased token is filtered against
    _HONORIFIC_DENYLIST.

    Why post-filter rather than encode "uppercase Unicode letter" in
    the regex?  Python's stdlib ``re`` has no \\p{Lu} property class.
    Enumerating Unicode uppercase ranges in a character class is
    error-prone (~600 codepoints across many ranges), so the cleanest
    correct implementation is regex-match + Python attribute check.
    """
    alts = "|".join(f"(?i:{re.escape(t)})" for t in tokens)
    # Optional period, then whitespace, then a Unicode word token of
    # length >= 2 (apostrophes and hyphens permitted to support
    # "O'Neill", "Smith-Jones").  The first-letter-uppercase
    # requirement is enforced in _honorific_fires.
    return re.compile(rf"\b(?:{alts})\.?\s+(\w[\w'\-]+)")


# Honorifics by gender.  Each list mixes English with the most common
# international forms encountered on multi-cultural resumes:
#   Spanish/Portuguese  Señor / Señora / Senor / Senora
#   French              Monsieur / Madame / Mme / Mlle / Maître
#   German              Herr / Frau
#   Italian             Signor / Signora
#   Religious/honorary  Rev / Reverend / Hon
# Accented and ASCII-fallback spellings are both listed so the audit
# works whether or not upstream text normalisation has stripped
# diacritics.
_HONORIFIC_PATTERNS: dict = {
    "male": _build_honorific_pattern([
        "Mr", "Mister", "Sir",
        "Señor", "Senor",       # Spanish/Portuguese
        "Herr",                 # German
        "Monsieur",             # French
        "Signor",               # Italian
        # NOTE: bare abbreviations "Sr", "Sr.", "M", "M." are
        # deliberately EXCLUDED — they collide with the English
        # suffix "Sr." (Senior) in "John Doe Sr." and the middle
        # initial "M" in "John M Smith".  Full forms only.
    ]),
    "female": _build_honorific_pattern([
        "Mrs", "Ms", "Miss", "Madam",
        "Madame", "Mme", "Mlle",            # French
        "Señora", "Senora",                 # Spanish/Portuguese
        "Frau",                             # German
        "Signora",                          # Italian
        # NOTE: "Sra", "Sra." excluded for the same reason as the
        # male side — abbreviation collisions outweigh coverage gain.
    ]),
    "neutral": _build_honorific_pattern([
        "Dr", "Prof", "Professor", "Mx",
        "Rev", "Reverend", "Hon", "Honorable",  # religious / honorary
        "Maître", "Maitre",                     # French legal/academic
    ]),
}


# Zero-width / format characters commonly used in confusable attacks.
# U+200B ZERO WIDTH SPACE, U+200C ZERO WIDTH NON-JOINER,
# U+200D ZERO WIDTH JOINER, U+2060 WORD JOINER, U+FEFF BOM /
# ZERO WIDTH NO-BREAK SPACE.  These render as nothing but split
# regex tokens — "M​r. Smith" is human-readable as "Mr. Smith"
# but the honorific scan sees "M", "​", "r", ".".
_ZERO_WIDTH_RE = re.compile(
    "[​‌‍⁠﻿]"
)


def _sanitise_for_detection(text: str) -> str:
    """NFKC-normalise the input and strip zero-width / BOM characters
    to neutralise the common Unicode-confusable bypasses against the
    honorific scan and the name-token extractor.

    NFKC collapses to ASCII / compatible forms:
        Fullwidth Latin            "Ｍｒ. Smith"  -> "Mr. Smith"
        Mathematical alphanumeric  "𝐌𝐫. Smith"  -> "Mr. Smith"
        Compatibility ligatures    "ﬁ"          -> "fi"

    Zero-width strip defeats:
        ZWSP inside salutation     "M\\u200Br. Smith"  -> "Mr. Smith"
        BOM at start of header     "\\uFEFFMr. Smith"  -> "Mr. Smith"

    KNOWN LIMITATION: this does NOT defend against Cyrillic-Latin
    confusables ("Мr. Smith" with U+041C Cyrillic capital Em).
    Closing that requires a Unicode confusables map (a la ICU's
    confusables.txt); it is documented but deliberately not
    implemented here because the false-positive risk of unified
    Latin/Cyrillic name handling is non-trivial.
    """
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = _ZERO_WIDTH_RE.sub("", text)
    return text


def _honorific_fires(pattern: re.Pattern, header_orig: str) -> bool:
    """Return True iff ``pattern`` matches inside ``header_orig`` with a
    follow-on token that (a) has an uppercase first character (Unicode-
    aware via ``str.isupper``), (b) contains no digits, and (c) is not
    on _HONORIFIC_DENYLIST.

    ``header_orig`` MUST preserve the resume's original case — lowercasing
    the input collapses "MS Office" (false positive) onto "Ms. Officer"
    (true positive) and defeats the uppercase-first-letter check.
    """
    for m in pattern.finditer(header_orig):
        follow = m.group(1)
        if not follow:
            continue
        # First-character check — Unicode-aware, catches "María", "Søren",
        # "Müller" while rejecting lowercase glue and digit-led tokens.
        if not follow[0].isupper():
            continue
        # Reject tokens with digits ("Mr. 1Smith" — almost certainly
        # spurious; legitimate names do not contain digits).
        if any(ch.isdigit() for ch in follow):
            continue
        clean = follow.rstrip(".,;:!?").lower()
        if clean and clean not in _HONORIFIC_DENYLIST:
            return True
    return False

# --- Name-based gender proxies (RELOCATED) --------------------------------
# The hand-curated GENDERED_NAMES and _UNISEX_NAMES sets used to live here
# (~140 lines).  At runtime they were superseded by the calibrated
# classifier in fairness/names/classifier.py; once the classifier shipped,
# this module stopped reading them.  Leaving a dead set in a security-
# critical file misleads future readers and invites accidental
# reintroduction of the cancellation bugs documented in task #2 of the
# security review.
#
# The sets now live in fairness/names/seed_lists.py and are imported only
# at corpus-build time by data/names/build_corpus.py.  The import-time
# vocab consistency invariant moved with them.  See seed_lists.py for
# the curatorial rationale.


# --- Resume-vocabulary denylist for the name scan -------------------------
# When sweeping header tokens through the name classifier, we exclude
# common resume-domain words.  Without this filter the OOV branch of
# the classifier produces confident-but-meaningless predictions on
# words like "Engineering", "Team", "Resume", "Senior" — short tokens
# whose char n-grams happen to overlap with one gender's name patterns.
#
# This list is intentionally conservative: it covers section headers,
# job-title nouns, and obvious non-name address words.  Anything not
# on the list still goes through the classifier, which is fine for
# real names (lookup will hit) and tolerable for surnames (the
# first-token rule below skips most of them).
_RESUME_VOCAB_DENYLIST: frozenset = frozenset({
    # Section / structural headers
    "resume", "cv", "vitae", "summary", "objective", "profile",
    "professional", "personal", "contact", "address", "phone", "email",
    "linkedin", "github", "website", "location", "education",
    "experience", "skills", "projects", "certifications", "awards",
    "publications", "references", "languages", "interests",
    # Job-title nouns
    "engineer", "engineering", "developer", "scientist", "analyst",
    "manager", "director", "consultant", "specialist", "architect",
    "designer", "researcher", "lead", "senior", "junior", "principal",
    "associate", "assistant", "officer", "executive", "head", "chief",
    "data", "team", "software", "systems", "product", "project",
    "technical", "technology", "business", "marketing", "sales",
    # Bare honorifics that escape the honorific scan when they appear
    # alone without a following name (e.g. a resume that says "Mr" in
    # the header).
    "mr", "mrs", "ms", "miss", "dr", "sir", "madam", "prof", "mister",
    # Academic degree suffixes that frequently appear after a comma
    # in the resume header ("John Doe, PhD" / "Jane Smith, MD").
    # Without these, the comma-cascade strategy that takes the
    # right-of-comma part would feed the suffix to the classifier
    # and produce a spurious gender signal.
    "phd", "msc", "mba", "ba", "bs", "ma", "md", "jd", "dphil",
    "llm", "llb", "edd", "dvm", "dds", "pe", "esq", "cfa", "cpa",
    "rn", "np", "do",
})

# Minimum classifier confidence (=|p - 0.5| * 2) required for a token
# to drive the legacy male_name / female_name boolean signals.  Below
# this floor we still record name_p_female and set unisex_name=True,
# but we DO NOT vote for either gender — the classifier is too
# uncertain about this token to bias the categorical decision.
_NAME_SIGNAL_CONFIDENCE_FLOOR: float = 0.40

# --- Calibration-drift gate thresholds -------------------------------------
# The classifier's published overall ECE is 0.012 (well-calibrated by the
# field-convention threshold of 0.05).  But per-culture ECE varies: Arab
# 0.090, south_asian 0.080, european_other 0.024.  An audit whose corpus
# composition skews heavily toward high-ECE cultures has a weighted-ECE
# above 0.05, which means the predicted P(female|name) values are NOT
# trustworthy at face value — pass/fail verdicts under these conditions
# should not be published as if the math is fully calibrated.
#
# Three tiers:
#   weighted_ece <= 0.05    "ok"          publish verdict as-is
#   0.05 < we <= 0.10       "warn"        publish + add caveat recommendation
#   weighted_ece > 0.10     "inconclusive" override verdict to inconclusive
#
# Coverage gate (separate): if less than half the audit's candidates are
# in cultures with ANY measured ECE, the weighted_ece is itself unreliable
# and the verdict is forced to inconclusive.
_CALIBRATION_DRIFT_OK_CEILING: float    = 0.05
_CALIBRATION_DRIFT_WARN_CEILING: float  = 0.10
_CALIBRATION_ECE_COVERAGE_FLOOR: float  = 0.50


# --- Model card cache for per-culture ECE disclosure ----------------------
# Loaded once and reused across audits.  Returns {} when the card file is
# absent — disclosure simply omits the ECE field in that case.  See the
# culture_distribution block of audit_ranking_bias.
_MODEL_CARD_ECE_CACHE: dict = None  # type: ignore[assignment]


def _load_model_card_ece() -> dict:
    """Read per-culture ECE from fairness/names/model_card.json once."""
    global _MODEL_CARD_ECE_CACHE
    if _MODEL_CARD_ECE_CACHE is not None:
        return _MODEL_CARD_ECE_CACHE
    out: dict = {}
    try:
        import json
        card_path = (
            Path(__file__).resolve().parent.parent
            / "fairness" / "names" / "model_card.json"
        )
        if card_path.exists():
            card = json.loads(card_path.read_text(encoding="utf-8"))
            by_culture = card.get("metrics", {}).get("by_culture", {})
            for culture, m in by_culture.items():
                if "ece" in m:
                    out[culture] = m["ece"]
    except Exception:
        # Defensive: model card load is decorative.  If anything goes
        # wrong we still produce a usable audit, just without the ECE
        # column in the culture_distribution table.
        out = {}
    _MODEL_CARD_ECE_CACHE = out
    return out


class BiasDetector:
    """
    Detects potential demographic bias in resume ranking results.

    Supports:
    - Gender bias detection via text proxies (names, titles, pronouns)
    - Adverse impact ratio (4/5 rule)
    - Demographic parity and equalized odds
    - Statistical significance testing (Mann-Whitney U)

    See module-level LIMITATIONS before using gender detection.
    """

    def __init__(self, threshold: float = FAIRNESS_ADVERSE_IMPACT_THRESHOLD):
        self.adverse_impact_threshold = threshold

    # --- Gender Detection -------------------------------------------------

    @staticmethod
    def _extract_header_token_strategies(text: str) -> list:
        """Return one candidate-token list per cascade strategy, in
        priority order.  Pulled out of detect_gender_proxy_scored so
        that audit_ranking_bias can batch the classifier call across
        every resume in the corpus (one model.predict_proba on ALL
        distinct header tokens, instead of one per resume).

        Cascade strategies (see detect_gender_proxy_scored for the
        rationale):

          1. Right-of-first-comma on line 1
             (academic-CV "Lastname, Firstname" format)
          2. Line 1 as-is
          3. Line 1 + line 2 concatenated
             (fallback for "Job Title\\nCandidate Name" layouts)

        Each list contains Title-cased tokens of length >= 2 that are
        not on _RESUME_VOCAB_DENYLIST, in source order.
        """
        header_lines = text.strip().split("\n")[:2]
        line1 = header_lines[0] if header_lines else ""
        blocks: list = []
        if "," in line1:
            blocks.append(line1.split(",", 1)[1])
        blocks.append(line1)
        if len(header_lines) > 1:
            blocks.append(" ".join(header_lines))

        strategy_lists: list = []
        for block in blocks:
            cands: list = []
            for raw in block.split():
                cleaned = re.sub(r"[^A-Za-z]", "", raw)
                if len(cleaned) < 2 or not cleaned[0].isupper():
                    continue
                if cleaned.lower() in _RESUME_VOCAB_DENYLIST:
                    continue
                cands.append(cleaned)
            strategy_lists.append(cands)
        return strategy_lists

    @staticmethod
    def _pick_name_signal(strategy_lists: list, results_by_token: dict):
        """Walk the cascade strategies against a precomputed
        ``results_by_token`` map (keyed by lower-cased token).
        Returns ``(chosen, first_surname)`` — either may be None.

        Cascade STOPS at the first strategy that yields any candidates:
        a non-surname-only result becomes ``chosen``; if every result
        is surname-only the first one becomes ``first_surname`` for
        the diagnostic.  See detect_gender_proxy_scored's main
        cascade comment for the why.
        """
        chosen = None
        first_surname = None
        for cands in strategy_lists:
            if not cands:
                continue
            results = [results_by_token[c.lower()] for c in cands
                       if c.lower() in results_by_token]
            if not results:
                continue
            non_surnames = [r for r in results if not r.is_surname_only]
            if non_surnames:
                chosen = non_surnames[0]
            else:
                first_surname = results[0]
            break
        return chosen, first_surname

    @staticmethod
    def detect_gender_proxy_scored(
        text: str,
        _precomputed_name_results: dict = None,
    ) -> dict:
        """
        Detect gender from resume text and return a scored result.

        Returns:
            {
                "gender":     "male" | "female" | "unknown",
                "confidence": float in [0.0, 1.0],
                "signals": {
                    "male_pronoun":  int,   # count of male pronoun matches
                    "female_pronoun": int,
                    "male_title":    bool,
                    "female_title":  bool,
                    "neutral_title": bool,
                    "male_name":     bool,
                    "female_name":   bool,
                }
            }

        Confidence heuristic (uncalibrated):
            - Gendered title fired:            0.95
            - Pronoun + name agree on gender:  0.85
            - Name only:                       0.65
            - Pronoun only (gap >= 3):          0.55
            - Otherwise:                       0.00  ("unknown")
        """
        # Defence against Unicode-confusable bypasses BEFORE any other
        # processing — see _sanitise_for_detection.  Without this,
        # attackers flip detected gender with one zero-width insertion
        # ("M​r. Smith") or a fullwidth honorific ("Ｍｒ. Smith")
        # because the ASCII-anchored regex misses the bypassed token.
        text = _sanitise_for_detection(text)

        text_lower = text.lower()
        header_orig = text[:200]   # honorific scan needs ORIGINAL case

        signals: dict = {
            "male_pronoun": 0,
            "female_pronoun": 0,
            "male_title": False,
            "female_title": False,
            "neutral_title": False,
            # Legacy boolean signals — derived from name_p_female via
            # the DEFAULT_HARD_THRESHOLD of the name classifier.  Kept
            # for downstream code that expects the old shape.
            "male_name":   False,
            "female_name": False,
            "unisex_name": False,
            # New probabilistic name signal — calibrated P(female) from
            # the highest-confidence header token, per the trained
            # char-ngram + isotonic-calibration classifier in
            # fairness/names/classifier.py.  0.5 = no name signal.
            "name_p_female": 0.5,
            # Provenance of the name signal: "lookup" (corpus row hit),
            # "model" (OOV, used classifier fallback), or "empty" (no
            # usable token found in the header).
            "name_source": "empty",
            # The header token that produced name_p_female, lower-cased.
            "name_token": "",
            # True when the only Title-cased header tokens we could find
            # are on the surname denylist (data/names/surnames.csv).
            # In that case name_source stays "empty" — surnames carry
            # no reliable gender signal — but the token is recorded
            # here so audit reports can disclose the situation.
            "name_is_surname": False,
            # Culture cluster of the chosen name token, derived from
            # the lookup row in training_corpus.csv.  "unknown" for OOV
            # model results and empty / surname-only headers.  Audit
            # report uses this to surface per-culture composition.
            "name_culture": "unknown",
        }

        # 1. Pronoun scan (full text, lowercased)
        for pat in GENDER_INDICATORS["male"]:
            signals["male_pronoun"] += len(re.findall(pat, text_lower))
        for pat in GENDER_INDICATORS["female"]:
            signals["female_pronoun"] += len(re.findall(pat, text_lower))

        # 2. Honorific scan (original-case header only, strict pattern + denylist).
        # See _HONORIFIC_PATTERNS for the false-positive defences.
        signals["male_title"]    = _honorific_fires(_HONORIFIC_PATTERNS["male"],    header_orig)
        signals["female_title"]  = _honorific_fires(_HONORIFIC_PATTERNS["female"],  header_orig)
        signals["neutral_title"] = _honorific_fires(_HONORIFIC_PATTERNS["neutral"], header_orig)

        # 3. Name scan: classify the FIRST plausible name token in the
        # first two header lines through the calibrated name classifier.
        #
        # Design rationale (see _RESUME_VOCAB_DENYLIST and
        # _NAME_SIGNAL_CONFIDENCE_FLOOR for the supporting constants):
        #
        #   - "First plausible token" rather than "max-confidence token"
        #     because resumes overwhelmingly place the candidate's given
        #     name first.  Picking max-confidence instead made surnames
        #     dominate — "Mary Jones" classified male because Jones is
        #     a confident OOV male prediction, "Sarah Chen" classified
        #     by Chen, etc.  The first-token rule is the standard
        #     resume convention and recovers the expected behaviour.
        #
        #   - "Plausible" means: starts with an uppercase letter in the
        #     original text (resumes Title-case names), length >= 2 after
        #     stripping non-letters, and not in the resume-vocab denylist
        #     (Engineer, Resume, Team, Senior, ...).  Without these
        #     filters the OOV branch of the classifier produces
        #     confident-but-meaningless predictions on common resume
        #     vocabulary whose char n-grams happen to overlap with
        #     one gender's name patterns.
        #
        #   - Below _NAME_SIGNAL_CONFIDENCE_FLOOR the classifier is
        #     too uncertain to vote for either gender; the candidate is
        #     marked unisex_name=True but neither male_name nor
        #     female_name fires.
        # Extract candidate-token lists for each cascade strategy.
        # See _extract_header_token_strategies / _pick_name_signal
        # for the cascade rationale.  The two-phase split (extract
        # then resolve) lets audit_ranking_bias batch the classifier
        # call across every resume in the corpus — one big
        # model.predict_proba instead of one per resume.
        strategy_lists = BiasDetector._extract_header_token_strategies(text)

        if _precomputed_name_results is not None:
            results_by_token = _precomputed_name_results
        else:
            # Standalone path: batch the classifier on this resume's
            # tokens only.  Slightly more overhead than the old direct
            # predict_many on the chosen strategy (we resolve every
            # cascade level upfront) but trivially small per resume.
            from fairness.names.classifier import predict_many
            distinct = sorted({
                t.lower() for cands in strategy_lists for t in cands
            })
            if distinct:
                batch = predict_many(distinct)
                results_by_token = {r.name: r for r in batch}
            else:
                results_by_token = {}

        chosen, first_surname = BiasDetector._pick_name_signal(
            strategy_lists, results_by_token,
        )

        if chosen is not None and chosen.source != "empty":
            signals["name_p_female"]   = round(float(chosen.p_female), 4)
            signals["name_source"]     = chosen.source
            signals["name_token"]      = chosen.name
            signals["name_is_surname"] = False
            # Culture is set on lookup hits, None for model fallback.
            # Surfacing it here lets audit_ranking_bias build a
            # per-culture composition table for the report.
            signals["name_culture"]    = chosen.culture or "unknown"
        elif first_surname is not None:
            signals["name_token"]      = first_surname.name
            signals["name_is_surname"] = True
            signals["name_culture"]    = "unknown"
        else:
            signals["name_culture"]    = "unknown"

        # Derived legacy booleans.  Lookup hits with high distance from
        # 0.5, OR model hits that clear the confidence floor, vote for
        # the corresponding gender.  Everything in between is unisex.
        if signals["name_source"] != "empty":
            p_f = signals["name_p_female"]
            confidence = abs(p_f - 0.5) * 2.0
            if confidence >= _NAME_SIGNAL_CONFIDENCE_FLOOR:
                if p_f > 0.5:
                    signals["female_name"] = True
                else:
                    signals["male_name"] = True
            else:
                signals["unisex_name"] = True

        # 4. Score aggregation
        male_score = signals["male_pronoun"]
        female_score = signals["female_pronoun"]

        if signals["male_title"]:
            male_score += 8
        if signals["female_title"]:
            female_score += 8

        if signals["male_name"] and not signals["female_name"]:
            male_score += 5
        elif signals["female_name"] and not signals["male_name"]:
            female_score += 5

        # 5. Decision + confidence
        if male_score > female_score and male_score >= 2:
            gender = "male"
            dominant, other = male_score, female_score
        elif female_score > male_score and female_score >= 2:
            gender = "female"
            dominant, other = female_score, male_score
        else:
            gender = "unknown"
            dominant, other = 0, 0

        if gender == "unknown":
            confidence = 0.0
        elif signals["male_title"] or signals["female_title"]:
            confidence = 0.95
        elif (signals["male_pronoun"] >= 2 or signals["female_pronoun"] >= 2) and (
            signals["male_name"] or signals["female_name"]
        ):
            confidence = 0.85
        elif signals["male_name"] or signals["female_name"]:
            confidence = 0.65
        elif abs(dominant - other) >= 3:
            confidence = 0.55
        else:
            confidence = 0.35

        return {"gender": gender, "confidence": round(confidence, 3), "signals": signals}

    @staticmethod
    def detect_gender_proxy(text: str) -> str:
        """
        Backward-compatible wrapper: returns 'male', 'female', or 'unknown'.
        See detect_gender_proxy_scored() for confidence and signal details.
        """
        return BiasDetector.detect_gender_proxy_scored(text)["gender"]

    # --- Formal Fairness Metrics ------------------------------------------

    @staticmethod
    def demographic_parity_distance(
        group_selection_rates: dict[str, float],
    ) -> float:
        """
        Demographic Parity Distance (DPD).
        DPD = max|P(Y=1|G=g) - P(Y=1)| over all groups g.
        Returns 0 when perfectly fair.
        """
        if not group_selection_rates:
            return 0.0
        rates = list(group_selection_rates.values())
        overall_rate = float(np.mean(rates))
        return float(max(abs(r - overall_rate) for r in rates))

    @staticmethod
    def equalized_odds(
        group_tpr: dict[str, float],
        group_fpr: dict[str, float],
    ) -> dict:
        """
        Equalized Odds.
        Returns tpr_gap, fpr_gap, and equalized_odds_gap (max of the two).
        """
        tprs = list(group_tpr.values())
        fprs = list(group_fpr.values())
        tpr_gap = max(tprs) - min(tprs) if tprs else 0.0
        fpr_gap = max(fprs) - min(fprs) if fprs else 0.0
        return {
            "tpr_gap": round(tpr_gap, 4),
            "fpr_gap": round(fpr_gap, 4),
            "equalized_odds_gap": round(max(tpr_gap, fpr_gap), 4),
        }

    @staticmethod
    def statistical_parity_difference(
        group_selection_rates: dict[str, float],
    ) -> float:
        """
        Statistical Parity Difference: max_rate - min_rate across groups.
        0 means perfect parity.
        """
        if not group_selection_rates:
            return 0.0
        rates = list(group_selection_rates.values())
        return float(max(rates) - min(rates))

    def adverse_impact_ratio(
        self,
        group_a_selected: int,
        group_a_total: int,
        group_b_selected: int,
        group_b_total: int,
    ) -> dict:
        """
        Compute Adverse Impact Ratio (4/5 Rule).
        AIR = selection_rate_minority / selection_rate_majority.
        AIR < 0.8 may indicate adverse impact.
        """
        rate_a = group_a_selected / group_a_total if group_a_total > 0 else 0.0
        rate_b = group_b_selected / group_b_total if group_b_total > 0 else 0.0

        if rate_a == 0 and rate_b == 0:
            air = 1.0
        elif max(rate_a, rate_b) == 0:
            air = 0.0
        else:
            air = min(rate_a, rate_b) / max(rate_a, rate_b)

        return {
            "group_a_rate": round(rate_a, 4),
            "group_b_rate": round(rate_b, 4),
            "adverse_impact_ratio": round(air, 4),
            "passes_4_5_rule": air >= self.adverse_impact_threshold,
            "risk_level": self._risk_level(air),
        }

    @staticmethod
    def _risk_level(air: float) -> str:
        if air >= 0.8:
            return "LOW"
        elif air >= 0.6:
            return "MODERATE"
        elif air >= 0.4:
            return "HIGH"
        else:
            return "CRITICAL"

    # --- Soft (probability-weighted) AIR ----------------------------------

    @staticmethod
    def _air_soft(candidate_records: list) -> dict:
        """Compute the probability-weighted Adverse Impact Ratio.

        Each candidate with a known P(female | name) contributes that
        probability to the female group's mass and (1 - p) to the male
        group's mass.  Candidates whose name signal is "empty" (no
        useful classifier output) are excluded — they contribute to
        neither group total, which keeps the soft AIR comparable to
        the hard AIR which also excludes "unknown" candidates.

        Returns a dict with the male/female totals, selected masses,
        per-group selection rates, and the resulting AIR_soft.  When
        either group has zero total mass the AIR is reported as 1.0
        (cannot detect adverse impact with one group).
        """
        male_total = female_total = 0.0
        male_selected = female_selected = 0.0
        for rec in candidate_records:
            p_f = rec["p_female_soft"]
            if p_f is None:
                continue  # name_source == "empty" — exclude
            p_m = 1.0 - p_f
            male_total   += p_m
            female_total += p_f
            if rec["selected"]:
                male_selected   += p_m
                female_selected += p_f

        def _rate(num: float, den: float) -> float:
            return num / den if den > 0 else 0.0

        male_rate   = _rate(male_selected,   male_total)
        female_rate = _rate(female_selected, female_total)

        if male_total == 0 or female_total == 0:
            air_soft = 1.0
        elif max(male_rate, female_rate) == 0:
            air_soft = 0.0
        else:
            air_soft = min(male_rate, female_rate) / max(male_rate, female_rate)

        return {
            "male_total_mass":       round(male_total, 4),
            "female_total_mass":     round(female_total, 4),
            "male_selected_mass":    round(male_selected, 4),
            "female_selected_mass":  round(female_selected, 4),
            "male_rate":             round(male_rate, 4),
            "female_rate":           round(female_rate, 4),
            "adverse_impact_ratio":  round(air_soft, 4),
        }

    # --- Full Bias Audit --------------------------------------------------

    def audit_ranking_bias(
        self,
        resume_texts: dict[str, str],
        scores: dict[str, float],
        selection_threshold: Optional[float] = None,
    ) -> dict:
        """
        Comprehensive bias audit across detected demographic groups.

        Args:
            resume_texts: {resume_filename: text}
            scores: {resume_filename: ranking_score}
            selection_threshold: Score threshold for selected vs not selected.
                                 If None, uses the median score.
        """
        if selection_threshold is None:
            selection_threshold = float(np.median(list(scores.values())))

        # --- Phase 1: batched classifier call ---------------------------
        # Walk every resume once to collect the union of header tokens,
        # then invoke the calibrated classifier ONCE on the deduped
        # list.  This collapses what was N small predict_proba calls
        # (one per resume, ~0.5ms each) into a single batched call —
        # ~10-20x faster on corpora with thousands of resumes.
        from fairness.names.classifier import predict_many
        all_strategies: dict = {}
        token_union: set = set()
        for filename, text in resume_texts.items():
            if filename not in scores:
                continue
            sl = BiasDetector._extract_header_token_strategies(text)
            all_strategies[filename] = sl
            for cands in sl:
                for t in cands:
                    token_union.add(t.lower())
        if token_union:
            batch = predict_many(sorted(token_union))
            results_by_token = {r.name: r for r in batch}
        else:
            results_by_token = {}

        # --- Phase 2: per-resume aggregation ----------------------------
        # For each candidate we record BOTH the hard categorical
        # ("male"/"female"/"unknown" via detect_gender_proxy_scored) AND
        # the calibrated soft probability (name_p_female from the
        # classifier signals).  The hard label drives the legacy AIR;
        # the soft probabilities drive AIR_soft (see _air_dual below).
        gender_groups: dict[str, list] = defaultdict(list)
        candidate_records: list = []  # used for soft-AIR aggregation
        culture_records: list = []    # used for per-culture disclosure
        for filename, text in resume_texts.items():
            if filename in scores:
                result = self.detect_gender_proxy_scored(
                    text,
                    _precomputed_name_results=results_by_token,
                )
                gender = result["gender"]
                selected = scores[filename] >= selection_threshold
                signals = result["signals"]
                # If the classifier produced no name signal we treat the
                # candidate as unknown for BOTH views.  Otherwise the
                # soft view uses the calibrated P(female|name).
                if signals.get("name_source", "empty") == "empty":
                    p_female_soft = None  # marker: candidate is unknown
                else:
                    p_female_soft = float(signals.get("name_p_female", 0.5))
                gender_groups[gender].append({
                    "filename":   filename,
                    "score":      scores[filename],
                    "selected":   selected,
                    "confidence": result["confidence"],
                })
                candidate_records.append({
                    "filename":      filename,
                    "selected":      selected,
                    "hard_gender":   gender,
                    "p_female_soft": p_female_soft,
                })
                culture_records.append({
                    "filename":      filename,
                    "selected":      selected,
                    "culture":       signals.get("name_culture", "unknown"),
                    "name_source":   signals.get("name_source", "empty"),
                    "p_female":      float(signals.get("name_p_female", 0.5)),
                })

        results: dict = {
            "threshold": selection_threshold,
            "total_resumes": len(scores),
            "gender_distribution": {},
            "gender_bias_analysis": {},
            "score_distribution": {},
            "detection_coverage": {},
            "culture_distribution": {},
            "recommendations": [],
        }

        # Detection coverage stats
        n_known = sum(len(v) for k, v in gender_groups.items() if k != "unknown")
        n_unknown = len(gender_groups.get("unknown", []))
        results["detection_coverage"] = {
            "detected": n_known,
            "undetected": n_unknown,
            "coverage_rate": round(n_known / max(n_known + n_unknown, 1), 4),
        }

        # Per-group stats
        for gender, resumes in gender_groups.items():
            if not resumes:
                continue
            results["gender_distribution"][gender] = {
                "count": len(resumes),
                "mean_score": round(float(np.mean([r["score"] for r in resumes])), 4),
                "selected_count": sum(1 for r in resumes if r["selected"]),
                "selection_rate": round(
                    sum(1 for r in resumes if r["selected"]) / len(resumes), 4
                ),
                "mean_confidence": round(
                    float(np.mean([r["confidence"] for r in resumes])), 3
                ),
            }

        # --- Per-culture audit composition -------------------------------
        # Surface which culture clusters drove this audit, so reviewers
        # can interpret the AIR numbers in light of the per-culture
        # calibration ECE recorded in fairness/names/model_card.json.
        # Without this disclosure, "AIR=0.85 with 70% Arab candidates"
        # looks identical to "AIR=0.85 with 70% European candidates"
        # even though Arab has ~3x the calibration error.
        culture_groups: dict = defaultdict(list)
        for rec in culture_records:
            culture_groups[rec["culture"]].append(rec)
        per_culture_ece = _load_model_card_ece()
        for culture, recs in culture_groups.items():
            n = len(recs)
            sel = sum(1 for r in recs if r["selected"])
            results["culture_distribution"][culture] = {
                "count":             n,
                "selected_count":    sel,
                "selection_rate":    round(sel / n, 4) if n else 0.0,
                "mean_p_female":     round(
                    float(np.mean([r["p_female"] for r in recs])), 4
                ),
                "lookup_share":      round(
                    sum(1 for r in recs if r["name_source"] == "lookup") / n,
                    4,
                ),
                # ECE from the model card for this culture cluster.  None
                # for "unknown" (no per-culture metric applies) and for
                # cultures the model card didn't have enough samples to
                # measure.  When this is materially above 0.05, the AIR
                # numbers below should be interpreted with caution.
                "model_card_ece":    per_culture_ece.get(culture),
            }

        # --- Corpus-weighted calibration drift gate ----------------------
        # We weight per-culture ECE by the audit's own culture composition.
        # This produces a single number representing how well-calibrated
        # the classifier IS on THIS particular audit corpus — which is
        # what governs whether the AIR pass/fail can be trusted.  See
        # _CALIBRATION_DRIFT_* constants for the three tier thresholds.
        ece_weighted_sum = 0.0
        ece_weight_total = 0   # count of candidates in cultures with ECE
        for culture, recs in culture_groups.items():
            ece = per_culture_ece.get(culture)
            if ece is None:
                continue
            n = len(recs)
            ece_weighted_sum += n * ece
            ece_weight_total += n
        total_audited = sum(len(v) for v in culture_groups.values())
        ece_coverage = (ece_weight_total / total_audited) if total_audited else 0.0
        weighted_ece = (
            ece_weighted_sum / ece_weight_total
            if ece_weight_total > 0 else None
        )

        if weighted_ece is None:
            drift_status = "unknown"  # no ECE data at all
        elif ece_coverage < _CALIBRATION_ECE_COVERAGE_FLOOR:
            drift_status = "inconclusive_low_ece_coverage"
        elif weighted_ece > _CALIBRATION_DRIFT_WARN_CEILING:
            drift_status = "inconclusive_high_drift"
        elif weighted_ece > _CALIBRATION_DRIFT_OK_CEILING:
            drift_status = "warn"
        else:
            drift_status = "ok"

        results["calibration_drift"] = {
            "weighted_ece":     (round(weighted_ece, 4)
                                 if weighted_ece is not None else None),
            "ece_coverage":     round(ece_coverage, 4),
            "ok_ceiling":       _CALIBRATION_DRIFT_OK_CEILING,
            "warn_ceiling":     _CALIBRATION_DRIFT_WARN_CEILING,
            "coverage_floor":   _CALIBRATION_ECE_COVERAGE_FLOOR,
            "status":           drift_status,
        }

        # --- Dual AIR computation ----------------------------------------
        # We report TWO views of the same audit:
        #
        #   AIR_hard  — each candidate is assigned exactly one group via
        #               the threshold-driven categorical label.  This is
        #               the legacy / EEOC-style 4/5 rule computation.
        #
        #   AIR_soft  — each candidate contributes P(group) mass to each
        #               group, using the calibrated probabilities from
        #               the name classifier.  Borderline candidates
        #               (e.g. p_female=0.6) contribute 0.6 to female
        #               and 0.4 to male instead of being forced into one.
        #
        # Pass/fail uses min(AIR_hard, AIR_soft) — the more conservative
        # of the two — so an adversary cannot cherry-pick the view that
        # makes the system look fair.  See task #15 in the security review.
        male_data = gender_groups.get("male", [])
        female_data = gender_groups.get("female", [])

        if male_data and female_data:
            air_hard = self.adverse_impact_ratio(
                group_a_selected=sum(1 for r in male_data if r["selected"]),
                group_a_total=len(male_data),
                group_b_selected=sum(1 for r in female_data if r["selected"]),
                group_b_total=len(female_data),
            )
            air_soft = self._air_soft(candidate_records)
            conservative = min(
                air_hard["adverse_impact_ratio"],
                air_soft["adverse_impact_ratio"],
            )
            raw_passes = conservative >= self.adverse_impact_threshold
            # The publish-ready verdict folds in the calibration drift
            # gate.  raw passes_4_5_rule is left unchanged so callers
            # that want the unadjusted math can still read it.
            if drift_status in ("inconclusive_low_ece_coverage",
                                "inconclusive_high_drift"):
                verdict = f"inconclusive ({drift_status})"
            elif raw_passes:
                verdict = "pass" if drift_status in ("ok", "unknown") else "pass_with_drift_warning"
            else:
                verdict = "fail"

            results["gender_bias_analysis"] = {
                **air_hard,
                "adverse_impact_ratio_hard": air_hard["adverse_impact_ratio"],
                "adverse_impact_ratio_soft": air_soft["adverse_impact_ratio"],
                "soft_male_mass":            air_soft["male_total_mass"],
                "soft_female_mass":          air_soft["female_total_mass"],
                "soft_male_rate":            air_soft["male_rate"],
                "soft_female_rate":          air_soft["female_rate"],
                # Raw conservative-of-both AIR — unchanged by the
                # calibration drift gate.  Use this when you want the
                # math; use `verdict` when you want the publishable answer.
                "adverse_impact_ratio":      round(conservative, 4),
                "passes_4_5_rule":           raw_passes,
                "risk_level":                self._risk_level(conservative),
                "agreement_gap":             round(
                    abs(air_hard["adverse_impact_ratio"]
                        - air_soft["adverse_impact_ratio"]),
                    4,
                ),
                # The verdict the audit RECOMMENDS publishing.
                # "pass" / "fail"                         — math is trusted
                # "pass_with_drift_warning"               — math passes, drift
                #                                           in warn band (0.05<we<=0.10)
                # "inconclusive_high_drift"               — drift > 0.10
                # "inconclusive_low_ece_coverage"         — < 50% of audit
                #                                           is in cultures with measured ECE
                "verdict":                   verdict,
            }

            if "inconclusive" in verdict:
                results["recommendations"].append(
                    f"[INCONCLUSIVE] {verdict}: weighted ECE="
                    f"{(results['calibration_drift']['weighted_ece'] or 0):.3f}, "
                    f"coverage={results['calibration_drift']['ece_coverage']:.0%}. "
                    f"AIR={conservative:.2f}, but the classifier's "
                    f"calibration on this culture mix is too poor to "
                    f"publish a pass/fail verdict."
                )
            elif not raw_passes:
                results["recommendations"].append(
                    f"[FAIL] Gender AIR (conservative of hard/soft) = "
                    f"{conservative:.2f} (below "
                    f"{self.adverse_impact_threshold:.2f} threshold). "
                    f"Hard AIR={air_hard['adverse_impact_ratio']:.2f}, "
                    f"Soft AIR={air_soft['adverse_impact_ratio']:.2f}. "
                    f"Potential gender bias detected."
                )
            else:
                if drift_status == "warn":
                    results["recommendations"].append(
                        f"[NOTE] Verdict is PASS but classifier "
                        f"calibration on this culture mix is degraded "
                        f"(weighted ECE="
                        f"{results['calibration_drift']['weighted_ece']:.3f}, "
                        f"above the 0.05 target).  Treat the AIR number "
                        f"as a lower-precision estimate."
                    )
                if (results["gender_bias_analysis"]["agreement_gap"]
                        > 0.10):
                    results["recommendations"].append(
                        f"[NOTE] Hard and soft AIR disagree by "
                        f"{results['gender_bias_analysis']['agreement_gap']:.2f}. "
                        f"Borderline name predictions are material — review "
                        f"the per-candidate name_source breakdown."
                    )

        # Score distribution
        all_scores = list(scores.values())
        results["score_distribution"] = {
            "mean": round(float(np.mean(all_scores)), 4),
            "std": round(float(np.std(all_scores)), 4),
            "median": round(float(np.median(all_scores)), 4),
            "min": round(float(min(all_scores)), 4),
            "max": round(float(max(all_scores)), 4),
        }

        # Mann-Whitney U test for gender score difference
        if len(male_data) >= 5 and len(female_data) >= 5:
            from scipy import stats
            male_scores = [r["score"] for r in male_data]
            female_scores = [r["score"] for r in female_data]
            try:
                _, p_value = stats.mannwhitneyu(
                    male_scores, female_scores, alternative="two-sided"
                )
                results["gender_bias_analysis"]["mann_whitney_p_value"] = round(p_value, 6)
                if p_value < 0.05:
                    results["recommendations"].append(
                        f"[WARN] Statistically significant score difference between "
                        f"genders (p={p_value:.4f}). Further investigation recommended."
                    )
            except Exception:
                pass

        if not results["recommendations"]:
            results["recommendations"].append(
                "[OK] No significant bias detected in this evaluation."
            )

        if n_unknown > n_known:
            results["recommendations"].append(
                f"[NOTE] {n_unknown}/{n_known + n_unknown} resumes had undetectable "
                f"gender proxies. Interpret AIR with caution."
            )

        return results

    def print_audit_report(self, audit: dict) -> None:
        """Print a formatted bias audit report (ASCII-safe)."""
        sep = "=" * 60
        print(f"\n{sep}")
        print("  FAIRNESS & BIAS AUDIT REPORT")
        print(sep)
        print(f"  Total resumes analyzed: {audit['total_resumes']}")
        print(f"  Selection threshold:    {audit['threshold']:.4f}")

        cov = audit.get("detection_coverage", {})
        if cov:
            print(f"  Gender detection:       "
                  f"{cov.get('detected', 0)} detected, "
                  f"{cov.get('undetected', 0)} unknown "
                  f"({cov.get('coverage_rate', 0):.1%} coverage)")

        print(f"\n  Gender Distribution:")
        for gender, s in audit["gender_distribution"].items():
            conf_str = (f", mean_confidence={s['mean_confidence']:.2f}"
                        if "mean_confidence" in s else "")
            print(f"    {gender.upper():>8}: {s['count']} resumes, "
                  f"mean_score={s['mean_score']:.4f}, "
                  f"selection_rate={s['selection_rate']:.1%}"
                  f"{conf_str}")

        if audit["gender_bias_analysis"]:
            bias = audit["gender_bias_analysis"]
            print(f"\n  Adverse Impact Analysis (4/5 Rule):")
            print(f"    Male selection rate:   {bias.get('group_a_rate', 0):.1%}")
            print(f"    Female selection rate: {bias.get('group_b_rate', 0):.1%}")
            print(f"    AIR:                   {bias.get('adverse_impact_ratio', 0):.4f}")
            passed = bias.get('passes_4_5_rule', False)
            print(f"    Passes 4/5 rule:       {'[PASS]' if passed else '[FAIL]'}")
            print(f"    Risk level:            {bias.get('risk_level', 'N/A')}")
            if "mann_whitney_p_value" in bias:
                print(f"    Mann-Whitney p:        {bias['mann_whitney_p_value']:.6f}")

        print(f"\n  Score Distribution:")
        sd = audit["score_distribution"]
        print(f"    Mean:   {sd['mean']:.4f} +/- {sd['std']:.4f}")
        print(f"    Median: {sd['median']:.4f}")
        print(f"    Range:  [{sd['min']:.4f}, {sd['max']:.4f}]")

        print(f"\n  Recommendations:")
        for rec in audit["recommendations"]:
            print(f"    {rec}")

        print(f"\n{sep}\n")


if __name__ == "__main__":
    detector = BiasDetector()

    resume_texts = {
        "john_smith_resume.txt": "John Smith\nSenior Engineer with Python, Java",
        "mary_jones_resume.txt": "Mary Jones\nData Scientist with Python, R skills",
        "alex_unknown_resume.txt": "Alex Morgan\nProject Manager with Agile, Scrum",
        "priya_sharma_resume.txt": "Ms. Priya Sharma\nML Engineer with TensorFlow",
        "james_wilson_resume.txt": "James Wilson\nHe is a Backend Developer with Go",
        "dr_chen_resume.txt": "Dr. Wei Chen\nResearcher with publications in NLP",
    }

    scores = {
        "john_smith_resume.txt": 0.85,
        "mary_jones_resume.txt": 0.72,
        "alex_unknown_resume.txt": 0.60,
        "priya_sharma_resume.txt": 0.90,
        "james_wilson_resume.txt": 0.78,
        "dr_chen_resume.txt": 0.88,
    }

    print("--- Scored detection ---")
    for name, text in resume_texts.items():
        r = detector.detect_gender_proxy_scored(text)
        print(f"  {name}: {r['gender']} (conf={r['confidence']:.2f})")

    print()
    audit = detector.audit_ranking_bias(resume_texts, scores)
    detector.print_audit_report(audit)
