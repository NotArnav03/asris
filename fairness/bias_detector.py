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


# --- Cyrillic / Greek -> Latin confusable map -----------------------------
# Visually-identical (or near-identical) characters that attackers use to
# spoof Latin honorifics: "Мr. Smith" uses U+041C CYRILLIC CAPITAL LETTER
# EM, which renders identically to ASCII "M" but bypasses the strict-Latin
# honorific regex.
#
# Curated from the Unicode Consortium's confusables.txt and the IANA
# IDNA confusables list — only the high-confidence visually-identical
# mappings are retained.  Lookalikes that diverge in some fonts
# (e.g. Cyrillic д vs Latin d) are deliberately NOT mapped.
#
# Applied UNCONDITIONALLY in _sanitise_for_detection.  Trade-off:
# a resume body legitimately written in Russian / Greek will have these
# characters mapped to their Latin equivalents.  For our audit purposes
# (honorific scan + Title-cased name token extraction), this is
# acceptable — the alternative (selective context-aware replacement)
# admits the "Мс. Smith" attack where the entire honorific word is
# wholly-Cyrillic and a naive heuristic leaves it alone.
_CONFUSABLES_MAP: dict = {
    # Cyrillic uppercase -> Latin uppercase
    "А": "A", "В": "B", "Е": "E", "К": "K", "М": "M", "Н": "H",
    "О": "O", "Р": "P", "С": "C", "Т": "T", "У": "Y", "Х": "X",
    "І": "I", "Ј": "J", "Ѕ": "S",
    # Cyrillic lowercase -> Latin lowercase
    "а": "a", "е": "e", "о": "o", "р": "p", "с": "c", "у": "y",
    "х": "x", "і": "i", "ј": "j", "ѕ": "s",
    # Greek uppercase -> Latin uppercase (visually identical)
    "Α": "A", "Β": "B", "Ε": "E", "Ζ": "Z", "Η": "H", "Ι": "I",
    "Κ": "K", "Μ": "M", "Ν": "N", "Ο": "O", "Ρ": "P", "Τ": "T",
    "Υ": "Y", "Χ": "X",
    # Greek lowercase
    "ο": "o", "ν": "v",  # nu looks like v in many fonts
}

_CONFUSABLES_TRANS = str.maketrans(_CONFUSABLES_MAP)


def _sanitise_for_detection(text: str) -> str:
    """NFKC-normalise the input, strip zero-width / BOM characters, and
    fold Cyrillic / Greek confusables onto their Latin look-alikes.

    Three defences against Unicode-confusable bypasses, applied in order:

      NFKC collapses:
          Fullwidth Latin            "Ｍｒ. Smith"  -> "Mr. Smith"
          Mathematical alphanumeric  "𝐌𝐫. Smith"  -> "Mr. Smith"
          Compatibility ligatures    "ﬁ"          -> "fi"

      Zero-width strip defeats:
          ZWSP inside salutation     "M\\u200Br. Smith"  -> "Mr. Smith"
          BOM at start of header     "\\uFEFFMr. Smith"  -> "Mr. Smith"

      Confusables fold (Cyrillic / Greek -> Latin) defeats:
          "Мr. Smith"   (U+041C M)  -> "Mr. Smith"
          "Ms. Smith"   (Greek Mu)  -> "Ms. Smith"
          "Mс. Khan"    (U+0441 c)  -> "Ms. Khan"
          "Мс. Khan"    (both Cyr)  -> "Ms. Khan"

    Trade-off documented in _CONFUSABLES_MAP: this fold is applied
    UNCONDITIONALLY, so legitimate Russian / Greek resume body
    content will see those characters mapped to ASCII.  For an
    audit tool that only looks at the header for honorifics and
    Title-cased name tokens, this is the right call — selective
    context-aware replacement still admits the "Мс." wholly-Cyrillic
    attack.
    """
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = _ZERO_WIDTH_RE.sub("", text)
    text = text.translate(_CONFUSABLES_TRANS)
    return text


# Section headers commonly used in resumes — when one of these appears
# we know the header window has ended and any honorific further down
# is in the body, not the candidate's salutation.  Pattern is anchored
# to start-of-line and matches case-insensitively.
_SECTION_HEADER_RE = re.compile(
    r"^\s*(EXPERIENCE|EDUCATION|SKILLS|SUMMARY|OBJECTIVE|PROFILE|"
    r"CONTACT|PROJECTS|CERTIFICATIONS|AWARDS|PUBLICATIONS|"
    r"REFERENCES|EMPLOYMENT|WORK\s+HISTORY|TECHNICAL\s+SKILLS|"
    r"PROFESSIONAL\s+EXPERIENCE|EDUCATION\s+AND\s+TRAINING|"
    r"LANGUAGES|INTERESTS|HOBBIES|ACCOMPLISHMENTS)\s*:?\s*$",
    re.IGNORECASE | re.MULTILINE,
)


def _extract_header_window(
    text: str,
    max_lines: int = 8,
    max_chars: int = 1000,
    long_line_cutoff: int = 200,
) -> str:
    """Adaptive header detection — replaces the previous ``text[:200]``
    hard cap.

    Scans line-by-line from the start, accumulating non-empty lines
    until one of these terminates the window:

      * ``max_lines`` non-empty lines have been collected (default 8)
      * total character count reaches ``max_chars`` (default 1000)
      * a line longer than ``long_line_cutoff`` chars (typical body
        paragraph; default 200) is encountered
      * a recognised section header (EXPERIENCE, EDUCATION, ...)
        is encountered — anything after this is body content

    Closes the "pad the resume with 250 chars of address before the
    salutation to evade the 200-char honorific scan" bypass.  Real
    headers fit comfortably inside 8 lines / 1000 chars; body content
    triggers one of the terminating conditions.

    Returns the joined header text (preserving newlines for the
    section-header regex on downstream calls).
    """
    if not text:
        return ""
    kept: list = []
    char_count = 0
    line_count = 0
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            # Blank lines do not count against limits but are kept
            # so multi-paragraph headers (address block then name)
            # remain delimited.
            kept.append(line)
            continue
        if _SECTION_HEADER_RE.match(stripped):
            break
        if len(line) > long_line_cutoff:
            break
        kept.append(line)
        char_count += len(line) + 1
        line_count += 1
        if line_count >= max_lines or char_count >= max_chars:
            break
    return "\n".join(kept)


# --- Right-to-left (Arabic, Hebrew) honorifics ----------------------------
# Pure-substring patterns rather than the strict-regex Latin pipeline:
#
#   1. Word-boundary semantics in Arabic / Hebrew are non-trivial (no
#      space-separated word stems; compounds attach prefixes).  Sub-
#      string matching on a salutation phrase is the standard approach.
#
#   2. We do NOT try to extract / classify the candidate's name from
#      RTL scripts here.  The trained char-ngram classifier was fit
#      on romanised forms; running it on raw Arabic / Hebrew tokens
#      would produce uncalibrated probabilities.  Honorific signal
#      alone is what we lift.
#
# Sources: salutation conventions documented in
#   https://en.wikipedia.org/wiki/Arabic_honorifics
#   https://en.wikipedia.org/wiki/Hebrew_honorifics
_RTL_HONORIFICS: dict = {
    "male": [
        # Arabic
        "السيد",  "السّيد",  "سيد",
        # Hebrew
        "מר", "אדון",
    ],
    "female": [
        # Arabic
        "السيدة",  "السّيدة",  "سيدة", "الآنسة", "آنسة",
        # Hebrew
        "מרת", "גברת", "העלמה",
    ],
    "neutral": [
        # Arabic
        "بروفيسور", "أستاذ", "الدكتور",
        # Hebrew
        "פרופ", "דוקטור",
    ],
}


def _has_rtl_script(text: str) -> bool:
    """True if the text contains any character in the Hebrew, Arabic,
    Syriac, or Thaana Unicode blocks.  Used to decide whether to run
    the RTL honorific scan in addition to the Latin scan."""
    if not text:
        return False
    for c in text:
        cp = ord(c)
        if (0x0590 <= cp <= 0x05FF      # Hebrew
                or 0x0600 <= cp <= 0x06FF   # Arabic
                or 0x0700 <= cp <= 0x074F   # Syriac
                or 0x0780 <= cp <= 0x07BF   # Thaana
                or 0xFB1D <= cp <= 0xFDFF   # Hebrew/Arabic presentation
                or 0xFE70 <= cp <= 0xFEFF):  # Arabic presentation B
            return True
    return False


def _rtl_honorific_fires(honorifics: list, header: str) -> bool:
    """Substring match for any RTL honorific in the header window."""
    return any(h in header for h in honorifics)


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


# --- Language detection + per-locale resume vocab -------------------------
# The default _RESUME_VOCAB_DENYLIST is English.  A Spanish, French,
# German, Portuguese, or Italian resume with locale-specific section
# headers and job-title vocabulary needs its own denylist or those
# tokens leak into the classifier as OOV "names".
#
# Detection is a lightweight stop-word frequency heuristic — counts
# the occurrences of ~15 stopwords per language in the resume body,
# picks the language with the most matches (minimum 3 to claim
# anything other than English).  No external dependency.
_LANGUAGE_STOPWORDS: dict = {
    "en": frozenset({
        "the", "and", "of", "to", "in", "for", "with", "at", "from",
        "by", "on", "as", "an", "is", "was", "are", "were",
    }),
    "es": frozenset({
        "el", "la", "los", "las", "y", "de", "en", "para", "con",
        "por", "del", "que", "una", "un", "es", "son", "como",
    }),
    "fr": frozenset({
        "le", "la", "les", "et", "de", "du", "des", "en", "pour",
        "avec", "par", "sur", "que", "une", "un", "est", "sont",
    }),
    "de": frozenset({
        "der", "die", "das", "und", "in", "von", "mit", "bei",
        "für", "auf", "ein", "eine", "ist", "sind", "zu", "auch",
    }),
    "pt": frozenset({
        "o", "a", "os", "as", "e", "de", "do", "da", "em", "para",
        "com", "por", "no", "na", "uma", "um", "que", "é",
    }),
    "it": frozenset({
        "il", "la", "i", "le", "e", "di", "in", "per", "con",
        "da", "una", "un", "che", "è", "sono", "del", "della",
    }),
}


def _detect_language(text: str) -> str:
    """Return the most likely language of ``text``.

    Counts the occurrence of each language's stopword set, returns the
    winner.  Defaults to "en" when no language clears 3 matches —
    English is what the default _RESUME_VOCAB_DENYLIST handles, so the
    fallback is safe.
    """
    if not text:
        return "en"
    words = re.findall(r"[a-zàáâãäåçèéêëìíîïñòóôõöùúûüýÿ]+", text.lower())
    if not words:
        return "en"
    counts: dict = {}
    for lang, stops in _LANGUAGE_STOPWORDS.items():
        counts[lang] = sum(1 for w in words if w in stops)
    best_lang = max(counts, key=counts.get)
    if counts[best_lang] < 3:
        return "en"
    return best_lang


# Per-language resume vocab.  Each set is MERGED with the English base
# at runtime so locale-specific tokens add to (not replace) the English
# denylist, protecting the typical case where a multilingual resume
# mixes English with another language.
_RESUME_VOCAB_BY_LANG: dict = {
    "es": frozenset({
        "resumen", "currículum", "curriculum", "vitae", "perfil",
        "objetivo", "experiencia", "educación", "formación",
        "habilidades", "competencias", "idiomas", "proyectos",
        "referencias", "contacto", "dirección", "teléfono",
        "ingeniero", "ingeniera", "desarrollador", "desarrolladora",
        "gerente", "director", "directora", "analista", "consultor",
        "consultora", "diseñador", "diseñadora", "arquitecto",
        "arquitecta", "investigador", "investigadora", "asesor",
        "asesora", "jefe", "jefa", "líder", "señor", "señora",
        "señorita",
    }),
    "fr": frozenset({
        "résumé", "curriculum", "vitae", "profil", "objectif",
        "expérience", "formation", "éducation", "compétences",
        "langues", "projets", "références", "contact", "adresse",
        "téléphone", "ingénieur", "ingénieure", "développeur",
        "développeuse", "directeur", "directrice", "responsable",
        "analyste", "consultant", "consultante", "architecte",
        "chercheur", "chercheuse", "conseiller", "conseillère",
        "chef", "monsieur", "madame", "mademoiselle",
    }),
    "de": frozenset({
        "lebenslauf", "profil", "ziel", "berufserfahrung",
        "ausbildung", "kenntnisse", "sprachen", "projekte",
        "referenzen", "kontakt", "adresse", "telefon", "ingenieur",
        "ingenieurin", "entwickler", "entwicklerin", "leiter",
        "leiterin", "manager", "managerin", "berater", "beraterin",
        "architekt", "architektin", "forscher", "forscherin",
        "chef", "chefin", "herr", "frau",
    }),
    "pt": frozenset({
        "currículo", "perfil", "objetivo", "experiência",
        "formação", "educação", "habilidades", "competências",
        "idiomas", "projetos", "referências", "contato",
        "endereço", "telefone", "engenheiro", "engenheira",
        "desenvolvedor", "desenvolvedora", "gerente", "diretor",
        "diretora", "analista", "consultor", "consultora",
        "arquiteto", "arquiteta", "pesquisador", "pesquisadora",
        "chefe", "senhor", "senhora", "senhorita",
    }),
    "it": frozenset({
        "curriculum", "vitae", "profilo", "obiettivo", "esperienza",
        "formazione", "istruzione", "competenze", "lingue",
        "progetti", "referenze", "contatto", "indirizzo",
        "telefono", "ingegnere", "ingegnera", "sviluppatore",
        "sviluppatrice", "direttore", "direttrice", "responsabile",
        "analista", "consulente", "architetto", "ricercatore",
        "ricercatrice", "capo", "signor", "signora", "signorina",
    }),
}

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

# --- Detection coverage gate ---------------------------------------------
# The audit can only meaningfully report AIR for the candidates whose
# gender we could classify.  When detection_coverage drops below this
# floor, the AIR numerator/denominator are based on a small fraction of
# the corpus and any "pass" verdict is statistically suspect.  We force
# the verdict to inconclusive_low_detection_coverage and emit a
# recommendation pointing the operator at the per-resume trail so they
# can investigate why so many resumes failed detection.
_DETECTION_COVERAGE_FLOOR: float = 0.50

# --- Deduplication ---------------------------------------------------------
# Default Hamming-distance threshold for SimHash near-duplicate detection.
# A 64-bit SimHash with Hamming distance <= 3 means the two documents share
# at least ~95% of their content fingerprint — close enough that we treat
# them as the same submission for the purposes of AIR computation.
# Submit-the-same-resume-100-times is the explicit attack vector this
# closes (was task #8 in the original review).
_SIMHASH_BITS: int = 64
_NEAR_DUPLICATE_HAMMING_THRESHOLD: int = 3
# Fraction of input that being a duplicate trips a ballot-stuffing alert.
# At 20% dedup rate the AIR numerator could be doubled by one bad actor.
_BALLOT_STUFFING_THRESHOLD: float = 0.20


def _simhash(text: str, bits: int = _SIMHASH_BITS) -> int:
    """Charikar SimHash fingerprint for near-duplicate detection.

    Tokens are alphabetic word-grams from the lower-cased text.  Each
    token's SHA-256 hash contributes +freq to each set bit position
    and -freq to each unset bit position.  The final fingerprint has
    bit i = 1 iff the accumulator at position i is positive.

    Two documents with Hamming distance <= ``_NEAR_DUPLICATE_HAMMING_THRESHOLD``
    are considered near-duplicates.  Default 64 bits with threshold 3
    is the canonical Charikar configuration.
    """
    import hashlib
    from collections import Counter
    tokens = re.findall(r"\w+", text.lower())
    if not tokens:
        return 0
    counter = Counter(tokens)
    v = [0] * bits
    for token, freq in counter.items():
        h = int(hashlib.sha256(token.encode("utf-8")).hexdigest(), 16)
        for i in range(bits):
            if (h >> i) & 1:
                v[i] += freq
            else:
                v[i] -= freq
    fingerprint = 0
    for i in range(bits):
        if v[i] > 0:
            fingerprint |= 1 << i
    return fingerprint


def _hamming_distance(a: int, b: int) -> int:
    """Number of differing bits in two integers' binary representations."""
    return bin(a ^ b).count("1")


# --- Model card cache for per-culture ECE disclosure ----------------------
# Loaded once and reused across audits.  Returns {} when the card file is
# absent — disclosure simply omits the ECE field in that case.  See the
# culture_distribution block of audit_ranking_bias.
_MODEL_CARD_ECE_CACHE: dict = None  # type: ignore[assignment]


def reset_model_card_ece_cache() -> None:
    """Reset the in-memory cache of per-culture ECE values.  Exposed
    for tests that monkey-patch _load_model_card_ece or rewrite the
    model card on disk — without this they would silently see stale
    cached values.
    """
    global _MODEL_CARD_ECE_CACHE
    _MODEL_CARD_ECE_CACHE = None


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
    def _resolve_denylist_for_text(text: str) -> tuple:
        """Return (detected_language, denylist) for the input.

        Merges the English base _RESUME_VOCAB_DENYLIST with any
        locale-specific vocab in _RESUME_VOCAB_BY_LANG so multilingual
        resumes (English + Spanish / French / German / Portuguese /
        Italian) get both sets of denylisted vocabulary.
        """
        lang = _detect_language(text)
        if lang == "en":
            return lang, _RESUME_VOCAB_DENYLIST
        extra = _RESUME_VOCAB_BY_LANG.get(lang, frozenset())
        merged = _RESUME_VOCAB_DENYLIST | extra
        return lang, merged

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

        # Locale-aware denylist: English base + detected-language
        # additions.  See _resolve_denylist_for_text.
        _lang, denylist = BiasDetector._resolve_denylist_for_text(text)

        strategy_lists: list = []
        for block in blocks:
            cands: list = []
            for raw in block.split():
                # Keep hyphens and apostrophes INSIDE the token — these
                # are part of legitimate names like "O'Brien",
                # "Smith-Jones", "D'Angelo", "Anne-Marie".  Strip every
                # other non-letter character.  Strip surrounding punct
                # (commas / quotes from "Doe," / "'Mary'") to avoid
                # spurious tokens.
                cleaned = re.sub(r"[^A-Za-zÀ-ÿ'\-]", "", raw)
                cleaned = cleaned.strip("'-")
                if len(cleaned) < 2 or not cleaned[0].isupper():
                    continue
                # Denylist check uses the lowercased alpha-only form so
                # localised tokens with diacritics ("Ingénieur" ->
                # "ingenieur") are still caught by the locale denylist.
                stripped = re.sub(r"[^a-zàáâãäåçèéêëìíîïñòóôõöùúûüýÿ]", "", cleaned.lower())
                if stripped in denylist:
                    continue
                # Also try the stripped diacritics-removed form for
                # cross-locale matching ("Ingénieur" -> "ingenieur"
                # also caught by English "ingenieur" if added).
                normalised = (
                    unicodedata.normalize("NFKD", stripped)
                    .encode("ascii", "ignore").decode("ascii")
                )
                if normalised in denylist:
                    continue
                cands.append(cleaned)
            strategy_lists.append(cands)
        return strategy_lists

    @staticmethod
    def _pick_name_signal(strategy_lists: list, results_by_token: dict):
        """Walk the cascade strategies against a precomputed
        ``results_by_token`` map (keyed by the LOWER-CASED INPUT FORM
        of each candidate token, preserving hyphens / apostrophes).
        Returns ``(chosen, first_surname)`` — either may be None.

        Keying by the input form (not the classifier's internal
        ``result.name``) avoids a subtle bug: compound names like
        "Smith-Jones" classify via a part lookup that emits
        result.name="jones", but the cascade still sees the original
        "Smith-Jones" token.  Dict lookup by result.name would miss;
        keying by the input form aligns the two.

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
        # Adaptive header window: scans up to 8 short lines or 1000
        # chars, terminating on section headers or long paragraphs.
        # Replaces the prior fixed text[:200] cap that an attacker
        # could evade by padding the resume with 200+ chars of
        # address / contact info before the salutation.
        header_orig = _extract_header_window(text)

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
            # ISO 639-1 code of the detected resume language (en, es,
            # fr, de, pt, it).  Defaults to "en" when no language
            # clears the stopword-frequency threshold.  Surfaced so
            # downstream callers can branch on locale (and so audit
            # reports can show the language mix).
            "detected_language": _detect_language(text),
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

        # 2b. RTL honorific scan — Arabic / Hebrew salutations.  We
        # OR into the existing signals so a resume with both Latin
        # and RTL salutations produces signal from whichever fires.
        # Name extraction from RTL scripts is NOT attempted; only the
        # honorific signal is lifted (the classifier was trained on
        # romanised forms).
        if _has_rtl_script(header_orig):
            if _rtl_honorific_fires(_RTL_HONORIFICS["male"], header_orig):
                signals["male_title"] = True
            if _rtl_honorific_fires(_RTL_HONORIFICS["female"], header_orig):
                signals["female_title"] = True
            if _rtl_honorific_fires(_RTL_HONORIFICS["neutral"], header_orig):
                signals["neutral_title"] = True

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
            #
            # Key by the LOWER-CASED INPUT FORM (with hyphens / apostrophes
            # preserved) so the cascade can look up "Anne-Marie" /
            # "Smith-Jones" tokens directly.  See _pick_name_signal.
            from fairness.names.classifier import predict_many
            distinct = sorted({
                t.lower() for cands in strategy_lists for t in cands
            })
            if distinct:
                batch = predict_many(distinct)
                results_by_token = dict(zip(distinct, batch))
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
        group_selection_rates: dict,
        group_sizes: Optional[dict] = None,
    ) -> float:
        """Demographic Parity Distance (DPD).

        DPD = max |P(Y=1 | G=g) - P(Y=1)| over all groups g.
        Returns 0 when perfectly fair.

        When ``group_sizes`` is provided, the overall rate P(Y=1) is
        computed as the SIZE-WEIGHTED Σ_g (n_g/N) * rate_g — the
        statistically correct population rate.  Without sizes, falls
        back to the unweighted mean of rates (which over-weights small
        groups and was the bug flagged in the security review).

        For full statistical breakdown (chi-squared independence test,
        Theil index, weighted overall rate), see
        demographic_parity_full().
        """
        if not group_selection_rates:
            return 0.0
        rates = list(group_selection_rates.values())
        if group_sizes:
            total_size = sum(group_sizes.get(g, 0) for g in group_selection_rates)
            if total_size > 0:
                overall_rate = sum(
                    group_sizes.get(g, 0) * r
                    for g, r in group_selection_rates.items()
                ) / total_size
            else:
                overall_rate = float(np.mean(rates))
        else:
            overall_rate = float(np.mean(rates))
        return float(max(abs(r - overall_rate) for r in rates))

    @staticmethod
    def demographic_parity_full(
        group_selected: dict,
        group_total: dict,
    ) -> dict:
        """Comprehensive demographic-parity statistics.

        Returns a dict with:

          overall_selection_rate
            Size-weighted Σ_g n_g r_g / N — the population's overall
            selection rate.  THIS is the statistically correct P(Y=1)
            value to compare per-group rates against.

          group_rates
            {group: r_g}, each in [0, 1].

          dpd_weighted
            max_g |r_g - overall_rate| using the weighted overall rate.

          dpd_unweighted
            Same metric using the unweighted mean of rates (the legacy
            value).  Surfaced for backwards comparability.

          chi_squared
            {statistic, p_value, dof}
            Chi-squared independence test on the 2 x G contingency
            table (selected vs not, per group).  Low p_value (e.g.
            < 0.05) means the rates differ more than chance would
            explain.

          theil_t
            Theil's T inequality index of selection rates.  Sensitive
            to deviations in the upper tail (high-selection groups).
            0 means perfect equality; higher means more inequality.

        All values rounded to 4 decimals.
        """
        groups = [g for g in group_total if group_total[g] > 0]
        if not groups:
            return {
                "overall_selection_rate": 0.0,
                "group_rates":            {},
                "dpd_weighted":           0.0,
                "dpd_unweighted":         0.0,
                "chi_squared":            {"statistic": 0.0, "p_value": 1.0, "dof": 0},
                "theil_t":                0.0,
            }

        rates = {g: group_selected[g] / group_total[g] for g in groups}
        N = sum(group_total[g] for g in groups)
        total_selected = sum(group_selected[g] for g in groups)
        overall = total_selected / N if N > 0 else 0.0

        dpd_weighted   = max(abs(r - overall) for r in rates.values())
        dpd_unweighted = max(
            abs(r - float(np.mean(list(rates.values()))))
            for r in rates.values()
        )

        # Chi-squared on the 2 x G contingency table.
        chi_block: dict = {"statistic": 0.0, "p_value": 1.0, "dof": 0}
        try:
            from scipy.stats import chi2_contingency
            observed = np.array([
                [group_selected[g] for g in groups],
                [group_total[g] - group_selected[g] for g in groups],
            ])
            chi2, p, dof, _exp = chi2_contingency(observed)
            chi_block = {
                "statistic": round(float(chi2), 4),
                "p_value":   round(float(p), 6),
                "dof":       int(dof),
            }
        except Exception:
            pass

        # Theil's T (entropy-based inequality).  Defined as
        # T = (1/G) Σ_g (r_g/mean) ln(r_g/mean).  Skip groups with
        # zero rate (ln 0 undefined); they contribute 0 by convention.
        theil_t = 0.0
        mean_rate = float(np.mean(list(rates.values()))) if rates else 0.0
        if mean_rate > 0:
            terms = []
            for r in rates.values():
                if r > 0:
                    terms.append((r / mean_rate) * float(np.log(r / mean_rate)))
            if terms:
                theil_t = float(np.mean(terms))

        return {
            "overall_selection_rate": round(overall, 4),
            "group_rates":            {g: round(r, 4) for g, r in rates.items()},
            "dpd_weighted":           round(dpd_weighted, 4),
            "dpd_unweighted":         round(dpd_unweighted, 4),
            "chi_squared":            chi_block,
            "theil_t":                round(theil_t, 4),
        }

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

    @staticmethod
    def _wilson_interval(selected: int, total: int,
                         z: float = 1.96) -> tuple:
        """Two-sided Wilson score confidence interval for a binomial
        proportion.  At z=1.96 -> 95% CI.

        Wilson is preferred over normal-approximation Wald because it
        stays inside [0, 1] and is well-behaved at extreme rates and
        small sample sizes — exactly the regime AIR audits live in.

        Returns (lower, upper) both clipped to [0, 1].
        """
        if total <= 0:
            return (0.0, 0.0)
        p = selected / total
        n = total
        denom = 1 + z * z / n
        centre = (p + z * z / (2 * n)) / denom
        margin = (z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
        return (max(0.0, centre - margin), min(1.0, centre + margin))

    def adverse_impact_ratio(
        self,
        group_a_selected: int,
        group_a_total: int,
        group_b_selected: int,
        group_b_total: int,
        protected_group: Optional[str] = None,
        group_a_name: str = "a",
        group_b_name: str = "b",
    ) -> dict:
        """Compute Adverse Impact Ratio with directional + symmetric views.

        The classic EEOC 4/5 rule is DIRECTIONAL:
            AIR = selection_rate_protected / selection_rate_reference
        with the "protected" group being the one that historically
        faced discrimination (or the one with lower selection rate
        in the audit if no prior protected designation is given).

        Args:
            group_a_selected, group_a_total: counts for group A
            group_b_selected, group_b_total: counts for group B
            protected_group: one of group_a_name or group_b_name to
                designate as protected.  When None we auto-pick the
                group with the lower selection rate (the conservative
                "find the disadvantaged group" interpretation).
            group_a_name, group_b_name: human-readable labels.

        Returns a dict with BOTH views:

          group_*_rate, group_*_count, group_*_total, group_*_wilson_ci
            Descriptive per-group stats, including the Wilson 95% CI
            on the selection rate.

          protected_group, reference_group     Names of the two roles.
          directional_air = sel_protected / sel_reference
            One-directional AIR — < 1 means protected group is
            disadvantaged.  THIS is the EEOC-style number.
          adverse_impact_ratio_symmetric = min(rates) / max(rates)
            Symmetric form retained for backwards compat (this is
            what the prior implementation returned).
          adverse_impact_ratio
            ALIAS for directional_air — this is the publish-ready
            value going forward.
          passes_4_5_rule
            directional_air >= threshold (default 0.80).
          risk_level                            LOW/MODERATE/HIGH/CRITICAL
          air_lower_ci, air_upper_ci
            Wilson-based interval on the AIR ratio computed by
            propagating each rate's CI through the ratio (uses the
            conservative endpoint combinations).
        """
        rate_a = group_a_selected / group_a_total if group_a_total > 0 else 0.0
        rate_b = group_b_selected / group_b_total if group_b_total > 0 else 0.0
        ci_a   = BiasDetector._wilson_interval(group_a_selected, group_a_total)
        ci_b   = BiasDetector._wilson_interval(group_b_selected, group_b_total)

        # Symmetric AIR (legacy)
        if rate_a == 0 and rate_b == 0:
            air_sym = 1.0
        elif max(rate_a, rate_b) == 0:
            air_sym = 0.0
        else:
            air_sym = min(rate_a, rate_b) / max(rate_a, rate_b)

        # Pick protected group
        if protected_group is None:
            protected = group_a_name if rate_a <= rate_b else group_b_name
        elif protected_group in (group_a_name, group_b_name):
            protected = protected_group
        else:
            protected = group_a_name if rate_a <= rate_b else group_b_name
        reference = group_b_name if protected == group_a_name else group_a_name

        sel_protected = rate_a if protected == group_a_name else rate_b
        sel_reference = rate_b if protected == group_a_name else rate_a

        if sel_reference == 0:
            air_dir = 1.0 if sel_protected == 0 else float("inf")
        else:
            air_dir = sel_protected / sel_reference
        # Clip infinity for downstream JSON serialisation.
        if air_dir == float("inf"):
            air_dir = 999.0

        # CI propagation: AIR = p_prot / p_ref.  Conservative interval:
        # lower = (lower bound of protected) / (upper bound of reference)
        # upper = (upper bound of protected) / (lower bound of reference)
        ci_protected = ci_a if protected == group_a_name else ci_b
        ci_reference = ci_b if protected == group_a_name else ci_a
        if ci_reference[1] > 0:
            air_lower = ci_protected[0] / ci_reference[1]
        else:
            air_lower = 0.0
        if ci_reference[0] > 0:
            air_upper = ci_protected[1] / ci_reference[0]
        else:
            air_upper = 999.0

        return {
            f"{group_a_name}_rate":      round(rate_a, 4),
            f"{group_b_name}_rate":      round(rate_b, 4),
            f"{group_a_name}_count":     int(group_a_selected),
            f"{group_b_name}_count":     int(group_b_selected),
            f"{group_a_name}_total":     int(group_a_total),
            f"{group_b_name}_total":     int(group_b_total),
            f"{group_a_name}_wilson_ci": (round(ci_a[0], 4), round(ci_a[1], 4)),
            f"{group_b_name}_wilson_ci": (round(ci_b[0], 4), round(ci_b[1], 4)),
            # Legacy keys preserved
            "group_a_rate":                  round(rate_a, 4),
            "group_b_rate":                  round(rate_b, 4),
            "adverse_impact_ratio_symmetric": round(air_sym, 4),
            # New directional answer
            "protected_group":               protected,
            "reference_group":               reference,
            "directional_air":               round(air_dir, 4),
            "adverse_impact_ratio":          round(air_dir, 4),
            "passes_4_5_rule":               air_dir >= self.adverse_impact_threshold,
            "risk_level":                    self._risk_level(air_dir),
            "air_lower_ci":                  round(air_lower, 4),
            "air_upper_ci":                  round(min(air_upper, 999.0), 4),
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
        resume_texts: dict,
        scores: dict,
        selection_threshold: Optional[float] = None,
        cutoff_method: str = "median",
        top_k: Optional[int] = None,
        percentile: Optional[float] = None,
        dedup: bool = True,
        near_dup_hamming: int = _NEAR_DUPLICATE_HAMMING_THRESHOLD,
        scorer=None,
        jd_text: Optional[str] = None,
        counterfactual_sample_size: int = 10,
        audit_log_path: Optional[Path] = None,
        write_baseline: bool = False,
    ) -> dict:
        """Comprehensive bias audit across detected demographic groups.

        Args:
            resume_texts: {resume_filename: text}
            scores: {resume_filename: ranking_score}
            selection_threshold: Score threshold for selected vs not selected.
                                 If None, uses the median score.
            scorer: Optional callable ``scorer(jd, resume) -> float`` —
                if provided alongside jd_text, the audit additionally
                runs the counterfactual name-swap robustness harness on
                a sample of candidates and surfaces the result under
                audit["counterfactual_robustness"].  See
                evaluation/counterfactual_robustness.py.
            jd_text: Job-description text required when ``scorer`` is
                provided.  Without it the counterfactual check cannot run.
            counterfactual_sample_size: Number of candidates to sample
                for the counterfactual check (default 10).  Each sampled
                candidate triggers 16 scorer calls (8 male + 8 female
                name swaps); the harness is cheap but not free.
        """
        # --- Pre-step: deduplication -----------------------------------
        # Drop exact duplicate resume bodies (SHA-256 of normalised text)
        # AND near-duplicates (SimHash Hamming <= near_dup_hamming).
        # Without this, the same resume submitted 100 times inflates
        # one group's denominator and the AIR "pass" verdict.  This is
        # the explicit attack vector closed by task #8.
        dedup_report: dict = {
            "applied":              dedup,
            "input_resumes":        len(resume_texts),
            "exact_duplicate_sets": 0,
            "exact_dropped":        0,
            "near_duplicate_pairs": 0,
            "kept":                 len(resume_texts),
            "ballot_stuffing_alert": False,
        }
        if dedup and resume_texts:
            import hashlib
            from collections import defaultdict as _dd

            # Exact dedup by SHA-256 of normalised (whitespace-collapsed) text.
            hash_to_files: dict = _dd(list)
            for fn, text in resume_texts.items():
                norm = " ".join((text or "").split()).lower()
                h = hashlib.sha256(norm.encode("utf-8")).hexdigest()
                hash_to_files[h].append(fn)
            kept_filenames: set = set()
            exact_dropped = 0
            exact_dup_sets = 0
            for files in hash_to_files.values():
                kept_filenames.add(files[0])
                if len(files) > 1:
                    exact_dup_sets += 1
                    exact_dropped += len(files) - 1

            # Near-dup SimHash pass on the kept-after-exact set.
            kept_list = list(kept_filenames)
            simhashes = {fn: _simhash(resume_texts[fn]) for fn in kept_list}
            near_pairs: list = []
            survivors = set(kept_list)
            for i, fn_i in enumerate(kept_list):
                if fn_i not in survivors:
                    continue
                for fn_j in kept_list[i + 1:]:
                    if fn_j not in survivors:
                        continue
                    if _hamming_distance(
                        simhashes[fn_i], simhashes[fn_j]
                    ) <= near_dup_hamming:
                        near_pairs.append((fn_i, fn_j))
                        survivors.discard(fn_j)

            new_resume_texts = {
                fn: resume_texts[fn]
                for fn in kept_list if fn in survivors
            }
            new_scores = {fn: scores[fn] for fn in new_resume_texts if fn in scores}
            dropped_total = len(resume_texts) - len(new_resume_texts)
            ballot_stuffing = (
                dropped_total / max(len(resume_texts), 1)
                >= _BALLOT_STUFFING_THRESHOLD
            )
            dedup_report.update({
                "exact_duplicate_sets": exact_dup_sets,
                "exact_dropped":        exact_dropped,
                "near_duplicate_pairs": len(near_pairs),
                "kept":                 len(new_resume_texts),
                "ballot_stuffing_alert": ballot_stuffing,
            })
            resume_texts = new_resume_texts
            scores = new_scores

        # --- Cutoff resolution -----------------------------------------
        # The recruiter's OPERATIONAL cutoff governs whose "selected"
        # status the audit measures.  Four supported modes:
        #
        #   median       Selected = scores >= median(scores)   (default,
        #                legacy behaviour; useful for "balanced 50/50"
        #                pass/fail framing).
        #   top_k        Selected = top-k by score (requires top_k=N).
        #                Mirrors how recruiters actually use ranked lists.
        #   percentile   Selected = scores >= p-th percentile
        #                (requires percentile=0..100).
        #   explicit     Selected = scores >= selection_threshold
        #                (requires selection_threshold=float).
        #
        # All modes resolve to a single ``selection_threshold`` float
        # internally so downstream code is unchanged.  The chosen method
        # and its concrete threshold are surfaced in the audit report
        # under ``cutoff_method``, ``cutoff_threshold``, ``cutoff_top_k``,
        # and ``cutoff_percentile`` for full reviewer disclosure.
        score_values = list(scores.values())
        if not score_values:
            cutoff_method = "median"  # nothing to do — degenerate
        cutoff_top_k = None
        cutoff_percentile = None
        if cutoff_method == "top_k":
            if top_k is None or top_k <= 0:
                raise ValueError("cutoff_method='top_k' requires top_k > 0")
            sorted_scores = sorted(score_values, reverse=True)
            cutoff_top_k = min(int(top_k), len(sorted_scores))
            # Threshold = the score AT rank top_k.  >= this is selected.
            selection_threshold = float(sorted_scores[cutoff_top_k - 1])
        elif cutoff_method == "percentile":
            if percentile is None or not (0 <= percentile <= 100):
                raise ValueError(
                    "cutoff_method='percentile' requires percentile in [0, 100]"
                )
            cutoff_percentile = float(percentile)
            selection_threshold = float(
                np.percentile(score_values, 100 - percentile)
            )
        elif cutoff_method == "explicit":
            if selection_threshold is None:
                raise ValueError(
                    "cutoff_method='explicit' requires selection_threshold=float"
                )
            selection_threshold = float(selection_threshold)
        else:  # median
            cutoff_method = "median"
            if selection_threshold is None:
                selection_threshold = float(np.median(score_values))

        # --- Phase 0: classifier integrity verification -----------------
        # Force the classifier to load (idempotent) so its integrity
        # check has run, then surface the result.  A mismatch means
        # model.pkl has been swapped or corrupted since training; the
        # rest of the audit still runs (so callers can compare against
        # historical baselines) but the report prepends a CRITICAL
        # recommendation and exposes the hash mismatch.
        from fairness.names.classifier import get_classifier
        _clf = get_classifier()
        _clf._ensure_loaded()
        _integrity_violated = bool(
            getattr(_clf, "integrity_violated", False)
        )

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
            distinct = sorted(token_union)
            batch = predict_many(distinct)
            # Key by INPUT form (not result.name) so the cascade
            # lookups in _pick_name_signal land — compound names like
            # "Smith-Jones" have result.name="jones" via part lookup
            # but the cascade still sees the original token.
            results_by_token = dict(zip(distinct, batch))
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
                # Forensic per-resume trail — full signal snapshot for
                # individual-case review without having to re-run
                # detect_gender_proxy_scored separately.  Surfaced
                # under audit["per_resume"] below.
                if "per_resume" not in locals():
                    per_resume: dict = {}
                per_resume[filename] = {
                    "score":              scores[filename],
                    "selected":           selected,
                    "hard_gender":        gender,
                    "name_token":         signals.get("name_token", ""),
                    "name_source":        signals.get("name_source", "empty"),
                    "name_p_female":      signals.get("name_p_female", 0.5),
                    "name_is_surname":    signals.get("name_is_surname", False),
                    "name_culture":       signals.get("name_culture", "unknown"),
                    "detected_language":  signals.get("detected_language", "en"),
                    "male_pronoun":       signals.get("male_pronoun", 0),
                    "female_pronoun":     signals.get("female_pronoun", 0),
                    "male_title":         signals.get("male_title", False),
                    "female_title":       signals.get("female_title", False),
                    "neutral_title":      signals.get("neutral_title", False),
                    "confidence":         result.get("confidence", 0.0),
                }

        results: dict = {
            "threshold": selection_threshold,
            "cutoff_method":      cutoff_method,
            "cutoff_threshold":   selection_threshold,
            "cutoff_top_k":       cutoff_top_k,
            "cutoff_percentile":  cutoff_percentile,
            "dedup":              dedup_report,
            "total_resumes": len(scores),
            "gender_distribution": {},
            "gender_bias_analysis": {},
            "score_distribution": {},
            "detection_coverage": {},
            "culture_distribution": {},
            # Forensic per-resume audit trail.  Same signals
            # detect_gender_proxy_scored produced, surfaced keyed by
            # filename so reviewers can drill into individual cases
            # without having to re-invoke the detector on each resume.
            "per_resume": locals().get("per_resume", {}),
            "integrity": {
                # SHA-256 of the loaded model.pkl compared against the
                # value recorded in model_card.json at training time.
                # See fairness/names/classifier.py for the verification.
                "model_integrity_violated": _integrity_violated,
                "expected_sha256":          getattr(_clf, "expected_sha", None),
                "actual_sha256":            getattr(_clf, "actual_sha", None),
            },
            "recommendations": [],
        }

        if dedup_report["ballot_stuffing_alert"]:
            dropped = (
                dedup_report["exact_dropped"]
                + dedup_report["near_duplicate_pairs"]
            )
            results["recommendations"].append(
                f"[SUSPECT] {dropped}/{dedup_report['input_resumes']} "
                f"resumes were duplicates of each other "
                f"({100 * dropped / max(dedup_report['input_resumes'], 1):.0f}%). "
                f"This may be ballot-stuffing — inspect "
                f"audit['dedup'] for the affected sets."
            )

        if _integrity_violated:
            results["recommendations"].append(
                f"[CRITICAL] Classifier integrity check FAILED. "
                f"Expected SHA-256 starts with "
                f"{(_clf.expected_sha or '?')[:16]}..., got "
                f"{(_clf.actual_sha or '?')[:16]}.... The model file "
                f"has been modified since training.  Audit verdicts "
                f"below are computed from an UNVERIFIED classifier."
            )
        # Card schema validation results — surfaced so a malformed
        # card (typically caused by a partial retrain that dropped
        # a field) is visible in the audit instead of crashing
        # downstream metric consumers.
        _card_errors = getattr(_clf, "card_validation_errors", [])
        results["model_card_validation"] = {
            "valid":  not _card_errors,
            "errors": list(_card_errors),
        }
        if _card_errors:
            results["recommendations"].append(
                f"[CRITICAL] model_card.json failed schema validation: "
                f"{_card_errors}.  Some audit fields may be missing or "
                f"silently broken; re-run train_classifier.py to "
                f"regenerate the card."
            )

        # Detection coverage stats
        n_known = sum(len(v) for k, v in gender_groups.items() if k != "unknown")
        n_unknown = len(gender_groups.get("unknown", []))
        coverage_rate = n_known / max(n_known + n_unknown, 1)
        results["detection_coverage"] = {
            "detected":             n_known,
            "undetected":           n_unknown,
            "coverage_rate":        round(coverage_rate, 4),
            "coverage_floor":       _DETECTION_COVERAGE_FLOOR,
            "coverage_floor_met":   coverage_rate >= _DETECTION_COVERAGE_FLOOR,
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

        # --- Comprehensive parity statistics (Task #5) -----------------
        # Adds size-weighted DPD, chi-squared independence test, and
        # Theil-T inequality on the male × female × selected
        # contingency table.  Surfaced regardless of whether the AIR
        # block fires; small-sample audits still benefit from the
        # descriptive view.
        if male_data and female_data:
            results["parity_statistics"] = self.demographic_parity_full(
                group_selected={
                    "male":   sum(1 for r in male_data if r["selected"]),
                    "female": sum(1 for r in female_data if r["selected"]),
                },
                group_total={
                    "male":   len(male_data),
                    "female": len(female_data),
                },
            )

        if male_data and female_data:
            air_hard = self.adverse_impact_ratio(
                group_a_selected=sum(1 for r in male_data if r["selected"]),
                group_a_total=len(male_data),
                group_b_selected=sum(1 for r in female_data if r["selected"]),
                group_b_total=len(female_data),
                group_a_name="male",
                group_b_name="female",
            )
            air_soft = self._air_soft(candidate_records)
            conservative = min(
                air_hard["adverse_impact_ratio"],
                air_soft["adverse_impact_ratio"],
            )
            raw_passes = conservative >= self.adverse_impact_threshold
            # The publish-ready verdict folds in multiple gates in
            # PRIORITY ORDER.  Earlier gates dominate later ones —
            # an inconclusive verdict due to low coverage is more
            # informative than a "fail" verdict computed from the
            # same too-small denominator.  raw passes_4_5_rule is
            # left unchanged so callers that want the unadjusted
            # math can still read it.
            coverage_rate = results["detection_coverage"]["coverage_rate"]
            if coverage_rate < _DETECTION_COVERAGE_FLOOR:
                verdict = "inconclusive_low_detection_coverage"
            elif drift_status in ("inconclusive_low_ece_coverage",
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

            if verdict == "inconclusive_low_detection_coverage":
                results["recommendations"].append(
                    f"[INCONCLUSIVE] Detection coverage "
                    f"{coverage_rate:.0%} is below the "
                    f"{_DETECTION_COVERAGE_FLOOR:.0%} floor.  Only "
                    f"{n_known} of {n_known + n_unknown} resumes had a "
                    f"usable gender signal; AIR computed from the "
                    f"remaining sample is too small to publish a "
                    f"verdict.  Inspect audit['per_resume'] to find "
                    f"the candidates with name_source='empty'."
                )
            elif "inconclusive" in verdict:
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

        # --- Counterfactual robustness on the SCORER itself --------------
        # Audit-time integration of evaluation/counterfactual_robustness.py.
        # When a scorer (and JD) are provided we sample candidates and
        # run the name-swap harness on each, then aggregate.  This is
        # what closes the "we built the harness but never used it
        # automatically" gap.
        # --- Historical drift detection (audit log) ---------------------
        # Optional baseline-comparison block.  When audit_log_path is
        # given AND the file already exists with at least one prior
        # record, we compare the current weighted_ece / AIR against
        # the LAST recorded baseline and surface the deltas.  When
        # write_baseline=True we append the current run to the log.
        if audit_log_path is not None:
            import json
            from datetime import datetime, timezone
            log_p = Path(audit_log_path)
            if log_p.exists():
                try:
                    last_line = None
                    with log_p.open(encoding="utf-8") as fh:
                        for line in fh:
                            if line.strip():
                                last_line = line
                    if last_line:
                        baseline = json.loads(last_line)
                        cur_ece = results["calibration_drift"].get("weighted_ece")
                        baseline_ece = baseline.get("weighted_ece")
                        cur_air = None
                        baseline_air = baseline.get("adverse_impact_ratio")
                        if results.get("gender_bias_analysis"):
                            cur_air = results["gender_bias_analysis"].get(
                                "adverse_impact_ratio"
                            )
                        ece_delta = (
                            None if cur_ece is None or baseline_ece is None
                            else round(cur_ece - baseline_ece, 4)
                        )
                        air_delta = (
                            None if cur_air is None or baseline_air is None
                            else round(cur_air - baseline_air, 4)
                        )
                        results["drift_since_baseline"] = {
                            "baseline_timestamp":   baseline.get("timestamp"),
                            "baseline_weighted_ece": baseline_ece,
                            "current_weighted_ece":  cur_ece,
                            "weighted_ece_delta":    ece_delta,
                            "baseline_air":          baseline_air,
                            "current_air":           cur_air,
                            "air_delta":             air_delta,
                        }
                        # Material drift trips a recommendation.  Tunable
                        # via the constants but the defaults are
                        # conservative — small per-audit fluctuations
                        # shouldn't fire.
                        if (ece_delta is not None and abs(ece_delta) > 0.02):
                            results["recommendations"].append(
                                f"[DRIFT] Weighted ECE changed by "
                                f"{ece_delta:+.4f} since baseline "
                                f"({baseline.get('timestamp')}).  "
                                f"Audited corpus composition may have "
                                f"shifted; review culture_distribution."
                            )
                        if (air_delta is not None and abs(air_delta) > 0.05):
                            results["recommendations"].append(
                                f"[DRIFT] AIR changed by {air_delta:+.4f} "
                                f"since baseline.  Investigate whether "
                                f"the ranker behaviour or candidate mix "
                                f"changed materially."
                            )
                except Exception:
                    pass

            if write_baseline:
                try:
                    record = {
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "n_resumes": len(scores),
                        "weighted_ece": results["calibration_drift"].get(
                            "weighted_ece"
                        ),
                        "ece_coverage": results["calibration_drift"].get(
                            "ece_coverage"
                        ),
                        "adverse_impact_ratio": (
                            results.get("gender_bias_analysis", {}).get(
                                "adverse_impact_ratio"
                            )
                        ),
                        "verdict": (
                            results.get("gender_bias_analysis", {}).get(
                                "verdict"
                            )
                        ),
                        "culture_distribution": {
                            c: stats["count"]
                            for c, stats in results.get(
                                "culture_distribution", {}
                            ).items()
                        },
                    }
                    log_p.parent.mkdir(parents=True, exist_ok=True)
                    with log_p.open("a", encoding="utf-8") as fh:
                        fh.write(json.dumps(record) + "\n")
                except Exception:
                    pass

        if scorer is not None and jd_text is not None and resume_texts:
            from evaluation.counterfactual_robustness import (
                name_swap_robustness,
            )
            sampled = list(resume_texts.items())[:counterfactual_sample_size]
            reports: list = []
            for _, body in sampled:
                try:
                    rep = name_swap_robustness(
                        scorer=scorer, jd=jd_text, base_resume=body,
                    )
                except Exception:
                    continue
                if rep.male_scores and rep.female_scores:
                    reports.append(rep)

            if reports:
                gaps = [r.score_gap for r in reports]
                deltas = [r.max_swap_delta for r in reports]
                mean_gap = sum(gaps) / len(gaps)
                max_swap = max(deltas)
                n_robust = sum(1 for r in reports if r.robust)
                results["counterfactual_robustness"] = {
                    "n_sampled":              len(reports),
                    "mean_score_gap":         round(mean_gap, 4),
                    "max_swap_delta":         round(max_swap, 4),
                    "fraction_robust":        round(n_robust / len(reports), 4),
                    "all_robust":             n_robust == len(reports),
                }
                if n_robust < len(reports):
                    results["recommendations"].append(
                        f"[FAIRNESS] Counterfactual robustness audit "
                        f"FAILED on {len(reports) - n_robust}/{len(reports)} "
                        f"sampled candidates.  Mean name-swap score gap = "
                        f"{mean_gap:.4f}, max single-swap delta = "
                        f"{max_swap:.4f}.  The scorer is sensitive to "
                        f"the candidate's NAME, not just their content."
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
