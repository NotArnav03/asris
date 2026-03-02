"""
ASRIS — Text Normalizer
Comprehensive text cleaning, normalization, and lemmatization pipeline.
"""

import re
import unicodedata
from typing import Optional

# Try to load spaCy for lemmatization; fall back to basic if unavailable
try:
    import spacy
    _nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
    _HAS_SPACY = True
except (ImportError, OSError):
    _HAS_SPACY = False

import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))
from config import get_logger

logger = get_logger("preprocessing.normalizer")


# ─── Regex Patterns (compiled once) ─────────────────────────────
_EMAIL_RE = re.compile(r"\S+@\S+\.\S+")
_PHONE_RE = re.compile(r"(\+?\d{1,3}[\s\-]?)?\(?\d{2,4}\)?[\s\-]?\d{3,4}[\s\-]?\d{3,4}")
_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_LINKEDIN_RE = re.compile(r"linkedin\.com/\S+", re.IGNORECASE)
_GITHUB_RE = re.compile(r"github\.com/\S+", re.IGNORECASE)
_MULTIPLE_SPACES_RE = re.compile(r"[ \t]+")
_MULTIPLE_NEWLINES_RE = re.compile(r"\n{3,}")
_SPECIAL_CHARS_RE = re.compile(r"[^\w\s\.\,\;\:\-\+\#\(\)\/\&]")
_BULLET_RE = re.compile(r"^[\s]*[•●○▪▸►◆★✦✓✔→\-\*]\s*", re.MULTILINE)


def normalize_unicode(text: str) -> str:
    """Normalize unicode characters to ASCII equivalents."""
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("ascii")
    return text


def remove_pii(text: str) -> str:
    """Remove personally identifiable information (emails, phones, URLs)."""
    text = _EMAIL_RE.sub("", text)
    text = _PHONE_RE.sub("", text)
    text = _URL_RE.sub("", text)
    text = _LINKEDIN_RE.sub("", text)
    text = _GITHUB_RE.sub("", text)
    return text


def clean_whitespace(text: str) -> str:
    """Normalize spaces, tabs, and excessive newlines."""
    text = _MULTIPLE_SPACES_RE.sub(" ", text)
    text = _MULTIPLE_NEWLINES_RE.sub("\n\n", text)
    lines = [line.strip() for line in text.split("\n")]
    return "\n".join(lines)


def normalize_bullets(text: str) -> str:
    """Standardize bullet point characters to a uniform format."""
    return _BULLET_RE.sub("- ", text)


def lemmatize(text: str) -> str:
    """Lemmatize text using spaCy (if available)."""
    if not _HAS_SPACY:
        return text

    doc = _nlp(text)
    tokens = []
    for token in doc:
        if not token.is_stop and not token.is_punct and not token.is_space:
            tokens.append(token.lemma_.lower())
    return " ".join(tokens)


def normalize_text(
    text: str,
    remove_personal_info: bool = True,
    do_lemmatize: bool = False,
    lowercase: bool = False,
) -> str:
    """
    Full text normalization pipeline.

    Args:
        text: Raw input text
        remove_personal_info: Strip emails, phones, URLs
        do_lemmatize: Apply lemmatization (requires spaCy)
        lowercase: Convert to lowercase

    Returns:
        Cleaned, normalized text
    """
    if not text or not isinstance(text, str):
        return ""

    text = normalize_unicode(text)

    if remove_personal_info:
        text = remove_pii(text)

    text = normalize_bullets(text)
    text = _SPECIAL_CHARS_RE.sub(" ", text)
    text = clean_whitespace(text)

    if do_lemmatize:
        text = lemmatize(text)

    if lowercase:
        text = text.lower()

    text = text.strip()
    return text


# ─── Batch Processing ───────────────────────────────────────────
def normalize_batch(texts: list[str], **kwargs) -> list[str]:
    """Normalize a list of texts with the same settings."""
    results = []
    for i, text in enumerate(texts):
        results.append(normalize_text(text, **kwargs))
        if (i + 1) % 500 == 0:
            logger.info(f"Normalized {i + 1}/{len(texts)} texts")
    return results


if __name__ == "__main__":
    sample = """
    John Doe  |  john.doe@email.com  |  +1-555-123-4567
    LinkedIn: linkedin.com/in/johndoe  |  GitHub: github.com/johndoe

    ★ PROFESSIONAL SUMMARY
    Experienced software engineer with 5+ years in Python, Java, C++

    ● EXPERIENCE
    • Senior Developer at TechCorp (2020–2024)
      ▸ Built microservices architecture serving 1M+ users
      ► Led team of 5 engineers

    ○ EDUCATION
    ▪ B.S. Computer Science, MIT (2016–2020)
    """

    print("=== ORIGINAL ===")
    print(sample)
    print("\n=== NORMALIZED ===")
    print(normalize_text(sample))
    print("\n=== LEMMATIZED ===")
    print(normalize_text(sample, do_lemmatize=True, lowercase=True))
