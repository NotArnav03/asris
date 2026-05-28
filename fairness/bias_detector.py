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

    The honorific itself is matched case-insensitively (scoped flag), but
    the follow-on word MUST start with an uppercase letter and contain
    only name-like characters (letters, apostrophes, hyphens).  This is
    the primary defence against "MS in CS" / "MR-aware" style strings.
    The denylist (`_HONORIFIC_DENYLIST`) is the secondary defence against
    the residual case where the follow-on word is Title-cased but not a
    name ("MS Office").
    """
    alts = "|".join(f"(?i:{re.escape(t)})" for t in tokens)
    # Optional period, then whitespace, then a Title-cased name token.
    # Apostrophes and hyphens are permitted to support "O'Neill", "Smith-Jones".
    return re.compile(rf"\b(?:{alts})\.?\s+([A-Z][A-Za-z'\-]+)")


_HONORIFIC_PATTERNS: dict = {
    "male":    _build_honorific_pattern(["Mr", "Mister", "Sir"]),
    "female":  _build_honorific_pattern(["Mrs", "Ms", "Miss", "Madam"]),
    "neutral": _build_honorific_pattern(["Dr", "Prof", "Professor", "Mx"]),
}


def _honorific_fires(pattern: re.Pattern, header_orig: str) -> bool:
    """Return True iff ``pattern`` matches inside ``header_orig`` with a
    follow-on token that is not on the denylist.

    ``header_orig`` MUST preserve the resume's original case — lowercasing
    the input collapses "MS Office" (false positive) onto "Ms. Officer"
    (true positive) and defeats the strict-case capture group.
    """
    for m in pattern.finditer(header_orig):
        follow = m.group(1)
        if not follow:
            continue
        clean = follow.rstrip(".,;:!?").lower()
        if clean and clean not in _HONORIFIC_DENYLIST:
            return True
    return False

# --- Name-based gender proxies (common gendered first names) ---------------
# Organised by cultural cluster for transparency.
# Sources: US Social Security Administration top-1000 lists,
#          common South Asian, East Asian, and Arab given names.
GENDERED_NAMES: dict[str, set[str]] = {
    "male": {
        # Western English
        "james", "john", "robert", "michael", "william", "david",
        "richard", "joseph", "thomas", "charles", "daniel", "matthew",
        "anthony", "mark", "donald", "steven", "paul", "andrew",
        "kenneth", "george", "joshua", "kevin", "brian", "edward",
        "ronald", "timothy", "jason", "jeffrey", "ryan", "gary",
        "jacob", "nicholas", "eric", "jonathan", "stephen", "larry",
        "justin", "scott", "brandon", "benjamin", "samuel", "patrick",
        "frank", "raymond", "gregory", "jack", "dennis", "jerry",
        "tyler", "aaron", "adam", "henry", "nathan", "douglas",
        "zachary", "peter", "kyle", "walter", "ethan", "jeremy",
        "harold", "terry", "sean", "arthur", "christian", "austin",
        "bruce", "ralph", "roy", "noah", "russell", "alan", "philip",
        "todd", "carl", "cameron", "logan", "hunter", "mason", "liam",
        "oliver", "elijah", "lucas", "aiden", "owen", "caleb",
        "connor", "wyatt", "jayden", "gabriel", "dylan", "jordan",
        # "lee" removed — predominantly used as a surname and as a
        # unisex given name; see _UNISEX_NAMES.
        "bryan", "billy", "marcus", "christopher", "alexander",
        "sebastian", "leo", "julian", "evan", "isaac", "dominic",
        "parker", "cooper", "lincoln", "xavier", "eli", "colton",
        "nolan", "jaxon", "hudson", "levi", "landon", "jackson",
        "carson", "jameson", "grayson", "maverick", "roman", "bryson",
        "ivan", "victor", "felix", "max", "charlie", "theo", "harry",
        "oscar", "george", "freddie", "alfie", "archie", "reuben",
        # South Asian (male)
        "rahul", "amit", "vikram", "arun", "suresh", "rajesh",
        "arjun", "ravi", "sanjay", "deepak", "manish", "ajay",
        "akash", "anand", "aniket", "ankur", "aditya", "abhishek",
        "ashish", "atul", "gaurav", "harsh", "kunal", "mayank",
        "mohit", "nikhil", "nishant", "piyush", "pratik", "prateek",
        "rohit", "sachin", "sahil", "shubham", "siddharth", "sumit",
        "vaibhav", "vivek", "yash", "karan", "rohan", "sandeep",
        "vikas", "aarav", "dev", "harish", "krishna", "vishnu",
        "santosh", "ramesh", "naresh", "mahesh", "dinesh", "ganesh",
        # East Asian (male)
        # NOTE: Chinese family names (chen, li, wang, zhang, liu) and
        # the unisex Korean syllable "hyun" were removed from this list.
        # Family names carry no given-name gender signal, and including
        # them caused every East Asian candidate to be misclassified
        # male regardless of actual gender; "hyun" appeared in BOTH the
        # male and female lists, which silently cancelled to "unknown".
        # See _UNISEX_NAMES for unisex Korean/Chinese tokens.
        "wei", "ming", "jun", "yang", "xiao", "lei", "fang", "hao",
        "long", "tao", "ping", "bo", "zhen", "jian", "hiro",
        "kenji", "takashi", "naoki", "daisuke", "ryo", "yuto",
        "seung", "jae", "sung", "dong", "tae",
        # Arab / Middle Eastern (male)
        "mohammed", "omar", "hassan", "ali", "ahmed", "khalid",
        "yusuf", "ibrahim", "mustafa", "tariq", "walid", "bilal",
        "kareem", "faris", "zaid", "nabil", "rami", "samir",
        "karim", "jamal", "nasser",
    },
    "female": {
        # Western English
        "mary", "patricia", "jennifer", "linda", "barbara", "elizabeth",
        "susan", "jessica", "sarah", "karen", "nancy", "lisa",
        "margaret", "betty", "sandra", "ashley", "dorothy", "kimberly",
        "emily", "donna", "michelle", "carol", "amanda", "melissa",
        "deborah", "stephanie", "rebecca", "sharon", "laura", "cynthia",
        "kathleen", "amy", "angela", "shirley", "anna", "brenda",
        "pamela", "emma", "nicole", "helen", "samantha", "katherine",
        "christine", "debra", "rachel", "carolyn", "janet", "catherine",
        "maria", "heather", "diane", "julie", "joyce", "victoria",
        "kelly", "christina", "lauren", "joan", "evelyn", "olivia",
        "judith", "megan", "cheryl", "martha", "andrea", "frances",
        "hannah", "jacqueline", "ann", "gloria", "teresa", "kathryn",
        "sara", "janice", "jean", "alice", "julia", "grace", "judy",
        "theresa", "rose", "beverly", "denise", "amber", "marilyn",
        "danielle", "crystal", "brittany", "natalie", "sophia",
        "madison", "isabella", "aria", "scarlett", "zoe", "chloe",
        "hazel", "lily", "mia", "ellie", "avery", "ella", "abigail",
        "aaliyah", "nora", "charlotte", "amelia", "ava", "harper",
        "luna", "camila", "sofia", "gianna", "violet", "aurora",
        "savannah", "audrey", "brooklyn", "bella", "claire", "skylar",
        "lucy", "paisley", "everly", "caroline", "nova", "emilia",
        "kennedy", "maya", "willow", "kinsley", "naomi", "elena",
        "ariel", "leah", "stella", "zara", "eva", "ivy", "ruby",
        "poppy", "daisy", "freya", "isla", "florence", "imogen",
        # South Asian (female)
        "priya", "anita", "sunita", "kavita", "neha", "pooja",
        "divya", "meena", "rekha", "anjali", "deepa", "geeta",
        "jyoti", "kritika", "lakshmi", "manisha", "nisha", "poonam",
        "radha", "rani", "shalini", "shruti", "swati", "tanvi",
        "uma", "vandana", "vineeta", "rashmi", "preeti", "pallavi",
        "namrata", "mamta", "komal", "kiran", "isha", "chandni",
        "archana", "aparna", "shreya", "riya", "tanya", "sangita",
        "namita", "sarita", "bharati",
        # East Asian (female)
        # NOTE: the Korean syllables hyun / young / min / ji / soo were
        # removed because they are routinely used across genders in
        # modern Korean given names (and "hyun" was simultaneously in
        # the male list, producing a silent cancellation).  They now
        # live in _UNISEX_NAMES and contribute no gender signal.
        "mei", "ling", "xiu", "yan", "fei", "qian", "jing", "yun",
        "shu", "xia", "akiko", "yoko", "haruko", "noriko", "keiko",
        "sachiko", "tomoko", "yuki", "sakura", "hana", "aiko",
        "eun", "na",
        "hua", "hong", "qing",
        # Arab / Middle Eastern (female)
        "fatima", "amira", "nadia", "layla", "yasmin", "nour",
        "rania", "zainab", "mariam", "hana", "dina", "lina",
        "rana", "mona", "huda", "asmaa", "salma", "aisha",
        "maryam", "sara",
    },
}


# --- Unisex given names ----------------------------------------------------
# Tokens that are statistically used across genders in their source culture.
# These are MATCHED so that a name like "Hyun Park" or "Jordan Smith" is
# correctly recognised as a first name (and so it can be excluded from
# pronoun-only or title-only fallbacks downstream), but they vote NEITHER
# male nor female from the name channel.
#
# Curated conservatively — every token here was either (a) present in both
# the male and female lists in a prior revision (a structural bug) or
# (b) routinely used across genders in the source culture per public
# naming statistics.  Western unisex names (jordan, avery, taylor, ...)
# remain hard-classified for now and are handled by the probabilistic
# classifier introduced in task #3.
_UNISEX_NAMES: set = {
    # Korean syllables that appear as unisex given names.
    # ("eun" is intentionally LEFT in the female list — it is strongly
    # female-coded in modern Korean usage despite occasional male use,
    # and the import-time invariant guards against re-adding it here.)
    "hyun", "young", "min", "ji", "soo", "jin", "joon", "hye",
    # Common Chinese given-name characters used across genders
    "yu", "an",
    # Western unisex (most-flagrant cases — extended in task #3)
    "lee",
}


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
})

# Minimum classifier confidence (=|p - 0.5| * 2) required for a token
# to drive the legacy male_name / female_name boolean signals.  Below
# this floor we still record name_p_female and set unisex_name=True,
# but we DO NOT vote for either gender — the classifier is too
# uncertain about this token to bias the categorical decision.
_NAME_SIGNAL_CONFIDENCE_FLOOR: float = 0.40


# --- Vocab consistency assertion ------------------------------------------
# Fail fast at import time if the name vocabulary develops cross-list
# contamination.  These invariants are LOAD-BEARING for the fairness audit:
# a single token appearing in two sets silently produces "unknown" for
# every candidate whose first name happens to match it, which then drops
# them from the AIR denominator.  See task #2 in the security review.
def _assert_name_vocab_invariants() -> None:
    male = GENDERED_NAMES["male"]
    female = GENDERED_NAMES["female"]
    overlap_mf = male & female
    overlap_mu = male & _UNISEX_NAMES
    overlap_fu = female & _UNISEX_NAMES
    if overlap_mf:
        raise AssertionError(
            f"GENDERED_NAMES: male/female collision: {sorted(overlap_mf)}"
        )
    if overlap_mu:
        raise AssertionError(
            f"GENDERED_NAMES: male/unisex collision: {sorted(overlap_mu)}"
        )
    if overlap_fu:
        raise AssertionError(
            f"GENDERED_NAMES: female/unisex collision: {sorted(overlap_fu)}"
        )


_assert_name_vocab_invariants()


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
    def detect_gender_proxy_scored(text: str) -> dict:
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
        from fairness.names.classifier import predict

        header_lines = text.strip().split("\n")[:2]
        first_plausible: str = ""
        for raw in " ".join(header_lines).split():
            cleaned = re.sub(r"[^A-Za-z]", "", raw)
            if len(cleaned) < 2:
                continue
            if not cleaned[0].isupper():
                continue  # resumes write names Title-cased or all-caps
            if cleaned.lower() in _RESUME_VOCAB_DENYLIST:
                continue
            first_plausible = cleaned
            break

        if first_plausible:
            result = predict(first_plausible)
            if result.source != "empty":
                signals["name_p_female"] = round(float(result.p_female), 4)
                signals["name_source"]   = result.source
                signals["name_token"]    = result.name

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

        gender_groups: dict[str, list] = defaultdict(list)
        for filename, text in resume_texts.items():
            if filename in scores:
                result = self.detect_gender_proxy_scored(text)
                gender = result["gender"]
                gender_groups[gender].append({
                    "filename": filename,
                    "score": scores[filename],
                    "selected": scores[filename] >= selection_threshold,
                    "confidence": result["confidence"],
                })

        results: dict = {
            "threshold": selection_threshold,
            "total_resumes": len(scores),
            "gender_distribution": {},
            "gender_bias_analysis": {},
            "score_distribution": {},
            "detection_coverage": {},
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

        # Adverse impact: male vs female
        male_data = gender_groups.get("male", [])
        female_data = gender_groups.get("female", [])

        if male_data and female_data:
            air_result = self.adverse_impact_ratio(
                group_a_selected=sum(1 for r in male_data if r["selected"]),
                group_a_total=len(male_data),
                group_b_selected=sum(1 for r in female_data if r["selected"]),
                group_b_total=len(female_data),
            )
            results["gender_bias_analysis"] = air_result

            if not air_result["passes_4_5_rule"]:
                results["recommendations"].append(
                    f"[WARN] Gender AIR = {air_result['adverse_impact_ratio']:.2f} "
                    f"(below 0.80 threshold). Potential gender bias detected."
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
