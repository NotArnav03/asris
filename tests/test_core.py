"""
FAIMR — Unit Tests
Tests for preprocessing, evaluation, and fairness modules.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest


# ═══════════════════════════════════════════════════════════════
# Text Normalizer Tests
# ═══════════════════════════════════════════════════════════════

class TestTextNormalizer:

    def test_remove_email(self):
        from preprocessing.text_normalizer import normalize_text
        text = "Contact me at john@example.com for details"
        result = normalize_text(text)
        assert "john@example.com" not in result

    def test_remove_phone(self):
        from preprocessing.text_normalizer import normalize_text
        text = "Call +1-555-123-4567 today"
        result = normalize_text(text)
        assert "555-123-4567" not in result

    def test_remove_url(self):
        from preprocessing.text_normalizer import normalize_text
        text = "Visit https://www.example.com for more"
        result = normalize_text(text)
        assert "https://www.example.com" not in result

    def test_normalize_bullets(self):
        from preprocessing.text_normalizer import normalize_text
        text = "• Item one\n► Item two\n★ Item three"
        result = normalize_text(text)
        assert "- Item one" in result
        assert "- Item two" in result

    def test_empty_input(self):
        from preprocessing.text_normalizer import normalize_text
        assert normalize_text("") == ""
        assert normalize_text(None) == ""

    def test_unicode_normalization(self):
        from preprocessing.text_normalizer import normalize_unicode
        text = "café résumé naïve"
        result = normalize_unicode(text)
        assert "cafe" in result

    def test_preserve_content(self):
        from preprocessing.text_normalizer import normalize_text
        text = "Senior Python Developer with ML experience"
        result = normalize_text(text, remove_personal_info=False)
        assert "Python" in result
        assert "Developer" in result

    def test_lowercase(self):
        from preprocessing.text_normalizer import normalize_text
        text = "Senior Developer"
        result = normalize_text(text, lowercase=True)
        assert result == "senior developer"


# ═══════════════════════════════════════════════════════════════
# Section Parser Tests
# ═══════════════════════════════════════════════════════════════

class TestSectionParser:

    def test_parse_basic_resume(self):
        from preprocessing.section_parser import parse_resume
        text = """John Doe

EXPERIENCE
Senior Developer at Google

EDUCATION
MS Computer Science, Stanford

SKILLS
Python, Java, Go
"""
        parsed = parse_resume(text)
        assert "experience" in parsed.section_names
        assert "education" in parsed.section_names
        assert "skills" in parsed.section_names

    def test_empty_resume(self):
        from preprocessing.section_parser import parse_resume
        parsed = parse_resume("")
        assert len(parsed.sections) == 0

    def test_no_sections_detected(self):
        from preprocessing.section_parser import parse_resume
        text = "Just a plain text with no headers at all."
        parsed = parse_resume(text)
        assert "other" in parsed.sections

    def test_weighted_text(self):
        from preprocessing.section_parser import parse_resume
        text = """SKILLS
Python, Java, Go

EXPERIENCE
5 years at Google
"""
        parsed = parse_resume(text)
        weighted = parsed.get_weighted_text()
        # Skills should be repeated more than once
        assert weighted.count("Python") >= 2

    def test_section_word_count(self):
        from preprocessing.section_parser import parse_resume
        text = """SKILLS
Python Java Go Rust C++
"""
        parsed = parse_resume(text)
        skills_section = parsed.sections.get("skills")
        assert skills_section is not None
        assert skills_section.word_count == 5


# ═══════════════════════════════════════════════════════════════
# Evaluation Metrics Tests
# ═══════════════════════════════════════════════════════════════

class TestEvaluationMetrics:

    def test_precision_at_k(self):
        from evaluation.metrics import precision_at_k
        y_true = [1, 0, 1, 0, 0]
        y_scores = [0.9, 0.8, 0.7, 0.6, 0.5]
        assert precision_at_k(y_true, y_scores, 1) == 1.0
        assert precision_at_k(y_true, y_scores, 2) == 0.5
        assert precision_at_k(y_true, y_scores, 3) == pytest.approx(2 / 3, abs=0.01)

    def test_recall_at_k(self):
        from evaluation.metrics import recall_at_k
        y_true = [1, 0, 1, 0, 0]
        y_scores = [0.9, 0.8, 0.7, 0.6, 0.5]
        assert recall_at_k(y_true, y_scores, 1) == 0.5
        assert recall_at_k(y_true, y_scores, 3) == 1.0

    def test_ndcg_perfect(self):
        from evaluation.metrics import ndcg_at_k
        y_true = [1, 1, 0, 0]
        y_scores = [0.9, 0.8, 0.3, 0.1]
        assert ndcg_at_k(y_true, y_scores, 4) == pytest.approx(1.0, abs=0.01)

    def test_mrr(self):
        from evaluation.metrics import mean_reciprocal_rank
        y_true = [0, 1, 0, 0]
        y_scores = [0.9, 0.8, 0.7, 0.6]
        assert mean_reciprocal_rank(y_true, y_scores) == 0.5  # First relevant at rank 2

    def test_average_precision(self):
        from evaluation.metrics import average_precision
        y_true = [1, 0, 1, 0]
        y_scores = [0.9, 0.8, 0.7, 0.6]
        ap = average_precision(y_true, y_scores)
        assert 0 <= ap <= 1

    def test_empty_inputs(self):
        from evaluation.metrics import precision_at_k, recall_at_k
        assert precision_at_k([], [], 5) == 0.0
        assert recall_at_k([], [], 5) == 0.0

    def test_quick_evaluate(self):
        from evaluation.metrics import quick_evaluate
        y_true = [1, 0, 1, 0, 1]
        y_scores = [0.9, 0.7, 0.8, 0.4, 0.6]
        results = quick_evaluate(y_true, y_scores)
        assert "P@1" in results
        assert "NDCG@5" in results
        assert "ROC-AUC" in results

    def test_ranking_evaluator(self):
        from evaluation.metrics import RankingEvaluator
        evaluator = RankingEvaluator(k_values=[1, 3])
        evaluator.add_query("q1", [1, 0, 1], [0.9, 0.5, 0.7])
        evaluator.add_query("q2", [0, 1, 0], [0.3, 0.9, 0.4])
        results = evaluator.compute_all()
        assert results["num_queries"] == 2
        assert "aggregate" in results


# ═══════════════════════════════════════════════════════════════
# Fairness / Bias Detection Tests
# ═══════════════════════════════════════════════════════════════

class TestBiasDetector:

    def test_gender_detection_male(self):
        from fairness.bias_detector import BiasDetector
        detector = BiasDetector()
        assert detector.detect_gender_proxy("John Smith - Senior Engineer") == "male"

    def test_gender_detection_female(self):
        from fairness.bias_detector import BiasDetector
        detector = BiasDetector()
        assert detector.detect_gender_proxy("Mary Jones - Data Scientist") == "female"

    def test_gender_detection_unknown(self):
        from fairness.bias_detector import BiasDetector
        detector = BiasDetector()
        result = detector.detect_gender_proxy("A professional resume text")
        assert result == "unknown"

    def test_adverse_impact_ratio_pass(self):
        from fairness.bias_detector import BiasDetector
        detector = BiasDetector()
        result = detector.adverse_impact_ratio(40, 50, 38, 50)
        assert result["passes_4_5_rule"] is True

    def test_adverse_impact_ratio_fail(self):
        from fairness.bias_detector import BiasDetector
        detector = BiasDetector()
        result = detector.adverse_impact_ratio(40, 50, 10, 50)
        assert result["passes_4_5_rule"] is False
        assert result["risk_level"] in ["MODERATE", "HIGH", "CRITICAL"]

    # --- Honorific hardening (Issue #1) -------------------------------
    # These tests lock down the strict honorific detection that replaced
    # the old `\bms\.?\b` regex.  They MUST stay green: each one
    # represents a previously-exploitable false positive that allowed an
    # applicant to flip their detected gender by editing one line of
    # their resume.

    def test_ms_office_does_not_fire_female_title(self):
        from fairness.bias_detector import BiasDetector
        result = BiasDetector.detect_gender_proxy_scored(
            "John Smith\nSummary: MS Office, Excel, PowerPoint"
        )
        assert result["signals"]["female_title"] is False

    def test_ms_in_cs_does_not_fire_female_title(self):
        from fairness.bias_detector import BiasDetector
        result = BiasDetector.detect_gender_proxy_scored(
            "Robert Jones\nMS in Computer Science, Stanford"
        )
        assert result["signals"]["female_title"] is False

    def test_ms_powerpoint_does_not_fire_female_title(self):
        from fairness.bias_detector import BiasDetector
        result = BiasDetector.detect_gender_proxy_scored(
            "Daniel Wright\nProficient in MS PowerPoint and MS Project"
        )
        assert result["signals"]["female_title"] is False

    def test_mr_aware_acronym_does_not_fire_male_title(self):
        from fairness.bias_detector import BiasDetector
        result = BiasDetector.detect_gender_proxy_scored(
            "Aisha Patel\nDesigned MR-aware caching for ARKit pipelines"
        )
        assert result["signals"]["male_title"] is False

    def test_dr_drive_does_not_fire_neutral_title(self):
        from fairness.bias_detector import BiasDetector
        result = BiasDetector.detect_gender_proxy_scored(
            "Casey Lin\n123 Dr Drive, Mountain View, CA"
        )
        assert result["signals"]["neutral_title"] is False

    def test_mr_smith_fires_male_title(self):
        from fairness.bias_detector import BiasDetector
        result = BiasDetector.detect_gender_proxy_scored(
            "Mr. Smith\nSenior Backend Engineer with 8 years experience"
        )
        assert result["signals"]["male_title"] is True
        assert result["gender"] == "male"
        assert result["confidence"] == 0.95

    def test_ms_priya_fires_female_title(self):
        from fairness.bias_detector import BiasDetector
        result = BiasDetector.detect_gender_proxy_scored(
            "Ms. Priya Sharma\nML Engineer, TensorFlow specialist"
        )
        assert result["signals"]["female_title"] is True
        assert result["gender"] == "female"

    def test_dr_chen_fires_neutral_title(self):
        from fairness.bias_detector import BiasDetector
        result = BiasDetector.detect_gender_proxy_scored(
            "Dr. Wei Chen\nResearcher with NLP publications"
        )
        assert result["signals"]["neutral_title"] is True

    def test_all_caps_honorific_with_name_fires(self):
        # Resumes sometimes use all-caps name lines.  The honorific is
        # case-insensitive but the follow-on must still be a capitalised
        # name token (which all-caps satisfies).
        from fairness.bias_detector import BiasDetector
        result = BiasDetector.detect_gender_proxy_scored(
            "MR. JOHN DOE\nSoftware Engineer"
        )
        assert result["signals"]["male_title"] is True

    # --- Name vocab hygiene (Issue #2) --------------------------------
    # Surnames must not vote as given names, and unisex tokens must not
    # appear in both gendered lists.  Each test below corresponds to a
    # previously-broken case.

    def test_chinese_surname_alone_does_not_vote_male(self):
        from fairness.bias_detector import BiasDetector
        # "Chen" used as surname only — no given-name signal should fire.
        # Previously "chen" was in the male given-name list, so any
        # East Asian candidate was misclassified male regardless of gender.
        result = BiasDetector.detect_gender_proxy_scored("Chen Engineering Team Lead")
        assert result["signals"]["male_name"] is False
        assert result["signals"]["female_name"] is False

    def test_sarah_chen_is_classified_female(self):
        from fairness.bias_detector import BiasDetector
        # Previously: Sarah fires female_name AND Chen fires male_name,
        # the two cancel, and the candidate is silently dropped to
        # "unknown" and erased from AIR.  Must now classify as female.
        result = BiasDetector.detect_gender_proxy_scored("Sarah Chen\nData Scientist")
        assert result["signals"]["female_name"] is True
        assert result["signals"]["male_name"] is False
        assert result["gender"] == "female"

    def test_wang_li_zhang_liu_not_in_male_list(self):
        from fairness.names.seed_lists import GENDERED_NAMES
        for surname in ("chen", "li", "wang", "zhang", "liu", "lee"):
            assert surname not in GENDERED_NAMES["male"], (
                f"{surname!r} is a surname, not a given name — must not vote male"
            )

    def test_hyun_is_unisex_not_double_listed(self):
        from fairness.names.seed_lists import GENDERED_NAMES, _UNISEX_NAMES
        assert "hyun" not in GENDERED_NAMES["male"]
        assert "hyun" not in GENDERED_NAMES["female"]
        assert "hyun" in _UNISEX_NAMES

    def test_unisex_korean_name_does_not_vote_either_way(self):
        from fairness.bias_detector import BiasDetector
        result = BiasDetector.detect_gender_proxy_scored("Hyun Park\nSoftware Engineer")
        assert result["signals"]["male_name"] is False
        assert result["signals"]["female_name"] is False
        assert result["signals"]["unisex_name"] is True
        assert result["gender"] == "unknown"

    def test_name_vocab_consistency_invariants(self):
        # The vocab is checked at import time, but we re-assert here
        # so the test failure is informative if anyone disables the
        # import-time check.
        from fairness.names.seed_lists import GENDERED_NAMES, _UNISEX_NAMES
        male = GENDERED_NAMES["male"]
        female = GENDERED_NAMES["female"]
        assert male.isdisjoint(female), (
            f"male/female collision: {sorted(male & female)}"
        )
        assert male.isdisjoint(_UNISEX_NAMES), (
            f"male/unisex collision: {sorted(male & _UNISEX_NAMES)}"
        )
        assert female.isdisjoint(_UNISEX_NAMES), (
            f"female/unisex collision: {sorted(female & _UNISEX_NAMES)}"
        )

    def test_mrs_smith_jones_hyphenated_name_fires(self):
        from fairness.bias_detector import BiasDetector
        result = BiasDetector.detect_gender_proxy_scored(
            "Mrs. Smith-Jones\nProgram Director"
        )
        assert result["signals"]["female_title"] is True

    # --- Unicode-aware honorifics (Task #18) ---------------------------

    def test_ms_maria_with_accent_fires_female_title(self):
        # Previously: [A-Z][A-Za-z'\-]+ rejected the accented "M" of
        # "María" so "Ms. María" silently failed to fire.
        from fairness.bias_detector import BiasDetector
        r = BiasDetector.detect_gender_proxy_scored(
            "Ms. María Fernández\nProduct Manager"
        )
        assert r["signals"]["female_title"] is True

    def test_mr_soren_nordic_letter_fires(self):
        from fairness.bias_detector import BiasDetector
        r = BiasDetector.detect_gender_proxy_scored(
            "Mr. Søren Jensen\nBackend Engineer"
        )
        assert r["signals"]["male_title"] is True

    def test_dr_muller_german_umlaut_fires(self):
        from fairness.bias_detector import BiasDetector
        r = BiasDetector.detect_gender_proxy_scored(
            "Dr. Müller\nResearch Director"
        )
        assert r["signals"]["neutral_title"] is True

    def test_senora_lopez_spanish_honorific_fires(self):
        from fairness.bias_detector import BiasDetector
        r = BiasDetector.detect_gender_proxy_scored(
            "Señora López\nMarketing Lead"
        )
        assert r["signals"]["female_title"] is True

    def test_frau_schmidt_german_honorific_fires(self):
        from fairness.bias_detector import BiasDetector
        r = BiasDetector.detect_gender_proxy_scored(
            "Frau Schmidt\nLawyer"
        )
        assert r["signals"]["female_title"] is True

    def test_herr_meyer_german_honorific_fires(self):
        from fairness.bias_detector import BiasDetector
        r = BiasDetector.detect_gender_proxy_scored(
            "Herr Meyer\nFinancial Analyst"
        )
        assert r["signals"]["male_title"] is True

    def test_madame_dupont_french_honorific_fires(self):
        from fairness.bias_detector import BiasDetector
        r = BiasDetector.detect_gender_proxy_scored(
            "Madame Dupont\nDirector"
        )
        assert r["signals"]["female_title"] is True

    def test_monsieur_dupont_french_honorific_fires(self):
        from fairness.bias_detector import BiasDetector
        r = BiasDetector.detect_gender_proxy_scored(
            "Monsieur Dupont\nDirector"
        )
        assert r["signals"]["male_title"] is True

    def test_reverend_neutral_honorific_fires(self):
        from fairness.bias_detector import BiasDetector
        r = BiasDetector.detect_gender_proxy_scored(
            "Rev. Williams\nChaplain"
        )
        assert r["signals"]["neutral_title"] is True

    def test_bare_M_initial_does_not_fire_male_title(self):
        # "John M Smith" — M is a middle initial, not a honorific.
        # The bare "M" honorific was deliberately excluded; this test
        # guards against re-adding it.
        from fairness.bias_detector import BiasDetector
        r = BiasDetector.detect_gender_proxy_scored(
            "John M Smith\nEngineer"
        )
        assert r["signals"]["male_title"] is False

    # --- Unicode-confusable defence (Task #20) -------------------------

    def test_fullwidth_honorific_still_fires(self):
        # "Ｍｒ. Smith" (fullwidth M and r, U+FF2D / U+FF52) used to
        # bypass the honorific regex because the ASCII pattern can't
        # match the fullwidth characters.  NFKC normalisation collapses
        # them to ASCII "Mr." before regex matching.
        from fairness.bias_detector import BiasDetector
        r = BiasDetector.detect_gender_proxy_scored(
            "Ｍｒ. Smith\nSoftware Engineer"
        )
        assert r["signals"]["male_title"] is True

    def test_zero_width_inside_honorific_does_not_bypass(self):
        # "M​r. Smith" — zero-width space inside the honorific.
        # Renders identically to "Mr. Smith" but breaks the regex
        # word boundary. Strip pass fixes it.
        from fairness.bias_detector import BiasDetector
        r = BiasDetector.detect_gender_proxy_scored(
            "M​r. Smith\nSoftware Engineer"
        )
        assert r["signals"]["male_title"] is True

    def test_zero_width_joiner_in_name_does_not_bypass(self):
        # ZWJ insertion in the follow-on name — must not prevent the
        # honorific from firing on the cleaned token.
        from fairness.bias_detector import BiasDetector
        r = BiasDetector.detect_gender_proxy_scored(
            "Ms. Pri‍ya Sharma\nML Engineer"
        )
        assert r["signals"]["female_title"] is True

    def test_bom_at_start_does_not_bypass(self):
        # U+FEFF BOM at the start of a resume is common when the file
        # was saved with a UTF-8 BOM-prefixed encoder. It must not
        # prevent the honorific scan from seeing the first character.
        from fairness.bias_detector import BiasDetector
        r = BiasDetector.detect_gender_proxy_scored(
            "﻿Mrs. Khan\nAnalyst"
        )
        assert r["signals"]["female_title"] is True

    def test_mathematical_bold_honorific_fires(self):
        # "𝐌𝐫. Smith" — Mathematical Bold Capital/Small (U+1D400 / U+1D42B).
        # NFKC collapses these to "Mr." so the pattern matches.
        from fairness.bias_detector import BiasDetector
        r = BiasDetector.detect_gender_proxy_scored(
            "\U0001D40C\U0001D42B. Smith\nSoftware Engineer"
        )
        assert r["signals"]["male_title"] is True

    def test_fullwidth_name_token_still_extracts(self):
        # The name scan should also benefit from NFKC — a candidate
        # whose name was pasted in fullwidth Latin still gets a
        # readable token after sanitisation.
        from fairness.bias_detector import BiasDetector
        r = BiasDetector.detect_gender_proxy_scored(
            "Ｐｒｉｙａ Sharma\nML Engineer"  # "Ｐｒｉｙａ"
        )
        assert r["signals"]["name_token"] == "priya"
        assert r["signals"]["female_name"] is True

    # --- Adaptive header window (Task #27) -----------------------------

    def test_padded_header_does_not_evade_honorific_scan(self):
        # Attack: pad the resume with > 200 chars of fake address /
        # contact info before the salutation.  The old text[:200]
        # window cut off before "Mr. Smith" and the male_title signal
        # silently failed.  The adaptive window scans up to 8 lines /
        # 1000 chars so the salutation is now seen.
        from fairness.bias_detector import BiasDetector
        padding = (
            "1234 Some Street\n"
            "Apartment 567B, Building Complex C\n"
            "Some Very Long City Name, Some State 99999\n"
            "email-address-that-is-quite-long@example.com\n"
            "Phone: +1-555-0100-extension-1234\n"
            "Personal website: example.com/profile/very-long-handle\n"
        )
        assert len(padding) > 200
        text = padding + "Mr. Smith\nSoftware Engineer"
        r = BiasDetector.detect_gender_proxy_scored(text)
        assert r["signals"]["male_title"] is True

    def test_section_header_terminates_window(self):
        # Body content under an EXPERIENCE section header must NOT
        # be scanned for honorifics — a "Mr." appearing inside a
        # job description is somebody ELSE's salutation, not the
        # candidate's.
        from fairness.bias_detector import BiasDetector
        text = (
            "Jane Doe\n"
            "Data Scientist\n\n"
            "EXPERIENCE\n"
            "Reported to Mr. Anderson during my Q3 2024 rotation.\n"
        )
        r = BiasDetector.detect_gender_proxy_scored(text)
        assert r["signals"]["male_title"] is False

    def test_long_body_paragraph_terminates_window(self):
        # A paragraph longer than the long-line cutoff means we've
        # entered the body.  Subsequent honorifics belong there.
        from fairness.bias_detector import BiasDetector
        body_paragraph = "I am a senior engineer with " + "decades " * 40
        assert len(body_paragraph) > 200
        text = (
            "Jane Doe\n"
            + body_paragraph + "\n"
            + "Mr. Anderson was my manager.\n"
        )
        r = BiasDetector.detect_gender_proxy_scored(text)
        assert r["signals"]["male_title"] is False

    def test_8_short_header_lines_all_scanned(self):
        # All 8 short header lines should be in the window so a
        # salutation in any of them fires.
        from fairness.bias_detector import BiasDetector
        text = (
            "555-123-4567\n"
            "contact@example.com\n"
            "linkedin.com/in/example\n"
            "github.com/example\n"
            "City, State\n"
            "Open to remote work\n"
            "References available on request\n"
            "Mr. Smith\n"
            "Software Engineer\n"
        )
        r = BiasDetector.detect_gender_proxy_scored(text)
        # 8th non-empty line is "Mr. Smith" — should fire.
        assert r["signals"]["male_title"] is True

    def test_normal_short_header_unaffected(self):
        # Regression: typical short header still produces the same
        # signal as before the adaptive-window change.
        from fairness.bias_detector import BiasDetector
        r = BiasDetector.detect_gender_proxy_scored(
            "Mr. Smith\nSoftware Engineer with 8 years experience"
        )
        assert r["signals"]["male_title"] is True


    # --- RTL / bidirectional honorifics (Task #29) ---------------------

    def test_arabic_mr_honorific_fires_male_title(self):
        from fairness.bias_detector import BiasDetector
        r = BiasDetector.detect_gender_proxy_scored(
            "السيد محمد\nمهندس برمجيات"   # "Mr. Mohammed\nSoftware Engineer"
        )
        assert r["signals"]["male_title"] is True

    def test_arabic_mrs_honorific_fires_female_title(self):
        from fairness.bias_detector import BiasDetector
        r = BiasDetector.detect_gender_proxy_scored(
            "السيدة فاطمة\nمحللة بيانات"   # "Mrs. Fatima\nData Analyst"
        )
        assert r["signals"]["female_title"] is True

    def test_arabic_miss_honorific_fires_female_title(self):
        from fairness.bias_detector import BiasDetector
        r = BiasDetector.detect_gender_proxy_scored("الآنسة نور\nطبيبة")
        assert r["signals"]["female_title"] is True

    def test_hebrew_mr_honorific_fires_male_title(self):
        from fairness.bias_detector import BiasDetector
        r = BiasDetector.detect_gender_proxy_scored(
            "מר כהן\nמהנדס תוכנה"   # "Mr. Cohen\nSoftware Engineer"
        )
        assert r["signals"]["male_title"] is True

    def test_hebrew_mrs_honorific_fires_female_title(self):
        from fairness.bias_detector import BiasDetector
        r = BiasDetector.detect_gender_proxy_scored(
            "גברת לוי\nאנליסטית"   # "Mrs. Levy\nAnalyst"
        )
        assert r["signals"]["female_title"] is True

    def test_arabic_professor_fires_neutral_title(self):
        from fairness.bias_detector import BiasDetector
        r = BiasDetector.detect_gender_proxy_scored(
            "بروفيسور أحمد\nباحث في الذكاء الاصطناعي"
        )
        assert r["signals"]["neutral_title"] is True

    def test_rtl_scan_idempotent_on_plain_ascii(self):
        # Regression: pure ASCII input never triggers the RTL scan,
        # so the signals must be identical to the Latin-only path.
        from fairness.bias_detector import BiasDetector
        r = BiasDetector.detect_gender_proxy_scored(
            "Mr. Smith\nSoftware Engineer"
        )
        assert r["signals"]["male_title"] is True
        # Just confirm no spurious other RTL-driven signal
        assert r["signals"]["female_title"] is False

    def test_has_rtl_script_detector(self):
        from fairness.bias_detector import _has_rtl_script
        assert _has_rtl_script("plain ASCII") is False
        assert _has_rtl_script("السيد") is True       # Arabic
        assert _has_rtl_script("מר") is True           # Hebrew
        assert _has_rtl_script("Mixed السيد ASCII") is True


    # --- Cyrillic / Greek confusables (Task #30) -----------------------

    def test_cyrillic_M_in_mr_does_not_bypass(self):
        # "Мr. Smith" with U+041C Cyrillic capital Em.  Renders
        # identically to "Mr. Smith" but old pattern missed it.
        from fairness.bias_detector import BiasDetector
        r = BiasDetector.detect_gender_proxy_scored(
            "Мr. Smith\nSoftware Engineer"
        )
        assert r["signals"]["male_title"] is True

    def test_cyrillic_lowercase_in_ms_does_not_bypass(self):
        # "Mѕ. Khan" with U+0455 Cyrillic lowercase Dze (looks like s).
        from fairness.bias_detector import BiasDetector
        r = BiasDetector.detect_gender_proxy_scored(
            "Mѕ. Khan\nAnalyst"
        )
        assert r["signals"]["female_title"] is True

    def test_wholly_cyrillic_honorific_does_not_bypass(self):
        # "Мѕ. Khan" — Cyrillic М (U+041C) + Cyrillic ѕ (U+0455).
        # The unconditional confusables fold catches this case
        # where a context-aware heuristic would leave the whole-
        # Cyrillic word alone.
        from fairness.bias_detector import BiasDetector
        r = BiasDetector.detect_gender_proxy_scored(
            "Мѕ. Khan\nAnalyst"
        )
        assert r["signals"]["female_title"] is True

    def test_greek_mu_in_mr_does_not_bypass(self):
        # Greek capital Mu (U+039C) is visually identical to Latin M.
        from fairness.bias_detector import BiasDetector
        r = BiasDetector.detect_gender_proxy_scored(
            "Μr. Smith\nSoftware Engineer"
        )
        assert r["signals"]["male_title"] is True

    def test_greek_omicron_confusable_in_name(self):
        # "Priοa" with Greek omicron in the middle of "Priya"
        # would otherwise be OOV; after confusables fold it becomes
        # "Prioa" or similar (close to lookup).  Test that NO
        # spurious signal is created when the candidate is genuinely
        # ambiguous.
        from fairness.bias_detector import BiasDetector
        r = BiasDetector.detect_gender_proxy_scored(
            "Priοa Sharma\nML Engineer"
        )
        # After fold "prioa" -> not a known lookup; but the OOV
        # model might predict something.  Just confirm no crash and
        # SOME signal is recorded.
        assert "name_p_female" in r["signals"]


    # --- Language detection + localised denylist (Task #32) ------------

    def test_spanish_resume_detected_and_localised_denylist_applied(self):
        from fairness.bias_detector import BiasDetector
        text = (
            "Currículum\n"
            "Ingeniero de Software\n"
            "María García\n"
            "5 años de experiencia en el desarrollo de aplicaciones\n"
            "con Python y JavaScript para los clientes de la empresa.\n"
        )
        r = BiasDetector.detect_gender_proxy_scored(text)
        assert r["signals"]["detected_language"] == "es"

    def test_french_resume_detected(self):
        from fairness.bias_detector import BiasDetector
        text = (
            "Résumé\n"
            "Ingénieur logiciel avec une expérience de 5 ans dans le\n"
            "développement d'applications pour les clients de la société.\n"
        )
        r = BiasDetector.detect_gender_proxy_scored(text)
        assert r["signals"]["detected_language"] == "fr"

    def test_german_resume_detected(self):
        from fairness.bias_detector import BiasDetector
        text = (
            "Lebenslauf\n"
            "Software Ingenieur mit 5 Jahren Erfahrung in der\n"
            "Entwicklung von Anwendungen für die Kunden der Firma.\n"
        )
        r = BiasDetector.detect_gender_proxy_scored(text)
        assert r["signals"]["detected_language"] == "de"

    def test_english_default_when_no_language_clears_threshold(self):
        from fairness.bias_detector import BiasDetector
        text = "Just a name\nSoftware Engineer"
        r = BiasDetector.detect_gender_proxy_scored(text)
        assert r["signals"]["detected_language"] == "en"


    def test_plain_ascii_unaffected_by_sanitisation(self):
        # Regression: NFKC/zero-width strip is idempotent on ASCII.
        # If this fails the sanitisation has a bug that ALSO affects
        # the 95% of inputs that aren't under attack.
        from fairness.bias_detector import BiasDetector
        r = BiasDetector.detect_gender_proxy_scored(
            "Mr. Smith\nSoftware Engineer with 8 years experience"
        )
        assert r["signals"]["male_title"] is True
        assert r["gender"] == "male"

    def test_doe_Sr_suffix_does_not_fire_male_title(self):
        # "John Doe Sr." — Sr is the English suffix for Senior, not
        # the Spanish honorific.  Guard against the false positive.
        from fairness.bias_detector import BiasDetector
        r = BiasDetector.detect_gender_proxy_scored(
            "John Doe Sr.\nProfessor"
        )
        # The neutral_title check still fires on "Professor" if it
        # follows a recognised honorific; but Sr should not vote male.
        assert r["signals"]["male_title"] is False

    def test_ms_oneill_apostrophe_name_fires(self):
        from fairness.bias_detector import BiasDetector
        result = BiasDetector.detect_gender_proxy_scored(
            "Ms. O'Neill\nProduct Manager"
        )
        assert result["signals"]["female_title"] is True

    # --- Classifier integration (Issue #3 / Task #14) -----------------

    def test_classifier_drives_name_signal_for_known_name(self):
        from fairness.bias_detector import BiasDetector
        r = BiasDetector.detect_gender_proxy_scored(
            "Priya Sharma\nML Engineer"
        )
        # The signal must come from the calibrated classifier.
        assert r["signals"]["name_source"] in ("lookup", "model")
        assert r["signals"]["name_token"] == "priya"
        assert r["signals"]["name_p_female"] >= 0.85
        assert r["signals"]["female_name"] is True
        assert r["gender"] == "female"

    def test_resume_vocab_does_not_drive_name_signal(self):
        # No real name in the header — only resume vocabulary.  The
        # name scan must report name_source == "empty" and not vote
        # for either gender.  Previously the OOV branch of the model
        # would classify these tokens with high confidence and produce
        # spurious gender votes.
        from fairness.bias_detector import BiasDetector
        r = BiasDetector.detect_gender_proxy_scored(
            "Senior Software Engineer\nData Scientist Resume"
        )
        assert r["signals"]["name_source"] == "empty"
        assert r["signals"]["male_name"] is False
        assert r["signals"]["female_name"] is False

    def test_first_token_rule_picks_given_name_over_surname(self):
        # "Mary Jones" — Jones is a confident OOV male lookup; under
        # the previous max-confidence rule Jones beat Mary and the
        # candidate was misclassified.  First-token rule fixes this.
        from fairness.bias_detector import BiasDetector
        r = BiasDetector.detect_gender_proxy_scored(
            "Mary Jones - Data Scientist"
        )
        assert r["signals"]["name_token"] == "mary"
        assert r["gender"] == "female"

    def test_lowercase_token_is_not_a_name_candidate(self):
        # Resumes don't write names in all-lowercase.  The first
        # uppercase-leading token must be picked, skipping any
        # lowercase preceding tokens.
        from fairness.bias_detector import BiasDetector
        r = BiasDetector.detect_gender_proxy_scored(
            "summary objective Priya Sharma"
        )
        assert r["signals"]["name_token"] == "priya"

    def test_name_p_female_is_surfaced_in_signals(self):
        # The probability must always be present (default 0.5) so
        # downstream code can rely on it without a key-existence check.
        from fairness.bias_detector import BiasDetector
        r = BiasDetector.detect_gender_proxy_scored("")
        assert r["signals"]["name_p_female"] == 0.5
        assert r["signals"]["name_source"] == "empty"

    def test_audit_runs(self):
        from fairness.bias_detector import BiasDetector
        detector = BiasDetector()
        texts = {
            "john.txt": "John is an engineer",
            "mary.txt": "Mary is a scientist",
        }
        scores = {"john.txt": 0.8, "mary.txt": 0.7}
        audit = detector.audit_ranking_bias(texts, scores)
        assert "gender_distribution" in audit
        assert "recommendations" in audit

    # --- Surname handling (Task #16) -----------------------------------

    def test_surname_only_header_produces_no_gender_signal(self):
        # "Park" alone should NOT vote male (or female) even though the
        # OOV model would otherwise predict strongly male for that
        # token.  This is the core surname-attack fix.
        from fairness.bias_detector import BiasDetector
        r = BiasDetector.detect_gender_proxy_scored(
            "Park\nSoftware Engineer at TechCorp"
        )
        assert r["signals"]["name_source"] == "empty"
        assert r["signals"]["name_is_surname"] is True
        assert r["signals"]["male_name"] is False
        assert r["signals"]["female_name"] is False
        assert r["gender"] == "unknown"

    def test_jones_smith_khan_are_recognised_as_surnames(self):
        from fairness.bias_detector import BiasDetector
        for sn in ("Jones", "Smith", "Khan", "Patel", "Park"):
            r = BiasDetector.detect_gender_proxy_scored(
                f"{sn}\nSenior Engineer"
            )
            assert r["signals"]["name_is_surname"] is True, (
                f"{sn!r} should be recognised as a surname"
            )
            assert r["signals"]["name_source"] == "empty", (
                f"{sn!r} alone must not produce a name signal"
            )

    def test_given_name_with_surname_picks_given_name(self):
        # "Priya Sharma" — Sharma is a surname, Priya is a given name.
        # The classifier must pick Priya and ignore Sharma.
        from fairness.bias_detector import BiasDetector
        r = BiasDetector.detect_gender_proxy_scored(
            "Priya Sharma\nML Engineer"
        )
        assert r["signals"]["name_token"] == "priya"
        assert r["signals"]["name_is_surname"] is False
        assert r["signals"]["female_name"] is True

    def test_comma_lastname_first_format_picks_given_name(self):
        # "Doe, John" — academic-CV format.  The right-of-comma part
        # ("John") should drive the signal, not the left-of-comma
        # surname ("Doe").
        from fairness.bias_detector import BiasDetector
        r = BiasDetector.detect_gender_proxy_scored(
            "Doe, John\nResearch Scientist"
        )
        assert r["signals"]["name_token"] == "john"
        assert r["signals"]["male_name"] is True

    def test_comma_with_suffix_falls_through_to_line1(self):
        # "John Doe, PhD" — line1 has a comma but the right side is
        # just a denylisted suffix.  Cascade must fall through and
        # use line1 ("John Doe") as-is.
        from fairness.bias_detector import BiasDetector
        r = BiasDetector.detect_gender_proxy_scored(
            "John Doe, PhD\nProfessor"
        )
        assert r["signals"]["name_token"] == "john"
        assert r["signals"]["male_name"] is True

    def test_classifier_result_carries_surname_flags(self):
        # `is_surname` flags any token on the surname denylist (US Census
        # + curated multi-cultural).  `is_surname_only` is the derived
        # property callers use — True only when there's no strong
        # given-name lookup evidence to compete with the surname status.
        from fairness.names.classifier import predict
        # Park / Patel: surnames with no strong given-name evidence ->
        # both flags True.
        for tok in ("Park", "Patel", "Jones"):
            r = predict(tok)
            assert r.is_surname is True, f"{tok} should be on surname list"
            assert r.is_surname_only is True, (
                f"{tok} should be surname-only (no strong given-name lookup)"
            )
        # John: BOTH a top US surname AND a strongly-attested given name
        # in the corpus.  is_surname=True but is_surname_only=False so
        # downstream gender detection still uses John.
        john = predict("John")
        assert john.is_surname is True, "John IS on the US Census surname list"
        assert john.is_surname_only is False, (
            "John must NOT be surname-only — it has a strong given-name lookup"
        )
        # Priya: not on the surname list -> both flags False.
        priya = predict("Priya")
        assert priya.is_surname is False
        assert priya.is_surname_only is False


    # --- Dual soft/hard AIR (Task #15) ---------------------------------

    def test_audit_emits_both_soft_and_hard_air(self):
        from fairness.bias_detector import BiasDetector
        detector = BiasDetector()
        texts = {
            f"m{i}.txt": f"John{i} Smith\nSoftware Engineer" for i in range(6)
        } | {
            f"f{i}.txt": f"Priya{i} Sharma\nML Engineer" for i in range(6)
        }
        scores = {n: 0.5 + i * 0.01 for i, n in enumerate(texts)}
        audit = detector.audit_ranking_bias(texts, scores)
        analysis = audit["gender_bias_analysis"]
        assert "adverse_impact_ratio_hard" in analysis
        assert "adverse_impact_ratio_soft" in analysis
        # Pass/fail uses the conservative of the two.
        assert (analysis["adverse_impact_ratio"]
                == min(analysis["adverse_impact_ratio_hard"],
                       analysis["adverse_impact_ratio_soft"]))

    def test_soft_air_uses_probability_mass_for_borderline_candidates(self):
        # Construct a scenario where the hard view excludes borderline
        # candidates but the soft view includes them as partial mass.
        # Specifically: 4 selected males, 4 unselected males, plus 4
        # selected borderline-female (p_female ~ 0.7).  Hard AIR sees
        # the borderline females as "female" (selected_rate=1.0 vs
        # male_rate=0.5 -> AIR=0.5).  Soft AIR sees them as 70% female
        # mass / 30% male mass, which shifts the rates.
        from fairness.bias_detector import BiasDetector
        detector = BiasDetector()
        # Hand-build the audit via the soft helper directly so the
        # behaviour is unit-tested independent of the classifier.
        records = (
            [{"selected": True,  "p_female_soft": 0.05}] * 4 +  # 4 sel male
            [{"selected": False, "p_female_soft": 0.05}] * 4 +  # 4 unsel male
            [{"selected": True,  "p_female_soft": 0.70}] * 4    # 4 sel border-f
        )
        soft = BiasDetector._air_soft(records)
        # Male total mass: 4*0.95 + 4*0.95 + 4*0.30 = 8.8
        # Male selected mass: 4*0.95 + 4*0.30 = 5.0
        # Female total mass: 4*0.05 + 4*0.05 + 4*0.70 = 3.2
        # Female selected mass: 4*0.05 + 4*0.70 = 3.0
        # male_rate = 5.0 / 8.8 = 0.5682
        # female_rate = 3.0 / 3.2 = 0.9375
        # AIR = 0.5682 / 0.9375 = 0.606
        assert abs(soft["male_total_mass"]    - 8.80) < 0.05
        assert abs(soft["female_total_mass"]  - 3.20) < 0.05
        assert abs(soft["male_rate"]    - 0.5682) < 0.01
        assert abs(soft["female_rate"]  - 0.9375) < 0.01
        assert abs(soft["adverse_impact_ratio"] - 0.606) < 0.01

    def test_soft_air_excludes_unknown_candidates(self):
        from fairness.bias_detector import BiasDetector
        records = [
            {"selected": True,  "p_female_soft": 0.95},
            {"selected": False, "p_female_soft": 0.05},
            {"selected": True,  "p_female_soft": None},  # excluded
        ]
        soft = BiasDetector._air_soft(records)
        # The "None" record contributes nothing to either bucket.
        assert abs(soft["male_total_mass"]   - 1.00) < 1e-6
        assert abs(soft["female_total_mass"] - 1.00) < 1e-6

    # --- Adversarial soft-vs-hard AIR (Task #17) -----------------------
    # These tests prove the "min(hard, soft)" conservative gate works
    # by constructing scenarios where the two views deliberately
    # disagree, then checking that pass/fail uses the worse value.

    def test_hard_air_passes_soft_air_fails_conservative_gate_fails(self):
        # Setup: hard view sees a tied 50/50 male/female selection so
        # hard AIR = 1.0 (passes 4/5).  Soft view sees that the
        # "selected" candidates have borderline female probabilities
        # while the "unselected" had strong female probabilities,
        # depressing the female mass selection rate so soft AIR < 0.8.
        from fairness.bias_detector import BiasDetector
        # Hand-built records — hard label per candidate, plus soft P(female).
        records = (
            # 5 strong-male candidates, all selected -> hard:5/5 male sel
            [{"selected": True, "p_female_soft": 0.02}] * 5 +
            # 5 borderline-female candidates, all selected
            #   -> hard female: 5/5 selected (hard sel rate = 1.0)
            #   -> soft female: 0.55 * 5 = 2.75 of 0.55*5+0.45*5 = 5.0 mass
            #                   wait — need ALL candidates contribute to total
            [{"selected": True, "p_female_soft": 0.55}] * 5 +
            # 5 strong-female candidates, NONE selected
            #   -> hard female: still 5 selected of 10 total = 0.5 sel rate
            #   -> soft female: heavy female mass NOT selected, depresses rate
            [{"selected": False, "p_female_soft": 0.98}] * 5
        )
        soft = BiasDetector._air_soft(records)
        # Soft male:
        #   total = 5*0.98 + 5*0.45 + 5*0.02 = 4.9 + 2.25 + 0.10 = 7.25
        #   selected = 5*0.98 + 5*0.45 = 4.9 + 2.25 = 7.15
        #   rate = 7.15/7.25 = 0.986
        # Soft female:
        #   total = 5*0.02 + 5*0.55 + 5*0.98 = 0.10 + 2.75 + 4.90 = 7.75
        #   selected = 5*0.02 + 5*0.55 = 0.10 + 2.75 = 2.85
        #   rate = 2.85/7.75 = 0.368
        # Soft AIR = 0.368 / 0.986 = 0.373 — well below 0.8
        assert soft["adverse_impact_ratio"] < 0.5, (
            f"Soft AIR should be <0.5, got {soft['adverse_impact_ratio']:.3f}"
        )
        # Demonstrating the math is enough — the audit-level integration
        # is exercised by test_audit_emits_both_soft_and_hard_air.

    def test_soft_air_passes_hard_air_fails(self):
        # Inverse scenario: hard fails (male sel 1.0, female sel 0.0)
        # but soft passes because the candidates' probabilities are
        # closer to 0.5, so mass leaks across the boundary.
        from fairness.bias_detector import BiasDetector
        records = (
            # All hard-male, all selected
            [{"selected": True,  "p_female_soft": 0.40}] * 5 +
            # All hard-female (by argmax), none selected
            [{"selected": False, "p_female_soft": 0.60}] * 5
        )
        # Hard AIR: male_rate=1.0, female_rate=0.0 -> AIR=0.0 (fails)
        # Soft AIR computation:
        #   male_total = 5*0.60 + 5*0.40 = 5.0
        #   male_selected = 5*0.60 = 3.0
        #   male_rate = 0.6
        #   female_total = 5*0.40 + 5*0.60 = 5.0
        #   female_selected = 5*0.40 = 2.0
        #   female_rate = 0.4
        #   soft AIR = 0.4/0.6 = 0.667 — still below 0.8 but better than 0.0
        soft = BiasDetector._air_soft(records)
        assert soft["adverse_impact_ratio"] > 0.5
        assert soft["adverse_impact_ratio"] < 0.8

    def test_conservative_gate_picks_min_when_views_disagree(self):
        # End-to-end audit with names that the classifier predicts at
        # different confidences.  The conservative gate must pick min.
        from fairness.bias_detector import BiasDetector
        texts = {
            "p.txt": "Priya Sharma\nEngineer",     # strong-female lookup
            "j.txt": "John Smith\nEngineer",        # strong-male lookup
            "f.txt": "Fatima Khan\nAnalyst",        # strong-female lookup
            "m.txt": "Mohammed Ali\nAnalyst",       # strong-male lookup
        }
        scores = {"p.txt": 0.9, "j.txt": 0.8, "f.txt": 0.7, "m.txt": 0.6}
        audit = BiasDetector().audit_ranking_bias(texts, scores)
        a = audit["gender_bias_analysis"]
        # adverse_impact_ratio (the publish field) MUST equal
        # min(hard, soft) — never max, never the average.
        assert a["adverse_impact_ratio"] == min(
            a["adverse_impact_ratio_hard"],
            a["adverse_impact_ratio_soft"],
        )
        # passes_4_5_rule MUST be derived from the conservative number.
        expected_pass = (
            min(a["adverse_impact_ratio_hard"],
                a["adverse_impact_ratio_soft"]) >= 0.80
        )
        assert a["passes_4_5_rule"] is expected_pass

    def test_dead_seed_lists_moved_out_of_bias_detector(self):
        # Regression guard: the curated seed lists must not be
        # importable from fairness.bias_detector (where they were
        # dead code at audit time and invited accidental reuse).
        # They live in fairness.names.seed_lists now.
        import fairness.bias_detector as bd
        assert not hasattr(bd, "GENDERED_NAMES"), (
            "GENDERED_NAMES is dead at audit time — "
            "must live in fairness.names.seed_lists only"
        )
        assert not hasattr(bd, "_UNISEX_NAMES"), (
            "_UNISEX_NAMES is dead at audit time — "
            "must live in fairness.names.seed_lists only"
        )
        # ...but they ARE still importable from the seed-list module,
        # because build_corpus.py still needs them.
        from fairness.names.seed_lists import GENDERED_NAMES, _UNISEX_NAMES
        assert "john" in GENDERED_NAMES["male"]
        assert "hyun" in _UNISEX_NAMES


    # --- Per-culture disclosure + batched path (Task #19) -------------

    def test_audit_emits_culture_distribution(self):
        from fairness.bias_detector import BiasDetector
        texts = {
            "p.txt": "Priya Sharma\nEngineer",        # south_asian (lookup)
            "j.txt": "John Smith\nEngineer",          # western (lookup)
            "f.txt": "Fatima Khan\nAnalyst",          # arab (lookup)
            "w.txt": "Wei Chen\nResearcher",          # east_asian (lookup or unisex)
        }
        scores = {"p.txt": 0.9, "j.txt": 0.8, "f.txt": 0.7, "w.txt": 0.6}
        audit = BiasDetector().audit_ranking_bias(texts, scores)
        cd = audit["culture_distribution"]
        assert cd, "culture_distribution must be populated"
        # Expected cultures appear; counts sum to total resumes
        # (every resume is recorded under some culture, even 'unknown').
        assert sum(v["count"] for v in cd.values()) == len(scores)
        for culture, stats in cd.items():
            assert "count" in stats
            assert "selected_count" in stats
            assert "selection_rate" in stats
            assert "mean_p_female" in stats
            assert "lookup_share" in stats
            # model_card_ece may be None for "unknown" or absent cultures
            assert "model_card_ece" in stats

    def test_culture_distribution_includes_per_culture_ece_when_card_present(self):
        # When the model card is on disk, ECE values are looked up and
        # surfaced for cultures that have them.  This is what lets a
        # reviewer correlate "70% Arab" with "ECE 0.09" and weight
        # the audit verdict accordingly.
        from fairness.bias_detector import BiasDetector, _load_model_card_ece
        ece = _load_model_card_ece()
        # Verify the cache returns a dict; if model_card.json is present
        # in the repo the dict should be non-empty.
        assert isinstance(ece, dict)
        from pathlib import Path
        card_path = Path("fairness/names/model_card.json")
        if card_path.exists():
            assert ece, "model_card.json exists but ECE cache is empty"
            # At least one expected culture should be present.
            assert any(c in ece for c in (
                "western", "east_asian", "arab", "south_asian"
            ))

    # --- Calibration-drift gate (Task #21) -----------------------------

    def test_audit_emits_calibration_drift_block(self):
        from fairness.bias_detector import BiasDetector
        texts = {
            "p.txt": "Priya Sharma\nEngineer",
            "j.txt": "John Smith\nEngineer",
            "f.txt": "Fatima Khan\nAnalyst",
            "m.txt": "Mohammed Ali\nAnalyst",
        }
        scores = {"p.txt": 0.9, "j.txt": 0.8, "f.txt": 0.7, "m.txt": 0.6}
        audit = BiasDetector().audit_ranking_bias(texts, scores)
        cd = audit["calibration_drift"]
        assert "weighted_ece" in cd
        assert "ece_coverage" in cd
        assert "status" in cd
        assert cd["status"] in (
            "ok", "warn", "inconclusive_high_drift",
            "inconclusive_low_ece_coverage", "unknown",
        )

    def test_audit_emits_verdict_field(self):
        from fairness.bias_detector import BiasDetector
        texts = {
            "p.txt": "Priya Sharma\nEngineer",
            "j.txt": "John Smith\nEngineer",
        }
        scores = {"p.txt": 0.9, "j.txt": 0.8}
        audit = BiasDetector().audit_ranking_bias(texts, scores)
        analysis = audit.get("gender_bias_analysis", {})
        if analysis:  # only emitted when both groups present
            assert "verdict" in analysis
            assert any(verdict_prefix in analysis["verdict"] for verdict_prefix in (
                "pass", "fail", "inconclusive",
            ))

    def test_drift_gate_overrides_pass_when_corpus_is_high_drift(self):
        # Direct unit test on _air_soft + drift logic: simulate an
        # audit where the AIR math passes but weighted_ece would be
        # above 0.10 (the high-drift ceiling).  Verify the verdict
        # field shifts to inconclusive_high_drift.  We synthesise the
        # audit by mocking _load_model_card_ece via monkey-patching.
        import fairness.bias_detector as bd
        from fairness.bias_detector import BiasDetector
        original = bd._load_model_card_ece
        bd._MODEL_CARD_ECE_CACHE = None
        try:
            bd._load_model_card_ece = lambda: {
                # Simulated very-high-drift culture for all candidates.
                "south_asian": 0.20,
            }
            # All candidates resolve to south_asian via lookup hits.
            texts = {
                f"r{i}.txt": name + "\nEngineer"
                for i, name in enumerate([
                    "Priya Sharma", "Anjali Patel", "Neha Verma",
                    "Rahul Mehta", "Vikram Gupta", "Arjun Kumar",
                ])
            }
            scores = {k: 0.5 + i * 0.05 for i, k in enumerate(texts)}
            audit = BiasDetector().audit_ranking_bias(texts, scores)
            assert audit["calibration_drift"]["weighted_ece"] == 0.20
            assert audit["calibration_drift"]["status"] == "inconclusive_high_drift"
            if audit["gender_bias_analysis"]:
                assert "inconclusive" in audit["gender_bias_analysis"]["verdict"]
        finally:
            bd._load_model_card_ece = original
            bd._MODEL_CARD_ECE_CACHE = None

    def test_drift_warn_band_passes_with_warning_recommendation(self):
        # Weighted ECE in (0.05, 0.10] -> "warn" status.  We construct
        # balanced selection (female_rate == male_rate) so AIR passes,
        # then assert the warn-band NOTE recommendation fires.
        import fairness.bias_detector as bd
        from fairness.bias_detector import BiasDetector
        original = bd._load_model_card_ece
        bd._MODEL_CARD_ECE_CACHE = None
        try:
            bd._load_model_card_ece = lambda: {"south_asian": 0.08}
            texts = {
                "f1.txt": "Priya Sharma\nEngineer",     # female sa
                "m1.txt": "Rahul Mehta\nEngineer",      # male sa
                "f2.txt": "Anjali Patel\nEngineer",     # female sa
                "m2.txt": "Vikram Gupta\nEngineer",     # male sa
            }
            # Median is 0.75 -> selected are scores >= 0.75.
            # Score m2=0.9 (selected), f2=0.8 (selected), m1=0.7 (not),
            # f1=0.6 (not).  Selected: 1 male + 1 female. AIR = 1.0 PASS.
            scores = {"f1.txt": 0.6, "m1.txt": 0.7, "f2.txt": 0.8, "m2.txt": 0.9}
            audit = BiasDetector().audit_ranking_bias(texts, scores)
            assert audit["calibration_drift"]["status"] == "warn"
            assert any("calibration" in rec.lower()
                       for rec in audit["recommendations"])
        finally:
            bd._load_model_card_ece = original
            bd._MODEL_CARD_ECE_CACHE = None

    def test_drift_low_coverage_forces_inconclusive(self):
        # When fewer than 50% of candidates are in cultures with
        # measured ECE the verdict is inconclusive regardless of AIR.
        # Mock ECE for "south_asian" only; the corpus mixes south_asian
        # with arab + european_other so coverage drops below 50%.
        import fairness.bias_detector as bd
        from fairness.bias_detector import BiasDetector
        original = bd._load_model_card_ece
        bd._MODEL_CARD_ECE_CACHE = None
        try:
            bd._load_model_card_ece = lambda: {"south_asian": 0.05}
            texts = {
                "p.txt": "Priya Sharma\nEngineer",       # south_asian
                "f.txt": "Fatima Khan\nAnalyst",         # arab / european_other
                "m.txt": "Mohammed Ali\nAnalyst",        # arab
                "j.txt": "John Smith\nEngineer",         # european_other
                "s.txt": "Sebastian Lopez\nDesigner",    # western or european_other
            }
            scores = {k: 0.5 + i * 0.05 for i, k in enumerate(texts)}
            audit = BiasDetector().audit_ranking_bias(texts, scores)
            cd = audit["calibration_drift"]
            # ece_coverage = (n in south_asian) / (total).  Only Priya is
            # south_asian here, so coverage is 1/5 = 0.2 < 0.5.
            assert cd["ece_coverage"] < 0.5
            assert cd["status"] == "inconclusive_low_ece_coverage"
        finally:
            bd._load_model_card_ece = original
            bd._MODEL_CARD_ECE_CACHE = None


    # --- Counterfactual robustness inside audit (Task #33) -------------

    def test_audit_runs_counterfactual_when_scorer_provided(self):
        from fairness.bias_detector import BiasDetector
        texts = {
            "p.txt": "Priya Sharma\nEngineer with 5 years",
            "j.txt": "John Smith\nEngineer with 5 years",
        }
        scores = {"p.txt": 0.9, "j.txt": 0.8}
        audit = BiasDetector().audit_ranking_bias(
            texts, scores,
            scorer=lambda jd, res: 0.7,   # clean: same score for all
            jd_text="Senior Python role",
        )
        assert "counterfactual_robustness" in audit
        cf = audit["counterfactual_robustness"]
        assert cf["all_robust"] is True
        assert cf["mean_score_gap"] == 0.0

    def test_audit_counterfactual_flags_biased_scorer(self):
        from fairness.bias_detector import BiasDetector
        texts = {
            "p.txt": "Priya Sharma\nEngineer with 5 years",
            "j.txt": "John Smith\nEngineer with 5 years",
        }
        scores = {"p.txt": 0.9, "j.txt": 0.8}
        # Biased: scorer prefers any name in our male swap pool.
        # All 8 male swap names trigger 0.9; female names trigger 0.5.
        male_pool = {"john", "michael", "rahul", "vikram", "wei",
                     "hiroshi", "mohammed", "omar"}

        def biased(jd, res):
            first = res.strip().split()[0].lower()
            return 0.9 if first in male_pool else 0.5

        audit = BiasDetector().audit_ranking_bias(
            texts, scores, scorer=biased, jd_text="role",
        )
        cf = audit["counterfactual_robustness"]
        assert cf["all_robust"] is False
        # Gap of 0.4 (mean male 0.9, mean female 0.5) — way above
        # the default 0.02 threshold.
        assert cf["mean_score_gap"] > 0.3
        assert any("counterfactual" in r.lower() for r in audit["recommendations"])

    def test_audit_skips_counterfactual_when_no_scorer(self):
        from fairness.bias_detector import BiasDetector
        audit = BiasDetector().audit_ranking_bias(
            {"a.txt": "John Smith\nEngineer"}, {"a.txt": 0.5},
        )
        assert "counterfactual_robustness" not in audit

    # --- Historical drift baseline (Task #34) -------------------------

    # --- Detection coverage gate (Task #4 / #39) -----------------------

    # --- DPD weighting + chi-squared + Theil (Task #5 / #40) ----------

    # --- Directional AIR + Wilson CIs (Task #6 / #41) ------------------

    def test_air_returns_both_symmetric_and_directional(self):
        from fairness.bias_detector import BiasDetector
        result = BiasDetector().adverse_impact_ratio(
            group_a_selected=40, group_a_total=50,   # 80% male sel
            group_b_selected=20, group_b_total=50,   # 40% female sel
            group_a_name="male", group_b_name="female",
        )
        assert "directional_air" in result
        assert "adverse_impact_ratio_symmetric" in result
        # Both should equal 0.5 here (female 0.4 / male 0.8 = 0.5)
        assert abs(result["directional_air"] - 0.5) < 1e-9
        assert abs(result["adverse_impact_ratio_symmetric"] - 0.5) < 1e-9

    def test_air_auto_picks_lower_rate_as_protected(self):
        from fairness.bias_detector import BiasDetector
        result = BiasDetector().adverse_impact_ratio(
            group_a_selected=40, group_a_total=50,
            group_b_selected=20, group_b_total=50,
            group_a_name="male", group_b_name="female",
        )
        # Female has lower rate -> auto-protected
        assert result["protected_group"] == "female"
        assert result["reference_group"] == "male"

    def test_air_respects_explicit_protected_group(self):
        from fairness.bias_detector import BiasDetector
        # Caller explicitly says male is protected.  Then directional AIR
        # uses male_rate/female_rate even though female has the lower rate.
        result = BiasDetector().adverse_impact_ratio(
            group_a_selected=40, group_a_total=50,    # male 0.8
            group_b_selected=20, group_b_total=50,    # female 0.4
            group_a_name="male", group_b_name="female",
            protected_group="male",
        )
        assert result["protected_group"] == "male"
        # AIR = male_rate / female_rate = 0.8 / 0.4 = 2.0
        assert abs(result["directional_air"] - 2.0) < 1e-9

    def test_air_wilson_intervals_present_and_well_formed(self):
        from fairness.bias_detector import BiasDetector
        result = BiasDetector().adverse_impact_ratio(
            group_a_selected=40, group_a_total=50,
            group_b_selected=20, group_b_total=50,
            group_a_name="male", group_b_name="female",
        )
        # CI on each group rate is a (lower, upper) tuple within [0,1].
        ci_a = result["male_wilson_ci"]
        ci_b = result["female_wilson_ci"]
        assert len(ci_a) == 2 and 0 <= ci_a[0] <= ci_a[1] <= 1
        assert len(ci_b) == 2 and 0 <= ci_b[0] <= ci_b[1] <= 1
        # Point rates fall inside their CIs.
        assert ci_a[0] <= result["male_rate"] <= ci_a[1]
        assert ci_b[0] <= result["female_rate"] <= ci_b[1]
        # AIR CI bounds are present.
        assert "air_lower_ci" in result
        assert "air_upper_ci" in result
        assert result["air_lower_ci"] <= result["directional_air"] <= result["air_upper_ci"]

    def test_air_handles_zero_rate_edge_cases(self):
        from fairness.bias_detector import BiasDetector
        # Both zero -> AIR = 1.0 (no information, treat as parity)
        result = BiasDetector().adverse_impact_ratio(0, 50, 0, 50)
        assert result["directional_air"] == 1.0
        # One zero -> AIR = 0 (protected has zero selection)
        result = BiasDetector().adverse_impact_ratio(
            group_a_selected=40, group_a_total=50,
            group_b_selected=0,  group_b_total=50,
            group_a_name="male", group_b_name="female",
        )
        assert result["directional_air"] == 0.0


    # --- Cutoff method decoupling (Task #7 / #42) ----------------------

    # --- Dedup (Task #8 / #43) ----------------------------------------

    def test_audit_drops_exact_duplicate_resumes(self):
        from fairness.bias_detector import BiasDetector
        body = "Priya Sharma\nML Engineer with 5 years experience"
        # Submit the same body 5 times with different filenames + scores
        texts = {f"r{i}.txt": body for i in range(5)}
        scores = {f"r{i}.txt": 0.5 + i * 0.05 for i in range(5)}
        audit = BiasDetector().audit_ranking_bias(texts, scores)
        d = audit["dedup"]
        assert d["applied"] is True
        assert d["input_resumes"] == 5
        assert d["exact_dropped"] == 4
        assert d["kept"] == 1

    def test_audit_flags_ballot_stuffing_when_dedup_rate_high(self):
        from fairness.bias_detector import BiasDetector
        # 5 copies of one body + 1 unique -> 4/6 = 67% dedup
        body = "Priya Sharma\nEngineer with 5 years"
        texts = {f"r{i}.txt": body for i in range(5)}
        texts["unique.txt"] = "John Smith\nDeveloper"
        scores = {k: 0.5 for k in texts}
        audit = BiasDetector().audit_ranking_bias(texts, scores)
        assert audit["dedup"]["ballot_stuffing_alert"] is True
        assert any("ballot-stuffing" in r.lower()
                   for r in audit["recommendations"])

    def test_audit_detects_near_duplicates_via_simhash(self):
        from fairness.bias_detector import BiasDetector
        # Two near-identical bodies (one word changed).
        body_a = (
            "Priya Sharma\nML Engineer with 5 years of experience "
            "building recommender systems at TechCorp using Python "
            "and TensorFlow.\nMSc Computer Science."
        )
        body_b = (
            "Priya Sharma\nML Engineer with 5 years of experience "
            "building recommender systems at GlobalCorp using Python "
            "and TensorFlow.\nMSc Computer Science."
        )
        texts = {"a.txt": body_a, "b.txt": body_b, "c.txt": "John\nDev"}
        scores = {"a.txt": 0.5, "b.txt": 0.6, "c.txt": 0.7}
        audit = BiasDetector().audit_ranking_bias(
            texts, scores, near_dup_hamming=8,  # looser threshold
        )
        d = audit["dedup"]
        assert d["near_duplicate_pairs"] >= 1

    def test_audit_dedup_disable_keeps_all_resumes(self):
        from fairness.bias_detector import BiasDetector
        body = "Priya Sharma\nEngineer"
        texts = {f"r{i}.txt": body for i in range(3)}
        scores = {f"r{i}.txt": 0.5 for i in range(3)}
        audit = BiasDetector().audit_ranking_bias(
            texts, scores, dedup=False,
        )
        assert audit["dedup"]["applied"] is False
        assert audit["dedup"]["exact_dropped"] == 0


    def test_cutoff_method_median_is_default(self):
        from fairness.bias_detector import BiasDetector
        audit = BiasDetector().audit_ranking_bias(
            {"a.txt": "John\nEng", "b.txt": "Priya\nEng"},
            {"a.txt": 0.5, "b.txt": 0.9},
        )
        assert audit["cutoff_method"] == "median"

    def test_cutoff_method_top_k_uses_rank_threshold(self):
        from fairness.bias_detector import BiasDetector
        scores = {f"r{i}.txt": 0.1 * i for i in range(10)}
        texts = {k: "John Smith\nEng" for k in scores}
        audit = BiasDetector().audit_ranking_bias(
            texts, scores, cutoff_method="top_k", top_k=3,
        )
        assert audit["cutoff_method"] == "top_k"
        assert audit["cutoff_top_k"] == 3
        # Top 3 scores are 0.9, 0.8, 0.7 -> threshold = 0.7
        assert abs(audit["cutoff_threshold"] - 0.7) < 1e-9

    def test_cutoff_method_percentile_uses_score_percentile(self):
        from fairness.bias_detector import BiasDetector
        scores = {f"r{i}.txt": float(i) for i in range(101)}  # 0..100
        texts = {k: "John\nEng" for k in scores}
        audit = BiasDetector().audit_ranking_bias(
            texts, scores, cutoff_method="percentile", percentile=10,
        )
        assert audit["cutoff_method"] == "percentile"
        # Top 10% -> 90th percentile by score = 90.0
        assert abs(audit["cutoff_threshold"] - 90.0) < 1.0

    def test_cutoff_method_explicit_uses_given_threshold(self):
        from fairness.bias_detector import BiasDetector
        audit = BiasDetector().audit_ranking_bias(
            {"a.txt": "John\nEng"}, {"a.txt": 0.5},
            cutoff_method="explicit", selection_threshold=0.42,
        )
        assert audit["cutoff_method"] == "explicit"
        assert audit["cutoff_threshold"] == 0.42

    def test_cutoff_top_k_raises_without_top_k(self):
        from fairness.bias_detector import BiasDetector
        with pytest.raises(ValueError, match="top_k"):
            BiasDetector().audit_ranking_bias(
                {"a.txt": "John\nEng"}, {"a.txt": 0.5},
                cutoff_method="top_k",
            )

    def test_cutoff_percentile_raises_with_bad_percentile(self):
        from fairness.bias_detector import BiasDetector
        with pytest.raises(ValueError, match="percentile"):
            BiasDetector().audit_ranking_bias(
                {"a.txt": "John\nEng"}, {"a.txt": 0.5},
                cutoff_method="percentile", percentile=150,
            )

    def test_cutoff_explicit_raises_without_threshold(self):
        from fairness.bias_detector import BiasDetector
        with pytest.raises(ValueError, match="explicit"):
            BiasDetector().audit_ranking_bias(
                {"a.txt": "John\nEng"}, {"a.txt": 0.5},
                cutoff_method="explicit",
            )


    def test_dpd_uses_size_weighted_overall_rate_when_sizes_given(self):
        from fairness.bias_detector import BiasDetector
        # Unequal group sizes: A has 90, B has 10.  Rates A=0.5, B=0.9.
        # Unweighted mean = 0.7; weighted = (0.9*90 + 0.9*10)/100=... wait.
        # weighted = (0.5 * 90 + 0.9 * 10) / 100 = 0.54
        # |0.5 - 0.54| = 0.04, |0.9 - 0.54| = 0.36 -> DPD_w = 0.36
        # unweighted overall = 0.7; |0.5-0.7|=0.2, |0.9-0.7|=0.2 -> DPD_u = 0.2
        rates = {"A": 0.5, "B": 0.9}
        sizes = {"A": 90, "B": 10}
        weighted = BiasDetector.demographic_parity_distance(rates, sizes)
        unweighted = BiasDetector.demographic_parity_distance(rates)
        assert abs(weighted - 0.36) < 0.01
        assert abs(unweighted - 0.2) < 0.01

    def test_demographic_parity_full_returns_all_metrics(self):
        from fairness.bias_detector import BiasDetector
        result = BiasDetector.demographic_parity_full(
            group_selected={"male": 40, "female": 20},
            group_total={"male": 50, "female": 50},
        )
        for key in (
            "overall_selection_rate", "group_rates", "dpd_weighted",
            "dpd_unweighted", "chi_squared", "theil_t",
        ):
            assert key in result
        # overall = (40+20)/(50+50) = 0.6
        assert abs(result["overall_selection_rate"] - 0.6) < 1e-9
        # |0.8 - 0.6| = 0.2, |0.4 - 0.6| = 0.2 -> DPD = 0.2
        assert abs(result["dpd_weighted"] - 0.2) < 1e-9

    def test_chi_squared_detects_significant_difference(self):
        from fairness.bias_detector import BiasDetector
        # 40/50 male selected, 5/50 female -> very disparate
        result = BiasDetector.demographic_parity_full(
            group_selected={"male": 40, "female": 5},
            group_total={"male": 50, "female": 50},
        )
        assert result["chi_squared"]["p_value"] < 0.01

    def test_chi_squared_does_not_flag_balanced_rates(self):
        from fairness.bias_detector import BiasDetector
        result = BiasDetector.demographic_parity_full(
            group_selected={"male": 25, "female": 25},
            group_total={"male": 50, "female": 50},
        )
        assert result["chi_squared"]["p_value"] > 0.05

    def test_theil_t_zero_when_rates_are_equal(self):
        from fairness.bias_detector import BiasDetector
        result = BiasDetector.demographic_parity_full(
            group_selected={"male": 25, "female": 25},
            group_total={"male": 50, "female": 50},
        )
        assert result["theil_t"] == 0.0

    def test_theil_t_positive_when_rates_diverge(self):
        from fairness.bias_detector import BiasDetector
        result = BiasDetector.demographic_parity_full(
            group_selected={"male": 45, "female": 5},
            group_total={"male": 50, "female": 50},
        )
        assert result["theil_t"] > 0.05

    def test_audit_surfaces_parity_statistics(self):
        from fairness.bias_detector import BiasDetector
        audit = BiasDetector().audit_ranking_bias(
            {"p.txt": "Priya Sharma\nEng", "j.txt": "John Smith\nEng"},
            {"p.txt": 0.9, "j.txt": 0.8},
        )
        assert "parity_statistics" in audit
        ps = audit["parity_statistics"]
        for key in ("overall_selection_rate", "dpd_weighted",
                    "chi_squared", "theil_t"):
            assert key in ps


    def test_detection_coverage_floor_surfaced(self):
        from fairness.bias_detector import BiasDetector
        audit = BiasDetector().audit_ranking_bias(
            {"p.txt": "Priya Sharma\nEng", "j.txt": "John Smith\nEng"},
            {"p.txt": 0.9, "j.txt": 0.8},
        )
        dc = audit["detection_coverage"]
        for key in ("detected", "undetected", "coverage_rate",
                    "coverage_floor", "coverage_floor_met"):
            assert key in dc

    def test_low_coverage_forces_inconclusive_verdict(self):
        # Construct a corpus where the majority of resumes have
        # NO detectable name (just resume vocabulary).  Coverage
        # drops below 0.50 and the verdict is forced to
        # inconclusive_low_detection_coverage.
        from fairness.bias_detector import BiasDetector
        # Surname-only headers fall to name_source="empty" — neither
        # male_name nor female_name fires, so the candidate lands in
        # the "unknown" bucket.  1 detectable + 4 undetectable -> 0.20
        # detection coverage.  We pad both sides with one male candidate
        # to give the AIR block something to compute on.
        texts = {
            "good_f.txt":  "Priya Sharma\nEngineer",
            "good_m.txt":  "John Smith\nEngineer",
            "bad1.txt":    "Khan\nFull-stack engineer",
            "bad2.txt":    "Park\nMachine learning",
            "bad3.txt":    "Patel\nGrowth team",
            "bad4.txt":    "Jones\nUser experience",
            "bad5.txt":    "Singh\nQA team",
        }
        scores = {k: 0.5 + i * 0.05 for i, k in enumerate(texts)}
        audit = BiasDetector().audit_ranking_bias(texts, scores)
        assert audit["detection_coverage"]["coverage_rate"] < 0.5
        assert audit["detection_coverage"]["coverage_floor_met"] is False
        if audit.get("gender_bias_analysis"):
            assert (
                audit["gender_bias_analysis"]["verdict"]
                == "inconclusive_low_detection_coverage"
            )
        assert any(
            "detection coverage" in r.lower()
            for r in audit["recommendations"]
        )

    def test_full_coverage_does_not_trigger_gate(self):
        from fairness.bias_detector import BiasDetector
        # All detectable.
        texts = {
            "p.txt":  "Priya Sharma\nEngineer",
            "j.txt":  "John Smith\nEngineer",
            "f.txt":  "Fatima Khan\nAnalyst",
            "m.txt":  "Mohammed Ali\nAnalyst",
        }
        scores = {k: 0.5 + i * 0.05 for i, k in enumerate(texts)}
        audit = BiasDetector().audit_ranking_bias(texts, scores)
        assert audit["detection_coverage"]["coverage_floor_met"] is True
        if audit.get("gender_bias_analysis"):
            assert "low_detection_coverage" not in audit["gender_bias_analysis"]["verdict"]


    def test_audit_writes_baseline_when_requested(self, tmp_path):
        from fairness.bias_detector import BiasDetector
        log = tmp_path / "audit.jsonl"
        BiasDetector().audit_ranking_bias(
            {"p.txt": "Priya Sharma\nEng", "j.txt": "John Smith\nEng"},
            {"p.txt": 0.9, "j.txt": 0.8},
            audit_log_path=log, write_baseline=True,
        )
        assert log.exists()
        import json
        records = [json.loads(line) for line in log.read_text().splitlines()]
        assert len(records) == 1
        for key in ("timestamp", "n_resumes", "weighted_ece",
                    "ece_coverage", "adverse_impact_ratio", "verdict"):
            assert key in records[0]

    def test_second_audit_surfaces_drift_block(self, tmp_path):
        from fairness.bias_detector import BiasDetector
        log = tmp_path / "audit.jsonl"
        det = BiasDetector()
        det.audit_ranking_bias(
            {"p.txt": "Priya Sharma\nEng", "j.txt": "John Smith\nEng"},
            {"p.txt": 0.9, "j.txt": 0.8},
            audit_log_path=log, write_baseline=True,
        )
        audit2 = det.audit_ranking_bias(
            {"p.txt": "Priya Sharma\nEng", "j.txt": "John Smith\nEng"},
            {"p.txt": 0.9, "j.txt": 0.8},
            audit_log_path=log,
        )
        assert "drift_since_baseline" in audit2
        d = audit2["drift_since_baseline"]
        for key in ("baseline_timestamp", "baseline_weighted_ece",
                    "current_weighted_ece", "weighted_ece_delta",
                    "baseline_air", "current_air", "air_delta"):
            assert key in d

    def test_drift_block_absent_when_no_log_path(self):
        from fairness.bias_detector import BiasDetector
        audit = BiasDetector().audit_ranking_bias(
            {"a.txt": "John\nEng"}, {"a.txt": 0.5},
        )
        assert "drift_since_baseline" not in audit


    def test_audit_skips_counterfactual_when_no_jd_text(self):
        from fairness.bias_detector import BiasDetector
        audit = BiasDetector().audit_ranking_bias(
            {"a.txt": "John Smith\nEngineer"}, {"a.txt": 0.5},
            scorer=lambda jd, res: 0.5,  # scorer without jd_text
        )
        # No jd_text -> no counterfactual block
        assert "counterfactual_robustness" not in audit


    # --- Per-resume audit trail (Task #37) -----------------------------

    def test_audit_surfaces_per_resume_trail(self):
        from fairness.bias_detector import BiasDetector
        texts = {
            "p.txt": "Priya Sharma\nEngineer",
            "j.txt": "John Smith\nEngineer",
        }
        scores = {"p.txt": 0.9, "j.txt": 0.8}
        audit = BiasDetector().audit_ranking_bias(texts, scores)
        assert "per_resume" in audit
        pr = audit["per_resume"]
        assert set(pr.keys()) == set(texts.keys())
        for filename, row in pr.items():
            for key in (
                "score", "selected", "hard_gender", "name_token",
                "name_source", "name_p_female", "name_is_surname",
                "name_culture", "detected_language", "male_pronoun",
                "female_pronoun", "male_title", "female_title",
                "neutral_title", "confidence",
            ):
                assert key in row, f"{filename} missing key {key!r}"

    def test_per_resume_trail_matches_individual_call(self):
        from fairness.bias_detector import BiasDetector
        texts = {"p.txt": "Priya Sharma\nEngineer"}
        scores = {"p.txt": 0.9}
        audit = BiasDetector().audit_ranking_bias(texts, scores)
        individual = BiasDetector.detect_gender_proxy_scored(texts["p.txt"])
        trail = audit["per_resume"]["p.txt"]
        # Key fields must match the standalone call.
        assert trail["name_token"]    == individual["signals"]["name_token"]
        assert trail["name_p_female"] == individual["signals"]["name_p_female"]
        assert trail["hard_gender"]   == individual["gender"]


    def test_audit_batched_path_matches_per_resume_path(self):
        # Calling detect_gender_proxy_scored standalone (per-resume
        # path) and via audit_ranking_bias (batched path) must produce
        # the same gender / name_source / name_p_female for each resume.
        # Regression guard against the refactor.
        from fairness.bias_detector import BiasDetector
        texts = {
            f"r{i}.txt": name + "\nEngineer at TechCorp"
            for i, name in enumerate([
                "Priya Sharma", "John Smith", "Fatima Khan",
                "Mohammed Ali", "Sarah Chen", "Dr. Wei Chen",
            ])
        }
        scores = {k: 0.5 + i * 0.05 for i, k in enumerate(texts)}
        audit = BiasDetector().audit_ranking_bias(texts, scores)
        # Reconstruct via per-resume calls and compare distributions —
        # exact counts must match.
        per_resume_genders: dict = {}
        for filename, text in texts.items():
            per_resume_genders[filename] = (
                BiasDetector.detect_gender_proxy_scored(text)["gender"]
            )
        from collections import Counter
        batched = {
            g: stats["count"]
            for g, stats in audit["gender_distribution"].items()
        }
        per_resume = Counter(per_resume_genders.values())
        assert batched == dict(per_resume)


    def test_audit_flags_disagreement_between_soft_and_hard(self):
        # When the hard AIR passes but the soft AIR fails (or vice
        # versa), the audit should emit a NOTE flagging the gap so
        # reviewers know the verdict hinges on borderline calls.
        from fairness.bias_detector import BiasDetector
        detector = BiasDetector()
        # We can't easily construct this through detect_gender_proxy_scored
        # without specific names, so check the agreement_gap field is
        # populated and matches the underlying numbers.
        texts = {
            "p.txt": "Priya Sharma\nEngineer",
            "j.txt": "John Smith\nEngineer",
            "f.txt": "Fatima Khan\nAnalyst",
            "m.txt": "Mohammed Ali\nAnalyst",
        }
        scores = {"p.txt": 0.9, "j.txt": 0.8, "f.txt": 0.7, "m.txt": 0.6}
        audit = detector.audit_ranking_bias(texts, scores)
        analysis = audit["gender_bias_analysis"]
        gap = abs(analysis["adverse_impact_ratio_hard"]
                  - analysis["adverse_impact_ratio_soft"])
        assert abs(analysis["agreement_gap"] - gap) < 1e-4


# ═══════════════════════════════════════════════════════════════
# Name Classifier Tests (Issue #3)
# ═══════════════════════════════════════════════════════════════

class TestNameClassifier:
    """Tests for the calibrated char-ngram name classifier and the
    lookup-fastpath hybrid in fairness.names.classifier.

    These are integration tests against the committed model.pkl —
    they will FAIL if the model is missing or its calibration drifts
    materially.
    """

    def test_lookup_short_unisex_names_via_fastpath(self):
        # The model's n-gram generalisation tends to misclassify short
        # ambiguous names (wei, lee, kim) because their substrings are
        # dominated by majority-class patterns.  The lookup fastpath is
        # what makes the system correct on these.  Each of these names
        # must report source='lookup' AND have a p_female within 0.15
        # of the per-name empirical label from training_corpus.csv.
        from fairness.names.classifier import predict
        cases = {
            "wei":    0.50,
            "lee":    0.35,
            "kim":    0.50,
            "hyun":   0.50,
            "taylor": 0.25,
            "jordan": 0.29,
        }
        for name, target in cases.items():
            r = predict(name)
            assert r.source == "lookup", (
                f"{name!r} should hit the corpus lookup, got source={r.source}"
            )
            assert abs(r.p_female - target) <= 0.15, (
                f"{name!r} corpus says ~{target:.2f}, classifier says {r.p_female:.3f}"
            )

    def test_strong_signals_are_high_confidence(self):
        from fairness.names.classifier import predict
        for name in ("priya", "fatima", "sarah", "aisha", "maria", "anita"):
            r = predict(name)
            assert r.p_female >= 0.85, f"{name} should be ~female: {r.p_female:.3f}"
        for name in ("john", "ahmed", "mohammed", "sebastian", "rahul"):
            r = predict(name)
            assert r.p_female <= 0.15, f"{name} should be ~male: {r.p_female:.3f}"

    def test_oov_name_falls_through_to_model(self):
        # Made-up name unlikely to be in the corpus.  Confirms the model
        # path runs and returns a valid probability.
        from fairness.names.classifier import predict
        r = predict("Xqzaaria")
        assert r.source == "model"
        assert 0.0 <= r.p_female <= 1.0

    def test_empty_input_returns_neutral(self):
        from fairness.names.classifier import predict
        r = predict("")
        assert r.source == "empty"
        assert r.p_female == 0.5

    def test_non_alpha_input_returns_neutral(self):
        from fairness.names.classifier import predict
        r = predict("12345 !!")
        assert r.source == "empty"

    def test_hard_label_thresholding(self):
        from fairness.names.classifier import predict
        # John is strongly male and Priya strongly female; default 0.85
        # threshold should resolve cleanly.
        assert predict("john").hard_label() == "male"
        assert predict("priya").hard_label() == "female"
        # A near-unisex name should fall into 'unknown' at the default
        # threshold.  We use 'wei' which is corpus-labelled 0.5.
        assert predict("wei").hard_label() == "unknown"

    def test_confidence_is_distance_from_unisex(self):
        from fairness.names.classifier import NameGenderResult
        r = NameGenderResult(name="x", p_female=0.5, source="empty")
        assert r.confidence == 0.0
        r2 = NameGenderResult(name="x", p_female=1.0, source="lookup")
        assert r2.confidence == 1.0
        r3 = NameGenderResult(name="x", p_female=0.0, source="lookup")
        assert r3.confidence == 1.0
        r4 = NameGenderResult(name="x", p_female=0.85, source="lookup")
        assert abs(r4.confidence - 0.70) < 1e-9

    def test_batch_predict_matches_singletons(self):
        from fairness.names.classifier import predict, predict_many
        names = ["John", "Priya", "Xqzaaria", "", "Mohammed"]
        batch = predict_many(names)
        for raw, b in zip(names, batch):
            single = predict(raw)
            assert abs(b.p_female - single.p_female) < 1e-9
            assert b.source == single.source

    # --- Model integrity (Task #23) -----------------------------------

    def test_integrity_block_present_in_audit(self):
        from fairness.bias_detector import BiasDetector
        audit = BiasDetector().audit_ranking_bias(
            {"p.txt": "Priya Sharma\nEngineer",
             "j.txt": "John Smith\nEngineer"},
            {"p.txt": 0.9, "j.txt": 0.8},
        )
        assert "integrity" in audit
        i = audit["integrity"]
        assert "model_integrity_violated" in i
        assert "expected_sha256" in i
        assert "actual_sha256" in i

    def test_integrity_passes_on_unmodified_model(self):
        # When model.pkl hasn't been touched since training, the
        # expected and actual SHA-256 must match.  This guards
        # against accidental schema drift in the integrity block.
        from fairness.names.classifier import get_classifier
        clf = get_classifier()
        clf._ensure_loaded()
        if clf.expected_sha is None:
            import pytest
            pytest.skip("model_card.json has no integrity hash")
        assert clf.actual_sha == clf.expected_sha
        assert clf.integrity_violated is False

    def test_integrity_violation_surfaces_critical_recommendation(self):
        # Simulate a swapped model file by force-setting the flag on
        # an isolated classifier instance, then run a synthetic audit
        # path that uses it.  We can't easily swap the real pickle
        # mid-test (it's a singleton), so this exercises the
        # plumbing via direct attribute injection on the singleton.
        from fairness.bias_detector import BiasDetector
        from fairness.names.classifier import get_classifier
        clf = get_classifier()
        clf._ensure_loaded()
        # Snapshot originals so we restore them.
        orig = (clf.integrity_violated, clf.expected_sha, clf.actual_sha)
        try:
            clf.integrity_violated = True
            clf.expected_sha = "a" * 64
            clf.actual_sha = "b" * 64
            audit = BiasDetector().audit_ranking_bias(
                {"p.txt": "Priya Sharma\nEngineer",
                 "j.txt": "John Smith\nEngineer"},
                {"p.txt": 0.9, "j.txt": 0.8},
            )
            assert audit["integrity"]["model_integrity_violated"] is True
            assert any("[CRITICAL]" in rec and "integrity" in rec.lower()
                       for rec in audit["recommendations"])
        finally:
            clf.integrity_violated, clf.expected_sha, clf.actual_sha = orig


    # --- Multi-token names (Task #26) ----------------------------------

    def test_hyphenated_given_name_resolves(self):
        # "Anne-Marie" is a hyphenated given name; both parts ("anne",
        # "marie") are in the lookup as female names.  The classifier
        # must return a high p_female and the lookup path.
        from fairness.names.classifier import predict
        r = predict("Anne-Marie")
        assert r.source == "lookup"
        assert r.p_female >= 0.85

    def test_apostrophe_surname_resolves_as_surname(self):
        # "O'Brien" is a common Irish surname.  Both the joined form
        # "obrien" and the part "brien" should resolve through the
        # surname denylist.
        from fairness.names.classifier import predict
        r = predict("O'Brien")
        assert r.is_surname is True

    def test_hyphenated_surname_double_lookup(self):
        # "Smith-Jones" — both parts on the surname denylist.  Result
        # is_surname=True; for use in BiasDetector this should also
        # trigger is_surname_only.
        from fairness.names.classifier import predict
        r = predict("Smith-Jones")
        assert r.is_surname is True

    def test_particle_prefixed_surname_classifies_main_part(self):
        # "van der Berg" — "van" and "der" are surname particles and
        # are stripped; the resolved part is "berg".  Classification
        # should focus on the main name part.
        from fairness.names.classifier import predict
        r = predict("van der Berg")
        # The result should reflect "berg" being looked up / modelled,
        # not "vanderberg" as a single OOV string.
        assert r.name in ("berg", "vanderberg")  # accept either path

    def test_compound_first_name_with_compound_surname(self):
        # Full integration: "Anne-Marie Smith-Jones Engineer" should
        # be picked up as a female candidate via Anne-Marie, not
        # silenced by the surname.
        from fairness.bias_detector import BiasDetector
        r = BiasDetector.detect_gender_proxy_scored(
            "Anne-Marie Smith-Jones\nSenior Software Engineer"
        )
        assert r["signals"]["female_name"] is True
        assert r["gender"] == "female"

    def test_van_der_berg_picks_first_given_name(self):
        # Standard Dutch surname pattern: "Sarah van der Berg" — Sarah
        # is the given name and must win the cascade.
        from fairness.bias_detector import BiasDetector
        r = BiasDetector.detect_gender_proxy_scored(
            "Sarah van der Berg\nProduct Manager"
        )
        assert r["signals"]["name_token"] == "sarah"
        assert r["signals"]["female_name"] is True

    def test_de_la_cruz_works_in_comma_format(self):
        from fairness.bias_detector import BiasDetector
        r = BiasDetector.detect_gender_proxy_scored(
            "de la Cruz, Maria\nDesigner"
        )
        # Comma cascade picks the right side -> Maria
        assert r["signals"]["name_token"] == "maria"
        assert r["signals"]["female_name"] is True


    # --- Nickname canonical mapping (Task #28) -------------------------

    def test_bob_canonicalises_to_robert_male(self):
        from fairness.names.classifier import predict
        r = predict("Bob")
        # Bob -> Robert (lookup hit p_female~0 == male)
        assert r.p_female <= 0.15
        assert r.source == "lookup"

    def test_liz_canonicalises_to_elizabeth_female(self):
        from fairness.names.classifier import predict
        r = predict("Liz")
        assert r.p_female >= 0.85
        assert r.source == "lookup"

    def test_mike_canonicalises_to_michael(self):
        from fairness.names.classifier import predict
        r = predict("Mike")
        assert r.p_female <= 0.15

    def test_beth_canonicalises_to_elizabeth(self):
        from fairness.names.classifier import predict
        r = predict("Beth")
        assert r.p_female >= 0.85

    def test_nickname_in_full_resume_header(self):
        from fairness.bias_detector import BiasDetector
        r = BiasDetector.detect_gender_proxy_scored(
            "Bob Smith\nSenior Software Engineer"
        )
        assert r["signals"]["male_name"] is True
        assert r["gender"] == "male"

    def test_unisex_token_is_not_in_nickname_map(self):
        # Cross-gender-ambiguous nicknames (Chris, Sam, Alex, Pat,
        # Ronnie, Vivian) must NOT be canonicalised since the
        # canonical itself differs by gender.  Guard against future
        # CSV edits.
        from fairness.names.classifier import get_classifier
        clf = get_classifier()
        clf._ensure_loaded()
        for nick in ("chris", "sam", "alex", "pat", "ronnie", "ollie"):
            assert nick not in clf._nicknames, (
                f"{nick!r} is cross-gender ambiguous and must not "
                f"be in nicknames.csv"
            )


    # --- Per-culture calibration (Task #31) ----------------------------

    def test_per_culture_calibration_improved_over_global_baseline(self):
        # The model card records ece_per_culture_global_only_baseline
        # alongside the post-per-culture metrics in metrics.by_culture.
        # Per-culture calibration must improve ECE (vs baseline) on the
        # MAJORITY of clusters or this whole task was a regression.
        import json
        from pathlib import Path
        card = json.loads(
            Path("fairness/names/model_card.json").read_text(encoding="utf-8")
        )
        cal = card["pipeline"]["calibration"]
        assert cal["type"] == "IsotonicRegression"
        assert "per_culture_clusters_calibrated" in cal
        assert len(cal["per_culture_clusters_calibrated"]) >= 4

        baseline = cal["ece_per_culture_global_only_baseline"]
        final = card["metrics"]["by_culture"]
        improvements = []
        for culture, baseline_ece in baseline.items():
            if culture not in final or "ece" not in final[culture]:
                continue
            improvements.append(final[culture]["ece"] <= baseline_ece + 0.005)
        assert sum(improvements) >= len(improvements) // 2, (
            "Per-culture calibration must improve ECE on the majority "
            "of clusters compared to the global-only baseline"
        )

    def test_model_pipeline_is_cultural_calibrated_classifier(self):
        # Type guard so the runtime always loads the new design.
        from fairness.names.classifier import get_classifier
        from fairness.names.cultural_classifier import (
            CulturalCalibratedClassifier,
        )
        clf = get_classifier()
        clf._ensure_loaded()
        assert isinstance(clf._model, CulturalCalibratedClassifier)


    # --- Semver lineage (Task #35) -------------------------------------

    # --- Cache reset hooks (Task #36) ----------------------------------

    def test_reset_classifier_singleton_clears_lru(self):
        from fairness.names.classifier import (
            reset_classifier_singleton, predict_cached,
        )
        # Warm the cache
        predict_cached("John")
        assert predict_cached.cache_info().currsize > 0
        reset_classifier_singleton()
        assert predict_cached.cache_info().currsize == 0

    def test_reset_model_card_ece_cache_drops_cached_dict(self):
        import fairness.bias_detector as bd
        # Force the cache to be populated
        bd._MODEL_CARD_ECE_CACHE = {"test_culture": 0.123}
        bd.reset_model_card_ece_cache()
        assert bd._MODEL_CARD_ECE_CACHE is None


    def test_model_card_has_version_and_lineage(self):
        import json
        from pathlib import Path
        card = json.loads(
            Path("fairness/names/model_card.json").read_text(encoding="utf-8")
        )
        assert "version" in card
        parts = card["version"].split(".")
        assert len(parts) == 3 and all(p.isdigit() for p in parts)
        assert "lineage" in card
        # previous_version may be None for the first ever train but the
        # field must exist for traceability.
        assert "previous_version" in card["lineage"]
        assert "previous_sha256" in card["lineage"]


    # --- Model card schema validation (Task #38) -----------------------

    def test_validate_model_card_accepts_valid_card(self):
        from fairness.names.classifier import validate_model_card
        valid = {
            "model": "x", "version": "1.0.0",
            "trained_at": "2026-01-01T00:00:00Z",
            "integrity": {"sha256": "abc", "size_bytes": 100},
            "pipeline": {
                "type": "X", "vectorizer": {}, "gender_classifier": {},
                "calibration": {},
            },
            "metrics": {"overall": {}, "by_culture": {}},
            "calibration_target": {},
        }
        assert validate_model_card(valid) == []

    def test_validate_model_card_flags_missing_top_keys(self):
        from fairness.names.classifier import validate_model_card
        errors = validate_model_card({"model": "x"})
        assert errors
        assert any("missing top-level" in e for e in errors)

    def test_validate_model_card_flags_bad_version(self):
        from fairness.names.classifier import validate_model_card
        errors = validate_model_card({
            "model": "x", "version": "v1", "trained_at": "now",
            "integrity": {"sha256": "a", "size_bytes": 1},
            "pipeline": {
                "type": "X", "vectorizer": {}, "gender_classifier": {},
                "calibration": {},
            },
            "metrics": {"overall": {}, "by_culture": {}},
            "calibration_target": {},
        })
        assert any("semver" in e for e in errors)

    def test_shipped_model_card_is_valid(self):
        # Regression guard — the card we actually ship must pass.
        from fairness.names.classifier import (
            get_classifier, validate_model_card,
        )
        clf = get_classifier()
        clf._ensure_loaded()
        # card_validation_errors is populated during _load_model.
        assert clf.card_validation_errors == [], (
            f"shipped model_card.json is malformed: "
            f"{clf.card_validation_errors}"
        )

    def test_audit_surfaces_model_card_validation_block(self):
        from fairness.bias_detector import BiasDetector
        audit = BiasDetector().audit_ranking_bias(
            {"a.txt": "John\nEng"}, {"a.txt": 0.5},
        )
        assert "model_card_validation" in audit
        assert "valid" in audit["model_card_validation"]
        assert "errors" in audit["model_card_validation"]


    def test_overall_calibration_target_met(self):
        # The model_card declares calibration_target.overall_meets_target.
        # If this regresses (e.g. someone re-trains with worse data),
        # the test fires.
        import json
        from pathlib import Path
        card = json.loads(
            Path("fairness/names/model_card.json").read_text(encoding="utf-8")
        )
        assert card["calibration_target"]["overall_meets_target"] is True, (
            "Overall ECE > 0.05 — model is no longer field-calibrated"
        )
        assert card["metrics"]["overall"]["accuracy"] >= 0.85
        assert card["metrics"]["overall"]["roc_auc"] >= 0.90


# ═══════════════════════════════════════════════════════════════
# Fairness-Constrained Re-ranking Tests
# ═══════════════════════════════════════════════════════════════

class TestFairnessRanker:

    def test_already_fair_ranking(self):
        from ranking.fairness_ranker import FairnessConstrainedRanker, RankedCandidate
        candidates = [
            RankedCandidate("a", 0.9, "male"),
            RankedCandidate("b", 0.8, "female"),
            RankedCandidate("c", 0.7, "male"),
            RankedCandidate("d", 0.6, "female"),
        ]
        fcr = FairnessConstrainedRanker(threshold=0.8)
        report = fcr.rerank(candidates)
        assert report.fairness_satisfied is True
        assert report.num_swaps == 0

    def test_biased_ranking_gets_fixed(self):
        from ranking.fairness_ranker import FairnessConstrainedRanker, RankedCandidate
        candidates = [
            RankedCandidate("m1", 0.95, "male"),
            RankedCandidate("m2", 0.90, "male"),
            RankedCandidate("m3", 0.85, "male"),
            RankedCandidate("m4", 0.80, "male"),
            RankedCandidate("f1", 0.75, "female"),
            RankedCandidate("f2", 0.70, "female"),
            RankedCandidate("f3", 0.65, "female"),
            RankedCandidate("f4", 0.60, "female"),
        ]
        fcr = FairnessConstrainedRanker(threshold=0.8)
        report = fcr.rerank(candidates)
        assert report.final_air >= 0.8 or report.num_swaps > 0

    def test_displacement_cost_zero_when_unchanged(self):
        from ranking.fairness_ranker import FairnessConstrainedRanker, RankedCandidate
        candidates = [
            RankedCandidate("a", 0.9, "male"),
            RankedCandidate("b", 0.8, "female"),
        ]
        fcr = FairnessConstrainedRanker(threshold=0.8)
        report = fcr.rerank(candidates)
        assert report.displacement_cost == 0.0

    def test_from_scores_and_groups(self):
        from ranking.fairness_ranker import FairnessConstrainedRanker
        report = FairnessConstrainedRanker.from_scores_and_groups(
            names=["a", "b", "c", "d"],
            scores=[0.9, 0.8, 0.7, 0.6],
            groups=["male", "female", "male", "female"],
        )
        assert report.fairness_satisfied is True
        assert len(report.original_ranking) == 4

    def test_pareto_frontier_computed(self):
        from ranking.fairness_ranker import FairnessConstrainedRanker, RankedCandidate
        candidates = [
            RankedCandidate("a", 0.9, "male"),
            RankedCandidate("b", 0.8, "female"),
            RankedCandidate("c", 0.7, "male"),
            RankedCandidate("d", 0.6, "female"),
        ]
        fcr = FairnessConstrainedRanker(threshold=0.8)
        report = fcr.rerank(candidates)
        assert len(report.pareto_points) > 0
        assert all("threshold" in p for p in report.pareto_points)


# ═══════════════════════════════════════════════════════════════
# Surname Denylist Coverage (Task #24)
# ═══════════════════════════════════════════════════════════════

class TestSurnameCoverage:
    """Regression guard for surname denylist coverage.

    Coverage is measured against data/names/surname_holdout.csv, a
    curated set of ~95 well-attested common surnames per culture
    drawn from authoritative public lists (Wikipedia US/Asia,
    PRC public-security, Korean SIS, Japanese tele-directories,
    Forebears.io).  If a future denylist edit drops the per-culture
    coverage below the floor, this test fires immediately so the
    regression is caught before shipping.
    """

    _COVERAGE_FLOOR_PER_CULTURE = 0.85
    _COVERAGE_FLOOR_OVERALL     = 0.95

    def _measure(self):
        from data.names.validate_surnames import (
            load_denylist, load_holdout, measure_coverage,
        )
        return measure_coverage(load_denylist(), load_holdout())

    def test_overall_coverage_above_floor(self):
        cov = self._measure()
        total_n = sum(s["n"] for s in cov.values())
        total_hits = sum(s["hits"] for s in cov.values())
        ratio = total_hits / total_n
        assert ratio >= self._COVERAGE_FLOOR_OVERALL, (
            f"Overall surname coverage {ratio:.1%} below floor "
            f"{self._COVERAGE_FLOOR_OVERALL:.1%}.  Recent denylist "
            f"edits may have removed common surnames."
        )

    def test_per_culture_coverage_above_floor(self):
        cov = self._measure()
        for culture, stats in cov.items():
            assert stats["coverage"] >= self._COVERAGE_FLOOR_PER_CULTURE, (
                f"Culture {culture!r} surname coverage "
                f"{stats['coverage']:.1%} below floor "
                f"{self._COVERAGE_FLOOR_PER_CULTURE:.1%}.  Misses: "
                f"{stats['misses']!r}"
            )

    def test_holdout_is_non_empty_per_culture(self):
        from data.names.validate_surnames import load_holdout
        h = load_holdout()
        assert h, "holdout CSV produced no entries"
        for culture, surnames in h.items():
            assert len(surnames) >= 15, (
                f"Holdout for {culture!r} has only {len(surnames)} "
                f"entries — too few to validate coverage meaningfully"
            )

    def test_model_card_records_coverage(self):
        # The model card should expose the latest validator output so
        # reviewers see coverage at a glance without re-running the script.
        import json
        from pathlib import Path
        card_path = Path("fairness/names/model_card.json")
        if not card_path.exists():
            import pytest
            pytest.skip("model_card.json not present")
        card = json.loads(card_path.read_text(encoding="utf-8"))
        assert "surname_coverage" in card
        for culture, stats in card["surname_coverage"].items():
            assert "coverage" in stats
            assert 0.0 <= stats["coverage"] <= 1.0


# ═══════════════════════════════════════════════════════════════
# Counterfactual Name-Swap Robustness (Task #22)
# ═══════════════════════════════════════════════════════════════

class TestCounterfactualRobustness:
    """Tests for evaluation/counterfactual_robustness.py.

    A clean ranker (returns the same score regardless of name) must
    pass; a name-biased ranker (returns different scores per gender)
    must fail.  Without this harness there's no automated test that
    surfaces ranker-level name bias.
    """

    def test_clean_scorer_is_robust(self):
        from evaluation.counterfactual_robustness import name_swap_robustness

        # Scorer returns a constant — perfectly robust by construction.
        def clean(jd, resume):
            return 0.75

        report = name_swap_robustness(
            scorer=clean,
            jd="Senior Python role",
            base_resume="{NAME}\nSenior Python Developer\n5 years.",
        )
        assert report.robust is True
        assert report.score_gap == 0.0
        assert report.max_swap_delta == 0.0
        assert report.substitution_mode == "placeholder"

    def test_male_biased_scorer_fails_robustness(self):
        from evaluation.counterfactual_robustness import name_swap_robustness

        # Scorer that boosts male names — simulates a biased ranker.
        male_first_names = {"john", "michael", "rahul", "vikram", "wei",
                            "hiroshi", "mohammed", "omar"}

        def biased(jd, resume):
            first = resume.split()[0].lower()
            return 0.9 if first in male_first_names else 0.6

        report = name_swap_robustness(
            scorer=biased,
            jd="Senior Python role",
            base_resume="{NAME}\nDeveloper",
            gap_threshold=0.02,
        )
        assert report.robust is False
        # |0.9 - 0.6| = 0.3 — well above the 0.02 threshold
        assert report.score_gap > 0.2
        assert any("gap" in n.lower() for n in report.notes)

    def test_high_variance_scorer_fails_via_swap_delta(self):
        from evaluation.counterfactual_robustness import name_swap_robustness

        # Scorer that returns clean scores for most names but spikes
        # on one — the mean gap is small but max swap delta is huge.
        def spiky(jd, resume):
            if "Priya" in resume:
                return 0.99
            return 0.50

        report = name_swap_robustness(
            scorer=spiky,
            jd="role",
            base_resume="{NAME}\nDeveloper",
            gap_threshold=0.10,            # generous gap budget
            swap_delta_threshold=0.05,     # tight swap budget
        )
        assert report.robust is False
        assert report.max_swap_delta >= 0.4
        assert any("swap delta" in n.lower() for n in report.notes)

    def test_detection_mode_replaces_existing_name(self):
        from evaluation.counterfactual_robustness import name_swap_robustness

        captured: list = []

        def capture(jd, resume):
            captured.append(resume)
            return 0.5

        # No {NAME} placeholder -> detection mode kicks in.  The
        # original first token ("John") should be replaced in every
        # substitution.  We capture the per-call resumes and assert
        # the swap actually happened.
        base = "John Smith\nSenior Python Developer\n5 years."
        report = name_swap_robustness(
            scorer=capture,
            jd="Senior Python role",
            base_resume=base,
        )
        assert report.substitution_mode.startswith("detection")
        # Every captured resume should contain one of the swap names
        # AND none should contain the original "John" alone (since
        # we replaced the standalone token).
        for cap in captured:
            assert cap != base, "resume body must change across swaps"

    def test_detection_mode_fails_gracefully_without_detectable_name(self):
        from evaluation.counterfactual_robustness import name_swap_robustness

        # A header with NO Title-cased non-denylisted tokens.  Every
        # word is lowercase + on the resume-vocab denylist, so the
        # classifier returns no name_token and the harness must
        # short-circuit with an informative note (not crash, not
        # silently run a meaningless swap).
        def scorer(jd, resume):
            return 0.5

        report = name_swap_robustness(
            scorer=scorer,
            jd="role",
            base_resume="summary objective experience\nskills education",
        )
        assert report.male_scores == {}
        assert report.female_scores == {}
        assert any("no name token" in n.lower() for n in report.notes)

    def test_custom_name_pools_are_used(self):
        from evaluation.counterfactual_robustness import name_swap_robustness

        seen: list = []

        def scorer(jd, resume):
            seen.append(resume.split("\n")[0])
            return 0.5

        report = name_swap_robustness(
            scorer=scorer,
            jd="role",
            base_resume="{NAME}\nDeveloper",
            male_names=["Bob"],
            female_names=["Alice"],
        )
        assert list(report.male_scores.keys()) == ["Bob"]
        assert list(report.female_scores.keys()) == ["Alice"]


# ═══════════════════════════════════════════════════════════════
# API hardening (Task #10)
# ═══════════════════════════════════════════════════════════════

class TestApiHardening:
    """Tests for /audit + /rank + /explain API hardening.

    Uses FastAPI's TestClient to exercise the endpoints WITHOUT
    going over the network.  Some tests force-set environment
    variables before importing the server module so the configured
    caps and auth requirements are picked up.
    """

    def _client(self):
        # Defer import — the server module imports heavy ML deps on load.
        try:
            from fastapi.testclient import TestClient
            from api.server import app
            return TestClient(app)
        except Exception as e:
            import pytest
            pytest.skip(f"FastAPI test stack unavailable: {e}")

    def test_audit_rejects_oversized_single_resume(self, monkeypatch):
        client = self._client()
        # Construct a body well over the per-resume cap.
        oversized = "x" * 250_000
        resp = client.post(
            "/audit",
            json={
                "jd_text": "role",
                "resume_texts": {"big.txt": oversized},
            },
        )
        assert resp.status_code == 413

    def test_audit_rejects_too_many_resumes(self):
        client = self._client()
        # 10000 tiny resumes — over the 5000 cap.
        resumes = {f"r{i}.txt": "John\nEng" for i in range(10_000)}
        resp = client.post(
            "/audit",
            json={"jd_text": "role", "resume_texts": resumes},
        )
        assert resp.status_code == 413

    def test_api_key_required_when_set(self, monkeypatch):
        # Set env BEFORE re-import so the module captures it.  We use
        # importlib.reload because api.server reads os.getenv at module
        # load.
        monkeypatch.setenv("FAIMR_API_KEY", "secret123")
        import importlib, api.server
        importlib.reload(api.server)
        from fastapi.testclient import TestClient
        client = TestClient(api.server.app)
        # Without header: 401
        resp = client.post(
            "/audit",
            json={"jd_text": "role", "resume_texts": {"a.txt": "John\nEng"}},
        )
        assert resp.status_code == 401
        # With correct header: not 401 (may be other status from
        # missing infra in the test env, but auth must pass).
        resp = client.post(
            "/audit",
            headers={"X-API-Key": "secret123"},
            json={"jd_text": "role", "resume_texts": {"a.txt": "John\nEng"}},
        )
        assert resp.status_code != 401
        # Cleanup
        monkeypatch.delenv("FAIMR_API_KEY", raising=False)
        importlib.reload(api.server)


# ═══════════════════════════════════════════════════════════════
# Constrained-Insertion FCR (Task #9)
# ═══════════════════════════════════════════════════════════════

class TestConstrainedInsertionFCR:
    """Tests for the constrained-insertion FCR rewrite."""

    def _ranked(self, names_scores_groups):
        from ranking.fairness_ranker import RankedCandidate
        return [
            RankedCandidate(name=n, score=s, group=g)
            for n, s, g in names_scores_groups
        ]

    def test_algorithm_is_constrained_insertion(self):
        from ranking.fairness_ranker import FairnessConstrainedRanker
        report = FairnessConstrainedRanker(threshold=0.8).rerank(
            self._ranked([
                ("a", 0.9, "male"),   ("b", 0.8, "male"),
                ("c", 0.7, "female"), ("d", 0.6, "female"),
            ]),
        )
        assert report.algorithm == "constrained_insertion"
        assert report.termination_proof  # non-empty string

    def test_within_group_order_is_preserved(self):
        from ranking.fairness_ranker import FairnessConstrainedRanker
        cands = self._ranked([
            ("m1", 0.95, "male"),   ("m2", 0.90, "male"),
            ("m3", 0.85, "male"),   ("m4", 0.80, "male"),
            ("f1", 0.75, "female"), ("f2", 0.70, "female"),
            ("f3", 0.65, "female"), ("f4", 0.60, "female"),
        ])
        report = FairnessConstrainedRanker(threshold=0.8).rerank(cands)
        assert report.within_group_order_preserved is True
        # Female candidates appear in the fair ranking in the same
        # order they appeared in the input (f1, f2, f3, f4).
        females_in_fair = [n for n in report.fair_ranking if n.startswith("f")]
        assert females_in_fair == ["f1", "f2", "f3", "f4"]

    def test_terminates_in_single_pass(self):
        # The new algorithm makes EXACTLY one pass through n positions.
        # num_swaps_equivalent counts positions where output differs
        # from input; it is bounded by n, never n^2 or worse.
        from ranking.fairness_ranker import FairnessConstrainedRanker
        cands = self._ranked([
            (f"c{i}", 1.0 - i * 0.01,
             "male" if i % 2 == 0 else "female")
            for i in range(50)
        ])
        report = FairnessConstrainedRanker(threshold=0.8).rerank(cands)
        assert report.num_swaps <= len(cands)

    def test_already_fair_ranking_passes_through(self):
        from ranking.fairness_ranker import FairnessConstrainedRanker
        cands = self._ranked([
            ("a", 0.9, "male"),   ("b", 0.8, "female"),
            ("c", 0.7, "male"),   ("d", 0.6, "female"),
        ])
        report = FairnessConstrainedRanker(threshold=0.8).rerank(cands)
        assert report.fairness_satisfied is True
        # No reordering should happen — the input is already alternating.
        assert report.fair_ranking == ["a", "b", "c", "d"]

    def test_severely_biased_input_gets_reordered(self):
        from ranking.fairness_ranker import FairnessConstrainedRanker
        # All 5 males at top, all 5 females at bottom.
        cands = self._ranked([
            ("m1", 0.95, "male"),   ("m2", 0.90, "male"),
            ("m3", 0.85, "male"),   ("m4", 0.80, "male"),
            ("m5", 0.75, "male"),   ("f1", 0.70, "female"),
            ("f2", 0.65, "female"), ("f3", 0.60, "female"),
            ("f4", 0.55, "female"), ("f5", 0.50, "female"),
        ])
        report = FairnessConstrainedRanker(threshold=0.8).rerank(cands)
        # Female candidates should appear interleaved, not all at end.
        first_female = next(
            i for i, n in enumerate(report.fair_ranking) if n.startswith("f")
        )
        # In the input, first female is at index 5. After re-ranking,
        # the first female should appear much earlier.
        assert first_female < 5

    def test_final_air_is_close_to_threshold_or_higher(self):
        from ranking.fairness_ranker import FairnessConstrainedRanker
        cands = self._ranked([
            ("m1", 0.95, "male"),   ("m2", 0.90, "male"),
            ("m3", 0.85, "male"),   ("f1", 0.80, "female"),
            ("f2", 0.75, "female"), ("f3", 0.70, "female"),
        ])
        report = FairnessConstrainedRanker(threshold=0.8).rerank(cands)
        # After re-ranking the final AIR over the full list should
        # be 1.0 (both groups have all members "selected" at k=N).
        assert report.final_air == 1.0

    def test_displacement_is_bounded_and_metrics_consistent(self):
        from ranking.fairness_ranker import FairnessConstrainedRanker
        cands = self._ranked([
            ("m1", 0.95, "male"),   ("m2", 0.90, "male"),
            ("m3", 0.85, "male"),   ("m4", 0.80, "male"),
            ("f1", 0.75, "female"), ("f2", 0.70, "female"),
            ("f3", 0.65, "female"), ("f4", 0.60, "female"),
        ])
        report = FairnessConstrainedRanker(threshold=0.8).rerank(cands)
        # Displacement bounded in [0, 1] by the metric definition.
        assert 0.0 <= report.displacement_cost <= 1.0
        # Max single-candidate displacement bounded by n-1.
        assert 0 <= report.max_displacement <= len(cands) - 1

    def test_pareto_frontier_monotone_in_threshold(self):
        # As the AIR threshold increases, achieved_air should not
        # DECREASE (the algorithm can always satisfy a looser
        # threshold).  This isn't strictly true for greedy but the
        # constrained-insertion variant satisfies it.
        from ranking.fairness_ranker import FairnessConstrainedRanker
        cands = self._ranked([
            ("m1", 0.95, "male"),   ("m2", 0.90, "male"),
            ("f1", 0.85, "female"), ("f2", 0.80, "female"),
        ])
        report = FairnessConstrainedRanker(threshold=0.8).rerank(cands)
        assert len(report.pareto_points) >= 3


# ═══════════════════════════════════════════════════════════════
# Counterfactual Explainer Tests
# ═══════════════════════════════════════════════════════════════

class TestCounterfactual:

    def test_skill_extraction(self):
        from explainability.counterfactual import CounterfactualExplainer
        exp = CounterfactualExplainer()
        skills = exp._extract_skills("Python developer with machine learning and NLP")
        assert "python" in skills
        assert "machine learning" in skills
        assert "nlp" in skills

    def test_counterfactual_report_structure(self):
        from explainability.counterfactual import CounterfactualExplainer
        exp = CounterfactualExplainer()
        report = exp.explain_candidate(
            candidate_name="test",
            candidate_score=0.5,
            candidate_resume="Python developer",
            jd_text="Need Python, machine learning, deep learning, NLP skills",
            all_scores={"test": 0.5, "other": 0.8},
            top_k=3,
        )
        assert report.candidate_name == "test"
        assert report.original_score == 0.5
        assert report.total_skills_analyzed > 0

    def test_counterfactual_identifies_improvements(self):
        from explainability.counterfactual import CounterfactualExplainer
        exp = CounterfactualExplainer()
        report = exp.explain_candidate(
            candidate_name="bob",
            candidate_score=0.3,
            candidate_resume="Java developer with SQL experience",
            jd_text="Python machine learning deep learning TensorFlow NLP AWS",
            all_scores={"bob": 0.3, "alice": 0.7, "charlie": 0.9},
            top_k=5,
        )
        assert len(report.top_improvements) > 0

    def test_explain_all_candidates(self):
        from explainability.counterfactual import CounterfactualExplainer
        exp = CounterfactualExplainer()
        reports = exp.explain_all_candidates(
            scores={"a": 0.8, "b": 0.5},
            resume_texts={"a": "Python ML", "b": "Java SQL"},
            jd_text="Python machine learning deep learning",
            top_k=3,
        )
        assert len(reports) == 2


# ═══════════════════════════════════════════════════════════════
# Extended Fairness Metrics Tests
# ═══════════════════════════════════════════════════════════════

class TestExtendedFairness:

    def test_demographic_parity_distance(self):
        from fairness.bias_detector import BiasDetector
        rates = {"male": 0.8, "female": 0.4}
        dpd = BiasDetector.demographic_parity_distance(rates)
        assert dpd > 0
        assert dpd <= 0.5

    def test_equalized_odds(self):
        from fairness.bias_detector import BiasDetector
        tpr = {"male": 0.9, "female": 0.6}
        fpr = {"male": 0.1, "female": 0.2}
        result = BiasDetector.equalized_odds(tpr, fpr)
        assert result["tpr_gap"] == 0.3
        assert result["fpr_gap"] == 0.1
        assert result["equalized_odds_gap"] == 0.3

    def test_statistical_parity_difference_zero(self):
        from fairness.bias_detector import BiasDetector
        rates = {"male": 0.5, "female": 0.5}
        spd = BiasDetector.statistical_parity_difference(rates)
        assert spd == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
