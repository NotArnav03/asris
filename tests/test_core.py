"""
ASRIS — Unit Tests
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


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
