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
        from fairness.bias_detector import GENDERED_NAMES
        for surname in ("chen", "li", "wang", "zhang", "liu", "lee"):
            assert surname not in GENDERED_NAMES["male"], (
                f"{surname!r} is a surname, not a given name — must not vote male"
            )

    def test_hyun_is_unisex_not_double_listed(self):
        from fairness.bias_detector import GENDERED_NAMES, _UNISEX_NAMES
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
        from fairness.bias_detector import GENDERED_NAMES, _UNISEX_NAMES
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

    def test_ms_oneill_apostrophe_name_fires(self):
        from fairness.bias_detector import BiasDetector
        result = BiasDetector.detect_gender_proxy_scored(
            "Ms. O'Neill\nProduct Manager"
        )
        assert result["signals"]["female_title"] is True

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
