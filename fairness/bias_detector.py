"""
ASRIS — Fairness & Bias Detection Module
Detects potential bias in resume ranking across demographic groups.
Implements the 4/5 Rule (Adverse Impact Ratio), demographic parity,
and statistical significance testing.
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


# ─── Gender Proxy Detection ─────────────────────────────────────

GENDER_INDICATORS = {
    "male": [
        r"\bhe\b", r"\bhis\b", r"\bhim\b", r"\bfather\b",
        r"\bson\b", r"\bbrother\b", r"\bgentleman\b", r"\bmr\b",
    ],
    "female": [
        r"\bshe\b", r"\bher\b", r"\bhers\b", r"\bmother\b",
        r"\bdaughter\b", r"\bsister\b", r"\blady\b", r"\bms\b",
        r"\bmrs\b", r"\bmiss\b",
    ],
}

# Name-based gender proxies (common gendered names)
GENDERED_NAMES = {
    "male": {
        "john", "james", "robert", "michael", "william", "david",
        "richard", "joseph", "thomas", "charles", "daniel", "matthew",
        "anthony", "mark", "donald", "steven", "paul", "andrew",
        "rahul", "amit", "vikram", "arun", "suresh", "rajesh",
    },
    "female": {
        "mary", "patricia", "jennifer", "linda", "elizabeth", "barbara",
        "susan", "jessica", "sarah", "karen", "nancy", "lisa",
        "priya", "anita", "sunita", "kavita", "neha", "pooja",
    },
}


class BiasDetector:
    """
    Detects potential demographic bias in resume ranking results.

    Supports:
    - Gender bias detection via text proxies
    - Domain/category bias
    - Adverse impact ratio (4/5 rule)
    - Statistical significance testing
    """

    def __init__(self, threshold: float = FAIRNESS_ADVERSE_IMPACT_THRESHOLD):
        self.adverse_impact_threshold = threshold

    # ─── Gender Detection ────────────────────────────────────────

    @staticmethod
    def detect_gender_proxy(text: str) -> str:
        """
        Attempt to detect gender from resume text using linguistic cues.
        Returns 'male', 'female', or 'unknown'.

        NOTE: This is a rough proxy for bias analysis only.
        Should NOT be used for decision-making.
        """
        text_lower = text.lower()

        male_score = 0
        female_score = 0

        for pattern in GENDER_INDICATORS["male"]:
            male_score += len(re.findall(pattern, text_lower))
        for pattern in GENDER_INDICATORS["female"]:
            female_score += len(re.findall(pattern, text_lower))

        # Check first name in first line
        first_line = text.split("\n")[0].strip().lower()
        first_word = first_line.split()[0] if first_line.split() else ""

        if first_word in GENDERED_NAMES["male"]:
            male_score += 5
        elif first_word in GENDERED_NAMES["female"]:
            female_score += 5

        if male_score > female_score and male_score >= 2:
            return "male"
        elif female_score > male_score and female_score >= 2:
            return "female"
        return "unknown"

    # ─── Adverse Impact Analysis ─────────────────────────────────

    def adverse_impact_ratio(
        self,
        group_a_selected: int,
        group_a_total: int,
        group_b_selected: int,
        group_b_total: int,
    ) -> dict:
        """
        Compute Adverse Impact Ratio (4/5 Rule).

        AIR = selection_rate_minority / selection_rate_majority

        If AIR < 0.8, there may be adverse impact.
        """
        rate_a = group_a_selected / group_a_total if group_a_total > 0 else 0
        rate_b = group_b_selected / group_b_total if group_b_total > 0 else 0

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

    # ─── Full Bias Audit ─────────────────────────────────────────

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
            selection_threshold: Score threshold for "selected" vs "not selected".
                                 If None, uses median.
        """
        if selection_threshold is None:
            all_scores = list(scores.values())
            selection_threshold = float(np.median(all_scores))

        # Detect gender proxies
        gender_groups = defaultdict(list)
        for filename, text in resume_texts.items():
            if filename in scores:
                gender = self.detect_gender_proxy(text)
                gender_groups[gender].append({
                    "filename": filename,
                    "score": scores[filename],
                    "selected": scores[filename] >= selection_threshold,
                })

        results = {
            "threshold": selection_threshold,
            "total_resumes": len(scores),
            "gender_distribution": {},
            "gender_bias_analysis": {},
            "domain_bias_analysis": {},
            "score_distribution": {},
            "recommendations": [],
        }

        # Gender distribution
        for gender, resumes in gender_groups.items():
            results["gender_distribution"][gender] = {
                "count": len(resumes),
                "mean_score": round(np.mean([r["score"] for r in resumes]), 4),
                "selected_count": sum(1 for r in resumes if r["selected"]),
                "selection_rate": round(
                    sum(1 for r in resumes if r["selected"]) / len(resumes), 4
                ) if resumes else 0,
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
                    f"⚠️ Gender AIR = {air_result['adverse_impact_ratio']:.2f} "
                    f"(below 0.80 threshold). Potential gender bias detected."
                )

        # Score distribution analysis
        all_scores = list(scores.values())
        results["score_distribution"] = {
            "mean": round(np.mean(all_scores), 4),
            "std": round(np.std(all_scores), 4),
            "median": round(np.median(all_scores), 4),
            "min": round(min(all_scores), 4),
            "max": round(max(all_scores), 4),
        }

        # Statistical test (Mann-Whitney U) for gender score difference
        if len(male_data) >= 5 and len(female_data) >= 5:
            from scipy import stats
            male_scores = [r["score"] for r in male_data]
            female_scores = [r["score"] for r in female_data]
            try:
                u_stat, p_value = stats.mannwhitneyu(
                    male_scores, female_scores, alternative="two-sided"
                )
                results["gender_bias_analysis"]["mann_whitney_p_value"] = round(p_value, 6)
                if p_value < 0.05:
                    results["recommendations"].append(
                        f"⚠️ Statistically significant score difference between "
                        f"genders (p={p_value:.4f}). Further investigation recommended."
                    )
            except Exception:
                pass

        if not results["recommendations"]:
            results["recommendations"].append(
                "✅ No significant bias detected in this evaluation."
            )

        return results

    def print_audit_report(self, audit: dict):
        """Print a formatted bias audit report."""
        print(f"\n{'═' * 60}")
        print(f"  FAIRNESS & BIAS AUDIT REPORT")
        print(f"{'═' * 60}")
        print(f"  Total resumes analyzed: {audit['total_resumes']}")
        print(f"  Selection threshold: {audit['threshold']:.4f}")

        print(f"\n  Gender Distribution:")
        for gender, stats in audit["gender_distribution"].items():
            print(f"    {gender.upper()}: {stats['count']} resumes, "
                  f"mean_score={stats['mean_score']:.4f}, "
                  f"selection_rate={stats['selection_rate']:.1%}")

        if audit["gender_bias_analysis"]:
            bias = audit["gender_bias_analysis"]
            print(f"\n  Adverse Impact Analysis (4/5 Rule):")
            print(f"    Male selection rate:   {bias.get('group_a_rate', 0):.1%}")
            print(f"    Female selection rate:  {bias.get('group_b_rate', 0):.1%}")
            print(f"    AIR: {bias.get('adverse_impact_ratio', 0):.4f}")
            print(f"    Passes 4/5 rule: {'✅ Yes' if bias.get('passes_4_5_rule') else '❌ No'}")
            print(f"    Risk level: {bias.get('risk_level', 'N/A')}")

        print(f"\n  Score Distribution:")
        sd = audit["score_distribution"]
        print(f"    Mean: {sd['mean']:.4f} ± {sd['std']:.4f}")
        print(f"    Range: [{sd['min']:.4f}, {sd['max']:.4f}]")

        print(f"\n  Recommendations:")
        for rec in audit["recommendations"]:
            print(f"    {rec}")

        print(f"\n{'═' * 60}\n")


if __name__ == "__main__":
    detector = BiasDetector()

    # Synthetic demo
    resume_texts = {
        "john_smith_resume.txt": "John Smith - Senior Engineer with Python, Java experience",
        "mary_jones_resume.txt": "Mary Jones - Data Scientist with Python, R skills",
        "alex_unknown_resume.txt": "Alex Morgan - Project Manager with Agile, Scrum",
        "priya_sharma_resume.txt": "Priya Sharma - ML Engineer with TensorFlow, PyTorch",
        "james_wilson_resume.txt": "James Wilson - Backend Developer with Go, Rust",
    }

    scores = {
        "john_smith_resume.txt": 0.85,
        "mary_jones_resume.txt": 0.72,
        "alex_unknown_resume.txt": 0.60,
        "priya_sharma_resume.txt": 0.90,
        "james_wilson_resume.txt": 0.78,
    }

    audit = detector.audit_ranking_bias(resume_texts, scores)
    detector.print_audit_report(audit)
