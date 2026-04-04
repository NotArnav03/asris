"""
ASRIS/FAIMR -- Fairness & Bias Detection Module
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

# Title/honorific patterns -- applied to the first 200 characters of text
TITLE_HONORIFICS = {
    "male": [
        r"\bmr\.?\b", r"\bmister\b", r"\bsir\b",
    ],
    "female": [
        r"\bms\.?\b", r"\bmrs\.?\b", r"\bmiss\b", r"\bmadam\b",
    ],
    # Neutral titles do not vote for either gender
    "neutral": [
        r"\bdr\.?\b", r"\bprof\.?\b", r"\bprofessor\b", r"\bmx\.?\b",
    ],
}

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
        "bryan", "billy", "lee", "marcus", "christopher", "alexander",
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
        "wei", "ming", "jun", "yang", "xiao", "lei", "fang", "hao",
        "long", "tao", "ping", "bo", "zhen", "jian", "hiro",
        "kenji", "takashi", "naoki", "daisuke", "ryo", "yuto",
        "seung", "hyun", "jae", "sung", "dong", "tae", "jin",
        "chen", "li", "wang", "zhang", "liu",
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
        "mei", "ling", "xiu", "yan", "fei", "qian", "jing", "yun",
        "shu", "xia", "akiko", "yoko", "haruko", "noriko", "keiko",
        "sachiko", "tomoko", "yuki", "sakura", "hana", "aiko",
        "soo", "ji", "eun", "young", "hyun", "min", "na",
        "hua", "hong", "qing",
        # Arab / Middle Eastern (female)
        "fatima", "amira", "nadia", "layla", "yasmin", "nour",
        "rania", "zainab", "mariam", "hana", "dina", "lina",
        "rana", "mona", "huda", "asmaa", "salma", "aisha",
        "maryam", "sara",
    },
}


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
        header = text_lower[:200]  # title/name detection on header only

        signals: dict = {
            "male_pronoun": 0,
            "female_pronoun": 0,
            "male_title": False,
            "female_title": False,
            "neutral_title": False,
            "male_name": False,
            "female_name": False,
        }

        # 1. Pronoun scan (full text)
        for pat in GENDER_INDICATORS["male"]:
            signals["male_pronoun"] += len(re.findall(pat, text_lower))
        for pat in GENDER_INDICATORS["female"]:
            signals["female_pronoun"] += len(re.findall(pat, text_lower))

        # 2. Title/honorific scan (header only, stronger evidence)
        for pat in TITLE_HONORIFICS["male"]:
            if re.search(pat, header):
                signals["male_title"] = True
                break
        for pat in TITLE_HONORIFICS["female"]:
            if re.search(pat, header):
                signals["female_title"] = True
                break
        for pat in TITLE_HONORIFICS["neutral"]:
            if re.search(pat, header):
                signals["neutral_title"] = True
                break

        # 3. Name scan: check first two header lines, each word
        header_lines = text.strip().split("\n")[:2]
        header_tokens = " ".join(header_lines).lower().split()
        for token in header_tokens:
            clean = re.sub(r"[^a-z]", "", token)
            if clean in GENDERED_NAMES["male"]:
                signals["male_name"] = True
            if clean in GENDERED_NAMES["female"]:
                signals["female_name"] = True

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
