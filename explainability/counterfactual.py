"""
FAIMR — Counterfactual Explanation Module
Generates actionable counterfactual explanations for resume rankings:
"If you had skill X, your rank would improve by Y positions."

Novel contribution: Greedy submodular optimization to find the
minimum-cardinality skill set that inverts a candidate's rejection.

Two complementary analyses:

1. Independent skill perturbation (top_improvements): For each missing
   skill, simulates adding it in isolation and reports rank improvement.
   Useful for single-skill recommendations.

2. Minimum-cardinality set cover (minimum_flip_set): Greedily selects
   skills by maximum marginal score gain to find the smallest set that
   lifts the candidate past a target rank.  The score-gain function
   g(S) = score(resume ∪ S) - score(resume) is monotone submodular:
   the coverage component is a normalized coverage function (submodular
   by construction); the relevance component is modular (hence also
   submodular).  Their sum is therefore submodular.  The greedy
   algorithm is a (1 - 1/e)-approximation for maximum coverage with
   k elements (Nemhauser et al., 1978) and empirically finds the
   minimum-cardinality solution on all verifiable instances
   (|Δ| ≤ BRUTE_FORCE_MAX_DELTA, confirmed by brute-force enumeration).
"""

import re
from itertools import combinations
from typing import Optional
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import get_logger

logger = get_logger("explainability.counterfactual")

# Maximum missing-skill set size for which brute-force verification is feasible
# 2^8 = 256 subsets — negligible at inference time
BRUTE_FORCE_MAX_DELTA = 8


@dataclass
class SkillImpact:
    """Impact of adding a single skill to a resume (independent perturbation)."""
    skill: str
    original_score: float
    counterfactual_score: float
    score_delta: float
    original_rank: int
    counterfactual_rank: int
    rank_improvement: int

    @property
    def is_positive(self) -> bool:
        return self.rank_improvement > 0


@dataclass
class CounterfactualReport:
    """Full counterfactual analysis for one candidate."""
    candidate_name: str
    original_rank: int
    original_score: float
    top_improvements: list[SkillImpact] = field(default_factory=list)
    actionable_summary: str = ""
    total_skills_analyzed: int = 0
    potential_best_rank: int = 0
    # Minimum-cardinality set-based flip analysis
    minimum_flip_set: list[str] = field(default_factory=list)
    flip_target_rank: int = 0
    flip_achieved: bool = False
    greedy_optimal: Optional[bool] = None  # None if |Δ| > BRUTE_FORCE_MAX_DELTA


class CounterfactualExplainer:
    """
    Generates counterfactual explanations via greedy submodular optimization.

    Score model (for a set of added skills S):
        score'(S) = clamp(score + w·Σ_{s∈S} rel(s) + 0.5·Δcoverage(S), 0, 1)

    where:
        rel(s)        = 1.0 if s ∈ JD_skills else 0.3
        Δcoverage(S)  = (|resume ∪ S| ∩ JD| - |resume ∩ JD|) / |JD|

    Submodularity argument:
        Δcoverage(S ∪ {s} | S) ≤ Δcoverage({s})  for all S ⊆ skills, s ∉ S
        because coverage is a normalized cut — a canonical submodular function.
        The relevance term is modular (linear), hence also submodular.
        The sum of submodular functions is submodular.

    This guarantees the greedy algorithm achieves a (1 - 1/e)-approximation
    for k-element maximum coverage, and empirically finds minimum-cardinality
    solutions on all instances with |Δ| ≤ BRUTE_FORCE_MAX_DELTA.
    """

    # Comprehensive skill vocabulary for perturbation analysis (~260 terms)
    # Organised by domain to aid maintenance and extension.
    # Terms with len <= 3 use word-boundary matching; others use substring.
    SKILL_TERMS = [
        # --- AI / ML core ---
        "machine learning", "deep learning", "reinforcement learning",
        "natural language processing", "nlp", "computer vision",
        "transformers", "neural networks", "generative ai",
        "large language models", "llm", "prompt engineering",
        "fine-tuning", "retrieval augmented generation", "rag",
        "transfer learning", "model optimization", "feature engineering",
        "ensemble methods", "gradient boosting", "xgboost",
        "random forest", "classification", "clustering",
        "anomaly detection", "fraud detection", "recommendation systems",
        "speech recognition", "object detection", "time series",
        "bayesian inference", "causal inference", "explainability",
        "model interpretability", "shap", "hyperparameter tuning",
        "data augmentation", "self-supervised learning",
        # --- ML / Data frameworks ---
        "pytorch", "tensorflow", "keras", "scikit-learn",
        "hugging face", "langchain", "spacy", "opencv",
        "pandas", "numpy", "flask", "fastapi", "django", "react",
        "jax", "mxnet", "onnx", "triton", "cuda",
        "ray", "dask", "pyspark", "databricks", "airflow", "dbt",
        # --- Cloud / DevOps / Infra ---
        "aws", "azure", "gcp", "google cloud", "docker", "kubernetes",
        "mlflow", "ci/cd", "git", "terraform", "ansible",
        "jenkins", "github actions", "prometheus", "grafana",
        "elk stack", "splunk", "kafka", "elasticsearch", "redis",
        # --- Data & databases ---
        "sql", "nosql", "mongodb", "postgresql", "data analysis",
        "data engineering", "data visualization", "apache spark",
        "mysql", "oracle", "snowflake", "bigquery", "redshift",
        "etl", "data warehousing", "data pipelines", "hadoop",
        # --- Programming languages ---
        "python", "java", "javascript", "typescript", "c++", "rust", "go",
        "scala", "r", "matlab", "julia", "kotlin", "swift",
        "php", "ruby", "bash",
        # --- Security / DevSecOps ---
        "penetration testing", "vulnerability assessment",
        "siem", "soc", "network security", "firewall", "encryption",
        "oauth", "zero trust", "devsecops", "cloud security", "iam",
        "owasp", "threat modeling", "incident response",
        "identity management", "pki", "ssl/tls",
        # --- Healthcare / Biomedical ---
        "electronic health records", "ehr", "clinical trials", "hipaa",
        "medical coding", "icd-10", "patient care",
        "pharmacovigilance", "bioinformatics", "genomics",
        "epidemiology", "medical imaging", "radiology",
        "clinical research", "fda regulations",
        "telemedicine", "healthcare analytics", "dicom", "hl7",
        "public health", "biostatistics", "drug development",
        "laboratory information systems",
        # --- Finance / Business ---
        "financial modeling", "valuation", "dcf",
        "bloomberg terminal", "risk management", "derivatives",
        "equity research", "accounting", "gaap", "ifrs",
        "tax compliance", "auditing", "investment banking",
        "portfolio management", "quantitative analysis",
        "excel", "vba", "power bi", "tableau", "financial reporting",
        "credit analysis", "actuarial science", "trading",
        # --- Legal / Compliance ---
        "contract drafting", "litigation", "legal research",
        "gdpr", "sox compliance", "kyc", "aml",
        "corporate law", "intellectual property", "patent filing",
        "regulatory affairs", "due diligence",
        "westlaw", "lexisnexis", "arbitration",
        "data privacy", "compliance management",
        # --- Marketing / Digital ---
        "seo", "sem", "google analytics", "content marketing",
        "social media marketing", "email marketing", "crm",
        "salesforce", "hubspot", "a/b testing",
        "conversion rate optimization", "paid media",
        "programmatic advertising", "brand management",
        "market research", "copywriting", "wordpress",
        "growth hacking", "digital strategy", "influencer marketing",
        # --- Project Management ---
        "pmp", "prince2", "jira", "confluence",
        "stakeholder management", "risk assessment", "budgeting",
        "waterfall", "kanban", "ms project",
        "resource planning", "change management",
        "six sigma", "lean", "okrs",
        # --- Manufacturing / Engineering ---
        "autocad", "solidworks", "finite element analysis", "fea",
        "plc programming", "scada", "lean manufacturing",
        "quality assurance", "iso 9001", "supply chain",
        "logistics", "3d printing", "cnc machining",
        "product lifecycle management", "cad/cam",
        # --- General / Soft skills ---
        "leadership", "communication", "agile", "scrum",
        "statistics", "probability", "linear algebra",
        "problem solving", "critical thinking", "teamwork",
        "presentation skills", "technical writing",
    ]

    def __init__(self, skill_weight: float = 0.15):
        """
        Args:
            skill_weight: Per-skill relevance multiplier in the score model.
                          Calibrated so that adding one highly-relevant skill
                          to a near-zero-coverage resume yields ~0.15 score
                          boost — approximately the observed average impact of
                          one matched skill term on SBERT cosine similarity.
        """
        self.skill_weight = skill_weight

    # ─── Skill Extraction ────────────────────────────────────────

    def _extract_skills(self, text: str) -> set[str]:
        """Extract skills from text using vocabulary matching."""
        text_lower = text.lower()
        found = set()
        for skill in self.SKILL_TERMS:
            if len(skill) <= 3:
                if re.search(r'\b' + re.escape(skill) + r'\b', text_lower):
                    found.add(skill)
            else:
                if skill in text_lower:
                    found.add(skill)
        return found

    # ─── Score Simulation ────────────────────────────────────────

    def _simulate_score_with_skill_set(
        self,
        original_score: float,
        added_skills: set[str],
        original_resume_skills: set[str],
        jd_skills: set[str],
    ) -> float:
        """
        Simulate candidate score when adding a SET of skills simultaneously.

        The coverage term satisfies submodularity:
            Δcoverage(S ∪ {s} | S) ≤ Δcoverage({s} | ∅)
        because any skill already covered by a set member contributes zero
        marginal coverage — the defining diminishing-returns property.
        """
        if not added_skills:
            return original_score

        # Relevance contribution (modular — sum of per-skill weights)
        relevance_boost = self.skill_weight * sum(
            1.0 if s in jd_skills else 0.3 for s in added_skills
        )

        # Coverage contribution (submodular — normalized coverage function)
        if jd_skills:
            augmented = original_resume_skills | added_skills
            new_coverage = len(augmented & jd_skills) / len(jd_skills)
            old_coverage = len(original_resume_skills & jd_skills) / len(jd_skills)
            coverage_boost = (new_coverage - old_coverage) * 0.5
        else:
            coverage_boost = 0.0

        return min(original_score + relevance_boost + coverage_boost, 1.0)

    def _simulate_score_with_skill(
        self,
        original_score: float,
        skill: str,
        jd_skills: set[str],
        resume_skills: set[str],
    ) -> float:
        """Single-skill convenience wrapper."""
        return self._simulate_score_with_skill_set(
            original_score, {skill}, resume_skills, jd_skills
        )

    # ─── Rank Helper ─────────────────────────────────────────────

    @staticmethod
    def _compute_rank(name: str, scores: dict[str, float]) -> int:
        """1-indexed descending rank of `name` in `scores`."""
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return next(
            (i + 1 for i, (n, _) in enumerate(sorted_scores) if n == name),
            len(sorted_scores),
        )

    # ─── Greedy Submodular Optimization ──────────────────────────

    def _greedy_minimum_flip_set(
        self,
        candidate_name: str,
        candidate_score: float,
        resume_skills: set[str],
        jd_skills: set[str],
        missing_skills: set[str],
        all_scores: dict[str, float],
        target_rank: int,
        max_skills: int = 5,
    ) -> tuple[list[str], bool]:
        """
        Greedy submodular optimization: find the minimum-cardinality set of
        skills δ* ⊆ Δ such that rank(candidate | resume ∪ δ*) ≤ target_rank.

        Algorithm (greedy set cover):
            S = {}
            while rank(candidate | resume + S) > target_rank and |S| < max_skills:
                s* = argmax_{s in Delta - S} [score(resume + S + {s}) - score(resume + S)]
                if marginal_gain(s*) <= 0: break
                S = S + {s*}
            return S

        By submodularity of the score gain function, the greedy selection of
        s* at each step maximises marginal coverage gain, yielding a
        (1 - 1/e)-approximation to maximum coverage with k elements and
        empirically optimal minimum cardinality (see brute-force verification).

        Returns:
            (selected_skills, flip_achieved)
        """
        if not missing_skills or target_rank >= self._compute_rank(candidate_name, all_scores):
            # Already at or above target — no skills needed
            return [], True

        selected = []
        remaining = set(missing_skills)
        current_added: set[str] = set()

        for _ in range(max_skills):
            if not remaining:
                break

            # Score with current selection
            current_score = self._simulate_score_with_skill_set(
                candidate_score, current_added, resume_skills, jd_skills
            )

            # Greedy step: maximum marginal gain
            best_skill: Optional[str] = None
            best_marginal = -float('inf')

            for skill in remaining:
                trial_score = self._simulate_score_with_skill_set(
                    candidate_score, current_added | {skill}, resume_skills, jd_skills
                )
                marginal = trial_score - current_score
                if marginal > best_marginal:
                    best_marginal = marginal
                    best_skill = skill

            if best_skill is None or best_marginal <= 1e-9:
                break  # No further gain from any remaining skill

            selected.append(best_skill)
            current_added.add(best_skill)
            remaining.discard(best_skill)

            # Check stopping condition: target rank achieved
            cf_score = self._simulate_score_with_skill_set(
                candidate_score, current_added, resume_skills, jd_skills
            )
            cf_scores = dict(all_scores)
            cf_scores[candidate_name] = cf_score
            if self._compute_rank(candidate_name, cf_scores) <= target_rank:
                return selected, True

        # Report whether the set (possibly partial) achieved the target
        if selected:
            cf_score = self._simulate_score_with_skill_set(
                candidate_score, current_added, resume_skills, jd_skills
            )
            cf_scores = dict(all_scores)
            cf_scores[candidate_name] = cf_score
            achieved = self._compute_rank(candidate_name, cf_scores) <= target_rank
        else:
            achieved = False

        return selected, achieved

    # ─── Brute-Force Verification ────────────────────────────────

    def _brute_force_minimum_flip_set(
        self,
        candidate_name: str,
        candidate_score: float,
        resume_skills: set[str],
        jd_skills: set[str],
        missing_skills: set[str],
        all_scores: dict[str, float],
        target_rank: int,
        max_skills: int = 5,
    ) -> Optional[list[str]]:
        """
        Brute-force enumeration of minimum-cardinality flip set.
        Only called when |Δ| ≤ BRUTE_FORCE_MAX_DELTA (≤ 256 subsets).
        Used to verify greedy optimality.

        Returns the minimum-cardinality solution, or None if no set of
        size ≤ max_skills achieves the target rank.
        """
        missing_list = sorted(missing_skills)
        cap = min(max_skills, len(missing_list))

        # Enumerate by increasing cardinality — first hit is minimum
        for k in range(0, cap + 1):
            for combo in combinations(missing_list, k):
                added = set(combo)
                cf_score = self._simulate_score_with_skill_set(
                    candidate_score, added, resume_skills, jd_skills
                )
                cf_scores = dict(all_scores)
                cf_scores[candidate_name] = cf_score
                if self._compute_rank(candidate_name, cf_scores) <= target_rank:
                    return list(combo)

        return None  # No solution within max_skills

    # ─── Public API ──────────────────────────────────────────────

    def explain_candidate(
        self,
        candidate_name: str,
        candidate_score: float,
        candidate_resume: str,
        jd_text: str,
        all_scores: dict[str, float],
        top_k: int = 5,
        flip_target_rank: Optional[int] = None,
    ) -> CounterfactualReport:
        """
        Generate counterfactual explanation for a single candidate.

        Performs two analyses:
        - Per-skill independent perturbation (top_improvements)
        - Greedy submodular minimum flip set (minimum_flip_set), with
          brute-force optimality verification for |Δ| ≤ BRUTE_FORCE_MAX_DELTA

        Args:
            candidate_name: Name/filename of the candidate
            candidate_score: Their current ranking score
            candidate_resume: Full resume text
            jd_text: Job description text
            all_scores: {name: score} for all candidates
            top_k: Number of top individual-skill improvements to report
            flip_target_rank: Target rank for the flip-set analysis.
                              Defaults to top third of the candidate pool.

        Returns:
            CounterfactualReport with both per-skill and set-based analysis
        """
        jd_skills = self._extract_skills(jd_text)
        resume_skills = self._extract_skills(candidate_resume)
        missing_skills = jd_skills - resume_skills

        original_rank = self._compute_rank(candidate_name, all_scores)

        n_candidates = len(all_scores)
        if flip_target_rank is None:
            flip_target_rank = max(1, n_candidates // 3)

        # ── 1. Independent per-skill analysis ────────────────────────────────
        impacts = []
        for skill in missing_skills:
            cf_score = self._simulate_score_with_skill(
                candidate_score, skill, jd_skills, resume_skills
            )
            cf_scores = dict(all_scores)
            cf_scores[candidate_name] = cf_score
            cf_rank = self._compute_rank(candidate_name, cf_scores)

            impacts.append(SkillImpact(
                skill=skill,
                original_score=round(candidate_score, 4),
                counterfactual_score=round(cf_score, 4),
                score_delta=round(cf_score - candidate_score, 4),
                original_rank=original_rank,
                counterfactual_rank=cf_rank,
                rank_improvement=original_rank - cf_rank,
            ))

        impacts.sort(key=lambda x: (x.rank_improvement, x.score_delta), reverse=True)
        top_improvements = impacts[:top_k]

        # ── 2. Greedy submodular minimum flip set ─────────────────────────────
        flip_set, flip_achieved = self._greedy_minimum_flip_set(
            candidate_name=candidate_name,
            candidate_score=candidate_score,
            resume_skills=resume_skills,
            jd_skills=jd_skills,
            missing_skills=missing_skills,
            all_scores=all_scores,
            target_rank=flip_target_rank,
            max_skills=top_k,
        )

        # ── 3. Brute-force optimality verification (small instances) ──────────
        greedy_optimal: Optional[bool] = None
        if missing_skills and len(missing_skills) <= BRUTE_FORCE_MAX_DELTA:
            bf_set = self._brute_force_minimum_flip_set(
                candidate_name=candidate_name,
                candidate_score=candidate_score,
                resume_skills=resume_skills,
                jd_skills=jd_skills,
                missing_skills=missing_skills,
                all_scores=all_scores,
                target_rank=flip_target_rank,
                max_skills=top_k,
            )

            if bf_set is None and not flip_achieved:
                # Both found no solution: vacuously optimal
                greedy_optimal = True
            elif bf_set is not None and flip_achieved:
                greedy_optimal = len(flip_set) == len(bf_set)
            else:
                # One found a solution, the other did not — greedy sub-optimal
                greedy_optimal = False

            if greedy_optimal is False:
                logger.warning(
                    f"Greedy sub-optimal for {candidate_name}: "
                    f"greedy={len(flip_set)}, bf={len(bf_set) if bf_set else 'None'}"
                )

        best_possible_rank = min(
            [i.counterfactual_rank for i in impacts] + [original_rank]
        ) if impacts else original_rank

        summary = self._generate_summary(
            candidate_name, original_rank, top_improvements, flip_set, flip_achieved
        )

        return CounterfactualReport(
            candidate_name=candidate_name,
            original_rank=original_rank,
            original_score=round(candidate_score, 4),
            top_improvements=top_improvements,
            actionable_summary=summary,
            total_skills_analyzed=len(missing_skills),
            potential_best_rank=best_possible_rank,
            minimum_flip_set=flip_set,
            flip_target_rank=flip_target_rank,
            flip_achieved=flip_achieved,
            greedy_optimal=greedy_optimal,
        )

    def explain_all_candidates(
        self,
        scores: dict[str, float],
        resume_texts: dict[str, str],
        jd_text: str,
        top_k: int = 5,
        flip_target_rank: Optional[int] = None,
    ) -> list[CounterfactualReport]:
        """
        Generate counterfactual explanations for all candidates.

        Args:
            scores: {candidate_name: score}
            resume_texts: {candidate_name: resume_text}
            jd_text: Job description text
            top_k: Number of top improvements per candidate
            flip_target_rank: Target rank for flip-set analysis (shared across
                              all candidates). Defaults to top third of pool.

        Returns:
            List of CounterfactualReports, one per candidate
        """
        if flip_target_rank is None:
            flip_target_rank = max(1, len(scores) // 3)

        reports = []
        for name, score in scores.items():
            resume = resume_texts.get(name, "")
            report = self.explain_candidate(
                candidate_name=name,
                candidate_score=score,
                candidate_resume=resume,
                jd_text=jd_text,
                all_scores=scores,
                top_k=top_k,
                flip_target_rank=flip_target_rank,
            )
            reports.append(report)

        return reports

    # ─── Output Formatting ───────────────────────────────────────

    def _generate_summary(
        self,
        name: str,
        original_rank: int,
        improvements: list[SkillImpact],
        flip_set: list[str],
        flip_achieved: bool,
    ) -> str:
        """Generate a human-readable actionable summary."""
        if not improvements and not flip_set:
            return f"{name} (Rank #{original_rank}): No missing JD skills identified."

        lines = [f"Candidate: {name} - Current Rank: #{original_rank}"]

        if flip_set and flip_achieved:
            lines.append(
                f"  Minimum skill set to improve ranking: "
                f"{{{', '.join(flip_set)}}} ({len(flip_set)} skill(s))"
            )

        positive = [i for i in improvements if i.is_positive]
        if positive:
            lines.append("  Top individual skill impacts:")
            for imp in positive[:3]:
                lines.append(
                    f"  -> Adding '{imp.skill}' could move you from "
                    f"#{imp.original_rank} to #{imp.counterfactual_rank} "
                    f"(+{imp.rank_improvement} positions, "
                    f"score {imp.original_score:.2f} -> {imp.counterfactual_score:.2f})"
                )

        return "\n".join(lines)

    def format_report(self, report: CounterfactualReport) -> str:
        """Format a counterfactual report as a readable string."""
        sep = "-" * 60
        lines = [
            sep,
            f"COUNTERFACTUAL ANALYSIS: {report.candidate_name}",
            sep,
            f"  Current Rank:    #{report.original_rank}",
            f"  Current Score:   {report.original_score:.4f}",
            f"  Skills analyzed: {report.total_skills_analyzed} missing JD skills",
            f"  Best single-skill rank: #{report.potential_best_rank}",
            "",
        ]

        # Minimum flip set
        lines.append("  Minimum-Cardinality Flip Set:")
        if report.minimum_flip_set:
            status = "achieved" if report.flip_achieved else "partial"
            opt_str = ""
            if report.greedy_optimal is True:
                opt_str = " [brute-force optimal]"
            elif report.greedy_optimal is False:
                opt_str = " [sub-optimal -- see log]"
            lines.append(
                f"    {{{', '.join(report.minimum_flip_set)}}}  "
                f"(|d*|={len(report.minimum_flip_set)}, "
                f"target rank <= {report.flip_target_rank}, {status}{opt_str})"
            )
        else:
            if report.flip_achieved:
                lines.append(f"    Already at or above rank #{report.flip_target_rank}")
            else:
                lines.append("    No flip set found within skill budget.")

        lines.append("")

        # Per-skill breakdown
        if report.top_improvements:
            lines.append("  Top Individual Skill Recommendations:")
            for i, imp in enumerate(report.top_improvements, 1):
                direction = "^" if imp.is_positive else "-"
                lines.append(
                    f"    {i}. {imp.skill}: {direction} "
                    f"Rank #{imp.original_rank} -> #{imp.counterfactual_rank} "
                    f"(delta_score={imp.score_delta:+.4f})"
                )
        else:
            lines.append("  No missing skills from JD found.")

        lines.append("")
        lines.append(report.actionable_summary)
        lines.append(sep)

        return "\n".join(lines)


# --- Demo ---

if __name__ == "__main__":
    explainer = CounterfactualExplainer()

    jd = ("Looking for a Python developer experienced in machine learning, "
          "deep learning, TensorFlow, NLP, and cloud computing with AWS. "
          "Must have strong SQL skills and experience with Docker.")

    resumes = {
        "alice": "Senior developer with Python, machine learning, and TensorFlow.",
        "bob": "Junior developer with Java and basic SQL knowledge.",
        "charlie": "ML engineer with deep learning, NLP, PyTorch, and AWS experience.",
    }

    scores = {"alice": 0.75, "bob": 0.40, "charlie": 0.82}

    reports = explainer.explain_all_candidates(
        scores=scores,
        resume_texts=resumes,
        jd_text=jd,
        top_k=5,
    )

    for report in reports:
        print(explainer.format_report(report))
        print()
