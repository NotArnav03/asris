"""
FAIMR — Fairness-Constrained Re-ranking Module
Implements a post-processing re-ranker that balances relevance
with demographic parity using constrained optimization.

Novel contribution: Greedy swap-based algorithm that satisfies
the 4/5 Rule (AIR ≥ 0.8) while minimizing Kendall-τ displacement
from the original relevance ranking.
"""

import numpy as np
from typing import Optional
from dataclasses import dataclass, field
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import FAIRNESS_ADVERSE_IMPACT_THRESHOLD, get_logger

logger = get_logger("ranking.fairness_ranker")


@dataclass
class RankedCandidate:
    """A candidate with score and demographic info."""
    name: str
    score: float
    group: str  # demographic group label
    original_rank: int = 0
    new_rank: int = 0


@dataclass
class FairnessReport:
    """Report from the fairness-constrained re-ranking."""
    original_air: float = 0.0
    final_air: float = 0.0
    displacement_cost: float = 0.0
    max_displacement: int = 0
    num_swaps: int = 0
    original_ranking: list = field(default_factory=list)
    fair_ranking: list = field(default_factory=list)
    pareto_points: list = field(default_factory=list)
    fairness_satisfied: bool = False
    group_stats: dict = field(default_factory=dict)


class FairnessConstrainedRanker:
    """
    Post-processing fairness-constrained re-ranker.

    Given a relevance-ranked list and demographic group labels,
    produces a re-ranked list that satisfies the Adverse Impact
    Ratio (AIR) threshold at every prefix k of the ranking, while
    minimizing the total rank displacement (Kendall-τ distance).

    Algorithm:
        1. Scan the ranking top-down at each cutoff k.
        2. At each k, compute per-group selection rates.
        3. If AIR < threshold, find the highest-scored candidate
           from the underrepresented group below rank k.
        4. Swap them with the lowest-scored candidate from the
           overrepresented group at rank k.
        5. Repeat until AIR is satisfied or no valid swaps remain.
        6. Record displacement cost and Pareto frontier.
    """

    def __init__(
        self,
        threshold: float = FAIRNESS_ADVERSE_IMPACT_THRESHOLD,
        quality_threshold: float = 0.5,
    ):
        """
        Args:
            threshold: Minimum AIR to enforce (default 0.8 = 4/5 rule).
            quality_threshold: Minimum score ratio for a swap to be accepted.
                A swap is only performed if:
                    score(underrep_candidate) >= quality_threshold * score(overrep_candidate)
                This prevents extreme promotions (e.g., a 0.3-score candidate
                displacing a 0.9-score candidate), preserving the system's
                predictive validity and interpretability.  The default value
                of 0.5 is conservative: it allows a 2:1 score gap but blocks
                swaps where the quality difference would be indefensible to
                auditors.  Setting quality_threshold=0.0 disables the
                constraint and allows any swap that satisfies the AIR target.
        """
        self.threshold = threshold
        self.quality_threshold = quality_threshold

    # ─── Core Algorithm ──────────────────────────────────────────

    def rerank(
        self,
        candidates: list[RankedCandidate],
        min_group_size: int = 2,
        _compute_pareto: bool = True,
    ) -> FairnessReport:
        """
        Apply fairness-constrained re-ranking.

        Args:
            candidates: List of RankedCandidate, sorted by score descending.
            min_group_size: Minimum group size to enforce fairness
                            (skip groups smaller than this).

        Returns:
            FairnessReport with original and fair rankings.
        """
        if len(candidates) < 2:
            return FairnessReport(
                fairness_satisfied=True,
                original_ranking=[c.name for c in candidates],
                fair_ranking=[c.name for c in candidates],
            )

        # Assign original ranks
        for i, c in enumerate(candidates):
            c.original_rank = i + 1

        # Get valid groups (above min_group_size)
        group_counts = {}
        for c in candidates:
            group_counts[c.group] = group_counts.get(c.group, 0) + 1

        valid_groups = {g for g, cnt in group_counts.items()
                        if cnt >= min_group_size and g != "unknown"}

        if len(valid_groups) < 2:
            # Not enough groups to enforce fairness
            for i, c in enumerate(candidates):
                c.new_rank = i + 1
            return FairnessReport(
                fairness_satisfied=True,
                original_ranking=[c.name for c in candidates],
                fair_ranking=[c.name for c in candidates],
                group_stats=group_counts,
            )

        # Compute original AIR
        original_air = self._compute_air_at_k(
            candidates, len(candidates), valid_groups
        )

        # Greedy swap-based re-ranking
        fair_list = list(candidates)  # shallow copy
        num_swaps = 0
        max_swaps = len(candidates) * 2  # safety limit

        for iteration in range(max_swaps):
            # Find the first prefix k where AIR is violated
            violation_k = self._find_first_violation(fair_list, valid_groups)

            if violation_k is None:
                break  # All prefixes satisfy fairness

            # Identify under/over-represented groups at k
            underrep, overrep = self._identify_groups_at_k(
                fair_list, violation_k, valid_groups
            )

            if underrep is None or overrep is None:
                break

            # Find best swap: highest-scored underrep below k ↔ lowest-scored overrep at k
            swap_done = self._perform_swap(
                fair_list, violation_k, underrep, overrep,
                quality_threshold=self.quality_threshold,
            )

            if not swap_done:
                break

            num_swaps += 1

        # Assign new ranks
        for i, c in enumerate(fair_list):
            c.new_rank = i + 1

        # Compute metrics
        final_air = self._compute_air_at_k(
            fair_list, len(fair_list), valid_groups
        )
        displacement = self.displacement_cost(candidates, fair_list)
        max_disp = self.max_individual_displacement(candidates, fair_list)

        # Pareto frontier (skip during internal calls to avoid recursion)
        pareto = (self._compute_pareto_frontier(candidates, valid_groups)
                  if _compute_pareto else [])

        # Group statistics
        group_stats = self._compute_group_stats(fair_list, valid_groups)

        report = FairnessReport(
            original_air=round(original_air, 4),
            final_air=round(final_air, 4),
            displacement_cost=round(displacement, 4),
            max_displacement=max_disp,
            num_swaps=num_swaps,
            original_ranking=[c.name for c in candidates],
            fair_ranking=[c.name for c in fair_list],
            pareto_points=pareto,
            fairness_satisfied=final_air >= self.threshold,
            group_stats=group_stats,
        )

        logger.info(
            f"FCR complete: AIR {original_air:.3f} → {final_air:.3f}, "
            f"{num_swaps} swaps, displacement={displacement:.3f}"
        )

        return report

    # ─── AIR Computation ─────────────────────────────────────────

    def _compute_air_at_k(
        self,
        ranked: list[RankedCandidate],
        k: int,
        valid_groups: set,
    ) -> float:
        """Compute Adverse Impact Ratio at prefix k."""
        top_k = ranked[:k]
        total_counts = {}
        selected_counts = {}

        for c in ranked:
            if c.group in valid_groups:
                total_counts[c.group] = total_counts.get(c.group, 0) + 1

        for c in top_k:
            if c.group in valid_groups:
                selected_counts[c.group] = selected_counts.get(c.group, 0) + 1

        rates = {}
        for g in valid_groups:
            total = total_counts.get(g, 0)
            selected = selected_counts.get(g, 0)
            rates[g] = selected / total if total > 0 else 0

        if not rates or max(rates.values()) == 0:
            return 1.0

        min_rate = min(rates.values())
        max_rate = max(rates.values())

        return min_rate / max_rate if max_rate > 0 else 1.0

    def _find_first_violation(
        self,
        ranked: list[RankedCandidate],
        valid_groups: set,
    ) -> Optional[int]:
        """Find the first prefix k where AIR < threshold."""
        # Check at meaningful cutoff points (every candidate addition)
        min_k = sum(1 for g in valid_groups for _ in range(1))  # at least one per group
        for k in range(max(min_k, 3), len(ranked) + 1):
            air = self._compute_air_at_k(ranked, k, valid_groups)
            if air < self.threshold:
                return k
        return None

    def _identify_groups_at_k(
        self,
        ranked: list[RankedCandidate],
        k: int,
        valid_groups: set,
    ) -> tuple:
        """Identify under- and over-represented groups at prefix k."""
        top_k = ranked[:k]
        total_counts = {}
        selected_counts = {}

        for c in ranked:
            if c.group in valid_groups:
                total_counts[c.group] = total_counts.get(c.group, 0) + 1

        for c in top_k:
            if c.group in valid_groups:
                selected_counts[c.group] = selected_counts.get(c.group, 0) + 1

        rates = {}
        for g in valid_groups:
            total = total_counts.get(g, 0)
            selected = selected_counts.get(g, 0)
            rates[g] = selected / total if total > 0 else 0

        if not rates:
            return None, None

        underrep = min(rates, key=rates.get)
        overrep = max(rates, key=rates.get)

        if underrep == overrep:
            return None, None

        return underrep, overrep

    def _perform_swap(
        self,
        ranked: list[RankedCandidate],
        k: int,
        underrep: str,
        overrep: str,
        quality_threshold: float = 0.5,
    ) -> bool:
        """
        Swap the best underrep candidate below k with the worst
        overrep candidate within top-k.
        """
        # Find highest-scored underrep below position k
        best_underrep_idx = None
        best_underrep_score = -float('inf')

        for i in range(k, len(ranked)):
            if ranked[i].group == underrep and ranked[i].score > best_underrep_score:
                best_underrep_score = ranked[i].score
                best_underrep_idx = i

        # Find lowest-scored overrep within top-k
        worst_overrep_idx = None
        worst_overrep_score = float('inf')

        for i in range(k - 1, -1, -1):
            if ranked[i].group == overrep and ranked[i].score < worst_overrep_score:
                worst_overrep_score = ranked[i].score
                worst_overrep_idx = i

        if best_underrep_idx is None or worst_overrep_idx is None:
            return False

        # Quality gate: only swap if the underrep candidate's score is at least
        # quality_threshold × the overrep candidate's score.  This prevents
        # promotions so large that they undermine the system's predictive
        # validity (e.g., rank-1 displaced by rank-n with a 3× lower score).
        # The constraint is a design choice, not a mathematical requirement for
        # the AIR guarantee — setting quality_threshold=0.0 disables it.
        if best_underrep_score < worst_overrep_score * self.quality_threshold:
            return False

        # Perform swap
        ranked[best_underrep_idx], ranked[worst_overrep_idx] = (
            ranked[worst_overrep_idx], ranked[best_underrep_idx]
        )

        return True

    # ─── Displacement Metrics ────────────────────────────────────

    @staticmethod
    def displacement_cost(
        original: list[RankedCandidate],
        reranked: list[RankedCandidate],
    ) -> float:
        """
        Compute normalized Kendall-τ displacement cost.
        Measures how much the ranking changed (0 = identical, 1 = fully reversed).
        """
        n = len(original)
        if n <= 1:
            return 0.0

        # Build rank mapping
        orig_rank = {c.name: i for i, c in enumerate(original)}
        new_rank = {c.name: i for i, c in enumerate(reranked)}

        total_displacement = sum(
            abs(orig_rank[name] - new_rank[name])
            for name in orig_rank
        )

        # Normalize by maximum possible displacement
        max_displacement = n * n // 2
        return total_displacement / max_displacement if max_displacement > 0 else 0.0

    @staticmethod
    def max_individual_displacement(
        original: list[RankedCandidate],
        reranked: list[RankedCandidate],
    ) -> int:
        """Maximum rank change for any single candidate."""
        orig_rank = {c.name: i for i, c in enumerate(original)}
        new_rank = {c.name: i for i, c in enumerate(reranked)}

        return max(
            abs(orig_rank[name] - new_rank[name])
            for name in orig_rank
        ) if orig_rank else 0

    # ─── Pareto Frontier ─────────────────────────────────────────

    def _compute_pareto_frontier(
        self,
        candidates: list[RankedCandidate],
        valid_groups: set,
    ) -> list[dict]:
        """
        Compute the fairness-relevance Pareto frontier.
        Sweeps AIR thresholds from 0.5 to 1.0 and records
        the displacement cost at each level.
        """
        pareto_points = []
        thresholds = [0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]

        for t in thresholds:
            ranker = FairnessConstrainedRanker(threshold=t)
            candidates_copy = [
                RankedCandidate(c.name, c.score, c.group)
                for c in candidates
            ]
            report = ranker.rerank(candidates_copy, min_group_size=2, _compute_pareto=False)

            pareto_points.append({
                "threshold": t,
                "achieved_air": report.final_air,
                "displacement_cost": report.displacement_cost,
                "num_swaps": report.num_swaps,
            })

        return pareto_points

    # ─── Group Statistics ────────────────────────────────────────

    @staticmethod
    def _compute_group_stats(
        ranked: list[RankedCandidate],
        valid_groups: set,
    ) -> dict:
        """Compute per-group ranking statistics."""
        stats = {}
        for g in valid_groups:
            group_candidates = [c for c in ranked if c.group == g]
            if group_candidates:
                ranks = [c.new_rank for c in group_candidates]
                scores = [c.score for c in group_candidates]
                stats[g] = {
                    "count": len(group_candidates),
                    "mean_rank": round(np.mean(ranks), 2),
                    "mean_score": round(np.mean(scores), 4),
                    "median_rank": round(np.median(ranks), 2),
                    "top_3_count": sum(1 for r in ranks if r <= 3),
                }
        return stats

    # ─── Convenience ─────────────────────────────────────────────

    @classmethod
    def from_scores_and_groups(
        cls,
        names: list[str],
        scores: list[float],
        groups: list[str],
        threshold: float = FAIRNESS_ADVERSE_IMPACT_THRESHOLD,
    ) -> FairnessReport:
        """
        Convenience method: create candidates from parallel lists
        and run the re-ranking.

        Args:
            names: Candidate names/filenames
            scores: Relevance scores (higher = better)
            groups: Demographic group labels
            threshold: AIR threshold

        Returns:
            FairnessReport
        """
        # Sort by score descending
        indexed = sorted(
            zip(names, scores, groups),
            key=lambda x: x[1],
            reverse=True,
        )

        candidates = [
            RankedCandidate(name=n, score=s, group=g)
            for n, s, g in indexed
        ]

        ranker = cls(threshold=threshold)
        return ranker.rerank(candidates)


# ─── Demo / Test ─────────────────────────────────────────────────

if __name__ == "__main__":
    # Simulate a biased ranking scenario
    candidates = [
        RankedCandidate("Alice", 0.95, "female"),
        RankedCandidate("Bob", 0.92, "male"),
        RankedCandidate("Charlie", 0.88, "male"),
        RankedCandidate("David", 0.85, "male"),
        RankedCandidate("Eve", 0.83, "female"),
        RankedCandidate("Frank", 0.80, "male"),
        RankedCandidate("Grace", 0.78, "female"),
        RankedCandidate("Henry", 0.75, "male"),
        RankedCandidate("Ivy", 0.72, "female"),
        RankedCandidate("Jack", 0.70, "male"),
    ]

    ranker = FairnessConstrainedRanker(threshold=0.8)
    report = ranker.rerank(candidates)

    sep = "=" * 55
    print(sep)
    print("  FAIRNESS-CONSTRAINED RE-RANKING REPORT")
    print(sep)
    print(f"  Original AIR: {report.original_air:.4f}")
    print(f"  Final AIR:    {report.final_air:.4f}")
    print(f"  Swaps made:   {report.num_swaps}")
    print(f"  Displacement: {report.displacement_cost:.4f}")
    print(f"  Max displacement: {report.max_displacement}")
    print(f"  Fairness satisfied: {'YES' if report.fairness_satisfied else 'NO'}")
    print()
    print(f"  Original:  {report.original_ranking}")
    print(f"  Fair:      {report.fair_ranking}")
    print()
    print(f"  Group stats: {report.group_stats}")
    print()
    print("  Pareto frontier:")
    for p in report.pareto_points:
        print(f"    t={p['threshold']:.2f} -> AIR={p['achieved_air']:.3f}, "
              f"cost={p['displacement_cost']:.4f}, swaps={p['num_swaps']}")
    print(sep)
