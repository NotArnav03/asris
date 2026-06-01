"""
FA*IR baseline (Zehlike, Bonchi, Castillo, Hajian, Megahed, Baeza-Yates,
CIKM 2017): a fair top-k ranking algorithm that enforces a
statistically-significant minimum representation of a protected group
at every prefix.

Reference:
    Zehlike et al., "FA*IR: A Fair Top-k Ranking Algorithm",
    CIKM 2017, arXiv:1706.06368.

Implementation notes:

  * `m_alpha_table(k, p, alpha)` returns the per-prefix floor: at each
    prefix length k', the protected count must be >= m_table[k'].  The
    floor is the binomial-CDF inverse:

        m_table[k'] = F^{-1}_{Binom(k', p)}(alpha)

    so that under the null hypothesis "candidates drawn i.i.d. with
    protected probability p", a protected count below m_table[k']
    happens with probability <= alpha.  This is the same statistical
    floor the original FA*IR paper uses; for clarity we use the
    UN-adjusted alpha rather than the multiple-comparisons-adjusted
    alpha-c from the paper (the difference is small for k <= 1000 and
    not relevant for the head-to-head comparison vs FAIMR FCR).

  * The greedy selection step picks queue heads in score order, with
    the constraint that if `protected_count < m_table[next_pos]` we
    MUST pick from the protected queue.  This preserves within-group
    score order by construction (the protected/non-protected queues
    are pre-sorted by score).

  * Returns the re-ranked list AND a per-prefix audit log so the
    benchmark can verify the statistical floor is satisfied.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from scipy.stats import binom


@dataclass(frozen=True)
class FairItem:
    """Item with a score and protected-attribute label."""
    id: str
    score: float
    is_protected: bool


def m_alpha_table(k_max: int, p: float, alpha: float) -> list[int]:
    """Return m_table[k] = minimum protected count required at prefix k.

    m_table[0] is always 0.  For k >= 1, m_table[k] is the
    `binom.ppf(alpha, k, p)` -- the smallest m such that
    P(Binom(k, p) <= m) >= alpha.

    Args:
        k_max: list length to build the table for.
        p:     target proportion of the protected group.
        alpha: per-prefix significance level (smaller = stricter).
    """
    out = [0]
    for kk in range(1, k_max + 1):
        floor = int(binom.ppf(alpha, kk, p))
        out.append(max(0, floor))
    return out


def fair_rerank(
    items: Sequence[FairItem],
    p: float,
    alpha: float = 0.1,
    k: int | None = None,
) -> tuple[list[FairItem], dict]:
    """FA*IR fair top-k re-ranking.

    Args:
        items: list of FairItem sorted by score DESCENDING.
        p:     target proportion of the protected group.
        alpha: per-prefix significance level.
        k:     output list length (defaults to len(items)).

    Returns:
        ranked, audit
          ranked : the re-ranked list of FairItem
          audit  : dict with diagnostics --
              * m_table             : the floor list
              * prefix_protected    : protected count at every prefix
              * prefix_floor_ok     : bool per prefix
              * fail_first_prefix   : index of first failing prefix
                                       (None if all satisfied)
              * within_group_order_preserved : bool
              * n_protected, n_nonprotected : queue sizes consumed
    """
    if k is None:
        k = len(items)
    m_table = m_alpha_table(k, p, alpha)

    protected = [it for it in items if it.is_protected]
    nonprotected = [it for it in items if not it.is_protected]

    out: list[FairItem] = []
    pi = ni = 0
    pcount = 0
    for pos in range(1, k + 1):
        floor = m_table[pos]
        must_pick_protected = pcount < floor
        p_head = protected[pi] if pi < len(protected) else None
        n_head = nonprotected[ni] if ni < len(nonprotected) else None

        if must_pick_protected and p_head is not None:
            out.append(p_head); pi += 1; pcount += 1
        elif must_pick_protected and p_head is None:
            # Constraint impossible to satisfy here -- emit what's left.
            if n_head is None:
                break
            out.append(n_head); ni += 1
        else:
            if p_head is None and n_head is None:
                break
            if p_head is None:
                out.append(n_head); ni += 1
            elif n_head is None:
                out.append(p_head); pi += 1; pcount += 1
            elif p_head.score > n_head.score:
                out.append(p_head); pi += 1; pcount += 1
            else:
                out.append(n_head); ni += 1

    # Audit: count protected at every prefix.
    prefix_protected: list[int] = []
    running = 0
    for it in out:
        if it.is_protected:
            running += 1
        prefix_protected.append(running)
    prefix_floor_ok = [
        pp >= m_table[pos]
        for pos, pp in enumerate(prefix_protected, start=1)
    ]
    fail_first = next(
        (i + 1 for i, ok in enumerate(prefix_floor_ok) if not ok),
        None,
    )

    # Within-group order preservation: protected items in `out` should
    # appear in the same score order as in the input protected queue.
    protected_in_out = [it for it in out if it.is_protected]
    nonprotected_in_out = [it for it in out if not it.is_protected]
    within_group_order_preserved = (
        protected_in_out == protected[: len(protected_in_out)]
        and nonprotected_in_out == nonprotected[: len(nonprotected_in_out)]
    )

    return out, {
        "m_table":             m_table,
        "prefix_protected":    prefix_protected,
        "prefix_floor_ok":     prefix_floor_ok,
        "fail_first_prefix":   fail_first,
        "within_group_order_preserved": within_group_order_preserved,
        "n_protected_emitted": pi,
        "n_nonprotected_emitted": ni,
    }
