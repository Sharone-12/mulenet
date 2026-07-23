"""Combine detector candidates into a ranked ring queue.

Every detector is weak alone - measured lift over the base rate is 3-14x. The
original plan was to treat an account tripping several *different* signals as
qualitatively more suspicious. That was tested and does not hold: see
KIND_PRIOR below. Ranking is by measured per-detector reliability instead.

Candidates are ranked, not merged. Output is a queue rather than a yes/no
verdict, because at a 0.17% base rate no threshold gives both precision and
recall. Investigators work an alert queue top-down, so precision@k is the
metric that matters.
"""

from __future__ import annotations

from dataclasses import dataclass



from .detectors import Candidate

# A merged group larger than this is not a ring, it is the detector chaining
# through a hub. Ground-truth rings top out at 27 accounts.
MAX_RING_ACCOUNTS = 60

# Measured rate at which each detector's candidates contain any ground-truth
# account. pass_through is 3-25x better than everything else, and this is the
# only feature found that ranks better than chance.
#
# Multi-signal corroboration was tried first and abandoned. Correlating an
# account's distinct-signal count against ground truth gives |pearson| < 0.012
# for every variant (max, mean, fraction >=2, fraction >=3), and ranking by
# any of them is WORSE than ranking by raw candidate score. The premise that
# "3+ signals means a ring" does not hold on this dataset.
KIND_PRIOR = {
    "pass_through": 0.088,
    "fan_out": 0.028,
    "fan_in": 0.017,
    "cycle": 0.006,
    "velocity_burst": 0.003,
}


@dataclass
class Ring:
    ring_id: int
    accounts: set[str]
    score: float
    signals: set[str]
    evidence: list[dict]


def combine(candidates: list[Candidate]) -> list[Ring]:
    """Rank candidates, corroborating each against the other detectors.

    Candidates are deliberately NOT merged transitively. An earlier version
    unioned every candidate sharing an account: because hub accounts appear in
    many candidates, that chained the whole graph into a handful of blobs,
    which then blew past MAX_RING_ACCOUNTS and were dropped - collapsing
    ground-truth recall from 939 accounts to 47. Each candidate is already a
    ring hypothesis (a star, a cycle, a pass-through chain), so the unit of
    ranking is the candidate itself.
    """
    if not candidates:
        return []

    # Which detector kinds touch each account, across all candidates.
    kinds_by_account: dict[str, set[str]] = {}
    for cand in candidates:
        for account in cand.accounts:
            kinds_by_account.setdefault(account, set()).add(cand.kind)

    best: dict[frozenset, Ring] = {}
    for cand in candidates:
        if len(cand.accounts) > MAX_RING_ACCOUNTS:
            continue
        key = frozenset(cand.accounts)

        # Signals are recorded for the investigator's evidence trail, but they
        # deliberately do not drive the ranking - see KIND_PRIOR.
        signals: set[str] = set()
        for account in cand.accounts:
            signals |= kinds_by_account.get(account, set())
        score = KIND_PRIOR.get(cand.kind, 0.003) * cand.score

        existing = best.get(key)
        if existing is None:
            best[key] = Ring(
                ring_id=0,
                accounts=set(cand.accounts),
                score=score,
                signals=signals,
                evidence=[{"kind": cand.kind, "score": round(cand.score, 3), **cand.evidence}],
            )
        else:
            existing.score = max(existing.score, score)
            existing.signals |= signals
            existing.evidence.append(
                {"kind": cand.kind, "score": round(cand.score, 3), **cand.evidence}
            )

    rings = sorted(best.values(), key=lambda r: r.score, reverse=True)
    for rank, ring in enumerate(rings, start=1):
        ring.ring_id = rank
    return rings


def top_k(rings: list[Ring], k: int) -> list[set[str]]:
    """The account sets of the k highest-scoring rings, for scoring."""
    return [r.accounts for r in rings[:k]]
