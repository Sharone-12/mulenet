"""Precomputed detection state, loaded once and served to the tools.

A full detection run is ~25s plus a graph build. An agent that takes half a
minute to answer "scan the network" is not demoable, so the ranked queue is
computed offline by scripts/build_rings.py and the tools only ever read it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache

import pandas as pd

from . import config, detectors, graph as graph_mod, loader, rings as rings_mod

RINGS_CACHE = "detected_rings"


def build(split: str = config.DEFAULT_SPLIT, cycle_length: int = 3) -> pd.DataFrame:
    """Run the full pipeline and persist the ranked queue."""
    labelled = loader.load_cache(split)
    search = detectors.prefilter(labelled)
    g = graph_mod.build_graph(search)

    candidates = (
        detectors.fan_out(search)
        + detectors.fan_in(search)
        + detectors.pass_through(search)
        + detectors.velocity_burst(search)
        + detectors.cycles(g, max_length=cycle_length)
    )
    ranked = rings_mod.combine(candidates)

    df = pd.DataFrame(
        [
            {
                "ring_id": r.ring_id,
                "score": r.score,
                "accounts": json.dumps(sorted(r.accounts)),
                "signals": json.dumps(sorted(r.signals)),
                "evidence": json.dumps(r.evidence, default=str),
                "account_count": len(r.accounts),
            }
            for r in ranked
        ]
    )
    path = config.cache_path(split, RINGS_CACHE)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return df


@dataclass
class Engine:
    """Everything the three tools need, held in memory."""

    # Full corpus, plus the prefiltered slice detection actually searched.
    transactions: pd.DataFrame
    search: pd.DataFrame
    rings: pd.DataFrame
    _cross_ring: dict[str, int] | None = None

    def ring(self, ring_id: int) -> dict | None:
        hit = self.rings[self.rings["ring_id"] == ring_id]
        if hit.empty:
            return None
        row = hit.iloc[0]
        return {
            "ring_id": int(row["ring_id"]),
            "score": float(row["score"]),
            "accounts": json.loads(row["accounts"]),
            "signals": json.loads(row["signals"]),
            "evidence": json.loads(row["evidence"]),
        }

    def cross_ring_counts(self) -> dict[str, int]:
        """How many detected rings each account appears in.

        Appearing in several rings is the strongest controller signal
        available, so this is computed once over the whole queue and reused.
        """
        if self._cross_ring is None:
            counts: dict[str, int] = {}
            for payload in self.rings["accounts"]:
                for account in json.loads(payload):
                    counts[account] = counts.get(account, 0) + 1
            self._cross_ring = counts
        return self._cross_ring

    def ring_transactions(self, accounts: list[str]) -> pd.DataFrame:
        """Transactions with both endpoints inside the ring, oldest first.

        Served from the same prefiltered slice detection ran on. Querying the
        raw frame instead pulls in self-payments and other rails, which are
        not evidence for a finding they played no part in - one 3-account ring
        reported 339,006 of "traced" flow when its actual movement was ~10K.
        """
        accts = set(accounts)
        df = self.search
        inside = df["from_node"].isin(accts) & df["to_node"].isin(accts)
        return df[inside].sort_values("timestamp")


@lru_cache(maxsize=1)
def load(split: str = config.DEFAULT_SPLIT) -> Engine:
    """Load the precomputed state. Cached, so the cost is paid once."""
    rings_path = config.cache_path(split, RINGS_CACHE)
    if not rings_path.exists():
        raise FileNotFoundError(
            f"{rings_path} missing - run scripts/build_rings.py first"
        )
    labelled = loader.load_cache(split)
    return Engine(
        transactions=labelled,
        search=detectors.prefilter(labelled),
        rings=pd.read_parquet(rings_path),
    )
