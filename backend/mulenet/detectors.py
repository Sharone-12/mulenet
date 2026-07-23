"""Ring detection.

Design notes, all forced by what LI-Small actually looks like:

* All 117 labelled rings sit inside the single 504K-node component, so there
  is no cheap way to narrow the search by connectivity. Detection has to be a
  local structure search, not a global partition. This is why Louvain
  community detection was dropped: at mean degree 1.7 it returns ~100K
  communities that are mostly innocent pairs.

* No individual signal separates rings from noise. Measured lift over the
  0.17% base rate: degree 3-8x, amount-matching 14x. Against that base rate a
  binary classifier cannot be precise, so detectors emit scored *candidates*
  and the output is a ranked alert queue evaluated at precision@k.

* Ground-truth rings have out-degree 3-23 while the graph maxes out at 18,942.
  The mega-hubs are banks. Every star detector is therefore band-limited at
  both ends - too big is exculpatory, not incriminating.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import networkx as nx
import numpy as np
import pandas as pd

# Star sizes worth considering. The lower bound is the smallest thing that is
# structurally a fan; the upper bound excludes bank operational accounts.
MIN_STAR = 3
MAX_STAR = 30

# Rings run for days (median duration 39-160h by pattern), so every temporal
# window here is in days. The plan's original 2-hour threshold matches nothing
# in this dataset.
WINDOW_HOURS = 72.0

# 99.9% of labelled ring transactions are ACH against 11.5% of the corpus, so
# restricting to ACH discards 88% of the search space for a 0.1% recall cost.
#
# Be honest about what this is: IBM injected its synthetic laundering onto the
# ACH rail, so part of the effect is the generator rather than criminology.
# Payment rail is a defensible AML feature in its own right, but this
# particular filter should not be presented as a discovered fact about real
# money laundering, and it would need re-validating on any other dataset.
LAUNDERING_RAILS = ("ACH",)


def prefilter(df: pd.DataFrame, rails: tuple[str, ...] = LAUNDERING_RAILS) -> pd.DataFrame:
    """Restrict to the payment rails worth searching, and drop self-loops."""
    out = df[df["from_node"] != df["to_node"]]
    if rails:
        out = out[out["payment_format"].isin(rails)]
    return out


@dataclass
class Candidate:
    """One suspicious structure, scored in [0, 1]."""

    accounts: set[str]
    kind: str
    score: float
    evidence: dict = field(default_factory=dict)


def _star_candidates(df: pd.DataFrame, hub_col: str, leaf_col: str, kind: str) -> list[Candidate]:
    """Fan-out (hub=from) or fan-in (hub=to) stars, band-limited by degree.

    Fully vectorised: a per-hub Python loop over the ~75K hubs that clear the
    degree band does not finish in ten minutes.
    """
    real = df[df["from_node"] != df["to_node"]]
    counts = real.groupby(hub_col, sort=False)[leaf_col].nunique()
    hubs = counts[(counts >= MIN_STAR) & (counts <= MAX_STAR)].index
    if len(hubs) == 0:
        return []

    sub = real[real[hub_col].isin(hubs)]
    grouped = sub.groupby(hub_col, sort=False)
    stats = grouped.agg(
        first=("timestamp", "min"),
        last=("timestamp", "max"),
        mean_amt=("amount_paid", "mean"),
        std_amt=("amount_paid", "std"),
        total=("amount_paid", "sum"),
        n=("amount_paid", "size"),
    )
    stats["degree"] = counts.loc[stats.index]

    span_h = (stats["last"] - stats["first"]).dt.total_seconds() / 3600.0
    tight = (1.0 - span_h / (WINDOW_HOURS * 2)).clip(lower=0.0)
    cv = (stats["std_amt"] / stats["mean_amt"]).fillna(0.0)
    uniform = (1.0 - cv).clip(lower=0.0)
    # Structure alone is common; structure plus timing plus near-equal
    # slicing is rare.
    stats["score"] = 0.5 * tight + 0.5 * uniform
    stats["tight"] = tight
    stats["uniform"] = uniform
    stats = stats[(stats["score"] > 0) & (stats["n"] > 1)]
    if stats.empty:
        return []

    leaves = grouped[leaf_col].apply(set)
    return [
        Candidate(
            accounts=leaves[hub] | {hub},
            kind=kind,
            score=float(row.score),
            evidence={
                "hub": hub,
                "degree": int(row.degree),
                "tightness": round(float(row.tight), 3),
                "uniformity": round(float(row.uniform), 3),
                "total_amount": float(row.total),
            },
        )
        for hub, row in stats.iterrows()
    ]


def fan_out(df: pd.DataFrame) -> list[Candidate]:
    """One account paying an abnormal number of others in a short window."""
    return _star_candidates(df, "from_node", "to_node", "fan_out")


def fan_in(df: pd.DataFrame) -> list[Candidate]:
    """Many accounts converging on one - the cash-out end of a ring."""
    return _star_candidates(df, "to_node", "from_node", "fan_in")


def pass_through(
    df: pd.DataFrame, tolerance: float = 0.10, max_lag_hours: float = 240.0
) -> list[Candidate]:
    """Accounts where money arrives and near-identical money leaves.

    A real account holds a balance; a mule is a pipe. Volume is capped before
    the self-join because a degree-18,942 hub would otherwise generate a
    360M-row cartesian product.
    """
    real = df[df["from_node"] != df["to_node"]]
    outs = real.rename(
        columns={"from_node": "node", "amount_paid": "amt_out", "timestamp": "t_out"}
    )[["node", "amt_out", "t_out", "to_node"]]
    ins = real.rename(
        columns={"to_node": "node", "amount_paid": "amt_in", "timestamp": "t_in"}
    )[["node", "amt_in", "t_in", "from_node"]]

    volume_cap = MAX_STAR * 2
    ok = set(outs.groupby("node").size().pipe(lambda s: s[s <= volume_cap]).index) & set(
        ins.groupby("node").size().pipe(lambda s: s[s <= volume_cap]).index
    )
    outs = outs[outs["node"].isin(ok)]
    ins = ins[ins["node"].isin(ok)]

    pairs = ins.merge(outs, on="node")
    lag = (pairs["t_out"] - pairs["t_in"]).dt.total_seconds() / 3600.0
    ratio = pairs["amt_out"] / pairs["amt_in"].replace(0, np.nan)
    pairs = pairs[
        (lag >= 0) & (lag <= max_lag_hours) & ratio.between(1 - tolerance, 1 + tolerance)
    ]
    if pairs.empty:
        return []
    pairs = pairs.assign(lag=lag[pairs.index], ratio=ratio[pairs.index])

    out: list[Candidate] = []
    for node, g in pairs.groupby("node", sort=False):
        best = g.iloc[(g["ratio"] - 1.0).abs().argmin()]
        # Tighter amount match and faster turnaround both raise suspicion.
        amount_fit = 1.0 - abs(best["ratio"] - 1.0) / tolerance
        speed = 1.0 - best["lag"] / max_lag_hours
        out.append(
            Candidate(
                accounts={best["from_node"], node, best["to_node"]},
                kind="pass_through",
                score=float(0.6 * amount_fit + 0.4 * speed),
                evidence={
                    "node": node,
                    "amount_in": float(best["amt_in"]),
                    "amount_out": float(best["amt_out"]),
                    "hold_hours": round(float(best["lag"]), 1),
                },
            )
        )
    return out


def cycles(g: nx.DiGraph, max_length: int = 4) -> list[Candidate]:
    """Money returning to where it started.

    Directly targets CYCLE, STACK, SCATTER-GATHER and GATHER-SCATTER, which
    together are 41 of the 88 detectable rings.
    """
    out: list[Candidate] = []
    seen: set[frozenset] = set()
    for cycle in nx.simple_cycles(g, length_bound=max_length):
        if len(cycle) < 2:
            continue
        key = frozenset(cycle)
        if key in seen:
            continue
        seen.add(key)
        # A 2-cycle is a bare round trip and far more common than a longer
        # loop, so it scores lower.
        out.append(
            Candidate(
                accounts=set(cycle),
                kind="cycle",
                score=0.4 if len(cycle) == 2 else 0.8,
                evidence={"length": len(cycle), "path": list(cycle)},
            )
        )
    return out


def velocity_burst(df: pd.DataFrame, window_hours: float = 24.0) -> list[Candidate]:
    """Accounts whose lifetime activity collapses into one short window.

    Replaces the planned dormant-burst detector, which assumed a 90-day
    dormancy the 10-day dataset cannot express.
    """
    real = df[df["from_node"] != df["to_node"]]
    moves = pd.concat(
        [
            real[["from_node", "amount_paid", "timestamp"]].rename(
                columns={"from_node": "node"}
            ),
            real[["to_node", "amount_paid", "timestamp"]].rename(columns={"to_node": "node"}),
        ],
        ignore_index=True,
    )
    stats = moves.groupby("node", sort=False).agg(
        n=("amount_paid", "size"),
        total=("amount_paid", "sum"),
        first=("timestamp", "min"),
        last=("timestamp", "max"),
    )
    stats = stats[(stats["n"] >= MIN_STAR) & (stats["n"] <= MAX_STAR * 2)]
    span = (stats["last"] - stats["first"]).dt.total_seconds() / 3600.0
    burst = stats[(span > 0) & (span <= window_hours)]

    return [
        Candidate(
            accounts={node},
            kind="velocity_burst",
            score=float(1.0 - row_span / window_hours),
            evidence={"transactions": int(row.n), "total_amount": float(row.total)},
        )
        for (node, row), row_span in zip(burst.iterrows(), span[burst.index])
    ]
