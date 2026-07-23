"""The three tools the agent calls.

Each returns plain JSON-serialisable data. They are ordinary Python functions
so the agent loop, the tests and (later) the FastAPI layer can all share them
without going through HTTP.
"""

from __future__ import annotations

import json
from collections import Counter

import pandas as pd

from . import config, engine

# Roles are assigned by heuristic, NOT by a validated classifier. IBM never
# labels controller-vs-mule, and the only ground-truth proxy available is the
# 49 accounts appearing in more than one labelled ring - far too few to train
# or properly evaluate on. Treat the output as investigative triage and say so
# when presenting it.
CONTROLLER_THRESHOLD = 0.5


def _fmt_amount(x: float) -> float:
    return round(float(x), 2)


def _by_currency(txns: pd.DataFrame) -> dict[str, float]:
    """Totals split per currency.

    A bare total invites the model to render it with a currency symbol it
    picked itself - the agent reported a Euro ring as "$7,178.56". Amounts are
    also genuinely not summable across currencies, though only ~1% of rings
    mix them.
    """
    grouped = txns.groupby("payment_currency")["amount_paid"].sum()
    return {str(k): _fmt_amount(v) for k, v in grouped.items()}


def scan_network(top_k: int = 25, split: str = config.DEFAULT_SPLIT) -> dict:
    """Summarise the highest-scoring rings in the detection queue."""
    eng = engine.load(split)
    head = eng.rings.head(top_k)

    rings = []
    for _, row in head.iterrows():
        accounts = json.loads(row["accounts"])
        signals = json.loads(row["signals"])
        txns = eng.ring_transactions(accounts)
        rings.append(
            {
                "ring_id": int(row["ring_id"]),
                "accounts": len(accounts),
                "transactions": len(txns),
                "amount_by_currency": _by_currency(txns),
                "signals": signals,
                "score": round(float(row["score"]), 4),
            }
        )

    flagged = set()
    traced: Counter = Counter()
    for _, row in head.iterrows():
        flagged |= set(json.loads(row["accounts"]))
    for ring in rings:
        traced.update(ring["amount_by_currency"])

    return {
        "rings_detected": len(rings),
        "rings_in_queue": int(len(eng.rings)),
        "total_accounts_flagged": len(flagged),
        "total_traced_by_currency": {k: _fmt_amount(v) for k, v in traced.items()},
        "rings": rings,
    }


def investigate_ring(ring_id: int, split: str = config.DEFAULT_SPLIT) -> dict:
    """Full breakdown of one ring: flow, timing, and every transaction."""
    eng = engine.load(split)
    ring = eng.ring(ring_id)
    if ring is None:
        return {"error": f"ring {ring_id} not found"}

    accounts = ring["accounts"]
    txns = eng.ring_transactions(accounts)
    if txns.empty:
        return {"error": f"ring {ring_id} has no internal transactions"}

    inflow = Counter(txns["to_node"])
    outflow = Counter(txns["from_node"])
    sources = [a for a in accounts if outflow[a] and not inflow[a]]
    cashouts = [a for a in accounts if inflow[a] and not outflow[a]]

    duration = (txns["timestamp"].max() - txns["timestamp"].min()).total_seconds() / 3600

    return {
        "ring_id": ring_id,
        "accounts": accounts,
        "account_count": len(accounts),
        "source_accounts": sources,
        "cashout_accounts": cashouts,
        "amount_by_currency": _by_currency(txns),
        "transaction_count": len(txns),
        "duration_hours": round(duration, 1),
        "first_seen": str(txns["timestamp"].min()),
        "last_seen": str(txns["timestamp"].max()),
        "signals": ring["signals"],
        "currencies": sorted(set(txns["payment_currency"].dropna())),
        "transactions": [
            {
                "from": r.from_node,
                "to": r.to_node,
                "amount": _fmt_amount(r.amount_paid),
                "currency": r.payment_currency,
                "timestamp": str(r.timestamp),
            }
            # Long rings would blow the model's context; the agent can ask for
            # more if it needs them.
            for r in txns.head(50).itertuples()
        ],
        "transactions_truncated": max(0, len(txns) - 50),
    }


def _hold_hours(txns: pd.DataFrame, account: str) -> float | None:
    """Time between money first arriving and first leaving."""
    got = txns[txns["to_node"] == account]["timestamp"]
    sent = txns[txns["from_node"] == account]["timestamp"]
    if got.empty or sent.empty:
        return None
    after = sent[sent >= got.min()]
    if after.empty:
        return None
    return round((after.min() - got.min()).total_seconds() / 3600, 1)


def classify_roles(ring_id: int, split: str = config.DEFAULT_SPLIT) -> dict:
    """Separate likely controllers from likely recruited mules.

    Heuristic scoring - see CONTROLLER_THRESHOLD. Controller weight comes from
    sitting at an end of the flow (originating funds or taking the cash-out)
    and from appearing in more than one detected ring. Mules are
    pass-through intermediaries that appear once and hold briefly.
    """
    eng = engine.load(split)
    ring = eng.ring(ring_id)
    if ring is None:
        return {"error": f"ring {ring_id} not found"}

    accounts = ring["accounts"]
    txns = eng.ring_transactions(accounts)
    if txns.empty:
        return {"error": f"ring {ring_id} has no internal transactions"}

    cross_ring = eng.cross_ring_counts()
    inflow = Counter(txns["to_node"])
    outflow = Counter(txns["from_node"])
    first_move = txns.groupby("from_node")["timestamp"].min()
    earliest = txns["timestamp"].min()

    controllers, mules = [], []
    for account in accounts:
        gets, sends = inflow[account], outflow[account]
        rings_seen = int(cross_ring.get(account, 1))

        if sends and not gets:
            position = "source"
        elif gets and not sends:
            position = "cash_out"
        elif gets and sends:
            position = "intermediary"
        else:
            position = "peripheral"

        score = 0.0
        if position in ("source", "cash_out"):
            score += 0.4
        if rings_seen > 1:
            score += 0.3
        # Controllers set the pace; mules react to it.
        if account in first_move.index and first_move[account] == earliest:
            score += 0.3

        entry = {
            "account": account,
            "position": position,
            "rings_appeared": rings_seen,
            "hold_hours": _hold_hours(txns, account),
            "received": _fmt_amount(txns[txns["to_node"] == account]["amount_paid"].sum()),
            "sent": _fmt_amount(txns[txns["from_node"] == account]["amount_paid"].sum()),
            "confidence": round(score, 2),
        }

        if score >= CONTROLLER_THRESHOLD:
            entry["role"] = f"{position}_controller"
            entry["action"] = "freeze_and_investigate"
            controllers.append(entry)
        else:
            entry["role"] = "recruited_mule"
            entry["action"] = "contact_and_warn"
            mules.append(entry)

    controllers.sort(key=lambda e: e["confidence"], reverse=True)
    return {
        "ring_id": ring_id,
        "controllers": controllers,
        "mules": mules,
        "method": "heuristic scoring, not a validated classifier",
    }
