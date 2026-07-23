"""Both engines measured on identical terms.

The rule engine works per transaction; MuleNet works per ring. Comparing
"2 million flags" against "25 rings" is not a comparison at all, so this
flattens MuleNet's output onto the same transaction-level basis: a transaction
counts as flagged when both of its accounts sit inside a detected ring.

Everything is scored against the 3,565 rows IBM labels `is_laundering = 1`,
which is the fair denominator for a transaction-level engine.

Evaluation only - nothing here feeds detection.
"""

from __future__ import annotations

import json
from functools import lru_cache

import pandas as pd

from . import config, engine, rule_engine

# Queue depths worth reporting. An investigator works from the top, so the
# shallow rows matter more than the full queue.
DEPTHS = (10, 25, 100, 1000)


def _measure(flagged: pd.Series, truth: pd.Series, total: int) -> dict:
    """Precision, recall and workload for one set of flagged transactions."""
    n_flagged = int(flagged.sum())
    caught = int((flagged & truth).sum())
    n_truth = int(truth.sum())
    return {
        "flagged": n_flagged,
        "flag_rate": round(n_flagged / total, 5) if total else 0.0,
        "caught": caught,
        "missed": n_truth - caught,
        "recall": round(caught / n_truth, 4) if n_truth else 0.0,
        "precision": round(caught / n_flagged, 6) if n_flagged else 0.0,
        # The number an analyst actually feels: how many alerts they work
        # before finding one real case.
        "alerts_per_case": round(n_flagged / caught) if caught else None,
    }


@lru_cache(maxsize=1)
def compare(split: str = config.DEFAULT_SPLIT) -> dict:
    """Rule engine vs MuleNet, both scored per transaction."""
    eng = engine.load(split)
    df = eng.transactions
    truth = df["is_laundering"] == 1
    total = len(df)

    rules = rule_engine.evaluate(split)
    rule_flags = rules["total_flags"]

    def flagged_at(depth: int | None) -> pd.Series:
        rows = eng.rings if depth is None else eng.rings.head(depth)
        accounts: set[str] = set()
        for payload in rows["accounts"]:
            accounts |= set(json.loads(payload))
        return df["from_node"].isin(accounts) & df["to_node"].isin(accounts)

    by_depth = []
    for depth in DEPTHS:
        if depth > len(eng.rings):
            continue
        by_depth.append({"depth": depth, **_measure(flagged_at(depth), truth, total)})

    full = _measure(flagged_at(None), truth, total)
    baseline = _measure(
        pd.Series(True, index=df.index), truth, total
    )  # flag-everything reference

    return {
        "transactions_scanned": total,
        "laundering_total": int(truth.sum()),
        "base_rate": round(float(truth.mean()), 6),
        "rule_engine": {
            "flagged": rule_flags,
            "flag_rate": round(rule_flags / total, 5),
            "caught": rules["laundering_caught"],
            "missed": rules["laundering_missed"],
            "recall": rules["recall"],
            "precision": rules["precision"],
            "alerts_per_case": round(rule_flags / rules["laundering_caught"])
            if rules["laundering_caught"]
            else None,
        },
        "mulenet": full,
        "mulenet_by_depth": by_depth,
        "rings_in_queue": int(len(eng.rings)),
        # How much better than blind guessing the top of the queue is.
        "lift_at_top": (
            round(by_depth[0]["precision"] / float(truth.mean()))
            if by_depth and truth.mean()
            else None
        ),
        "flag_everything": baseline,
    }
