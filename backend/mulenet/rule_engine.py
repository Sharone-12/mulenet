"""A conventional threshold-based rule engine, for comparison.

This is the baseline MuleNet is argued against: five independent rules, each
checking one transaction (or one account-day) against a fixed threshold. It is
deliberately a faithful implementation of how legacy AML screening works, not
a strawman - the point is that thresholds cannot see structure, however well
they are tuned.

Nothing here feeds the detection engine. It is evaluation-only.
"""

from __future__ import annotations

from functools import lru_cache

import pandas as pd

from . import config, loader

SINGLE_TRANSACTION_LIMIT = 1_000_000
DAILY_VOLUME_LIMIT = 5_000_000
NIGHT_START_HOUR = 1
NIGHT_END_HOUR = 5
HOURLY_TRANSACTION_LIMIT = 10
CROSS_BORDER_LIMIT = 500_000

RULE_DESCRIPTIONS = {
    "large_single_transaction": f"Single transaction over {SINGLE_TRANSACTION_LIMIT:,}",
    "high_daily_volume": f"Account moves over {DAILY_VOLUME_LIMIT:,} in a day",
    "overnight_activity": f"Transaction between {NIGHT_START_HOUR}am and {NIGHT_END_HOUR}am",
    "high_frequency": f"Over {HOURLY_TRANSACTION_LIMIT} transactions from one account in an hour",
    "large_cross_border": f"Cross-border transfer over {CROSS_BORDER_LIMIT:,}",
}


def _apply_rules(df: pd.DataFrame) -> pd.DataFrame:
    """Boolean flag column per rule, one row per transaction."""
    flags = pd.DataFrame(index=df.index)

    flags["large_single_transaction"] = df["amount_paid"] > SINGLE_TRANSACTION_LIMIT

    day = df["timestamp"].dt.floor("D")
    daily = df.groupby([df["from_node"], day])["amount_paid"].transform("sum")
    flags["high_daily_volume"] = daily > DAILY_VOLUME_LIMIT

    hour_of_day = df["timestamp"].dt.hour
    flags["overnight_activity"] = hour_of_day.between(NIGHT_START_HOUR, NIGHT_END_HOUR - 1)

    hour = df["timestamp"].dt.floor("h")
    per_hour = df.groupby([df["from_node"], hour])["amount_paid"].transform("size")
    flags["high_frequency"] = per_hour > HOURLY_TRANSACTION_LIMIT

    # The dataset has no country column. A payment whose sent and received
    # currencies differ is the closest available proxy for a cross-border
    # transfer, and is how the currency columns are meant to be read.
    cross_border = df["payment_currency"] != df["receiving_currency"]
    flags["large_cross_border"] = cross_border & (df["amount_paid"] > CROSS_BORDER_LIMIT)

    return flags


@lru_cache(maxsize=1)
def evaluate(split: str = config.DEFAULT_SPLIT) -> dict:
    """Score the rule engine against the laundering labels.

    Recall is measured against the 3,565 transactions IBM marks as laundering,
    which is the fairest comparison for a transaction-level engine: unlike the
    graph engine it never claims to group them into rings.
    """
    df = loader.load_cache(split)
    flags = _apply_rules(df)
    any_flag = flags.any(axis=1)
    laundering = df["is_laundering"] == 1

    per_rule = {
        rule: {
            "description": RULE_DESCRIPTIONS[rule],
            "flags": int(flags[rule].sum()),
            "laundering_caught": int((flags[rule] & laundering).sum()),
        }
        for rule in flags.columns
    }

    caught = int((any_flag & laundering).sum())
    total_laundering = int(laundering.sum())
    total_flags = int(any_flag.sum())

    return {
        "engine": "rule_engine",
        "rules": per_rule,
        "total_flags": total_flags,
        "total_laundering_transactions": total_laundering,
        "laundering_caught": caught,
        "laundering_missed": total_laundering - caught,
        "recall": round(caught / total_laundering, 4) if total_laundering else 0.0,
        # Almost every flag is a false alarm; this is the number that makes
        # threshold engines unworkable in practice.
        "precision": round(caught / total_flags, 6) if total_flags else 0.0,
        "transactions_scanned": int(len(df)),
    }
