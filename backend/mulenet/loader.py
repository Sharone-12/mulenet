"""Load IBM AML transactions and attach the ground-truth ring labels.

The raw CSV is read once, joined against the answer key parsed from
*_Patterns.txt, and cached as parquet so downstream detection work does not
pay the parse cost again.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from . import config, patterns as patterns_mod

# A labelled row in Patterns.txt is matched back to its row in Trans.csv on
# this tuple. Amount has to be part of the key: 90 of the 1023 labelled rows
# share a (timestamp, from, to, format) tuple with an unrelated transaction,
# and without the amount those decoys get pulled into the answer key. It is
# rounded to 6dp so the Bitcoin rows (0.034277) survive the comparison.
AMOUNT_KEY = "_amount_key"
JOIN_KEYS = ["timestamp", "from_node", "to_node", "payment_format", AMOUNT_KEY]


def load_transactions(split: str = config.DEFAULT_SPLIT, path: Path | None = None) -> pd.DataFrame:
    """Read the raw transaction CSV with correct dtypes and node ids."""
    path = path or config.trans_path(split)
    df = pd.read_csv(
        path,
        skiprows=1,
        names=config.TRANS_COLUMNS,
        dtype=config.TRANS_DTYPES,
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"], format=config.TIMESTAMP_FORMAT)
    df["from_node"] = df["from_bank"] + "-" + df["from_account"]
    df["to_node"] = df["to_bank"] + "-" + df["to_account"]
    return df


def attach_ground_truth(
    trans: pd.DataFrame, patterns: pd.DataFrame
) -> tuple[pd.DataFrame, dict]:
    """Add `ring_id` / `pattern_type` to the transaction frame.

    Returns the annotated frame and a report describing how cleanly the two
    files reconciled. Unmatched pattern rows would silently shrink the answer
    key, so the caller is expected to check the report.
    """
    trans = trans.copy()
    patterns = patterns.copy()
    trans[AMOUNT_KEY] = trans["amount_paid"].round(6)
    patterns[AMOUNT_KEY] = patterns["amount_paid"].round(6)

    key = patterns[JOIN_KEYS + ["ring_id", "pattern_type"]]

    # A ring can legitimately repeat the same transaction key; collapsing to
    # the first keeps the join one-to-one.
    key = key.drop_duplicates(subset=JOIN_KEYS, keep="first")

    merged = trans.merge(key, on=JOIN_KEYS, how="left", validate="many_to_one")

    matched_keys = merged.loc[merged["ring_id"].notna(), JOIN_KEYS].drop_duplicates()
    unmatched = len(key) - len(matched_keys)
    merged = merged.drop(columns=[AMOUNT_KEY])

    labelled = merged["ring_id"].notna()
    report = {
        "transactions": len(merged),
        "pattern_rows": len(patterns),
        "pattern_keys": len(key),
        "unmatched_pattern_keys": int(unmatched),
        "rows_assigned_to_a_ring": int(labelled.sum()),
        "rings": int(merged["ring_id"].nunique()),
        # is_laundering=1 marks every laundering transaction; Patterns.txt only
        # attributes a subset of them to a named ring.
        "is_laundering_rows": int((merged["is_laundering"] == 1).sum()),
        "laundering_without_ring": int(
            ((merged["is_laundering"] == 1) & ~labelled).sum()
        ),
        "ring_rows_not_flagged": int((labelled & (merged["is_laundering"] != 1)).sum()),
    }
    return merged, report


def build_cache(split: str = config.DEFAULT_SPLIT, force: bool = False) -> tuple[Path, dict]:
    """Produce the annotated parquet cache for `split`."""
    out = config.cache_path(split, "trans_labelled")
    if out.exists() and not force:
        return out, {"skipped": "cache already exists"}

    pats = patterns_mod.parse_patterns(split=split)
    trans = load_transactions(split)
    merged, report = attach_ground_truth(trans, pats)

    out.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(out, index=False)

    rings = patterns_mod.ring_summary(pats)
    rings.to_parquet(config.cache_path(split, "rings"), index=False)
    patterns_mod.account_labels(pats).to_parquet(
        config.cache_path(split, "account_labels"), index=False
    )
    return out, report


def load_cache(split: str = config.DEFAULT_SPLIT) -> pd.DataFrame:
    """Read the annotated transactions back."""
    path = config.cache_path(split, "trans_labelled")
    if not path.exists():
        raise FileNotFoundError(f"{path} missing - run scripts/build_cache.py first")
    return pd.read_parquet(path)
