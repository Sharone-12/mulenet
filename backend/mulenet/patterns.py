"""Parse IBM AML *_Patterns.txt into a ground-truth answer key.

The file is a flat text log of labelled laundering rings:

    BEGIN LAUNDERING ATTEMPT - FAN-IN:  Max 3-degree Fan-In
    2022/09/01 02:38,001812,80279F810,0110,8000A94C0,10154.74,...,ACH,1
    ...
    END LAUNDERING ATTEMPT - FAN-IN

Each block is one ring. The rows inside are a subset of the rows in the
corresponding *_Trans.csv, in the same column order. Some headers carry a
":  <detail>" suffix describing the ring's size (degree or hop count) and
some do not, so the detail is optional.
"""

from __future__ import annotations

import io
import re
from pathlib import Path

import pandas as pd

from . import config

BEGIN_RE = re.compile(r"^BEGIN LAUNDERING ATTEMPT - ([A-Z-]+?)(?::\s*(.*))?$")
END_RE = re.compile(r"^END LAUNDERING ATTEMPT - ([A-Z-]+)$")


def parse_patterns(path: Path | None = None, split: str = config.DEFAULT_SPLIT) -> pd.DataFrame:
    """Return one row per labelled laundering transaction.

    Adds `ring_id`, `pattern_type` and `pattern_detail` to the standard
    transaction columns. `ring_id` is assigned in file order starting at 1.
    """
    path = path or config.patterns_path(split)

    rows: list[str] = []
    ring_ids: list[int] = []
    types: list[str] = []
    details: list[str] = []

    ring_id = 0
    current_type: str | None = None
    current_detail = ""

    with open(path, "r") as fh:
        for lineno, raw in enumerate(fh, start=1):
            line = raw.strip()
            if not line:
                continue

            begin = BEGIN_RE.match(line)
            if begin:
                if current_type is not None:
                    raise ValueError(f"{path}:{lineno}: nested BEGIN inside {current_type} ring")
                ring_id += 1
                current_type = begin.group(1)
                current_detail = (begin.group(2) or "").strip()
                continue

            end = END_RE.match(line)
            if end:
                if current_type is None:
                    raise ValueError(f"{path}:{lineno}: END without a matching BEGIN")
                if end.group(1) != current_type:
                    raise ValueError(
                        f"{path}:{lineno}: END {end.group(1)} closes BEGIN {current_type}"
                    )
                current_type = None
                current_detail = ""
                continue

            if current_type is None:
                raise ValueError(f"{path}:{lineno}: transaction row outside any ring block")

            rows.append(line)
            ring_ids.append(ring_id)
            types.append(current_type)
            details.append(current_detail)

    if current_type is not None:
        raise ValueError(f"{path}: file ended inside an unclosed {current_type} ring")

    df = pd.read_csv(
        io.StringIO("\n".join(rows)),
        names=config.TRANS_COLUMNS,
        dtype=config.TRANS_DTYPES,
    )
    df.insert(0, "ring_id", ring_ids)
    df.insert(1, "pattern_type", pd.Series(types, dtype="string"))
    df.insert(2, "pattern_detail", pd.Series(details, dtype="string"))
    df["timestamp"] = pd.to_datetime(df["timestamp"], format=config.TIMESTAMP_FORMAT)

    df["from_node"] = df["from_bank"] + "-" + df["from_account"]
    df["to_node"] = df["to_bank"] + "-" + df["to_account"]
    return df


def ring_summary(patterns: pd.DataFrame) -> pd.DataFrame:
    """Collapse the per-transaction answer key into one row per ring."""
    grouped = patterns.groupby("ring_id", sort=True)
    summary = grouped.agg(
        pattern_type=("pattern_type", "first"),
        pattern_detail=("pattern_detail", "first"),
        transactions=("ring_id", "size"),
        total_paid=("amount_paid", "sum"),
        first_seen=("timestamp", "min"),
        last_seen=("timestamp", "max"),
    )
    summary["duration_hours"] = (
        summary["last_seen"] - summary["first_seen"]
    ).dt.total_seconds() / 3600.0
    summary["accounts"] = grouped.apply(
        lambda g: len(set(g["from_node"]) | set(g["to_node"])), include_groups=False
    )
    return summary.reset_index()


def account_labels(patterns: pd.DataFrame) -> pd.DataFrame:
    """Map every account that appears in a labelled ring to its rings.

    An account touching more than one ring is the strongest available
    ground-truth signal for a controller, so `ring_count` is kept explicit.
    """
    endpoints = pd.concat(
        [
            patterns[["ring_id", "pattern_type", "from_node"]].rename(
                columns={"from_node": "node"}
            ),
            patterns[["ring_id", "pattern_type", "to_node"]].rename(columns={"to_node": "node"}),
        ],
        ignore_index=True,
    ).drop_duplicates(subset=["node", "ring_id"])

    grouped = endpoints.groupby("node", sort=True)
    return pd.DataFrame(
        {
            "ring_count": grouped["ring_id"].nunique(),
            "ring_ids": grouped["ring_id"].apply(lambda s: sorted(s.unique())),
            "pattern_types": grouped["pattern_type"].apply(lambda s: sorted(set(s))),
        }
    ).reset_index()
