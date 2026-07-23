"""Score predicted rings against the IBM ground truth.

Every detection algorithm returns the same shape - a list of predicted rings,
each a set of account node ids - so all of them can be measured the moment
they are written.

Two honesty constraints are baked in:

1. `Patterns.txt` attributes only 1023 of the 3565 laundering transactions to
   a named ring. The other 2542 are laundering with no ring identity, so an
   account flagged because of one is not really a false positive. Precision is
   therefore reported twice: `precision_strict` counts it against you,
   `precision_lenient` excludes it.

2. 8 of the 117 rings are a single transaction and 21 more are a bare
   round trip. No structural algorithm can recover a one-edge ring, so results
   are bucketed by ring size and `detectable` (>= 3 transactions) is the
   headline denominator.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

# A true ring counts as detected when a single predicted ring covers at least
# this fraction of its accounts. Partial credit is deliberate: recovering 4 of
# 5 mules is an investigative win, not a miss.
DEFAULT_COVERAGE = 0.5

DETECTABLE_MIN_TXNS = 3


@dataclass
class GroundTruth:
    """The answer key, in the shape the scorer needs."""

    ring_accounts: dict[int, set[str]]
    ring_types: dict[int, str]
    ring_sizes: dict[int, int]
    all_ring_accounts: set[str] = field(default_factory=set)
    # Accounts touching a laundering transaction that no ring claims.
    unattributed_accounts: set[str] = field(default_factory=set)

    @property
    def detectable_rings(self) -> set[int]:
        return {r for r, n in self.ring_sizes.items() if n >= DETECTABLE_MIN_TXNS}

    def size_bucket(self, ring_id: int) -> str:
        n = self.ring_sizes[ring_id]
        if n == 1:
            return "single_txn"
        if n == 2:
            return "round_trip"
        return "detectable"


def ground_truth(labelled: pd.DataFrame) -> GroundTruth:
    """Derive the answer key from the labelled transaction cache."""
    rings = labelled[labelled["ring_id"].notna()]

    ring_accounts: dict[int, set[str]] = {}
    ring_types: dict[int, str] = {}
    ring_sizes: dict[int, int] = {}
    for ring_id, grp in rings.groupby("ring_id"):
        rid = int(ring_id)
        ring_accounts[rid] = set(grp["from_node"]) | set(grp["to_node"])
        ring_types[rid] = grp["pattern_type"].iloc[0]
        ring_sizes[rid] = len(grp)

    all_accounts: set[str] = set()
    for accts in ring_accounts.values():
        all_accounts |= accts

    unattributed = labelled[(labelled["is_laundering"] == 1) & labelled["ring_id"].isna()]
    unattributed_accounts = (
        set(unattributed["from_node"]) | set(unattributed["to_node"])
    ) - all_accounts

    return GroundTruth(
        ring_accounts=ring_accounts,
        ring_types=ring_types,
        ring_sizes=ring_sizes,
        all_ring_accounts=all_accounts,
        unattributed_accounts=unattributed_accounts,
    )


def _f1(precision: float, recall: float) -> float:
    return 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)


def score(
    predicted: list[set[str]],
    truth: GroundTruth,
    coverage: float = DEFAULT_COVERAGE,
) -> dict:
    """Score predicted rings. Returns account-level and ring-level metrics."""
    flagged: set[str] = set()
    for ring in predicted:
        flagged |= set(ring)

    hits = flagged & truth.all_ring_accounts
    false_positives = flagged - truth.all_ring_accounts
    # Accounts caught for a laundering transaction that no ring claims.
    excused = false_positives & truth.unattributed_accounts

    recall = len(hits) / len(truth.all_ring_accounts) if truth.all_ring_accounts else 0.0
    precision_strict = len(hits) / len(flagged) if flagged else 0.0
    lenient_denominator = len(flagged) - len(excused)
    precision_lenient = len(hits) / lenient_denominator if lenient_denominator else 0.0

    # Ring level: best single predicted ring covering each true ring.
    per_ring = []
    for ring_id, accounts in truth.ring_accounts.items():
        best_cov, best_jaccard = 0.0, 0.0
        for pred in predicted:
            pred = set(pred)
            overlap = len(accounts & pred)
            if not overlap:
                continue
            best_cov = max(best_cov, overlap / len(accounts))
            best_jaccard = max(best_jaccard, overlap / len(accounts | pred))
        per_ring.append(
            {
                "ring_id": ring_id,
                "pattern_type": truth.ring_types[ring_id],
                "bucket": truth.size_bucket(ring_id),
                "transactions": truth.ring_sizes[ring_id],
                "coverage": best_cov,
                "jaccard": best_jaccard,
                "detected": best_cov >= coverage,
            }
        )
    rings_df = pd.DataFrame(per_ring)

    detectable = rings_df[rings_df["bucket"] == "detectable"]
    return {
        "accounts": {
            "flagged": len(flagged),
            "true_positives": len(hits),
            "false_positives": len(false_positives),
            "excused_false_positives": len(excused),
            "recall": recall,
            "precision_strict": precision_strict,
            "precision_lenient": precision_lenient,
            "f1_strict": _f1(precision_strict, recall),
        },
        "rings": {
            "predicted": len(predicted),
            "detected_of_detectable": int(detectable["detected"].sum()),
            "detectable_total": len(detectable),
            "detectable_recall": (
                detectable["detected"].mean() if len(detectable) else 0.0
            ),
            "detected_of_all": int(rings_df["detected"].sum()),
            "all_total": len(rings_df),
        },
        "by_bucket": rings_df.groupby("bucket")["detected"].agg(["sum", "size"]).to_dict("index"),
        "by_pattern": (
            detectable.groupby("pattern_type")["detected"]
            .agg(["sum", "size"])
            .to_dict("index")
        ),
        "per_ring": rings_df,
    }


def format_report(result: dict) -> str:
    """Human-readable summary for terminal runs."""
    acc, rings = result["accounts"], result["rings"]
    lines = [
        "ACCOUNTS",
        f"  flagged              {acc['flagged']}",
        f"  true positives       {acc['true_positives']}",
        f"  false positives      {acc['false_positives']}"
        f"  ({acc['excused_false_positives']} touch unattributed laundering)",
        f"  recall               {acc['recall']:.3f}",
        f"  precision (strict)   {acc['precision_strict']:.3f}",
        f"  precision (lenient)  {acc['precision_lenient']:.3f}",
        "",
        "RINGS",
        f"  predicted            {rings['predicted']}",
        f"  detectable recall    {rings['detected_of_detectable']}/{rings['detectable_total']}"
        f"  ({rings['detectable_recall']:.3f})",
        f"  all-rings recall     {rings['detected_of_all']}/{rings['all_total']}",
        "",
        "BY RING SIZE",
    ]
    for bucket, row in sorted(result["by_bucket"].items()):
        lines.append(f"  {bucket:<20} {int(row['sum'])}/{int(row['size'])}")
    lines += ["", "BY PATTERN (detectable rings only)"]
    for pattern, row in sorted(result["by_pattern"].items()):
        lines.append(f"  {pattern:<20} {int(row['sum'])}/{int(row['size'])}")
    return "\n".join(lines)
