"""Does a learned ranker beat the hand-set KIND_PRIOR?

    python scripts/learned_ranker.py

Detection is untouched. Only the *ranking* step is swapped: instead of
score = KIND_PRIOR[kind] * candidate_score, a gradient-boosted classifier
predicts whether a candidate contains a ground-truth laundering account.

The split is by RING, not by candidate. Candidates from the same ring share
accounts, so a random split would put near-duplicates on both sides and report
a fantasy score. Both rankers are then evaluated on the same held-out rings,
because the existing 0.786 precision@10 was measured with no holdout at all
and is not directly comparable to a test-set number.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mulenet import detectors, graph as graph_mod, loader, rings as rings_mod, scoring  # noqa: E402

KINDS = ["fan_out", "fan_in", "pass_through", "velocity_burst", "cycle"]
SEED = 0
TEST_FRACTION = 0.3


def featurise(candidates) -> pd.DataFrame:
    """One row per candidate. Missing features stay NaN - the model handles it."""
    rows = []
    for c in candidates:
        e = c.evidence
        rows.append(
            {
                "kind": c.kind,
                "score": c.score,
                "size": len(c.accounts),
                "degree": e.get("degree", np.nan),
                "tightness": e.get("tightness", np.nan),
                "uniformity": e.get("uniformity", np.nan),
                "total_amount": e.get("total_amount", np.nan),
                "hold_hours": e.get("hold_hours", np.nan),
                "amount_in": e.get("amount_in", np.nan),
                "cycle_length": e.get("length", np.nan),
            }
        )
    df = pd.DataFrame(rows)
    # Amounts span many orders of magnitude; log keeps splits meaningful.
    for col in ("total_amount", "amount_in"):
        df[col] = np.log1p(df[col].clip(lower=0))
    return pd.get_dummies(df, columns=["kind"]).reindex(
        columns=[
            "score", "size", "degree", "tightness", "uniformity",
            "total_amount", "hold_hours", "amount_in", "cycle_length",
            *[f"kind_{k}" for k in KINDS],
        ],
        fill_value=False,
    )


def precision_at_k(order, labels, k: int) -> float:
    top = labels[order[:k]]
    return float(top.mean()) if len(top) else 0.0


def main() -> int:
    from sklearn.ensemble import HistGradientBoostingClassifier

    labelled = loader.load_cache()
    truth = scoring.ground_truth(labelled)
    search = detectors.prefilter(labelled)
    g = graph_mod.build_graph(search)

    candidates = (
        detectors.fan_out(search)
        + detectors.fan_in(search)
        + detectors.pass_through(search)
        + detectors.velocity_burst(search)
        # length 2 only: length 3 costs 200s for 360 extra cycles.
        + detectors.cycles(g, max_length=2)
    )
    candidates = [c for c in candidates if len(c.accounts) <= rings_mod.MAX_RING_ACCOUNTS]
    print(f"candidates: {len(candidates):,}")

    # Which ground-truth ring (if any) each candidate touches.
    account_ring = {}
    for ring_id, accounts in truth.ring_accounts.items():
        for a in accounts:
            account_ring.setdefault(a, ring_id)

    touched = np.array(
        [next((account_ring[a] for a in c.accounts if a in account_ring), -1) for c in candidates]
    )
    labels = (touched >= 0).astype(int)
    print(f"positives: {labels.sum():,} ({labels.mean() * 100:.2f}%)")

    # Split by ring so no ring appears on both sides.
    rng = np.random.default_rng(SEED)
    ring_ids = np.array(sorted(truth.ring_accounts))
    rng.shuffle(ring_ids)
    n_test = int(len(ring_ids) * TEST_FRACTION)
    test_rings = set(ring_ids[:n_test].tolist())

    is_test = np.array(
        [
            (r in test_rings) if r >= 0 else bool(rng.random() < TEST_FRACTION)
            for r in touched
        ]
    )
    X = featurise(candidates).to_numpy(dtype=float)

    print(f"train {(~is_test).sum():,} candidates / test {is_test.sum():,}")
    print(f"test rings {len(test_rings)} of {len(ring_ids)}\n")

    model = HistGradientBoostingClassifier(
        max_iter=250, learning_rate=0.06, max_depth=4, random_state=SEED
    )
    model.fit(X[~is_test], labels[~is_test])
    learned = model.predict_proba(X[is_test])[:, 1]

    # The incumbent, scored on the same held-out candidates.
    heuristic = np.array(
        [
            rings_mod.KIND_PRIOR.get(c.kind, 0.003) * c.score
            for c, t in zip(candidates, is_test)
            if t
        ]
    )
    y_test = labels[is_test]

    order_l = np.argsort(-learned)
    order_h = np.argsort(-heuristic)

    print(f"{'k':>7} {'heuristic':>12} {'learned':>10} {'change':>10}")
    for k in (10, 25, 50, 100, 250):
        h = precision_at_k(order_h, y_test, k)
        l = precision_at_k(order_l, y_test, k)
        arrow = "better" if l > h else ("same" if l == h else "worse")
        print(f"{k:>7} {h:>11.1%} {l:>9.1%} {arrow:>10}")

    print(f"\nbase rate in test set: {y_test.mean():.2%}")
    names = [
        "score", "size", "degree", "tightness", "uniformity",
        "total_amount", "hold_hours", "amount_in", "cycle_length",
        *[f"kind_{k}" for k in KINDS],
    ]
    from sklearn.inspection import permutation_importance

    imp = permutation_importance(
        model, X[is_test], y_test, n_repeats=3, random_state=SEED, scoring="average_precision"
    )
    print("\ntop features:")
    for i in np.argsort(-imp.importances_mean)[:6]:
        print(f"  {names[i]:<16} {imp.importances_mean[i]:+.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
