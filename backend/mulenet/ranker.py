"""Learned ranking of detector candidates.

Detection stays entirely rule-based and explainable: five graph algorithms
decide *what* is a candidate ring. This module only decides *what order* an
investigator sees them in, which is where a model genuinely belongs - it is a
relevance problem with labels, not a legal-evidence problem.

Two rankers are produced side by side so the queue can be reordered live:

* heuristic - KIND_PRIOR[kind] * candidate_score, measured per-detector hit rates.
* learned   - gradient-boosted classifier over candidate features.

Evaluation is averaged over SPLITS, not run once. A single held-out split of
30 rings puts precision@10 anywhere between 10% and 80% purely on which rings
were held out - the first version of this reported one split and drew the
wrong conclusion from it. Averaged over 8 splits the learned ranker is ahead
at every depth and far steadier: 64% vs 46% at k=10 (sd 13% vs 29%), and 47%
vs 21% at k=50, where it wins every split.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .detectors import Candidate
from .rings import KIND_PRIOR

KINDS = ["fan_out", "fan_in", "pass_through", "velocity_burst", "cycle"]

FEATURES = [
    "score",
    "size",
    "degree",
    "tightness",
    "uniformity",
    "total_amount",
    "hold_hours",
    "amount_in",
    "cycle_length",
    *[f"kind_{k}" for k in KINDS],
]

SEED = 0


def featurise(candidates: list[Candidate]) -> pd.DataFrame:
    """One row per candidate; absent features stay NaN.

    HistGradientBoosting handles NaN natively, so a fan-out with no hold time
    and a pass-through with no degree can share one feature space without
    inventing zeros that would read as real measurements.
    """
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
    # Amounts span many orders of magnitude; log keeps tree splits meaningful.
    for col in ("total_amount", "amount_in"):
        df[col] = np.log1p(df[col].clip(lower=0))
    return pd.get_dummies(df, columns=["kind"]).reindex(columns=FEATURES, fill_value=False)


def heuristic_scores(candidates: list[Candidate]) -> np.ndarray:
    return np.array([KIND_PRIOR.get(c.kind, 0.003) * c.score for c in candidates])


def train_and_score(
    candidates: list[Candidate], ring_of_candidate: np.ndarray, n_splits: int = 8
) -> tuple[np.ndarray, dict]:
    """Fit the ranker and score every candidate.

    `ring_of_candidate` is the ground-truth ring each candidate touches, or -1.
    The train/test split is by RING: candidates from one ring share accounts,
    so splitting them randomly would put near-duplicates on both sides and
    report a fantasy score.

    Returns scores for ALL candidates plus metrics averaged over `n_splits`
    held-out splits. Reporting a single split is not enough here: precision@10
    on 30 held-out rings swings between 10% and 80% depending only on which
    rings were held out.
    """
    from sklearn.ensemble import HistGradientBoostingClassifier

    X = featurise(candidates).to_numpy(dtype=float)
    labels = (ring_of_candidate >= 0).astype(int)
    heuristic = heuristic_scores(candidates)
    ring_ids_all = np.unique(ring_of_candidate[ring_of_candidate >= 0])
    ks = (10, 25, 50, 100, 250)

    runs: list[dict] = []
    scores = np.zeros(len(candidates))

    for seed in range(n_splits):
        rng = np.random.default_rng(seed)
        ring_ids = ring_ids_all.copy()
        rng.shuffle(ring_ids)
        test_rings = set(ring_ids[: int(len(ring_ids) * 0.3)].tolist())
        is_test = np.array(
            [
                (r in test_rings) if r >= 0 else bool(rng.random() < 0.3)
                for r in ring_of_candidate
            ]
        )

        model = HistGradientBoostingClassifier(
            max_iter=250, learning_rate=0.06, max_depth=4, random_state=SEED
        )
        model.fit(X[~is_test], labels[~is_test])
        # Averaging predictions across folds gives the shipped ordering an
        # ensemble rather than one model's idiosyncrasies.
        scores += model.predict_proba(X)[:, 1] / n_splits

        y_test = labels[is_test]
        fold = model.predict_proba(X[is_test])[:, 1]
        order_l = np.argsort(-fold)
        order_h = np.argsort(-heuristic[is_test])
        runs.append(
            {
                "learned": {k: float(y_test[order_l[:k]].mean()) for k in ks},
                "heuristic": {k: float(y_test[order_h[:k]].mean()) for k in ks},
            }
        )

    def summarise(which: str, k: int) -> dict:
        vals = [r[which][k] for r in runs]
        return {"mean": round(float(np.mean(vals)), 4), "sd": round(float(np.std(vals)), 4)}

    metrics = {
        "splits": n_splits,
        "candidates": len(candidates),
        "positives": int(labels.sum()),
        "rings_covered": int(len(ring_ids_all)),
        "precision_at_k": [
            {
                "k": k,
                "heuristic": summarise("heuristic", k),
                "learned": summarise("learned", k),
                "learned_wins": sum(r["learned"][k] > r["heuristic"][k] for r in runs),
            }
            for k in ks
        ],
    }
    return scores, metrics
