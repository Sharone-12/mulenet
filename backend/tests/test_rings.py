"""Guards on candidate combination and ranking."""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mulenet import detectors, rings as rings_mod  # noqa: E402
from mulenet.detectors import Candidate  # noqa: E402


def test_candidates_are_not_merged_transitively():
    """The regression that cost 95% of recall.

    Two unrelated stars sharing one hub account must stay two rings. An
    earlier combiner unioned them, which chained the whole graph into blobs
    that then exceeded MAX_RING_ACCOUNTS and were dropped entirely.
    """
    cands = [
        Candidate({"H", "a1", "a2"}, "fan_out", 0.9),
        Candidate({"H", "b1", "b2"}, "fan_in", 0.9),
    ]
    out = rings_mod.combine(cands)
    assert len(out) == 2
    assert all(len(r.accounts) == 3 for r in out)


def test_oversized_candidates_are_dropped():
    big = {f"n{i}" for i in range(rings_mod.MAX_RING_ACCOUNTS + 1)}
    assert rings_mod.combine([Candidate(big, "fan_out", 1.0)]) == []


def test_identical_account_sets_collapse_and_keep_both_evidence():
    cands = [
        Candidate({"a", "b"}, "cycle", 0.5),
        Candidate({"a", "b"}, "pass_through", 0.5),
    ]
    out = rings_mod.combine(cands)
    assert len(out) == 1
    assert out[0].signals == {"cycle", "pass_through"}
    assert len(out[0].evidence) == 2


def test_ranking_prefers_the_stronger_detector():
    """pass_through measures 3-25x better than the others, so at equal
    candidate score it must outrank them."""
    out = rings_mod.combine(
        [
            Candidate({"x1", "x2"}, "velocity_burst", 0.9),
            Candidate({"y1", "y2"}, "pass_through", 0.9),
        ]
    )
    assert out[0].accounts == {"y1", "y2"}
    assert out[0].ring_id == 1


def test_ring_ids_are_rank_order():
    out = rings_mod.combine(
        [
            Candidate({"a", "b"}, "cycle", 0.1),
            Candidate({"c", "d"}, "pass_through", 0.9),
        ]
    )
    assert [r.ring_id for r in out] == [1, 2]
    assert out[0].score > out[1].score


def _row(frm, to, fmt="ACH", amount=100.0, ts="2022-09-01 00:00"):
    return {
        "from_node": frm,
        "to_node": to,
        "payment_format": fmt,
        "amount_paid": amount,
        "timestamp": pd.Timestamp(ts),
    }


def test_prefilter_drops_other_rails_and_self_loops():
    df = pd.DataFrame(
        [_row("A", "B"), _row("C", "D", fmt="Cheque"), _row("S", "S")]
    )
    out = detectors.prefilter(df)
    assert len(out) == 1
    assert out.iloc[0]["from_node"] == "A"


def test_star_detector_is_degree_banded():
    """Below MIN_STAR is not a fan; above MAX_STAR is a bank, not a ring."""
    small = pd.DataFrame([_row("H", f"n{i}") for i in range(detectors.MIN_STAR - 1)])
    assert detectors.fan_out(small) == []

    huge = pd.DataFrame([_row("H", f"n{i}") for i in range(detectors.MAX_STAR + 5)])
    assert detectors.fan_out(huge) == []


def test_fan_out_scores_uniform_and_tight_higher():
    tight = pd.DataFrame([_row("H", f"n{i}", amount=100.0) for i in range(5)])
    spread = pd.DataFrame(
        [
            _row("H", f"n{i}", amount=100.0 * (i + 1), ts=f"2022-09-0{i + 1} 00:00")
            for i in range(5)
        ]
    )
    assert detectors.fan_out(tight)[0].score > detectors.fan_out(spread)[0].score


def test_pass_through_matches_amount_in_to_amount_out():
    df = pd.DataFrame(
        [
            _row("A", "M", amount=1000.0, ts="2022-09-01 00:00"),
            _row("M", "B", amount=1000.0, ts="2022-09-02 00:00"),
        ]
    )
    out = detectors.pass_through(df)
    assert len(out) == 1
    assert out[0].accounts == {"A", "M", "B"}
    assert out[0].evidence["hold_hours"] == pytest.approx(24.0)


def test_pass_through_ignores_mismatched_amounts():
    df = pd.DataFrame(
        [
            _row("A", "M", amount=1000.0, ts="2022-09-01 00:00"),
            _row("M", "B", amount=5.0, ts="2022-09-02 00:00"),
        ]
    )
    assert detectors.pass_through(df) == []
