"""Calibration guards on the scoring harness.

If these drift, every precision/recall number the project reports is wrong,
so they use small hand-built fixtures rather than the real dataset.
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mulenet import graph as graph_mod, scoring  # noqa: E402


def _txn(frm, to, ring=None, laundering=0, amount=100.0, ts="2022-09-01 00:00"):
    return {
        "from_node": frm,
        "to_node": to,
        "ring_id": ring,
        "pattern_type": "FAN-OUT" if ring else None,
        "is_laundering": laundering,
        "amount_paid": amount,
        "timestamp": pd.Timestamp(ts),
    }


@pytest.fixture
def toy():
    """Ring 1 is detectable (3 txns), ring 2 is a single transaction.

    C->D is laundering that no ring claims, so flagging C or D must not be
    charged as a strict-precision false positive under the lenient measure.
    """
    return pd.DataFrame(
        [
            _txn("A", "M1", ring=1.0, laundering=1),
            _txn("A", "M2", ring=1.0, laundering=1),
            _txn("A", "M3", ring=1.0, laundering=1),
            _txn("X", "Y", ring=2.0, laundering=1),
            _txn("C", "D", laundering=1),
            _txn("N1", "N2"),
            _txn("N2", "N3"),
            _txn("S", "S"),
        ]
    )


def test_ground_truth_partitions_rings(toy):
    t = scoring.ground_truth(toy)
    assert t.ring_accounts[1] == {"A", "M1", "M2", "M3"}
    assert t.ring_sizes == {1: 3, 2: 1}
    assert t.detectable_rings == {1}
    assert t.size_bucket(1) == "detectable"
    assert t.size_bucket(2) == "single_txn"
    assert t.unattributed_accounts == {"C", "D"}


def test_oracle_scores_perfectly(toy):
    t = scoring.ground_truth(toy)
    r = scoring.score(list(t.ring_accounts.values()), t)
    assert r["accounts"]["recall"] == 1.0
    assert r["accounts"]["precision_strict"] == 1.0
    assert r["rings"]["detected_of_all"] == r["rings"]["all_total"]


def test_flag_none_scores_zero(toy):
    r = scoring.score([], scoring.ground_truth(toy))
    assert r["accounts"]["recall"] == 0.0
    assert r["rings"]["detected_of_detectable"] == 0


def test_unattributed_laundering_is_excused(toy):
    """Flagging C (laundering, but in no ring) hurts strict precision only."""
    t = scoring.ground_truth(toy)
    r = scoring.score([{"A", "M1", "M2", "M3", "C"}], t)
    assert r["accounts"]["excused_false_positives"] == 1
    assert r["accounts"]["precision_strict"] == pytest.approx(4 / 5)
    assert r["accounts"]["precision_lenient"] == 1.0


def test_innocent_false_positive_is_not_excused(toy):
    t = scoring.ground_truth(toy)
    r = scoring.score([{"A", "M1", "M2", "M3", "N1"}], t)
    assert r["accounts"]["excused_false_positives"] == 0
    assert r["accounts"]["precision_lenient"] == pytest.approx(4 / 5)


def test_partial_coverage_respects_threshold(toy):
    """Half a ring clears the default 0.5 bar; a quarter does not."""
    t = scoring.ground_truth(toy)
    assert scoring.score([{"A", "M1"}], t)["rings"]["detected_of_detectable"] == 1
    assert scoring.score([{"M1"}], t)["rings"]["detected_of_detectable"] == 0


def test_coverage_cannot_be_faked_by_splitting(toy):
    """Coverage is per predicted ring, so scattering a true ring across many
    one-account predictions must not count as a detection."""
    t = scoring.ground_truth(toy)
    r = scoring.score([{"A"}, {"M1"}, {"M2"}, {"M3"}], t)
    # All four of ring 1's accounts are flagged (ring 2's X/Y are not, hence
    # 4/6 overall) yet no single prediction covers the ring, so it is a miss.
    assert r["accounts"]["recall"] == pytest.approx(4 / 6)
    assert r["rings"]["detected_of_detectable"] == 0


def test_graph_excludes_self_loops_but_keeps_the_node(toy):
    g = graph_mod.build_graph(toy)
    assert "S" in g
    assert not g.has_edge("S", "S")
    assert g.nodes["S"][graph_mod.SELF_LOOP_ATTR] == 1
    assert g.nodes["A"][graph_mod.SELF_LOOP_ATTR] == 0


def test_edges_aggregate_repeat_transactions():
    df = pd.DataFrame(
        [
            _txn("A", "B", amount=10.0, ts="2022-09-01 00:00"),
            _txn("A", "B", amount=15.0, ts="2022-09-03 00:00"),
        ]
    )
    g = graph_mod.build_graph(df)
    assert g["A"]["B"]["count"] == 2
    assert g["A"]["B"]["total_amount"] == 25.0
    assert g["A"]["B"]["first_seen"] == pd.Timestamp("2022-09-01")
    assert g["A"]["B"]["last_seen"] == pd.Timestamp("2022-09-03")


def test_subgraph_around_expands_by_hops(toy):
    g = graph_mod.build_graph(toy)
    assert set(graph_mod.subgraph_around(g, ["N2"], hops=1)) == {"N1", "N2", "N3"}
    assert set(graph_mod.subgraph_around(g, ["N1"], hops=1)) == {"N1", "N2"}
