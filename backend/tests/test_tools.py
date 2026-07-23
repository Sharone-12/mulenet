"""Guards on the three agent tools.

These run against the real precomputed queue, because the failure mode that
matters - tools reporting evidence detection never saw - only shows up on
real data.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mulenet import agent, config, engine, tools  # noqa: E402

pytestmark = pytest.mark.skipif(
    not config.cache_path(config.DEFAULT_SPLIT, engine.RINGS_CACHE).exists(),
    reason="ring cache not built",
)


@pytest.fixture(scope="module")
def top_ring_id():
    return tools.scan_network(top_k=1)["rings"][0]["ring_id"]


def test_scan_returns_ranked_rings():
    out = tools.scan_network(top_k=5)
    assert len(out["rings"]) == 5
    scores = [r["score"] for r in out["rings"]]
    assert scores == sorted(scores, reverse=True)
    assert out["total_accounts_flagged"] > 0


def test_investigate_reports_only_evidence_detection_saw(top_ring_id):
    """The regression: querying the raw frame pulled in self-payments and
    other rails, inflating one 3-account ring to 339,006 of "traced" flow."""
    out = tools.investigate_ring(top_ring_id)
    for txn in out["transactions"]:
        assert txn["from"] != txn["to"]
    assert out["amount_by_currency"]
    assert out["transaction_count"] == len(out["transactions"]) + out["transactions_truncated"]


def test_investigate_flow_endpoints_are_consistent(top_ring_id):
    out = tools.investigate_ring(top_ring_id)
    accounts = set(out["accounts"])
    assert set(out["source_accounts"]) <= accounts
    assert set(out["cashout_accounts"]) <= accounts
    assert not set(out["source_accounts"]) & set(out["cashout_accounts"])


def test_classify_partitions_every_account(top_ring_id):
    out = tools.classify_roles(top_ring_id)
    ring = tools.investigate_ring(top_ring_id)
    classified = {e["account"] for e in out["controllers"] + out["mules"]}
    assert classified == set(ring["accounts"])


def test_classify_flags_flow_endpoints_as_controllers(top_ring_id):
    """Sitting at the source or the cash-out is the main controller signal."""
    ring = tools.investigate_ring(top_ring_id)
    out = tools.classify_roles(top_ring_id)
    controllers = {e["account"] for e in out["controllers"]}
    for account in ring["source_accounts"] + ring["cashout_accounts"]:
        assert account in controllers


def test_classify_declares_its_method(top_ring_id):
    """The heuristic must never be presented as a validated classifier."""
    assert "heuristic" in tools.classify_roles(top_ring_id)["method"]


def test_unknown_ring_returns_error_not_exception():
    assert "error" in tools.investigate_ring(10**9)
    assert "error" in tools.classify_roles(10**9)


def test_agent_tool_registry_matches_schemas():
    named = {s["function"]["name"] for s in agent.TOOL_SCHEMAS}
    assert named == set(agent.TOOL_IMPLS)


def test_run_tool_reports_errors_instead_of_raising():
    """A raised exception ends the conversation; an error dict lets the model
    read what went wrong and retry."""
    assert "error" in agent.run_tool("nope", {})
    assert "error" in agent.run_tool("investigate_ring", {"bogus": 1})
    assert "error" in agent.run_tool("investigate_ring", {"ring_id": 10**9})
