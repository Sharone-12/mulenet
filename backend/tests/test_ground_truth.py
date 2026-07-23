"""Guards on the ground-truth answer key.

The whole evaluation story rests on Patterns.txt reconciling exactly against
Trans.csv, so these assert the join stays lossless and unambiguous.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mulenet import config, loader, patterns as patterns_mod  # noqa: E402

pytestmark = pytest.mark.skipif(
    not config.patterns_path().exists(), reason="ibmdataset not present"
)


@pytest.fixture(scope="module")
def pats():
    return patterns_mod.parse_patterns()


@pytest.fixture(scope="module")
def labelled():
    return loader.load_cache()


def test_parses_every_ring(pats):
    assert pats["ring_id"].nunique() == 117
    assert len(pats) == 1023
    assert set(pats["pattern_type"]) == set(config.PATTERN_TYPES)


def test_ring_ids_are_contiguous(pats):
    assert sorted(pats["ring_id"].unique()) == list(range(1, 118))


def test_degenerate_rings_are_accounted_for(pats):
    """Not every labelled "ring" is a structure.

    8 rings are a single transaction (e.g. "Max 1-degree Fan-Out") and 21 more
    are a single A->B / B->A round trip. No graph algorithm can recover a
    one-edge ring, so ring-level recall is capped at 109/117 = 93% before any
    detection code is written. Evaluation must report against these buckets
    rather than a flat 117.
    """
    sizes = pats.groupby("ring_id").size()
    assert (sizes == 1).sum() == 8
    assert (sizes == 2).sum() == 21
    assert (sizes >= 3).sum() == 88


def test_self_loops_exist(pats):
    """Some labelled rows send an account to itself; graph construction must
    not silently drop or double-count these."""
    assert (pats["from_node"] == pats["to_node"]).sum() > 0


def test_join_is_lossless_and_unambiguous(labelled, pats):
    """Each labelled pattern row maps to exactly one transaction."""
    assert labelled["ring_id"].notna().sum() == len(pats)
    assert labelled["ring_id"].nunique() == 117


def test_every_ring_row_is_flagged_as_laundering(labelled):
    """A row attributed to a ring must also carry is_laundering=1."""
    ring_rows = labelled[labelled["ring_id"].notna()]
    assert (ring_rows["is_laundering"] == 1).all()


def test_patterns_are_a_subset_of_flagged_laundering(labelled):
    """Patterns.txt names 1023 of the 3565 laundering rows; the rest are
    flagged but unattributed, so recall must never be scored against 3565."""
    assert (labelled["is_laundering"] == 1).sum() == 3565
    assert labelled["ring_id"].notna().sum() == 1023


def test_node_ids_disambiguate_banks():
    assert config.node_id("011", "8000ECA90") != config.node_id("0110", "8000ECA90")


def test_account_labels_capture_cross_ring_accounts(pats):
    labels = patterns_mod.account_labels(pats)
    assert len(labels) == 1168
    # The only ground-truth controller signal available.
    assert (labels["ring_count"] > 1).sum() == 49
