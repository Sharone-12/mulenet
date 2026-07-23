"""Build the transaction graph.

Topology lives in a NetworkX DiGraph; the individual transactions stay in the
pandas frame. Detection algorithms that need timing (pass-through, bursts)
work off the frame, and the ones that need structure (degree, community,
cycles) work off the graph. Keeping them separate is what makes 6.9M
transactions tractable: the graph only carries 1.38M aggregated edges.
"""

from __future__ import annotations

import networkx as nx
import pandas as pd

# 11.6% of LI-Small rows send an account to itself, almost all of them
# "Reinvestment". They add no topological information and badly distort
# degree-based scores, so they are excluded from the graph by default and
# surfaced separately as a per-node attribute.
SELF_LOOP_ATTR = "self_loop_count"


def edge_table(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse transactions into one row per directed account pair."""
    real = df[df["from_node"] != df["to_node"]]
    agg = real.groupby(["from_node", "to_node"], sort=False).agg(
        count=("amount_paid", "size"),
        total_amount=("amount_paid", "sum"),
        first_seen=("timestamp", "min"),
        last_seen=("timestamp", "max"),
    )
    return agg.reset_index()


def build_graph(df: pd.DataFrame) -> nx.DiGraph:
    """Return the aggregated directed transaction graph.

    Edge attributes: count, total_amount, first_seen, last_seen.
    Node attributes: self_loop_count.
    """
    edges = edge_table(df)
    g = nx.from_pandas_edgelist(
        edges,
        source="from_node",
        target="to_node",
        edge_attr=["count", "total_amount", "first_seen", "last_seen"],
        create_using=nx.DiGraph,
    )

    # Isolated accounts whose only activity is paying themselves would
    # otherwise vanish from the graph entirely.
    loops = df[df["from_node"] == df["to_node"]]
    loop_counts = loops.groupby("from_node", sort=False).size()
    g.add_nodes_from(loop_counts.index)
    nx.set_node_attributes(g, 0, SELF_LOOP_ATTR)
    nx.set_node_attributes(g, loop_counts.to_dict(), SELF_LOOP_ATTR)
    return g


def graph_stats(g: nx.DiGraph) -> dict:
    return {
        "nodes": g.number_of_nodes(),
        "edges": g.number_of_edges(),
        "self_loop_nodes": sum(1 for _, c in g.nodes(data=SELF_LOOP_ATTR) if c),
        "weakly_connected_components": nx.number_weakly_connected_components(g),
        "largest_component": len(max(nx.weakly_connected_components(g), key=len)),
    }


def subgraph_around(g: nx.DiGraph, nodes, hops: int = 1) -> nx.DiGraph:
    """Induced subgraph over `nodes` plus their `hops`-step neighbourhood.

    The UI cannot render 705K nodes, so every visualisation is served from one
    of these rather than from the full graph.
    """
    frontier = set(nodes)
    seen = set(frontier)
    for _ in range(hops):
        nxt = set()
        for n in frontier:
            if n in g:
                nxt.update(g.successors(n))
                nxt.update(g.predecessors(n))
        frontier = nxt - seen
        seen |= nxt
    return g.subgraph(seen).copy()


def transactions_for(df: pd.DataFrame, nodes) -> pd.DataFrame:
    """Every transaction with both endpoints inside `nodes`."""
    nodes = set(nodes)
    return df[df["from_node"].isin(nodes) & df["to_node"].isin(nodes)]
