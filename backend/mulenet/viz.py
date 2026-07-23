"""Graph payloads sized for the browser.

react-force-graph-2d cannot render 705,907 nodes, so the frontend is served a
slice: every account in the top-ranked rings, plus the real transaction
neighbourhood around them. Presentation only - nothing here feeds detection.
"""

from __future__ import annotations

import json
from functools import lru_cache

from . import config, engine, graph as graph_mod, tools

DEFAULT_RING_COUNT = 25
# How far out from a flagged account we still call it context. 1 hop keeps the
# picture legible; 2 hops doubles the node count with accounts two steps
# removed from any finding.
DEFAULT_HOPS = 1

# Context accounts busier than this are payment hubs, not ring participants.
MAX_CONTEXT_DEGREE = 8


@lru_cache(maxsize=4)
def graph_sample(
    top_k: int = DEFAULT_RING_COUNT,
    hops: int = DEFAULT_HOPS,
    split: str = config.DEFAULT_SPLIT,
) -> dict:
    """Nodes and links for the network view.

    Every ring account is included and tagged with its role so the frontend
    can colour controllers and mules differently without a second call.
    """
    eng = engine.load(split)
    head = eng.rings.head(top_k)

    roles: dict[str, str] = {}
    ring_of: dict[str, int] = {}
    for _, row in head.iterrows():
        ring_id = int(row["ring_id"])
        classified = tools.classify_roles(ring_id, split=split)
        for entry in classified.get("controllers", []):
            roles[entry["account"]] = "controller"
            ring_of.setdefault(entry["account"], ring_id)
        for entry in classified.get("mules", []):
            roles.setdefault(entry["account"], "mule")
            ring_of.setdefault(entry["account"], ring_id)

    ring_accounts = set(roles)

    # Context is the REAL neighbourhood of the flagged accounts - every node
    # shown is an account that actually transacts with a ring, within `hops`
    # steps.
    #
    # This replaced an arbitrary head(2000) of the transaction file. That
    # filled the canvas with 3,800 unrelated accounts which had no
    # relationship to any finding: visually busier, but meaningless. A smaller
    # graph where every node earns its place is the honest picture.
    search = eng.search
    g = graph_mod.build_graph(search)
    neighbourhood = set(graph_mod.subgraph_around(g, ring_accounts, hops=hops))

    # Drop hub accounts from the context. One 2-hop neighbour had degree 63 and
    # rendered as a 60-spoke starburst that dominated the canvas while saying
    # nothing about any ring - a busy payment account, not a finding. Ring
    # accounts themselves are never filtered, whatever their degree.
    undirected = g.to_undirected()
    context = {
        n
        for n in neighbourhood - ring_accounts
        if undirected.degree(n) <= MAX_CONTEXT_DEGREE
    }

    keep = context | ring_accounts
    edges = (
        search[search["from_node"].isin(keep) & search["to_node"].isin(keep)]
        .groupby(["from_node", "to_node"], sort=False)["amount_paid"]
        .agg(["sum", "size"])
        .reset_index()
    )

    nodes_in_view = set(edges["from_node"]) | set(edges["to_node"]) | ring_accounts
    nodes = [
        {
            "id": account,
            "role": roles.get(account, "clean"),
            "ring_id": ring_of.get(account),
        }
        for account in sorted(nodes_in_view)
    ]

    return {
        "nodes": nodes,
        "links": [
            {
                "source": row.from_node,
                "target": row.to_node,
                "amount": round(float(row.sum), 2),
                "count": int(row.size),
            }
            for row in edges.itertuples()
        ],
        "ring_accounts": len(ring_accounts),
        "hops": hops,
        "total_accounts_in_graph": int(len(eng.transactions["from_node"].unique())),
    }


@lru_cache(maxsize=64)
def ring_subgraph(ring_id: int, split: str = config.DEFAULT_SPLIT) -> dict:
    """Just one ring's accounts and internal transfers, for the zoomed view."""
    eng = engine.load(split)
    ring = eng.ring(ring_id)
    if ring is None:
        return {"error": f"ring {ring_id} not found", "nodes": [], "links": []}

    classified = tools.classify_roles(ring_id, split=split)
    roles = {e["account"]: "controller" for e in classified.get("controllers", [])}
    roles.update({e["account"]: "mule" for e in classified.get("mules", [])})

    txns = eng.ring_transactions(ring["accounts"])
    return {
        "ring_id": ring_id,
        "nodes": [
            {"id": a, "role": roles.get(a, "clean"), "ring_id": ring_id}
            for a in ring["accounts"]
        ],
        "links": [
            {
                "source": r.from_node,
                "target": r.to_node,
                "amount": round(float(r.amount_paid), 2),
                "currency": r.payment_currency,
                "timestamp": str(r.timestamp),
            }
            for r in txns.itertuples()
        ],
    }


@lru_cache(maxsize=4)
def ring_cards(top_k: int = DEFAULT_RING_COUNT, split: str = config.DEFAULT_SPLIT) -> list[dict]:
    """Summary cards for the ring queue.

    Carries enough structure for each card to draw its own topology thumbnail,
    so the queue shows the shape of every ring in one request rather than
    firing `top_k` follow-up calls from the browser.
    """
    eng = engine.load(split)
    cards = []
    for _, row in eng.rings.head(top_k).iterrows():
        ring_id = int(row["ring_id"])
        accounts = json.loads(row["accounts"])
        txns = eng.ring_transactions(accounts)
        signals = json.loads(row["signals"])

        inflow = set(txns["to_node"])
        outflow = set(txns["from_node"])
        classified = tools.classify_roles(ring_id, split=split)

        cards.append(
            {
                "ring_id": ring_id,
                "accounts": len(accounts),
                "transactions": len(txns),
                "amount_by_currency": {
                    str(k): round(float(v), 2)
                    for k, v in txns.groupby("payment_currency")["amount_paid"].sum().items()
                },
                "pattern": ", ".join(signals),
                "score": round(float(row["score"]), 4),
                # Shape, for the thumbnail.
                "topology": {
                    "ring_id": ring_id,
                    "accounts": accounts,
                    "source_accounts": [a for a in accounts if a in outflow and a not in inflow],
                    "cashout_accounts": [a for a in accounts if a in inflow and a not in outflow],
                    "transactions": [
                        {"from": r.from_node, "to": r.to_node}
                        for r in txns.head(20).itertuples()
                    ],
                },
                "roles": {
                    "controllers": [{"account": c["account"]} for c in classified.get("controllers", [])],
                    "mules": [{"account": m["account"]} for m in classified.get("mules", [])],
                },
            }
        )
    return cards
