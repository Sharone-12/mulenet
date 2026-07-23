"""Export everything the UI needs as one static JSON bundle.

    python scripts/export_bundle.py

The detection engine needs pandas, NetworkX and a 284MB parquet cache. None of
that is needed to *serve* the results - the whole demo is ~110KB of JSON. This
export is what makes the app deployable to a serverless host: the heavy stack
runs here, offline, and production only reads the output.

Every figure in the bundle comes from the real detection run. Nothing is
mocked; it is precomputed, not invented.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mulenet import config, engine, rule_engine, tools, viz  # noqa: E402

DEFAULT_OUT = Path(__file__).resolve().parents[2] / "frontend" / "public" / "data" / "mulenet.json"


def build(top_k: int = 25, split: str = config.DEFAULT_SPLIT) -> dict:
    scan = tools.scan_network(top_k=top_k, split=split)
    ring_ids = [r["ring_id"] for r in scan["rings"]]

    # The benchmark view needs both engines' numbers side by side. This block
    # used to be assembled by the /tools/rule_engine endpoint, so exporting
    # only evaluate() left the view with no `comparison` key and a blank panel.
    rules = dict(rule_engine.evaluate(split))
    rules["comparison"] = {
        "rule_engine_flags": rules["total_flags"],
        "rule_engine_caught": rules["laundering_caught"],
        "rule_engine_missed": rules["laundering_missed"],
        "graph_engine_rings": int(len(engine.load(split).rings)),
        "graph_engine_top_rings": scan["rings_detected"],
        "graph_engine_accounts_traced": scan["total_accounts_flagged"],
    }

    return {
        "generated_from": split,
        "top_k": top_k,
        "scan": scan,
        "rings": viz.ring_cards(top_k=top_k, split=split),
        "graph": viz.graph_sample(top_k=top_k, split=split),
        "rule_engine": rules,
        "detail": {str(i): tools.investigate_ring(i, split=split) for i in ring_ids},
        "roles": {str(i): tools.classify_roles(i, split=split) for i in ring_ids},
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--top-k", type=int, default=25)
    ap.add_argument("--split", default=config.DEFAULT_SPLIT)
    args = ap.parse_args()

    bundle = build(args.top_k, args.split)
    payload = json.dumps(bundle, default=str, separators=(",", ":"))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(payload)

    # The agent function reads the same bundle, so it ships a copy rather than
    # reaching across to the frontend's public directory at runtime.
    api_copy = args.out.parents[2] / "api" / "mulenet.json"
    if api_copy.parent.exists():
        api_copy.write_text(payload)

    print(f"wrote {args.out}  ({len(payload) / 1024:.0f} KB)")
    print(f"  rings      {len(bundle['rings'])}")
    print(f"  graph      {len(bundle['graph']['nodes'])} nodes, {len(bundle['graph']['links'])} links")
    print(f"  detail for {len(bundle['detail'])} rings")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
