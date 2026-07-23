"""Build the graph and exercise the scoring harness against baselines.

    python scripts/evaluate.py

Until real detectors exist this runs three reference predictions, which
between them prove the scorer is calibrated:

  oracle      - hand back the ground truth; must score a perfect 1.0
  flag-all    - one ring containing every account; recall 1.0, precision ~0
  flag-none   - empty; all zeros

Any detector worth keeping has to land between flag-all and oracle.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mulenet import graph as graph_mod, loader, scoring  # noqa: E402


def main() -> int:
    started = time.time()
    labelled = loader.load_cache()
    print(f"loaded {len(labelled):,} transactions ({time.time() - started:.1f}s)\n")

    started = time.time()
    g = graph_mod.build_graph(labelled)
    print(f"GRAPH ({time.time() - started:.1f}s)")
    for key, value in graph_mod.graph_stats(g).items():
        print(f"  {key:<28} {value:,}")
    print()

    truth = scoring.ground_truth(labelled)
    print("GROUND TRUTH")
    print(f"  rings                        {len(truth.ring_accounts)}")
    print(f"  detectable rings (>=3 txns)  {len(truth.detectable_rings)}")
    print(f"  accounts in rings            {len(truth.all_ring_accounts):,}")
    print(f"  unattributed laundering accts {len(truth.unattributed_accounts):,}")
    print()

    all_accounts = set(g.nodes())
    baselines = {
        "oracle": list(truth.ring_accounts.values()),
        "flag-all": [all_accounts],
        "flag-none": [],
    }

    failures = 0
    for name, predicted in baselines.items():
        result = scoring.score(predicted, truth)
        print("=" * 60)
        print(f"BASELINE: {name}")
        print("=" * 60)
        print(scoring.format_report(result))
        print()

        if name == "oracle":
            acc, rings = result["accounts"], result["rings"]
            ok = (
                abs(acc["recall"] - 1.0) < 1e-9
                and abs(acc["precision_strict"] - 1.0) < 1e-9
                and rings["detected_of_all"] == rings["all_total"]
            )
            if not ok:
                print("FAIL: oracle did not score perfectly - the scorer is wrong.\n")
                failures += 1
        if name == "flag-all" and abs(result["accounts"]["recall"] - 1.0) > 1e-9:
            print("FAIL: flag-all should recall every ring account.\n")
            failures += 1

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
