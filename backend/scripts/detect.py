"""Run the full detection pipeline and score it as a ranked alert queue.

    python scripts/detect.py [--cycle-length 3] [--k 10 25 50 100]
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mulenet import detectors, graph as graph_mod, loader, rings as rings_mod, scoring  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cycle-length", type=int, default=3)
    ap.add_argument("--k", type=int, nargs="+", default=[10, 25, 50, 100, 250])
    args = ap.parse_args()

    labelled = loader.load_cache()
    truth = scoring.ground_truth(labelled)

    # Detection runs on the prefiltered slice; scoring stays against the full
    # ground truth so the recall cost of the filter is always visible.
    search = detectors.prefilter(labelled)
    print(f"search space {len(search):,} of {len(labelled):,} transactions\n")
    g = graph_mod.build_graph(search)

    candidates = []
    for name, fn in [
        ("fan_out", lambda: detectors.fan_out(search)),
        ("fan_in", lambda: detectors.fan_in(search)),
        ("pass_through", lambda: detectors.pass_through(search)),
        ("velocity_burst", lambda: detectors.velocity_burst(search)),
        ("cycle", lambda: detectors.cycles(g, max_length=args.cycle_length)),
    ]:
        started = time.time()
        found = fn()
        candidates += found
        print(f"{name:<16} {len(found):>7} candidates  {time.time() - started:6.1f}s")
        sys.stdout.flush()

    started = time.time()
    ranked = rings_mod.combine(candidates)
    print(f"{'combine':<16} {len(ranked):>7} rings       {time.time() - started:6.1f}s\n")

    multi = [r for r in ranked if len(r.signals) >= 2]
    print(f"rings with >=2 distinct signals: {len(multi)}")
    print(f"rings with >=3 distinct signals: {sum(1 for r in ranked if len(r.signals) >= 3)}\n")

    print(f"{'k':>6} {'prec@k':>8} {'recall':>8} {'rings hit':>10} {'accts hit':>10}")
    for k in args.k:
        result = scoring.score(rings_mod.top_k(ranked, k), truth)
        acc, rg = result["accounts"], result["rings"]
        print(
            f"{k:>6} {acc['precision_strict']:>8.3f} {acc['recall']:>8.3f}"
            f" {rg['detected_of_detectable']:>4}/{rg['detectable_total']:<5}"
            f" {acc['true_positives']:>5}/{len(truth.all_ring_accounts):<5}"
        )

    print("\nFULL QUEUE")
    print(scoring.format_report(scoring.score([r.accounts for r in ranked], truth)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
