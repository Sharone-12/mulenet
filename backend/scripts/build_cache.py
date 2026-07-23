"""Build the labelled transaction cache for a dataset split.

    python scripts/build_cache.py [--split LI-Small] [--force]
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mulenet import config, loader  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default=config.DEFAULT_SPLIT)
    ap.add_argument("--force", action="store_true", help="rebuild even if cached")
    args = ap.parse_args()

    started = time.time()
    out, report = loader.build_cache(args.split, force=args.force)
    elapsed = time.time() - started

    print(f"wrote {out}  ({elapsed:.1f}s)")
    for key, value in report.items():
        print(f"  {key:<28} {value}")

    if report.get("unmatched_pattern_keys"):
        print("\nWARNING: some labelled rings did not match any transaction row.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
