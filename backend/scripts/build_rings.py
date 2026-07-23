"""Precompute the ranked ring queue so the agent tools are instant.

    python scripts/build_rings.py [--split LI-Small] [--cycle-length 3]
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mulenet import config, engine  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default=config.DEFAULT_SPLIT)
    ap.add_argument("--cycle-length", type=int, default=3)
    args = ap.parse_args()

    started = time.time()
    df = engine.build(args.split, cycle_length=args.cycle_length)
    print(f"wrote {config.cache_path(args.split, engine.RINGS_CACHE)}")
    print(f"  rings              {len(df):,}")
    print(f"  accounts covered   {df['account_count'].sum():,}")
    print(f"  elapsed            {time.time() - started:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
