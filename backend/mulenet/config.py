"""Paths and dataset constants for MuleNet."""

import os
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATASET_DIR = REPO_ROOT / "ibmdataset"
CACHE_DIR = REPO_ROOT / "data" / "cache"

# Which IBM AML split to work against. LI-Small is the demo/dev default:
# 6.9M transactions, 712K accounts, 117 labelled rings, 10-day span.
DEFAULT_SPLIT = "LI-Small"

# Column names in *_Trans.csv. The raw header reuses the name "Account" for
# both sides of the transfer, so we always read with explicit names.
TRANS_COLUMNS = [
    "timestamp",
    "from_bank",
    "from_account",
    "to_bank",
    "to_account",
    "amount_received",
    "receiving_currency",
    "amount_paid",
    "payment_currency",
    "payment_format",
    "is_laundering",
]

# Bank and account ids carry meaningful leading zeros ("011" != "0110"),
# so they must never be inferred as integers.
TRANS_DTYPES = {
    "from_bank": "string",
    "from_account": "string",
    "to_bank": "string",
    "to_account": "string",
    "amount_received": "float64",
    "receiving_currency": "string",
    "amount_paid": "float64",
    "payment_currency": "string",
    "payment_format": "string",
    "is_laundering": "int8",
}

TIMESTAMP_FORMAT = "%Y/%m/%d %H:%M"

ACCOUNTS_COLUMNS = ["bank_name", "bank_id", "account", "entity_id", "entity_name"]

# The eight laundering topologies IBM labels in *_Patterns.txt.
PATTERN_TYPES = [
    "FAN-IN",
    "FAN-OUT",
    "GATHER-SCATTER",
    "SCATTER-GATHER",
    "CYCLE",
    "RANDOM",
    "BIPARTITE",
    "STACK",
]


def trans_path(split: str = DEFAULT_SPLIT) -> Path:
    return DATASET_DIR / f"{split}_Trans.csv"


def accounts_path(split: str = DEFAULT_SPLIT) -> Path:
    return DATASET_DIR / f"{split}_accounts.csv"


def patterns_path(split: str = DEFAULT_SPLIT) -> Path:
    return DATASET_DIR / f"{split}_Patterns.txt"


def cache_path(split: str, name: str) -> Path:
    return CACHE_DIR / f"{split}_{name}.parquet"


GROQ_MODEL = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")

# Tried in order when every key is rate-limited on the model above. Limits are
# per-model, so a smaller model is usually still answering when the large one
# is spent. Ordered most to least capable.
GROQ_FALLBACK_MODELS = [
    m.strip()
    for m in os.environ.get(
        "GROQ_FALLBACK_MODELS", "llama-3.1-8b-instant"
    ).split(",")
    if m.strip()
]


def load_env(path: Path | None = None) -> None:
    """Read `.env` into os.environ without clobbering real env vars.

    Kept dependency-free so the detection engine never needs the agent's
    packages installed.
    """
    path = path or REPO_ROOT / ".env"
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


# Groq keys are discovered by pattern rather than from a fixed list, so
# adding AI_KEY6 to .env needs no code change. The free tier rate-limits per
# key and a 429 mid-demo is indistinguishable from a crash, so the pool
# rotates across every key configured.
KEY_PATTERN = re.compile(r"^(GROQ_API_KEY|AI_KEY\d*)$")


def _key_sort(name: str) -> tuple[int, int]:
    """GROQ_API_KEY first, then AI_KEY, AI_KEY2, AI_KEY3 … numerically.

    Sorting the names as plain strings would put AI_KEY10 before AI_KEY2.
    """
    if name == "GROQ_API_KEY":
        return (0, 0)
    suffix = name[len("AI_KEY"):]
    return (1, int(suffix) if suffix.isdigit() else 1)


def groq_api_keys() -> list[str]:
    """Every Groq key configured, in rotation order, de-duplicated."""
    load_env()
    names = sorted((n for n in os.environ if KEY_PATTERN.match(n)), key=_key_sort)
    keys, seen = [], set()
    for name in names:
        key = os.environ.get(name, "").strip()
        if key and key not in seen:
            seen.add(key)
            keys.append(key)
    if not keys:
        raise RuntimeError(
            f"No Groq key found. Set GROQ_API_KEY or AI_KEY in {REPO_ROOT / '.env'}. "
            "Free keys: https://console.groq.com/keys"
        )
    return keys


def groq_api_key() -> str:
    """The primary Groq key."""
    return groq_api_keys()[0]


def node_id(bank: str, account: str) -> str:
    """Graph node key.

    Account numbers are not globally unique (LI-Small has 4 collisions across
    banks), so a node is always identified by the (bank, account) pair.
    """
    return f"{bank}-{account}"
