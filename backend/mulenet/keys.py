"""Groq key pool with failover.

The free tier limits each key independently, so a live demo that burns
through one key should slide onto the next rather than surfacing a 429. The
pool is shared process-wide: every agent session and the SAR endpoint draw
from the same state, so a key exhausted in one place is not immediately
retried in another.

Two failure modes are distinguished:

* rate-limited / quota exhausted - temporary. The key is benched for a
  cooldown and comes back automatically.
* invalid / revoked - permanent. The key is retired for the process, since
  retrying it only wastes a round trip on every future call.
"""

from __future__ import annotations

import re
import threading
import time
from dataclasses import dataclass, field

from . import config

# Groq states the exact reset time in its 429 ("Please try again in 1.23s"),
# and that is the only figure worth benching a key on. The previous flat 65s
# guess sidelined keys that were seconds from being usable, so a burst could
# take out all five and report "next free in 64s" when the true wait was
# under two.
#
# When no time can be parsed the key is NOT benched at all - it simply loses
# its turn in the rotation and is retried on the next pass.
FALLBACK_COOLDOWN_SECONDS = 0.0

# A short wait beats failing the question outright. Generous, because the
# waits are now real reset times rather than a padded guess.
MAX_WAIT_SECONDS = 25.0

# Parses "Please try again in 1.23s" / "in 2m30.5s" out of the error body.
_RETRY_AFTER = re.compile(
    r"try again in\s+(?:(\d+)m)?\s*([\d.]+)s",
    re.IGNORECASE,
)


def _retry_after(exc: Exception) -> float | None:
    """Seconds until this key resets, as reported by Groq itself."""
    match = _RETRY_AFTER.search(str(exc))
    if not match:
        return None
    minutes = float(match.group(1) or 0)
    seconds = float(match.group(2))
    # A small cushion, since the clock started before the response arrived.
    return minutes * 60 + seconds + 0.25

RATE_MARKERS = (
    "429",
    "rate limit",
    "rate_limit",
    "quota",
    "insufficient_quota",
    "too many requests",
    "capacity",
)

AUTH_MARKERS = (
    "401",
    "403",
    "invalid api key",
    "invalid_api_key",
    "unauthorized",
    "authentication",
)


def _classify(exc: Exception) -> str:
    text = str(exc).lower()
    if any(m in text for m in AUTH_MARKERS):
        return "auth"
    if any(m in text for m in RATE_MARKERS):
        return "rate"
    return "other"


@dataclass
class _Key:
    index: int
    value: str
    client: object
    # Benching is per MODEL, because Groq's limits are per model. A key spent
    # on the 70B still has full quota on a smaller one, so a single
    # benched_until would wrongly block the fallback path too.
    benched_until: dict[str, float] = field(default_factory=dict)
    retired: bool = False
    calls: int = 0
    failures: int = 0

    def available(self, now: float, model: str) -> bool:
        return not self.retired and now >= self.benched_until.get(model, 0.0)

    def free_at(self, model: str) -> float:
        return self.benched_until.get(model, 0.0)


@dataclass
class KeyPool:
    """Round-robins Groq keys, skipping any that are exhausted."""

    keys: list[_Key] = field(default_factory=list)
    cursor: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    @classmethod
    def from_env(cls) -> "KeyPool":
        from groq import Groq

        return cls(
            keys=[
                _Key(index=i, value=k, client=Groq(api_key=k))
                for i, k in enumerate(config.groq_api_keys())
            ]
        )

    def _next_available(self, now: float, model: str) -> _Key | None:
        """Round-robin from the cursor so load spreads across keys."""
        total = len(self.keys)
        for offset in range(total):
            key = self.keys[(self.cursor + offset) % total]
            if key.available(now, model):
                self.cursor = (self.cursor + offset) % total
                return key
        return None

    def complete(self, **kwargs):
        """One chat completion, failing over across keys and then models.

        Groq rate-limits per model, so when every key is spent on the large
        model a smaller one still has quota. Dropping to it keeps a live demo
        answering instead of erroring; the tools and the detection results are
        identical either way, only the narration gets terser.
        """
        requested = kwargs.get("model", config.GROQ_MODEL)
        chain = [requested] + [m for m in config.GROQ_FALLBACK_MODELS if m != requested]

        last: Exception | None = None
        for model in chain:
            try:
                return self._complete_on(dict(kwargs, model=model))
            except RuntimeError as exc:
                # Only key exhaustion is worth trying another model for.
                if "unavailable" not in str(exc):
                    raise
                last = exc
        raise last

    def _complete_on(self, kwargs: dict):
        """Run the whole pool against one model."""
        model = kwargs.get("model", config.GROQ_MODEL)
        attempts = 0
        while True:
            now = time.monotonic()
            with self._lock:
                key = self._next_available(now, model)
                if key is None:
                    soonest = min(
                        (k.free_at(model) for k in self.keys if not k.retired),
                        default=None,
                    )
                    wait = None if soonest is None else soonest - now

                if key is not None:
                    key.calls += 1

            if key is None:
                if wait is not None and 0 < wait <= MAX_WAIT_SECONDS:
                    time.sleep(wait + 0.1)
                    continue
                live = sum(1 for k in self.keys if not k.retired)
                raise RuntimeError(
                    f"All {len(self.keys)} Groq keys are unavailable "
                    f"({live} rate-limited, {len(self.keys) - live} invalid). "
                    + (f"Next free in {wait:.1f}s." if wait else "Check the keys in .env.")
                )

            try:
                response = key.client.chat.completions.create(**kwargs)
                with self._lock:
                    # Move on so the next call starts at a different key.
                    self.cursor = (key.index + 1) % len(self.keys)
                return response
            except Exception as exc:  # noqa: BLE001
                kind = _classify(exc)
                with self._lock:
                    key.failures += 1
                    if kind == "auth":
                        key.retired = True
                    elif kind == "rate":
                        wait = _retry_after(exc)
                        if wait is None:
                            wait = FALLBACK_COOLDOWN_SECONDS
                        key.benched_until[model] = time.monotonic() + wait
                    else:
                        # Not a key problem - the caller needs the real error.
                        raise

                    # Always hand the next attempt to a different key. Without
                    # this, a key benched for 0s stays "available" and
                    # _next_available returns it again, spinning on the same
                    # dead key until the attempt cap instead of failing over.
                    self.cursor = (key.index + 1) % len(self.keys)
                attempts += 1
                if attempts >= len(self.keys) * 2:
                    raise RuntimeError(
                        f"Groq failover exhausted after {attempts} attempts: {exc}"
                    ) from exc

    def status(self) -> dict:
        """Pool health, reported per model.

        Counting a key as benched because it is limited on ANY model was
        misleading: with the fallback in play the pool answered normally while
        /health claimed 0 of 5 available. What matters is whether some model
        can still be served, so that is what this reports.
        """
        now = time.monotonic()
        models = [config.GROQ_MODEL] + list(config.GROQ_FALLBACK_MODELS)
        live = [k for k in self.keys if not k.retired]

        per_model = {
            m: sum(1 for k in live if k.available(now, m)) for m in dict.fromkeys(models)
        }
        serving = next((m for m, n in per_model.items() if n), None)

        return {
            "total": len(self.keys),
            # Usable on the preferred model.
            "available": per_model.get(config.GROQ_MODEL, 0),
            # Usable on some model, which is what decides whether we can answer.
            "usable_any_model": sum(
                1 for k in live if any(k.available(now, m) for m in per_model)
            ),
            "benched": sum(
                1 for k in live if not any(k.available(now, m) for m in per_model)
            ),
            "retired": sum(1 for k in self.keys if k.retired),
            "by_model": per_model,
            "serving_model": serving,
            "degraded": serving is not None and serving != config.GROQ_MODEL,
            "calls": sum(k.calls for k in self.keys),
            "failures": sum(k.failures for k in self.keys),
        }


_pool: KeyPool | None = None
_pool_lock = threading.Lock()


def pool() -> KeyPool:
    """The process-wide pool, built on first use."""
    global _pool
    with _pool_lock:
        if _pool is None:
            _pool = KeyPool.from_env()
        return _pool
