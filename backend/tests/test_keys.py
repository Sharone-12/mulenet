"""Failover guards on the Groq key pool.

Rate-limit behaviour cannot be provoked on demand, so these drive fake
clients that raise the errors Groq returns.
"""

import os
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mulenet import config, keys as keys_mod  # noqa: E402


class FakeClient:
    """Raises `error` for the first `fail_times` calls, then succeeds."""

    def __init__(self, name, error=None, fail_times=0):
        self.name = name
        self.error = error
        self.fail_times = fail_times
        self.calls = 0
        self.chat = self

    @property
    def completions(self):
        return self

    def create(self, **kwargs):
        self.calls += 1
        if self.error and self.calls <= self.fail_times:
            raise self.error
        return f"ok:{self.name}"


def pool_of(*clients):
    return keys_mod.KeyPool(
        keys=[
            keys_mod._Key(index=i, value=f"k{i}", client=c)
            for i, c in enumerate(clients)
        ]
    )


def test_rate_limited_key_fails_over_to_the_next():
    a = FakeClient("a", Exception("429 rate limit exceeded"), fail_times=99)
    b = FakeClient("b")
    assert pool_of(a, b).complete(model="x") == "ok:b"


def test_exhausted_key_is_benched_not_retried():
    """A benched key must not be hit again on the next call - that is the
    whole point of sharing pool state across turns."""
    a = FakeClient("a", Exception("429 rate limit. Please try again in 600.0s."), fail_times=99)
    b = FakeClient("b")
    p = pool_of(a, b)
    for _ in range(4):
        p.complete(model="x")
    assert a.calls == 1
    assert not p.keys[0].available(time.monotonic(), "x")


def test_invalid_key_is_retired_permanently():
    a = FakeClient("a", Exception("401 Invalid API Key"), fail_times=99)
    b = FakeClient("b")
    p = pool_of(a, b)
    p.complete(model="x")
    assert p.status()["retired"] == 1
    assert p.keys[0].benched_until == {}


def test_quota_exhaustion_counts_as_rate_limited():
    a = FakeClient("a", Exception("insufficient_quota: daily limit reached"), fail_times=99)
    b = FakeClient("b")
    assert pool_of(a, b).complete(model="x") == "ok:b"


def test_load_spreads_across_keys():
    """Consecutive calls should not hammer one key into its limit."""
    clients = [FakeClient(str(i)) for i in range(3)]
    p = pool_of(*clients)
    for _ in range(6):
        p.complete(model="x")
    assert [c.calls for c in clients] == [2, 2, 2]


def test_non_key_errors_are_not_swallowed():
    """A malformed request must surface, not silently burn every key."""
    a = FakeClient("a", ValueError("bad model name"), fail_times=99)
    with pytest.raises(ValueError):
        pool_of(a, FakeClient("b")).complete(model="x")


def test_all_keys_down_raises_a_useful_message(monkeypatch):
    monkeypatch.setattr(keys_mod.config, "GROQ_FALLBACK_MODELS", [])
    err = Exception("429 rate limit. Please try again in 600.0s.")
    p = pool_of(FakeClient("a", err, 99), FakeClient("b", err, 99))
    with pytest.raises(RuntimeError, match="unavailable"):
        p.complete(model="x")


def test_lone_benched_key_recovers_rather_than_failing():
    """When the only key frees up shortly, waiting beats erroring at the user."""
    a = FakeClient("a", Exception("429 rate limit. Please try again in 0.05s."), fail_times=1)
    p = pool_of(a)
    started = time.monotonic()
    assert p.complete(model="x") == "ok:a"
    assert time.monotonic() - started >= 0.05
    assert p.status()["available"] == 1


def test_long_wait_errors_instead_of_stalling(monkeypatch):
    """Beyond MAX_WAIT_SECONDS an honest error beats a silent hang."""
    monkeypatch.setattr(keys_mod.config, "GROQ_FALLBACK_MODELS", [])
    a = FakeClient("a", Exception("429 rate limit. Please try again in 600.0s."), fail_times=99)
    with pytest.raises(RuntimeError, match="unavailable"):
        pool_of(a).complete(model="x")


@pytest.fixture
def clean_env(monkeypatch):
    """Isolate from the real .env, which groq_api_keys() otherwise reloads."""
    monkeypatch.setattr(config, "load_env", lambda *a, **k: None)
    for name in list(os.environ):
        if config.KEY_PATTERN.match(name):
            monkeypatch.delenv(name, raising=False)


def test_every_configured_key_is_discovered(clean_env, monkeypatch):
    """The earlier hardcoded list silently ignored AI_KEY4 and AI_KEY5."""
    for i, name in enumerate(("AI_KEY", "AI_KEY2", "AI_KEY3", "AI_KEY4", "AI_KEY5")):
        monkeypatch.setenv(name, f"gsk_test_{i}")
    assert len(config.groq_api_keys()) == 5


def test_keys_are_ordered_numerically(clean_env, monkeypatch):
    """String ordering would put AI_KEY10 ahead of AI_KEY2."""
    monkeypatch.setenv("AI_KEY2", "gsk_b")
    monkeypatch.setenv("AI_KEY10", "gsk_c")
    monkeypatch.setenv("GROQ_API_KEY", "gsk_a")
    assert config.groq_api_keys() == ["gsk_a", "gsk_b", "gsk_c"]


def test_duplicate_keys_are_collapsed(clean_env, monkeypatch):
    monkeypatch.setenv("AI_KEY", "gsk_same")
    monkeypatch.setenv("AI_KEY2", "gsk_same")
    assert config.groq_api_keys() == ["gsk_same"]


def test_no_keys_raises_a_directive_error(clean_env):
    with pytest.raises(RuntimeError, match="console.groq.com"):
        config.groq_api_keys()


def test_retry_after_is_read_from_the_error(monkeypatch):
    """Groq states the real reset time; a flat guess benched keys that were
    a second from usable and reported "next free in 64s"."""
    exc = Exception(
        "429 Rate limit reached for model `llama-3.3-70b-versatile` ... "
        "Please try again in 1.23s. Visit https://console.groq.com/docs"
    )
    assert keys_mod._retry_after(exc) == pytest.approx(1.48, abs=0.01)


def test_retry_after_handles_minutes():
    exc = Exception("Please try again in 2m30.5s.")
    assert keys_mod._retry_after(exc) == pytest.approx(150.75, abs=0.01)


def test_unparseable_rate_limit_does_not_bench_the_key():
    """With no stated time the key loses its turn but is not sidelined."""
    a = FakeClient("a", Exception("429 too many requests"), fail_times=1)
    p = pool_of(a, FakeClient("b"))
    p.complete(model="x")
    assert p.keys[0].free_at('x') <= time.monotonic()


def test_falls_back_to_a_smaller_model_when_all_keys_are_limited(monkeypatch):
    """Limits are per-model, so the demo should drop models rather than die."""
    monkeypatch.setattr(keys_mod.config, "GROQ_FALLBACK_MODELS", ["small-model"])

    class ModelAware:
        def __init__(self):
            self.chat = self
            self.models = []

        @property
        def completions(self):
            return self

        def create(self, **kwargs):
            self.models.append(kwargs["model"])
            if kwargs["model"] != "small-model":
                raise Exception("429 rate limit. Please try again in 600.0s.")
            return "ok:small"

    c = ModelAware()
    p = pool_of(c)
    assert p.complete(model="big-model") == "ok:small"
    assert c.models == ["big-model", "small-model"]


def test_status_does_not_report_dead_while_a_fallback_is_serving(monkeypatch):
    """The bug this replaced: /health said 0 of 5 available while the pool was
    answering every request on the fallback model."""
    monkeypatch.setattr(keys_mod.config, "GROQ_MODEL", "big")
    monkeypatch.setattr(keys_mod.config, "GROQ_FALLBACK_MODELS", ["small"])

    p = pool_of(FakeClient("a"), FakeClient("b"))
    for k in p.keys:
        k.benched_until["big"] = time.monotonic() + 600

    st = p.status()
    assert st["available"] == 0          # none on the preferred model
    assert st["usable_any_model"] == 2   # but both can still answer
    assert st["benched"] == 0
    assert st["serving_model"] == "small"
    assert st["degraded"] is True
