"""Tests for :class:`headroom.proxy.outcome.RequestOutcome` and the
:meth:`HeadroomProxy._record_request_outcome` funnel.

The point of this file is the *contract* — every behavioural assertion
here is a thing that, prior to the funnel, lived inline at one or more
of the 18 metrics-emit sites identified in
``docs/superpowers/specs/P0-proxy-pipeline-audit.md``. Locking the
contract in tests means future migrations onto the funnel cannot
silently regress the wire shape.
"""

from __future__ import annotations

import logging
from dataclasses import FrozenInstanceError
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from headroom.proxy.outcome import RequestOutcome

# ── Value-type contract ────────────────────────────────────────────────


def _outcome(**overrides: Any) -> RequestOutcome:
    """Construct a RequestOutcome with sensible defaults; override fields per test."""
    defaults: dict[str, Any] = {
        "request_id": "req-1",
        "provider": "anthropic",
        "model": "claude-sonnet-4",
        "original_tokens": 1000,
        "optimized_tokens": 300,
        "output_tokens": 50,
        "tokens_saved": 700,
        "attempted_input_tokens": 800,
    }
    defaults.update(overrides)
    return RequestOutcome(**defaults)


def test_outcome_is_frozen() -> None:
    """Mutability would let a handler patch the outcome after handing it
    to the funnel — bypassing the contract. Must error."""
    o = _outcome()
    with pytest.raises(FrozenInstanceError):
        o.cache_read_tokens = 999  # type: ignore[misc]


def test_cache_hit_is_derived_not_stored() -> None:
    """Pre-refactor, 9 of 18 ``RequestLog`` sites hardcoded ``cache_hit=False``
    even when ``cache_read_tokens > 0``. Deriving from the actual value
    makes "forgot to compute it" structurally impossible."""
    assert _outcome(cache_read_tokens=0).cache_hit is False
    assert _outcome(cache_read_tokens=1).cache_hit is True
    assert _outcome(cache_read_tokens=500).cache_hit is True


def test_cache_hit_pct_handles_zero_denominator() -> None:
    """No reads + no writes is a no-cache request, not a 0%-hit cache request.
    Returning 0 here is correct as long as dashboards distinguish via the
    absolute ``cache_read_tokens`` / ``cache_write_tokens`` values."""
    assert _outcome(cache_read_tokens=0, cache_write_tokens=0).cache_hit_pct == 0


def test_cache_hit_pct_rounds_to_int() -> None:
    """PERF log line consumed by ``headroom perf`` parses an integer here;
    keep the type contract tight."""
    o = _outcome(cache_read_tokens=2, cache_write_tokens=1)  # 66.66%
    assert o.cache_hit_pct == 67
    assert isinstance(o.cache_hit_pct, int)


def test_savings_pct_handles_zero_original() -> None:
    """A request with 0 original tokens — e.g. an empty body — should not
    raise ZeroDivisionError. Sites pre-refactor handled this inconsistently."""
    assert _outcome(original_tokens=0).savings_pct == 0.0


def test_savings_pct_basic() -> None:
    assert _outcome(original_tokens=1000, tokens_saved=300).savings_pct == 30.0


def test_provider_specific_fields_default_to_zero() -> None:
    """Anthropic's 5m/1h cache TTL splits don't exist on OpenAI / Gemini.
    The dataclass defaults them to 0 so non-Anthropic handlers don't
    have to know about them."""
    o = _outcome(provider="openai", cache_read_tokens=100, cache_write_tokens=200)
    assert o.cache_write_5m_tokens == 0
    assert o.cache_write_1h_tokens == 0
    # And OpenAI's "inferred" flag defaults False — only the OpenAI
    # handler sets it True after running _infer_openai_cache_write_tokens.
    assert o.cache_inferred is False


def test_optional_fields_default_to_neutral_values() -> None:
    """Handlers that don't have a field (e.g. Bedrock with no waste_signals)
    must not have to pass anything — defaults handle it."""
    o = _outcome()
    assert o.ttfb_ms == 0.0
    assert o.pipeline_timing is None
    assert o.waste_signals is None
    assert o.transforms_applied == ()
    assert o.turn_id is None
    assert o.request_messages is None
    assert o.tags == {}


# ── Funnel contract (_record_request_outcome) ──────────────────────────


class _CollectingLogger:
    """Minimal stand-in for ``RequestLogger``."""

    def __init__(self) -> None:
        self.logs: list[Any] = []

    def log(self, entry: Any) -> None:
        self.logs.append(entry)


class _FunnelHarness:
    """Pulls just enough of HeadroomProxy onto an object to exercise
    ``_record_request_outcome`` without instantiating the full proxy.

    The harness assigns the real method to ``self`` via descriptor
    binding so the implementation is exactly the production one — no
    forking, no mock-the-thing-you're-testing.
    """

    def __init__(self, *, with_cost_tracker: bool = True, with_logger: bool = True) -> None:
        from headroom.proxy.server import HeadroomProxy

        self.metrics = MagicMock()
        self.metrics.record_request = AsyncMock()
        self.cost_tracker = MagicMock() if with_cost_tracker else None
        self.logger = _CollectingLogger() if with_logger else None
        # Bind the real method to this harness.
        self._record_request_outcome = HeadroomProxy._record_request_outcome.__get__(
            self, type(self)
        )


@pytest.mark.asyncio
async def test_funnel_calls_metrics_with_full_kwargs() -> None:
    """The funnel must pass EVERY field that
    ``PrometheusMetrics.record_request`` knows about, not the
    pre-refactor "pass-what-was-convenient" subset. Otherwise a handler
    that forgets to populate a field silently degrades dashboard data."""
    h = _FunnelHarness()
    o = _outcome(
        provider="openai",
        model="gpt-4",
        optimized_tokens=300,
        output_tokens=50,
        tokens_saved=700,
        attempted_input_tokens=800,
        cache_read_tokens=200,
        cache_write_tokens=100,
        cache_write_5m_tokens=50,
        cache_write_1h_tokens=50,
        uncached_input_tokens=0,
        total_latency_ms=1234.5,
        overhead_ms=12.3,
        ttfb_ms=200.0,
        pipeline_timing={"phase": 1.0},
        waste_signals={"skipped": 3},
    )
    await h._record_request_outcome(o)

    h.metrics.record_request.assert_awaited_once()
    kwargs = h.metrics.record_request.await_args.kwargs
    assert kwargs["provider"] == "openai"
    assert kwargs["model"] == "gpt-4"
    assert kwargs["input_tokens"] == 300  # optimized → input
    assert kwargs["output_tokens"] == 50
    assert kwargs["tokens_saved"] == 700
    assert kwargs["latency_ms"] == 1234.5
    assert kwargs["cached"] is True  # derived from cache_read > 0
    assert kwargs["overhead_ms"] == 12.3
    assert kwargs["ttfb_ms"] == 200.0
    assert kwargs["pipeline_timing"] == {"phase": 1.0}
    assert kwargs["waste_signals"] == {"skipped": 3}
    assert kwargs["cache_read_tokens"] == 200
    assert kwargs["cache_write_tokens"] == 100
    assert kwargs["cache_write_5m_tokens"] == 50
    assert kwargs["cache_write_1h_tokens"] == 50
    assert kwargs["uncached_input_tokens"] == 0
    assert kwargs["attempted_input_tokens"] == 800


@pytest.mark.asyncio
async def test_funnel_passes_canonical_record_tokens_shape() -> None:
    """``cost_tracker.record_tokens`` takes ``(model, tokens_saved,
    optimized_tokens)`` positionally and the cache args as kwargs. The
    funnel preserves this — moving anything to positional would break
    sites that pass kwargs explicitly."""
    h = _FunnelHarness()
    o = _outcome(
        model="claude-sonnet-4",
        optimized_tokens=300,
        tokens_saved=700,
        cache_read_tokens=200,
        cache_write_tokens=100,
        cache_write_5m_tokens=80,
        cache_write_1h_tokens=20,
        uncached_input_tokens=0,
    )
    await h._record_request_outcome(o)

    h.cost_tracker.record_tokens.assert_called_once()
    args, kwargs = h.cost_tracker.record_tokens.call_args
    assert args == ("claude-sonnet-4", 700, 300)
    assert kwargs == {
        "cache_read_tokens": 200,
        "cache_write_tokens": 100,
        "cache_write_5m_tokens": 80,
        "cache_write_1h_tokens": 20,
        "uncached_tokens": 0,
    }


@pytest.mark.asyncio
async def test_funnel_skips_cost_tracker_when_absent() -> None:
    """When the proxy was started with ``--no-cost``, ``cost_tracker``
    is None and the funnel must skip step 2 silently."""
    h = _FunnelHarness(with_cost_tracker=False)
    await h._record_request_outcome(_outcome())
    # No crash, metrics still recorded.
    h.metrics.record_request.assert_awaited_once()


@pytest.mark.asyncio
async def test_funnel_logs_request_with_derived_cache_hit() -> None:
    """The RequestLog row needs cache_hit derived from cache_read>0, not
    the hardcoded False that 9 of 18 pre-refactor sites used."""
    h = _FunnelHarness()
    await h._record_request_outcome(_outcome(cache_read_tokens=200, cache_write_tokens=100))
    assert len(h.logger.logs) == 1
    log_entry = h.logger.logs[0]
    assert log_entry.cache_hit is True


@pytest.mark.asyncio
async def test_funnel_skips_request_log_when_logger_absent() -> None:
    """Same pattern as cost_tracker — optional surface."""
    h = _FunnelHarness(with_logger=False)
    await h._record_request_outcome(_outcome())
    h.metrics.record_request.assert_awaited_once()  # still happens


@pytest.mark.asyncio
async def test_funnel_emits_perf_log_with_canonical_shape(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """``headroom perf`` parses this exact ``key=value`` format. Changing
    it breaks the analyzer. The contract: model, msgs, tok_before,
    tok_after, tok_saved, cache_read, cache_write, cache_hit_pct,
    opt_ms, transforms — in that order, space-separated."""
    h = _FunnelHarness()
    # Direct handler attach: caplog otherwise drops propagation-disabled
    # records (the proxy disables ``headroom.*`` propagation once started).
    target = logging.getLogger("headroom.proxy")
    captured: list[logging.LogRecord] = []

    class _H(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            captured.append(record)

    handler = _H(level=logging.INFO)
    target.addHandler(handler)
    prior_level = target.level
    target.setLevel(logging.INFO)
    try:
        await h._record_request_outcome(
            _outcome(
                request_id="req-perf",
                model="gpt-4",
                original_tokens=1000,
                optimized_tokens=300,
                tokens_saved=700,
                cache_read_tokens=200,
                cache_write_tokens=100,
                num_messages=5,
                overhead_ms=12.0,
                transforms_applied=("smart_crusher", "content_router"),
            )
        )
    finally:
        target.removeHandler(handler)
        target.setLevel(prior_level)

    perf_lines = [r.getMessage() for r in captured if " PERF " in r.getMessage()]
    assert len(perf_lines) == 1
    line = perf_lines[0]
    assert "[req-perf] PERF " in line
    assert "model=gpt-4" in line
    assert "msgs=5" in line
    assert "tok_before=1000" in line
    assert "tok_after=300" in line
    assert "tok_saved=700" in line
    assert "cache_read=200" in line
    assert "cache_write=100" in line
    assert "cache_hit_pct=67" in line  # 200/(200+100) * 100 = 67
    assert "opt_ms=12" in line
