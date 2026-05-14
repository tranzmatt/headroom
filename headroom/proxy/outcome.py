"""``RequestOutcome``: the canonical value type for "what happened during
one completed proxy request."

Per the P0 audit (``docs/superpowers/specs/P0-proxy-pipeline-audit.md``),
18 ``metrics.record_request`` call sites across four handler files
disagreed on argument shape — 9 of 18 omitted ``cached=``, 7 of 18
omitted ``attempted_input_tokens=``, only 4 sites emitted a structured
PERF log at all. This module is the structural fix: every handler
converges on building a :class:`RequestOutcome` at end-of-request and
hands it to :meth:`HeadroomProxy._record_request_outcome`, which owns
the four downstream effects (Prometheus, cost tracker, request logger,
PERF log).

Note: this is **output unification, not input unification**. Provider
APIs (Anthropic ``/v1/messages``, OpenAI Responses WS, Gemini
``generateContent``, Bedrock, Vertex) stay wildly different — the proxy
talks each upstream in its native dialect. This dataclass standardises
only the *observation* about a completed request. Provider-specific
concepts (Anthropic's 5m/1h cache TTL splits, OpenAI's
inferred-write flag, Gemini's read-only cache count) live as optional
fields with neutral defaults; handlers populate what their provider
actually reports.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class RequestOutcome:
    """Immutable, value-equal snapshot of a completed request.

    Construction policy: every field that downstream consumers read MUST
    be either required (no default) or have a neutral default that makes
    the consumer's behaviour identical to "field not present". This keeps
    the contract honest — a handler that forgets a field doesn't silently
    produce wrong metrics; it produces zeros, which the dashboard can
    surface as a missing-data condition (P3 follow-up).
    """

    # ── Identity ──────────────────────────────────────────────────────
    request_id: str
    provider: str
    model: str

    # ── Tokens (required — every site has these) ──────────────────────
    # original_tokens: pre-compression request size, for `tok_before`
    # optimized_tokens: post-compression bytes actually forwarded, for
    #     ``input_tokens`` and ``tok_after``
    # output_tokens: response tokens from upstream
    # tokens_saved: original - optimized (or 0 if compression bypassed)
    # attempted_input_tokens: denominator for active-savings-percent.
    #     The compressible portion only — excludes user messages, system
    #     prompts, prior assistant turns, frozen prefix bytes. This is the
    #     field 7 of 18 audit sites forgot to pass, collapsing
    #     ``active_savings_percent`` to 0 (#454 / #455).
    original_tokens: int
    optimized_tokens: int
    output_tokens: int
    tokens_saved: int
    attempted_input_tokens: int

    # ── Cache (provider-agnostic; unused fields stay 0) ───────────────
    # Anthropic populates all five (read + write + 5m + 1h + uncached).
    # OpenAI populates read + inferred-write + uncached, and sets
    # ``cache_inferred=True`` so the dashboard can warn that the write
    # column is an estimate rather than an upstream-reported counter.
    # Gemini populates read only.
    # Bedrock mirrors Anthropic (it forwards Anthropic-shape usage).
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    cache_write_5m_tokens: int = 0
    cache_write_1h_tokens: int = 0
    uncached_input_tokens: int = 0
    cache_inferred: bool = False

    # ── Timing ────────────────────────────────────────────────────────
    # total_latency_ms: wall-clock end-to-end for this request
    # overhead_ms: time spent in compression dispatch only (subset of total)
    # ttfb_ms: time to first upstream byte for streaming paths; 0 for
    #     non-streaming or when unmeasured (no None — convention is 0)
    # pipeline_timing: optional per-stage breakdown surfaced on dashboards
    total_latency_ms: float = 0.0
    overhead_ms: float = 0.0
    ttfb_ms: float = 0.0
    pipeline_timing: dict[str, float] | None = None

    # ── Transforms + diagnostics ──────────────────────────────────────
    # transforms_applied: tuple (immutable) of every transform that ran.
    #     RequestLog still wants list[str]; the funnel converts at the
    #     boundary.
    # waste_signals: per-router signals captured during routing (counts
    #     of skipped vs applied units etc.); dashboards summarise.
    # num_messages: messages in the original request (for ``msgs=N`` in
    #     PERF), counted from body.input/body.messages.
    # turn_id: stable hash of the conversation prefix; used by
    #     dashboards to group multi-turn sessions.
    # request_messages: only populated when ``config.log_full_messages``
    #     is enabled (off by default — message bodies are sensitive).
    # tags: client-provided routing/identification tags.
    transforms_applied: tuple[str, ...] = ()
    waste_signals: dict[str, int] | None = None
    num_messages: int = 0
    turn_id: str | None = None
    request_messages: list[dict[str, Any]] | None = None
    tags: dict[str, str] = field(default_factory=dict)

    # ── Derived (computed once, no caching needed — properties are cheap) ─

    @property
    def cache_hit(self) -> bool:
        """True iff upstream reported any cache read.

        Used by ``RequestLog.cache_hit`` and ``record_request(cached=...)``.
        Pre-refactor 9 of 18 sites hardcoded this to False — this property
        makes "I forgot to compute it" structurally impossible.
        """
        return self.cache_read_tokens > 0

    @property
    def cache_hit_pct(self) -> int:
        """Cache read share of (read + write), rounded to int percent.

        Returns 0 when neither read nor write fired (a request that did no
        cache work; distinguishing this from "0% hit rate on real cache
        work" requires looking at the absolute values, not the ratio).
        """
        denom = self.cache_read_tokens + self.cache_write_tokens
        if denom <= 0:
            return 0
        return round(self.cache_read_tokens / denom * 100)

    @property
    def savings_pct(self) -> float:
        """Compression savings as a fraction of the original request size.

        This is the proxy-side ratio: ``tokens_saved / original_tokens``.
        The dashboard headline "active savings percent" uses a different
        ratio (``tokens_saved / attempted_input_tokens``) — see the
        Prometheus metric for the active calculation.
        """
        if self.original_tokens <= 0:
            return 0.0
        return self.tokens_saved / self.original_tokens * 100.0
