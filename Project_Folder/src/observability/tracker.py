"""Resource Tracker for Span-Level Latency, TTFT, Token Accounting, and Cost Estimation.

Features:
- Sub-span latency profiling (waterfall execution timeline)
- Time to First Token (TTFT) measurement
- Unified token usage accounting (prompt, completion, total)
- Dynamic monetary cost ($ USD) estimation across major LLM vendors
- Generates comprehensive telemetry payload for WebSocket emission and observability platforms
"""

import time
from typing import Any

from src.core.logger import get_logger

logger = get_logger(__name__)

# Model pricing catalog: (Price per 1M prompt tokens, Price per 1M completion tokens) in USD
MODEL_PRICING_CATALOG: dict[str, tuple[float, float]] = {
    # OpenAI
    "gpt-4o": (2.50, 10.00),
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4-turbo": (10.00, 30.00),
    # Anthropic
    "claude-3-5-sonnet-20240620": (3.00, 15.00),
    "claude-3-5-sonnet-20241022": (3.00, 15.00),
    "claude-3-5-sonnet": (3.00, 15.00),
    "claude-3-haiku-20240307": (0.25, 1.25),
    "claude-3-haiku": (0.25, 1.25),
    # Google Gemini
    "gemini-1.5-pro": (1.25, 5.00),
    "gemini-1.5-flash": (0.075, 0.30),
    "gemini-2.0-flash": (0.10, 0.40),
    # Groq / Open Models
    "llama-3.1-70b-versatile": (0.59, 0.79),
    "llama-3.1-8b-instant": (0.05, 0.08),
    "mixtral-8x7b-32768": (0.24, 0.24),
}


class SpanMetric:
    """Represents a single step or operation span."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.start_time: float = time.perf_counter()
        self.end_time: float | None = None
        self.duration_ms: float = 0.0
        self.details: dict[str, Any] = {}

    def finish(self, details: dict[str, Any] | None = None) -> float:
        self.end_time = time.perf_counter()
        self.duration_ms = round((self.end_time - self.start_time) * 1000, 2)
        if details:
            self.details.update(details)
        return self.duration_ms

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "duration_ms": self.duration_ms,
            "details": self.details,
        }


class PipelineRunTracker:
    """Tracks latency, tokens, cost, and lifecycle spans for a single RAG pipeline execution."""

    def __init__(self, trace_id: str, model_name: str) -> None:
        self.trace_id = trace_id
        self.model_name = model_name
        self.start_time: float = time.perf_counter()
        self.end_time: float | None = None
        self.ttft_ms: float | None = None
        self.spans: list[SpanMetric] = []
        self._active_spans: dict[str, SpanMetric] = {}

        # Token counters
        self.prompt_tokens: int = 0
        self.completion_tokens: int = 0
        self.total_tokens: int = 0

    def start_span(self, name: str) -> SpanMetric:
        """Begin timing a sub-span."""
        span = SpanMetric(name)
        self._active_spans[name] = span
        return span

    def end_span(self, name: str, details: dict[str, Any] | None = None) -> float:
        """Complete timing a sub-span."""
        span = self._active_spans.pop(name, None)
        if span:
            duration = span.finish(details)
            self.spans.append(span)
            return duration
        return 0.0

    def record_first_token(self) -> float:
        """Record Time to First Token (TTFT)."""
        if self.ttft_ms is None:
            self.ttft_ms = round((time.perf_counter() - self.start_time) * 1000, 2)
            logger.debug(f"TTFT recorded: {self.ttft_ms}ms")
        return self.ttft_ms

    def add_tokens(self, prompt_tokens: int, completion_tokens: int) -> None:
        """Accumulate token counts from LLM calls."""
        self.prompt_tokens += prompt_tokens
        self.completion_tokens += completion_tokens
        self.total_tokens = self.prompt_tokens + self.completion_tokens

    def estimate_cost_usd(self) -> float:
        """Estimate total monetary cost in USD based on model pricing."""
        # Normalize model name match
        matched_pricing = None
        for key, pricing in MODEL_PRICING_CATALOG.items():
            if key in self.model_name.lower():
                matched_pricing = pricing
                break

        if not matched_pricing:
            return 0.0

        p_rate, c_rate = matched_pricing
        cost = (self.prompt_tokens / 1_000_000 * p_rate) + (self.completion_tokens / 1_000_000 * c_rate)
        return round(cost, 6)

    def finalize(self) -> dict[str, Any]:
        """Finalize total execution and return waterfall breakdown dictionary."""
        self.end_time = time.perf_counter()
        total_latency_ms = round((self.end_time - self.start_time) * 1000, 2)
        cost_usd = self.estimate_cost_usd()

        return {
            "trace_id": self.trace_id,
            "model_name": self.model_name,
            "total_latency_ms": total_latency_ms,
            "ttft_ms": self.ttft_ms,
            "tokens": {
                "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens,
                "total_tokens": self.total_tokens,
            },
            "estimated_cost_usd": cost_usd,
            "waterfall_spans": [span.to_dict() for span in self.spans],
        }
