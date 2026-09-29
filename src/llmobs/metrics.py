"""Latency percentiles and token/cost summarisation."""
from __future__ import annotations

from dataclasses import dataclass, field

from .cost import PRICING, ModelPricing, compute_cost


def percentile(values: list[float], p: float) -> float:
    """Linear-interpolation percentile (p in 0..100). Returns 0.0 for empty input."""
    if not values:
        return 0.0
    if not 0 <= p <= 100:
        raise ValueError("p must be between 0 and 100")
    data = sorted(values)
    if len(data) == 1:
        return round(data[0], 3)
    rank = (p / 100.0) * (len(data) - 1)
    lo = int(rank)
    hi = min(lo + 1, len(data) - 1)
    frac = rank - lo
    return round(data[lo] + (data[hi] - data[lo]) * frac, 3)


def latency_stats(durations_ms: list[float]) -> dict:
    """min / mean / max / p50 / p95 / p99 for a list of durations."""
    if not durations_ms:
        return {"count": 0, "min": 0.0, "mean": 0.0, "max": 0.0, "p50": 0.0, "p95": 0.0, "p99": 0.0}
    return {
        "count": len(durations_ms),
        "min": round(min(durations_ms), 3),
        "mean": round(sum(durations_ms) / len(durations_ms), 3),
        "max": round(max(durations_ms), 3),
        "p50": percentile(durations_ms, 50),
        "p95": percentile(durations_ms, 95),
        "p99": percentile(durations_ms, 99),
    }


@dataclass
class TokenSummary:
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    by_model: dict = field(default_factory=dict)

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


def summarize_tokens(records: list[dict], pricing: dict[str, ModelPricing] | None = None) -> TokenSummary:
    """Aggregate usage records.

    Each record: {"model": str, "input_tokens": int, "output_tokens": int, ...}
    """
    table = pricing if pricing is not None else PRICING
    s = TokenSummary()
    for r in records:
        model = r.get("model", "default")
        it, ot = int(r.get("input_tokens", 0)), int(r.get("output_tokens", 0))
        cost = compute_cost(model, it, ot, table)
        s.calls += 1
        s.input_tokens += it
        s.output_tokens += ot
        s.cost_usd = round(s.cost_usd + cost, 6)
        m = s.by_model.setdefault(model, {"calls": 0, "input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0})
        m["calls"] += 1
        m["input_tokens"] += it
        m["output_tokens"] += ot
        m["cost_usd"] = round(m["cost_usd"] + cost, 6)
    return s
