"""Token accounting and cost estimation with a pluggable pricing table."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelPricing:
    """Price in USD per 1,000 tokens."""

    input_per_1k: float
    output_per_1k: float


# Illustrative prices (USD / 1K tokens). Update freely — the shape matters, not the numbers.
PRICING: dict[str, ModelPricing] = {
    "gpt-4o-mini": ModelPricing(0.00015, 0.0006),
    "gpt-4o": ModelPricing(0.005, 0.015),
    "claude-3-5-sonnet": ModelPricing(0.003, 0.015),
    "llama-3.1-70b": ModelPricing(0.00059, 0.00079),
    "default": ModelPricing(0.001, 0.002),
}


def compute_cost(model: str, input_tokens: int, output_tokens: int,
                 pricing: dict[str, ModelPricing] | None = None) -> float:
    """Return USD cost for a call. Unknown models fall back to 'default'."""
    table = pricing if pricing is not None else PRICING
    p = table.get(model) or table.get("default") or ModelPricing(0.001, 0.002)
    cost = (input_tokens / 1000.0) * p.input_per_1k + (output_tokens / 1000.0) * p.output_per_1k
    return round(cost, 6)
