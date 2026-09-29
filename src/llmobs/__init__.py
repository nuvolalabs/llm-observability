"""LLM observability — tracing, token/cost accounting, latency stats, drift detection."""
from .tracing import Span, Tracer
from .cost import ModelPricing, compute_cost, PRICING
from .metrics import percentile, latency_stats, TokenSummary, summarize_tokens
from .drift import population_stability_index, drift_report
from .report import render_markdown, to_json

__all__ = [
    "Span", "Tracer",
    "ModelPricing", "compute_cost", "PRICING",
    "percentile", "latency_stats", "TokenSummary", "summarize_tokens",
    "population_stability_index", "drift_report",
    "render_markdown", "to_json",
]
__version__ = "1.0.0"
