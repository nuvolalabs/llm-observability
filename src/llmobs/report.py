"""Report rendering — tracing/metrics results to JSON or Markdown."""
from __future__ import annotations

import json


def to_json(obj) -> str:
    if hasattr(obj, "to_dict"):
        obj = obj.to_dict()
    return json.dumps(obj, indent=2, default=str)


def render_markdown(trace: dict, latency: dict | None = None,
                    tokens=None, drift: dict | None = None) -> str:
    """Render an observability summary as Markdown."""
    lines = [f"# LLM Observability Report", "", f"**Trace:** `{trace.get('trace_id', 'n/a')}`", ""]

    spans = trace.get("spans", [])
    lines += ["## Spans", "", "| span | parent | duration (ms) | status |", "|---|---|---|---|"]
    for s in spans:
        lines.append(f"| {s['name']} | {s.get('parent_id') or '-'} | {s['duration_ms']} | {s['status']} |")
    lines.append("")

    if latency:
        lines += ["## Latency", "",
                  f"- count: **{latency['count']}**",
                  f"- mean: **{latency['mean']} ms**",
                  f"- p50: **{latency['p50']} ms**  ·  p95: **{latency['p95']} ms**  ·  p99: **{latency['p99']} ms**",
                  f"- max: **{latency['max']} ms**", ""]

    if tokens is not None:
        total = tokens.total_tokens if hasattr(tokens, "total_tokens") else tokens["total_tokens"]
        lines += ["## Usage & cost", "",
                  f"- calls: **{tokens.calls}**",
                  f"- tokens: **{total}** (in {tokens.input_tokens} / out {tokens.output_tokens})",
                  f"- cost: **${tokens.cost_usd:.6f}**", ""]

    if drift:
        lines += ["## Drift", "",
                  f"- PSI: **{drift['psi']}** → **{drift['verdict']}**", ""]

    return "\n".join(lines)
