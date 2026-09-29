# LLM Observability

A dependency-free observability toolkit for LLM/agent systems: **hierarchical
tracing**, **token & cost accounting**, **latency percentiles**, and
**distribution-drift detection (PSI)** — with JSON/Markdown reports.

## Why

You can't operate an LLM product you can't see. This library gives the four
signals production AI teams watch: *where time went, what it cost, how slow it
was at p99, and whether inputs/responses drifted*.

## Architecture

```
  Tracer.span("agent")
    ├── span("retrieve")            ── nested spans, parent/child tree, error capture
    ├── span("llm_call")  ──────────▶ summarize_tokens()  → calls, tokens, $ cost
    └── span("tool_call")
                                   ▶ latency_stats()      → p50 / p95 / p99
   request distributions  ─────────▶ drift_report()       → PSI + verdict
                                   ▶ render_markdown()    → shareable report
```

## Quickstart

```bash
python3 tests/test_llmobs.py     # 18 tests, stdlib only

PYTHONPATH=src python3 - <<'PY'
import time
from llmobs import Tracer, summarize_tokens, latency_stats, drift_report, render_markdown

t = Tracer()
with t.span("agent"):
    with t.span("llm_call", model="gpt-4o-mini"):
        time.sleep(0.01)

tok = summarize_tokens([{"model":"gpt-4o-mini","input_tokens":1200,"output_tokens":300}])
lat = latency_stats([s.duration_ms for s in t.spans])
drift = drift_report([float(i) for i in range(100)], [float(i+40) for i in range(100)])
print(render_markdown(t.to_dict(), lat, tok, drift))
PY
```

## Design notes

- **Spans nest via context managers** and capture exceptions without suppressing
  them — errors propagate while still being recorded (`status="error"`).
- **Pricing is data** (`PRICING` table); unknown models fall back to `default`
  instead of crashing a cost report.
- **PSI uses the expected distribution's range for bucketing**, with epsilon
  smoothing so empty buckets don't produce `log(0)`.
- **Reports are plain Markdown/JSON** — drop them into a PR comment or a dashboard.

## Skills demonstrated

`LLM observability` · `distributed tracing` · `token/cost accounting`
· `latency percentiles (p50/p95/p99)` · `data-drift detection (PSI)`
· `CI/CD for AI` · `Python packaging + tests`

## License

MIT
