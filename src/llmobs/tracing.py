"""Hierarchical tracing for LLM/agent calls (OpenTelemetry-ish, stdlib only)."""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field


@dataclass
class Span:
    """A timed unit of work with attributes and a parent/child tree."""

    name: str
    trace_id: str
    span_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    parent_id: str | None = None
    start: float = field(default_factory=time.perf_counter)
    end: float | None = None
    attributes: dict = field(default_factory=dict)
    status: str = "ok"
    error: str | None = None

    @property
    def duration_ms(self) -> float:
        end = self.end if self.end is not None else time.perf_counter()
        return round((end - self.start) * 1000, 3)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "parent_id": self.parent_id,
            "duration_ms": self.duration_ms,
            "status": self.status,
            "error": self.error,
            "attributes": self.attributes,
        }


class Tracer:
    """Collects spans; supports nested `with tracer.span(...)` blocks and export."""

    def __init__(self, trace_id: str | None = None):
        self.trace_id = trace_id or uuid.uuid4().hex[:16]
        self.spans: list[Span] = []
        self._stack: list[Span] = []

    def span(self, name: str, **attributes) -> "_SpanContext":
        return _SpanContext(self, name, attributes)

    def _start(self, name: str, attributes: dict) -> Span:
        parent = self._stack[-1] if self._stack else None
        sp = Span(name=name, trace_id=self.trace_id,
                  parent_id=parent.span_id if parent else None, attributes=dict(attributes))
        self.spans.append(sp)
        self._stack.append(sp)
        return sp

    def _end(self, sp: Span, error: Exception | None = None) -> None:
        sp.end = time.perf_counter()
        if error is not None:
            sp.status = "error"
            sp.error = f"{type(error).__name__}: {error}"
        if self._stack and self._stack[-1] is sp:
            self._stack.pop()

    def roots(self) -> list[Span]:
        return [s for s in self.spans if s.parent_id is None]

    def by_name(self, name: str) -> list[Span]:
        return [s for s in self.spans if s.name == name]

    def to_dict(self) -> dict:
        return {"trace_id": self.trace_id, "spans": [s.to_dict() for s in self.spans]}


class _SpanContext:
    def __init__(self, tracer: Tracer, name: str, attributes: dict):
        self.tracer, self.name, self.attributes = tracer, name, attributes
        self.span: Span | None = None

    def __enter__(self) -> Span:
        self.span = self.tracer._start(self.name, self.attributes)
        return self.span

    def __exit__(self, exc_type, exc, tb) -> bool:
        assert self.span is not None
        self.tracer._end(self.span, exc if exc is not None else None)
        return False  # don't suppress exceptions
