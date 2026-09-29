"""Tests for LLM observability (stdlib unittest)."""
import os
import sys
import time
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from llmobs import (  # noqa: E402
    ModelPricing,
    Tracer,
    compute_cost,
    drift_report,
    latency_stats,
    percentile,
    population_stability_index,
    render_markdown,
    summarize_tokens,
    to_json,
)


class TestTracing(unittest.TestCase):
    def test_nested_spans_and_parents(self):
        t = Tracer()
        with t.span("agent"):
            with t.span("llm_call", model="gpt-4o-mini"):
                time.sleep(0.001)
            with t.span("tool_call"):
                pass
        self.assertEqual(len(t.spans), 3)
        agent = t.by_name("agent")[0]
        child = t.by_name("llm_call")[0]
        self.assertIsNone(agent.parent_id)
        self.assertEqual(child.parent_id, agent.span_id)
        self.assertEqual(len(t.roots()), 1)

    def test_attributes_and_duration(self):
        t = Tracer()
        with t.span("call", tokens=100) as sp:
            time.sleep(0.002)
        self.assertEqual(sp.attributes["tokens"], 100)
        self.assertGreater(sp.duration_ms, 0)

    def test_error_captured_and_propagated(self):
        t = Tracer()
        with self.assertRaises(ValueError):
            with t.span("boom"):
                raise ValueError("nope")
        self.assertEqual(t.spans[0].status, "error")
        self.assertIn("nope", t.spans[0].error)

    def test_export_dict(self):
        t = Tracer(trace_id="abc")
        with t.span("x"):
            pass
        d = t.to_dict()
        self.assertEqual(d["trace_id"], "abc")
        self.assertEqual(d["spans"][0]["name"], "x")


class TestCost(unittest.TestCase):
    def test_known_model(self):
        c = compute_cost("gpt-4o-mini", 1000, 1000)
        self.assertAlmostEqual(c, 0.00015 + 0.0006, places=6)

    def test_unknown_model_uses_default(self):
        c = compute_cost("some-new-model", 1000, 0)
        self.assertAlmostEqual(c, 0.001, places=6)

    def test_custom_pricing(self):
        table = {"m": ModelPricing(1.0, 2.0)}
        self.assertAlmostEqual(compute_cost("m", 1000, 1000, table), 3.0, places=6)


class TestMetrics(unittest.TestCase):
    def test_percentile(self):
        data = list(range(1, 101))  # 1..100
        self.assertAlmostEqual(percentile(data, 50), 50.5, places=1)
        self.assertAlmostEqual(percentile(data, 0), 1.0)
        self.assertAlmostEqual(percentile(data, 100), 100.0)

    def test_percentile_empty(self):
        self.assertEqual(percentile([], 95), 0.0)

    def test_percentile_bad_p(self):
        with self.assertRaises(ValueError):
            percentile([1, 2, 3], 150)

    def test_latency_stats(self):
        s = latency_stats([10, 20, 30, 40, 50])
        self.assertEqual(s["count"], 5)
        self.assertEqual(s["min"], 10)
        self.assertEqual(s["max"], 50)
        self.assertEqual(s["mean"], 30.0)
        self.assertLessEqual(s["p50"], s["p95"])
        self.assertLessEqual(s["p95"], s["p99"])

    def test_latency_stats_empty(self):
        self.assertEqual(latency_stats([])["count"], 0)

    def test_token_summary(self):
        records = [
            {"model": "gpt-4o-mini", "input_tokens": 1000, "output_tokens": 500},
            {"model": "gpt-4o-mini", "input_tokens": 1000, "output_tokens": 500},
        ]
        s = summarize_tokens(records)
        self.assertEqual(s.calls, 2)
        self.assertEqual(s.input_tokens, 2000)
        self.assertEqual(s.output_tokens, 1000)
        self.assertEqual(s.total_tokens, 3000)
        self.assertAlmostEqual(s.cost_usd, 2 * (0.00015 + 0.0003), places=6)
        self.assertIn("gpt-4o-mini", s.by_model)
        self.assertEqual(s.by_model["gpt-4o-mini"]["calls"], 2)


class TestDrift(unittest.TestCase):
    def test_identical_distribution_is_zero(self):
        data = [float(i) for i in range(100)]
        self.assertAlmostEqual(population_stability_index(data, data), 0.0, places=6)

    def test_shifted_distribution_flagged(self):
        expected = [float(i) for i in range(100)]
        actual = [float(i + 60) for i in range(100)]
        rep = drift_report(expected, actual)
        self.assertEqual(rep["verdict"], "significant_shift")
        self.assertGreater(rep["psi"], 0.25)

    def test_degenerate_range(self):
        self.assertEqual(population_stability_index([5, 5, 5], [5, 5, 5]), 0.0)

    def test_empty_raises(self):
        with self.assertRaises(ValueError):
            population_stability_index([], [1, 2])


class TestReport(unittest.TestCase):
    def test_markdown_and_json(self):
        t = Tracer(trace_id="t1")
        with t.span("root"):
            with t.span("child"):
                pass
        lat = latency_stats([s.duration_ms for s in t.spans])
        tok = summarize_tokens([{"model": "gpt-4o-mini", "input_tokens": 10, "output_tokens": 5}])
        md = render_markdown(t.to_dict(), lat, tok, drift_report([float(i) for i in range(50)],
                                                                [float(i) for i in range(50)]))
        self.assertIn("# LLM Observability Report", md)
        self.assertIn("| root |", md)
        self.assertIn("p95", md)
        js = to_json(t)
        self.assertIn("trace_id", js)


if __name__ == "__main__":
    unittest.main(verbosity=2)
