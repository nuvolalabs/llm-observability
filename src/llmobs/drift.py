"""Distribution drift detection — Population Stability Index (PSI)."""
from __future__ import annotations

import math


def population_stability_index(expected: list[float], actual: list[float],
                               bins: int = 10) -> float:
    """PSI between two numeric distributions (0 = identical).

    Common thresholds: <0.1 no shift, 0.1-0.25 moderate shift, >0.25 significant.
    Both inputs are bucketed against the *expected* distribution's range.
    """
    if not expected or not actual:
        raise ValueError("expected and actual must be non-empty")
    lo, hi = min(expected), max(expected)
    if hi == lo:  # degenerate: no spread to bucket
        return 0.0
    width = (hi - lo) / bins

    def bucket_counts(xs: list[float]) -> list[int]:
        counts = [0] * bins
        for x in xs:
            idx = int((x - lo) / width)
            idx = max(0, min(bins - 1, idx))
            counts[idx] += 1
        return counts

    e_counts = bucket_counts(expected)
    a_counts = bucket_counts(actual)
    n_e, n_a = len(expected), len(actual)
    eps = 1e-6
    psi = 0.0
    for i in range(bins):
        e = e_counts[i] / n_e
        a = a_counts[i] / n_a
        e = max(e, eps)
        a = max(a, eps)
        psi += (a - e) * math.log(a / e)
    return round(psi, 6)


def drift_report(expected: list[float], actual: list[float], bins: int = 10,
                 thresholds: tuple[float, float] = (0.1, 0.25)) -> dict:
    """PSI + a human-readable verdict."""
    psi = population_stability_index(expected, actual, bins)
    lo_t, hi_t = thresholds
    if psi < lo_t:
        verdict = "stable"
    elif psi < hi_t:
        verdict = "moderate_shift"
    else:
        verdict = "significant_shift"
    return {"psi": psi, "verdict": verdict, "bins": bins,
            "expected_n": len(expected), "actual_n": len(actual)}
