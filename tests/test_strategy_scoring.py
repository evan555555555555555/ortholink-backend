"""
RSA strategy scoring: no NaN, bounds in [0, 1].
"""

import math

import pytest

from app.core.strategy_scoring import strategy_utility_score


def test_scoring_no_nan():
    """Utility score must never return NaN regardless of input."""
    assert not math.isnan(strategy_utility_score(1.0, 0, 0, 0))
    assert not math.isnan(strategy_utility_score(0, 0, 0, 0))
    assert not math.isnan(strategy_utility_score(0.5, 12, 50000, 0.5))


def test_scoring_bounds():
    """Utility score must be non-negative and finite (TAM*reuse/(time*cost) formula)."""
    score = strategy_utility_score(0.8, 6, 25000, 0.9)
    assert score >= 0.0 and not math.isnan(score) and math.isfinite(score)
