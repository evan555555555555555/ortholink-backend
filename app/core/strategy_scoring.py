"""
RSA multi-objective country ranking (PRD §4.5.4).

Refactored: utility = (TAM_weight * reuse_percentage) / (time_months * cost_usd).
Prioritizes markets with high revenue potential (TAM), high document reusability,
and low regulatory friction (time, cost). Ron Sacher strategic optimization path.
"""


def strategy_utility_score(
    reuse_pct: float,
    time_months: float,
    cost_usd: float,
    tam_weight: float,
    *,
    time_floor: float = 1.0,
    cost_floor: float = 1000.0,
) -> float:
    """
    Country utility score (higher = better). Formula:
    (TAM_weight * reuse_pct) / (max(1, time_months) * max(1000, cost_usd)).

    This models the real strategic calculus for device manufacturers: we want markets
    with high revenue potential (TAM) and high document reusability, and we want to
    penalize long timelines and high cost. The previous cost-inversion proxy was
    replaced with hardcoded TAM baselines per country so ranking reflects actual
    market-prioritization logic. Used to order optimal_entry_sequence in StrategyReport.
    """
    denom = max(time_floor, time_months) * max(cost_floor, cost_usd)
    if denom <= 0:
        return 0.0
    tam = max(0.0, min(1.0, tam_weight))
    reuse = max(0.0, min(100.0, reuse_pct))
    raw = (tam * reuse) / denom
    return raw if not (raw != raw) else 0.0  # NaN guard
