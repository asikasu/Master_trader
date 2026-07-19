import math
from typing import List

from .types import EvaluationResult, FitnessScore


def compute_fitness(equity_curve: List[float]) -> FitnessScore:
    if not equity_curve or len(equity_curve) < 2:
        return FitnessScore()

    total_profit = equity_curve[-1] - equity_curve[0]

    returns = []
    for i in range(1, len(equity_curve)):
        prev = equity_curve[i - 1]
        if prev != 0:
            returns.append((equity_curve[i] - prev) / prev)

    n = len(returns)
    if n == 0:
        return FitnessScore(total_profit=total_profit)

    avg_return = sum(returns) / n
    variance = sum((r - avg_return) ** 2 for r in returns) / n
    std_dev = math.sqrt(variance) if variance > 0 else 1e-10
    risk_free_rate = 0.02 / 252
    sharpe_ratio = (avg_return - risk_free_rate) / std_dev * math.sqrt(252)

    peak = equity_curve[0]
    max_dd = 0.0
    for val in equity_curve:
        if val > peak:
            peak = val
        dd = (peak - val) / peak * 100.0 if peak != 0 else 0.0
        if dd > max_dd:
            max_dd = dd

    wins = sum(1 for r in returns if r > 0)
    win_rate = (wins / n * 100.0) if n > 0 else 0.0

    wins_val = sum(r for r in returns if r > 0)
    losses_val = abs(sum(r for r in returns if r < 0))
    profit_factor = wins_val / losses_val if losses_val > 0 else float('inf')

    return FitnessScore(
        total_profit=total_profit,
        sharpe_ratio=sharpe_ratio,
        max_drawdown_pct=max_dd,
        win_rate=win_rate,
        total_trades=n,
        profit_factor=profit_factor,
    )


def rank_results(results: List[EvaluationResult]) -> List[EvaluationResult]:
    return sorted(
        [r for r in results if r.error is None],
        key=lambda r: r.fitness.composite_score,
        reverse=True,
    )
