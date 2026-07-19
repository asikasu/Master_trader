import logging
import random
from typing import Dict, List, Tuple

from .types import (
    FitnessScore,
    ParamBounds,
    ParameterCombo,
    TradingConfig,
    XGBoostConfig,
)

logger = logging.getLogger(__name__)

MUTATION_RATE = 0.3
MUTATION_SCALE = 0.15

XGB_PARAM_BOUNDS: Dict[str, Tuple[float, float]] = {
    "n_estimators": (50, 2000),
    "max_depth": (3, 15),
    "learning_rate": (0.005, 0.3),
    "subsample": (0.5, 1.0),
    "colsample_bytree": (0.3, 1.0),
    "min_child_weight": (1, 20),
    "gamma": (0.0, 2.0),
    "scale_pos_weight": (1, 10),
}

TRADING_PARAM_BOUNDS: Dict[str, Tuple[float, float]] = {
    "stop_loss_pct": (0.001, 0.01),
    "take_profit_pct": (0.001, 0.02),
    "spread_bps": (0.1, 5.0),
    "commission": (0.0, 0.001),
    "slippage_bps": (0.0, 2.0),
    "buy_threshold": (0.55, 0.95),
    "sell_threshold": (0.05, 0.45),
}


def _mutate_value(current: float, low: float, high: float, is_int: bool = False) -> float:
    if random.random() > MUTATION_RATE:
        return current
    delta = random.uniform(-MUTATION_SCALE, MUTATION_SCALE) * (high - low)
    new_val = current + delta
    new_val = max(low, min(high, new_val))
    if is_int:
        new_val = round(new_val)
    return new_val


def mutate_combo(combo: ParameterCombo, combo_id: int) -> ParameterCombo:
    bounds = XGB_PARAM_BOUNDS
    new_xgb = XGBoostConfig(
        n_estimators=int(_mutate_value(
            float(combo.xgb.n_estimators), bounds["n_estimators"][0],
            bounds["n_estimators"][1], is_int=True)),
        max_depth=int(_mutate_value(
            float(combo.xgb.max_depth), bounds["max_depth"][0],
            bounds["max_depth"][1], is_int=True)),
        learning_rate=_mutate_value(
            combo.xgb.learning_rate, bounds["learning_rate"][0],
            bounds["learning_rate"][1]),
        subsample=_mutate_value(
            combo.xgb.subsample, bounds["subsample"][0],
            bounds["subsample"][1]),
        colsample_bytree=_mutate_value(
            combo.xgb.colsample_bytree, bounds["colsample_bytree"][0],
            bounds["colsample_bytree"][1]),
        min_child_weight=int(_mutate_value(
            float(combo.xgb.min_child_weight), bounds["min_child_weight"][0],
            bounds["min_child_weight"][1], is_int=True)),
        gamma=_mutate_value(
            combo.xgb.gamma, bounds["gamma"][0],
            bounds["gamma"][1]),
        scale_pos_weight=int(_mutate_value(
            float(combo.xgb.scale_pos_weight), bounds["scale_pos_weight"][0],
            bounds["scale_pos_weight"][1], is_int=True)),
    )
    t_bounds = TRADING_PARAM_BOUNDS
    new_trading = TradingConfig(
        stop_loss_pct=_mutate_value(
            combo.trading.stop_loss_pct, t_bounds["stop_loss_pct"][0],
            t_bounds["stop_loss_pct"][1]),
        take_profit_pct=_mutate_value(
            combo.trading.take_profit_pct, t_bounds["take_profit_pct"][0],
            t_bounds["take_profit_pct"][1]),
        spread_bps=_mutate_value(
            combo.trading.spread_bps, t_bounds["spread_bps"][0],
            t_bounds["spread_bps"][1]),
        commission=_mutate_value(
            combo.trading.commission, t_bounds["commission"][0],
            t_bounds["commission"][1]),
        slippage_bps=_mutate_value(
            combo.trading.slippage_bps, t_bounds["slippage_bps"][0],
            t_bounds["slippage_bps"][1]),
        buy_threshold=_mutate_value(
            combo.trading.buy_threshold, t_bounds["buy_threshold"][0],
            t_bounds["buy_threshold"][1]),
        sell_threshold=_mutate_value(
            combo.trading.sell_threshold, t_bounds["sell_threshold"][0],
            t_bounds["sell_threshold"][1]),
    )
    return ParameterCombo(xgb=new_xgb, trading=new_trading, combo_id=combo_id)


def evolve_population(
    ranked: List[Tuple[ParameterCombo, FitnessScore]],
    population_size: int,
    start_combo_id: int = 0,
) -> List[ParameterCombo]:
    keep_count = max(10, population_size // 2)
    survivors = [combo for combo, _ in ranked[:keep_count]]

    top_count = max(10, population_size // 4)
    top_combos = [combo for combo, _ in ranked[:top_count]]

    new_population: List[ParameterCombo] = list(survivors)
    next_id = start_combo_id

    while len(new_population) < population_size:
        parent = random.choice(top_combos)
        child = mutate_combo(parent, next_id)
        new_population.append(child)
        next_id += 1

    return new_population
