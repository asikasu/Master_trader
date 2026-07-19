import json
import logging
import os
from typing import Dict, List, Optional, Tuple

from .types import (
    EvaluationResult,
    FitnessScore,
    GenerationState,
    ParameterCombo,
    TradingConfig,
    XGBoostConfig,
)

logger = logging.getLogger(__name__)

BEST_PARAMS_FILE = "best_params.json"
STATE_FILE = "evolution_state.json"


def _combo_to_dict(combo: ParameterCombo) -> Dict:
    return {
        "combo_id": combo.combo_id,
        "xgb": {
            "n_estimators": combo.xgb.n_estimators,
            "max_depth": combo.xgb.max_depth,
            "learning_rate": combo.xgb.learning_rate,
            "subsample": combo.xgb.subsample,
            "colsample_bytree": combo.xgb.colsample_bytree,
            "min_child_weight": combo.xgb.min_child_weight,
            "gamma": combo.xgb.gamma,
            "scale_pos_weight": combo.xgb.scale_pos_weight,
        },
        "trading": {
            "stop_loss_pct": combo.trading.stop_loss_pct,
            "take_profit_pct": combo.trading.take_profit_pct,
            "spread_bps": combo.trading.spread_bps,
            "commission": combo.trading.commission,
            "slippage_bps": combo.trading.slippage_bps,
            "buy_threshold": combo.trading.buy_threshold,
            "sell_threshold": combo.trading.sell_threshold,
        },
    }


def _dict_to_combo(d: Dict) -> ParameterCombo:
    return ParameterCombo(
        xgb=XGBoostConfig(**d["xgb"]),
        trading=TradingConfig(**d["trading"]),
        combo_id=d.get("combo_id", 0),
    )


def _fitness_to_dict(f: FitnessScore) -> Dict:
    return {
        "total_profit": f.total_profit,
        "sharpe_ratio": f.sharpe_ratio,
        "max_drawdown_pct": f.max_drawdown_pct,
        "win_rate": f.win_rate,
        "total_trades": f.total_trades,
        "profit_factor": f.profit_factor,
        "composite_score": f.composite_score,
    }


def _dict_to_fitness(d: Dict) -> FitnessScore:
    return FitnessScore(
        total_profit=d.get("total_profit", 0.0),
        sharpe_ratio=d.get("sharpe_ratio", 0.0),
        max_drawdown_pct=d.get("max_drawdown_pct", 0.0),
        win_rate=d.get("win_rate", 0.0),
        total_trades=d.get("total_trades", 0),
        profit_factor=d.get("profit_factor", 0.0),
    )


def save_best_params(combos: List[Tuple[ParameterCombo, FitnessScore]],
                     filepath: str = BEST_PARAMS_FILE) -> None:
    try:
        data = []
        for combo, fitness in combos:
            entry = _combo_to_dict(combo)
            entry["fitness"] = _fitness_to_dict(fitness)
            data.append(entry)
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)
        logger.info("Saved %d best combos to %s", len(data), filepath)
    except Exception as e:
        logger.error("Failed to save best params: %s", e)


def load_best_params(filepath: str = BEST_PARAMS_FILE
                     ) -> List[Tuple[ParameterCombo, FitnessScore]]:
    if not os.path.exists(filepath):
        logger.info("No existing best params file found at %s", filepath)
        return []
    try:
        with open(filepath, "r") as f:
            data = json.load(f)
        result = []
        for entry in data:
            combo = _dict_to_combo(entry)
            fitness = _dict_to_fitness(entry.get("fitness", {}))
            result.append((combo, fitness))
        logger.info("Loaded %d best combos from %s", len(result), filepath)
        return result
    except Exception as e:
        logger.error("Failed to load best params: %s", e)
        return []


def save_state(state: GenerationState, filepath: str = STATE_FILE) -> None:
    try:
        data = {
            "generation": state.generation,
            "in_sample_start": state.in_sample_start,
            "in_sample_end": state.in_sample_end,
            "out_sample_start": state.out_sample_start,
            "out_sample_end": state.out_sample_end,
            "population": [_combo_to_dict(c) for c in state.population],
            "best_combo": (_combo_to_dict(state.best_combo)
                           if state.best_combo else None),
            "best_fitness": (_fitness_to_dict(state.best_fitness)
                             if state.best_fitness else None),
        }
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)
        logger.info("Saved generation state (gen %d) to %s",
                    state.generation, filepath)
    except Exception as e:
        logger.error("Failed to save state: %s", e)


def load_state(filepath: str = STATE_FILE) -> Optional[GenerationState]:
    if not os.path.exists(filepath):
        logger.info("No existing state file found at %s", filepath)
        return None
    try:
        with open(filepath, "r") as f:
            data = json.load(f)
        population = [_dict_to_combo(c) for c in data.get("population", [])]
        state = GenerationState(
            generation=data.get("generation", 0),
            population=population,
            best_combo=(_dict_to_combo(data["best_combo"])
                        if data.get("best_combo") else None),
            best_fitness=(_dict_to_fitness(data["best_fitness"])
                          if data.get("best_fitness") else None),
            in_sample_start=data.get("in_sample_start", 0),
            in_sample_end=data.get("in_sample_end", 0),
            out_sample_start=data.get("out_sample_start", 0),
            out_sample_end=data.get("out_sample_end", 0),
        )
        logger.info("Loaded generation state (gen %d) from %s",
                    state.generation, filepath)
        return state
    except Exception as e:
        logger.error("Failed to load state: %s", e)
        return None
