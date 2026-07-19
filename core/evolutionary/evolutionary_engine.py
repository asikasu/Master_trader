import logging
import os
import pandas as pd
import numpy as np
from typing import Callable, Dict, List, Optional, Tuple
from xgboost import XGBClassifier

from .types import (
    EvaluationResult,
    FitnessScore,
    GenerationState,
    ParameterCombo,
    TradingConfig,
    XGBoostConfig,
)
from .fitness import compute_fitness, rank_results
from .mutation import evolve_population
from .persistence import (
    BEST_PARAMS_FILE,
    STATE_FILE,
    load_best_params,
    load_state,
    save_best_params,
    save_state,
)
from .walkforward import WalkForwardSplitter

logger = logging.getLogger(__name__)


class EvolutionaryEngine:
    def __init__(
        self,
        data: pd.DataFrame,
        feature_columns: List[str],
        target_column: str = "Target",
        population_size: int = 100,
        max_generations: int = 20,
        state_dir: str = ".",
        resume: bool = True,
    ):
        self.data = data
        self.feature_columns = feature_columns
        self.target_column = target_column
        self.population_size = population_size
        self.max_generations = max_generations
        self.state_dir = state_dir
        self.resume = resume

        best_params_path = os.path.join(state_dir, BEST_PARAMS_FILE)
        state_path = os.path.join(state_dir, STATE_FILE)
        self.best_params_path = best_params_path
        self.state_path = state_path

        self.state: Optional[GenerationState] = None

    def _build_grid_population(self) -> List[ParameterCombo]:
        best_from_file = load_best_params(self.best_params_path)
        if best_from_file:
            logger.info("Seeding from best_params.json (%d combos)", len(best_from_file))
            seed = [c for c, _ in best_from_file]
            while len(seed) < self.population_size:
                parent = seed[len(seed) % len(seed)]
                from .mutation import mutate_combo
                child = mutate_combo(parent, len(seed))
                seed.append(child)
            return seed[:self.population_size]

        grid = []
        for i in range(10):
            xgb_variants = [
                XGBoostConfig(n_estimators=v, max_depth=d, learning_rate=lr)
                for v in [100, 300, 500, 700, 1000]
                for d in [4, 6, 8, 10, 12]
                for lr in [0.01, 0.02, 0.05, 0.1]
            ]
        n = min(self.population_size, len(xgb_variants))
        for i in range(n):
            xgb = xgb_variants[i]
            trading = TradingConfig(
                buy_threshold=0.75 + (i % 5) * 0.05,
                sell_threshold=0.25 - (i % 5) * 0.05,
                stop_loss_pct=0.002 + (i % 10) * 0.0005,
                take_profit_pct=0.005 + (i % 10) * 0.001,
            )
            grid.append(ParameterCombo(xgb=xgb, trading=trading, combo_id=i))
        return grid

    def _backtest(self, combo: ParameterCombo,
                  train_start: int, train_end: int,
                  test_start: int, test_end: int) -> List[float]:
        df = self.data.iloc[train_start:train_end].copy()
        X_train = df[self.feature_columns].values
        y_train = df[self.target_column].values

        model = XGBClassifier(
            n_estimators=combo.xgb.n_estimators,
            max_depth=combo.xgb.max_depth,
            learning_rate=combo.xgb.learning_rate,
            subsample=combo.xgb.subsample,
            colsample_bytree=combo.xgb.colsample_bytree,
            min_child_weight=combo.xgb.min_child_weight,
            gamma=combo.xgb.gamma,
            scale_pos_weight=combo.xgb.scale_pos_weight,
            random_state=42,
            eval_metric="logloss",
            n_jobs=-1,
        )
        model.fit(X_train, y_train)

        test_df = self.data.iloc[test_start:test_end].copy()
        X_test = test_df[self.feature_columns].values
        y_test = test_df[self.target_column].values
        probs = model.predict_proba(X_test)[:, 1]

        equity = [1000.0]
        for i in range(len(probs)):
            if probs[i] >= combo.trading.buy_threshold:
                if y_test[i] == 1:
                    equity.append(equity[-1] * (1 + combo.trading.take_profit_pct))
                else:
                    equity.append(equity[-1] * (1 - combo.trading.stop_loss_pct))
            elif probs[i] <= combo.trading.sell_threshold:
                if y_test[i] == 0:
                    equity.append(equity[-1] * (1 + combo.trading.take_profit_pct))
                else:
                    equity.append(equity[-1] * (1 - combo.trading.stop_loss_pct))
            else:
                equity.append(equity[-1])

        return equity

    def run(self) -> GenerationState:
        if self.resume:
            restored = load_state(self.state_path)
            if restored is not None:
                self.state = restored
                logger.info("Resumed from generation %d", self.state.generation)

        if self.state is None:
            population = self._build_grid_population()
            self.state = GenerationState(generation=0, population=population)

        total_rows = len(self.data)
        wf = WalkForwardSplitter(total_rows, train_ratio=0.6, val_ratio=0.2, num_windows=3)
        windows = wf.get_windows()

        for gen in range(self.state.generation, self.max_generations):
            logger.info("=== Generation %d / %d ===", gen + 1, self.max_generations)
            window_idx = gen % len(windows)
            ts, te, vs, ve = windows[window_idx]
            self.state.in_sample_start = ts
            self.state.in_sample_end = te
            self.state.out_sample_start = vs
            self.state.out_sample_end = ve
            logger.info("Train: [%d:%d], Test: [%d:%d]", ts, te, vs, ve)

            results: List[EvaluationResult] = []
            for combo in self.state.population:
                try:
                    eq = self._backtest(combo, ts, te, vs, ve)
                    fitness = compute_fitness(eq)
                    results.append(EvaluationResult(combo=combo, fitness=fitness, equity_curve=eq))
                except Exception as e:
                    logger.error("Combo %d failed: %s", combo.combo_id, e)
                    results.append(EvaluationResult(combo=combo, fitness=FitnessScore(), error=str(e)))

            ranked = rank_results(results)
            self.state.results = ranked
            if ranked:
                best = ranked[0]
                self.state.best_combo = best.combo
                self.state.best_fitness = best.fitness
                logger.info("Best: score=%.2f profit=%.2f sharpe=%.2f dd=%.2f%%",
                            best.fitness.composite_score, best.fitness.total_profit,
                            best.fitness.sharpe_ratio, best.fitness.max_drawdown_pct)

            top10 = [(c.combo, c.fitness) for c in ranked[:10]]
            save_best_params(top10, self.best_params_path)

            next_id = max((c.combo_id for c in self.state.population), default=0) + 1
            self.state.population = evolve_population(top10, self.population_size, next_id)
            self.state.generation = gen + 1
            save_state(self.state, self.state_path)

        return self.state

    def get_best_combo(self) -> Optional[ParameterCombo]:
        if self.state and self.state.best_combo:
            return self.state.best_combo
        best = load_best_params(self.best_params_path)
        return best[0][0] if best else None
