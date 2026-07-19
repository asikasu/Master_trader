import dataclasses
import enum
from typing import Dict, List, Optional, Tuple


class ParameterDomain(enum.Enum):
    CONTINUOUS = "continuous"
    DISCRETE = "discrete"
    CATEGORICAL = "categorical"


@dataclasses.dataclass(frozen=True)
class XGBoostConfig:
    n_estimators: int = 1000
    max_depth: int = 8
    learning_rate: float = 0.02
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    min_child_weight: int = 10
    gamma: float = 0.5
    scale_pos_weight: int = 1


@dataclasses.dataclass(frozen=True)
class TradingConfig:
    stop_loss_pct: float = 0.002
    take_profit_pct: float = 0.005
    spread_bps: float = 1.0
    commission: float = 0.0001
    slippage_bps: float = 0.5
    buy_threshold: float = 0.80
    sell_threshold: float = 0.20


@dataclasses.dataclass(frozen=True)
class ParameterCombo:
    xgb: XGBoostConfig
    trading: TradingConfig
    combo_id: int = 0


@dataclasses.dataclass
class FitnessScore:
    total_profit: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown_pct: float = 0.0
    win_rate: float = 0.0
    total_trades: int = 0
    profit_factor: float = 0.0

    @property
    def composite_score(self) -> float:
        if self.max_drawdown_pct >= 100.0:
            return -9999.0
        profit_factor = max(self.total_profit, 0.0) / (abs(self.total_profit) + 1.0)
        dd_penalty = max(0.0, 1.0 - self.max_drawdown_pct / 50.0)
        sharpe_boost = max(0.0, self.sharpe_ratio) * 0.3
        return (profit_factor * 0.4 + sharpe_boost * 0.4 + dd_penalty * 0.2) * 100.0


@dataclasses.dataclass
class EvaluationResult:
    combo: ParameterCombo
    fitness: FitnessScore
    equity_curve: List[float] = dataclasses.field(default_factory=list)
    error: Optional[str] = None


@dataclasses.dataclass
class GenerationState:
    generation: int = 0
    population: List[ParameterCombo] = dataclasses.field(default_factory=list)
    results: List[EvaluationResult] = dataclasses.field(default_factory=list)
    best_combo: Optional[ParameterCombo] = None
    best_fitness: Optional[FitnessScore] = None
    in_sample_start: int = 0
    in_sample_end: int = 0
    out_sample_start: int = 0
    out_sample_end: int = 0


ParamGrid = Dict[str, List[float | int | str]]
ParamBounds = Dict[str, Tuple[float, float]]
