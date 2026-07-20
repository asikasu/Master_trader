import logging
import pandas as pd
from typing import List

from core.data_loader import DataLoader
from core.feature_engine import FeatureEngine
from core.evolutionary.evolutionary_engine import EvolutionaryEngine
from core.evolutionary.persistence import load_best_params
from core.evolutionary.types import ParameterCombo

logger = logging.getLogger(__name__)

FEATURE_COLUMNS = [
    "EMA20", "EMA50", "EMA200", "EMA_DIFF", "EMA_SLOPE",
    "H1_EMA20", "H1_EMA50", "H1_TREND",
    "H4_EMA20", "H4_EMA50", "H4_TREND",
    "EMA20_50", "EMA50_200",
    "RSI14", "RSI_CHANGE",
    "ATR14", "ATR_PCT", "ATR_CHANGE",
    "Momentum", "Momentum10", "Momentum30",
    "Body", "Range", "BODY_PCT", "UPPER_WICK", "LOWER_WICK",
    "RET1", "RET5", "RET15", "RET60",
    "VOL20", "VOL60",
    "DIST_HH", "DIST_LL", "BREAK_HIGH", "BREAK_LOW",
    "HH_BREAK_5", "LL_BREAK_5",
    "TREND_UP", "TREND_DOWN",
    "WEEKDAY", "HOUR",
    "ASIA", "LONDON", "NEWYORK",
    "RSI_OVERBOUGHT", "RSI_OVERSOLD", "ATR_SPIKE",
    "MACD", "MACD_SIGNAL", "MACD_HIST",
    "ADX14", "DI_PLUS", "DI_MINUS"
]


class EvolutionaryTrainer:
    def __init__(self, data_dir: str = "data"):
        self.loader = DataLoader(data_dir)
        self.features = FeatureEngine()
        self._cached_data: pd.DataFrame = None

    def prepare_data(self, sample_size: int = 0) -> pd.DataFrame:
        if self._cached_data is not None:
            data = self._cached_data
            if sample_size > 0 and sample_size < len(data):
                data = data.iloc[:sample_size].copy()
            return data

        df = self.loader.load_gold_data()
        df = self.features.add_features(df)

        future_move = df["CLOSE"].shift(-60) - df["CLOSE"]
        df["Target"] = (future_move > df["ATR14"] * 0.5).astype(int)
        df = df.iloc[:-60].dropna(subset=["Target", "CLOSE"]).copy()

        self._cached_data = df
        logger.info("Prepared data: %d rows, %d features", len(df), len(FEATURE_COLUMNS))
        return df

    def run_evolution(self, generations: int = 10, population: int = 100,
                      resume: bool = True, state_dir: str = ".",
                      sample_size: int = 5000) -> EvolutionaryEngine:
        data = self.prepare_data(sample_size=sample_size)
        if sample_size > 0 and sample_size < len(data):
            logger.info("Sampling first %d rows for evolution (total: %d)", sample_size, len(data))
            data = data.iloc[:sample_size].copy()
        engine = EvolutionaryEngine(
            data=data,
            feature_columns=FEATURE_COLUMNS,
            target_column="Target",
            population_size=population,
            max_generations=generations,
            state_dir=state_dir,
            resume=resume,
        )
        engine.run()
        return engine
