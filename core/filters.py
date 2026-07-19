import logging
from datetime import datetime, timedelta
from pathlib import Path

logger = logging.getLogger(__name__)


class NewsFilter:
    NFP_DATES = None
    CPI_DATES = None
    FOMC_DATES = None

    @classmethod
    def _load_dates(cls):
        if cls.NFP_DATES is not None:
            return
        cls.NFP_DATES = set()
        cls.CPI_DATES = set()
        cls.FOMC_DATES = set()
        path = Path("data/news_events.csv")
        if path.exists():
            import pandas as pd
            df = pd.read_csv(path)
            for _, row in df.iterrows():
                d = row.get("date", "")
                t = row.get("type", "").upper()
                if "NFP" in t or "nonfarm" in t.lower():
                    cls.NFP_DATES.add(d)
                elif "CPI" in t or "cpi" in t.lower():
                    cls.CPI_DATES.add(d)
                elif "FOMC" in t or "fomc" in t.lower():
                    cls.FOMC_DATES.add(d)
            logger.info("Loaded %d NFP, %d CPI, %d FOMC dates", len(cls.NFP_DATES), len(cls.CPI_DATES), len(cls.FOMC_DATES))
        else:
            cls.NFP_DATES = {"2026-01-10", "2026-02-07", "2026-03-06", "2026-04-03", "2026-05-08", "2026-06-05", "2026-07-03"}
            cls.CPI_DATES = {"2026-01-15", "2026-02-12", "2026-03-12", "2026-04-10", "2026-05-13", "2026-06-10", "2026-07-15"}
            cls.FOMC_DATES = {"2026-01-28", "2026-03-18", "2026-05-06", "2026-06-17", "2026-07-29"}

    @classmethod
    def is_news_event(cls, dt: datetime, hours_buffer: int = 2) -> bool:
        cls._load_dates()
        date_str = dt.strftime("%Y-%m-%d")
        if date_str in cls.NFP_DATES or date_str in cls.CPI_DATES or date_str in cls.FOMC_DATES:
            hour = dt.hour
            if 7 <= hour <= 17:
                return True
        return False

    @classmethod
    def next_news(cls, dt: datetime) -> str:
        cls._load_dates()
        all_events = [(d, "NFP") for d in cls.NFP_DATES]
        all_events += [(d, "CPI") for d in cls.CPI_DATES]
        all_events += [(d, "FOMC") for d in cls.FOMC_DATES]
        all_events.sort()
        target = dt.strftime("%Y-%m-%d")
        for d, t in all_events:
            if d >= target:
                return f"{t} on {d}"
        return "unknown"


class SpreadFilter:
    def __init__(self, max_spread_bps: float = 2.0):
        self.max_spread_bps = max_spread_bps

    def check(self, bid: float, ask: float) -> bool:
        if bid <= 0 or ask <= 0:
            return False
        spread_bps = (ask - bid) / bid * 10000
        if spread_bps > self.max_spread_bps:
            logger.debug("Spread too high: %.2f bps (max: %.2f)", spread_bps, self.max_spread_bps)
            return False
        return True

    def check_from_tick(self, tick) -> bool:
        if tick is None:
            return False
        return self.check(tick.bid, tick.ask)


class SelfHealingEngine:
    def __init__(self, evolution_engine):
        self.evolution_engine = evolution_engine
        self.consecutive_losses = 0
        self.max_consecutive_losses = 10
        self.last_retrain_day = None

    def on_trade_result(self, profit: float):
        if profit < 0:
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0

        if self.consecutive_losses >= self.max_consecutive_losses:
            logger.warning("Consecutive losses=%d, triggering healing evolution", self.consecutive_losses)
            self._heal()

    def _heal(self):
        self.evolution_engine.run()
        self.consecutive_losses = 0
        logger.info("Healing evolution completed")

    def should_retrain(self, today: datetime) -> bool:
        if self.last_retrain_day is None:
            self.last_retrain_day = today
            return False
        if (today - self.last_retrain_day).days >= 7:
            self.last_retrain_day = today
            return True
        return False
