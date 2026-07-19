import logging
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class MTFFilter:
    """
    Multi-Timeframe + Price Parity Filter.
    Confirms signals across H1, H4, D1 timeframes using EMA alignment & ATR parity.
    """

    def __init__(self, ema_fast: int = 20, ema_slow: int = 50):
        self.ema_fast = ema_fast
        self.ema_slow = ema_slow

    def _resample(self, df: pd.DataFrame, freq: str) -> pd.DataFrame:
        idx = pd.to_datetime(df.index, unit="s") if df.index.dtype == "int64" else df.index
        resampled = df.copy()
        resampled.index = idx
        ohlc = resampled.resample(freq).agg({
            "OPEN": "first",
            "HIGH": "max",
            "LOW": "min",
            "CLOSE": "last",
            "VOLUME": "sum",
        }).dropna()
        ohlc["EMA20"] = ohlc["CLOSE"].ewm(span=self.ema_fast, adjust=False).mean()
        ohlc["EMA50"] = ohlc["CLOSE"].ewm(span=self.ema_slow, adjust=False).mean()
        tr = pd.concat([ohlc["HIGH"] - ohlc["LOW"],
                        (ohlc["HIGH"] - ohlc["CLOSE"].shift(1)).abs(),
                        (ohlc["LOW"] - ohlc["CLOSE"].shift(1)).abs()], axis=1).max(axis=1)
        ohlc["ATR14"] = tr.rolling(14).mean()
        return ohlc

    def confirm(self, df: pd.DataFrame, prob: float, signal: str) -> dict:
        """
        Returns dict with confirmed signal, confidence boost, and reason.
        """
        result = {"signal": signal, "confidence": prob, "confirmed": True, "reason": "OK"}

        if signal == "WAIT":
            return result

        h4 = self._resample(df, "4h")
        d1 = self._resample(df, "D")

        h4_ema20 = h4["EMA20"].iloc[-1] if len(h4) > 0 else None
        h4_ema50 = h4["EMA50"].iloc[-1] if len(h4) > 0 else None
        d1_ema20 = d1["EMA20"].iloc[-1] if len(d1) > 0 else None
        d1_ema50 = d1["EMA50"].iloc[-1] if len(d1) > 0 else None

        trend_h4 = 1 if h4_ema20 and h4_ema50 and h4_ema20 > h4_ema50 else (-1 if h4_ema20 and h4_ema50 and h4_ema20 < h4_ema50 else 0)
        trend_d1 = 1 if d1_ema20 and d1_ema50 and d1_ema20 > d1_ema50 else (-1 if d1_ema20 and d1_ema50 and d1_ema20 < d1_ema50 else 0)

        if signal == "BUY":
            if trend_h4 == -1 or trend_d1 == -1:
                result["signal"] = "WAIT"
                result["confirmed"] = False
                result["reason"] = f"H4={trend_h4}, D1={trend_d1} trend против BUY"
                return result
            if trend_h4 == 1 and trend_d1 == 1:
                result["confidence"] = min(prob * 1.15, 0.95)
                result["reason"] = f"BUY confirmed across H4+ D1"

        elif signal == "SELL":
            if trend_h4 == 1 or trend_d1 == 1:
                result["signal"] = "WAIT"
                result["confirmed"] = False
                result["reason"] = f"H4={trend_h4}, D1={trend_d1} trend против SELL"
                return result
            if trend_h4 == -1 and trend_d1 == -1:
                result["confidence"] = min(prob * 1.15, 0.95)
                result["reason"] = f"SELL confirmed across H4+ D1"

        h4_atr = h4["ATR14"].iloc[-1] if len(h4) > 0 and "ATR14" in h4.columns else None
        h1_atr = df["ATR14"].iloc[-1] if "ATR14" in df.columns else None
        if h4_atr and h1_atr and h4_atr > 0:
            parity = h1_atr / h4_atr
            if parity < 0.3 or parity > 3.0:
                result["signal"] = "WAIT"
                result["confirmed"] = False
                result["reason"] = f"ATR parity abnormal: {parity:.2f}"
                return result

        return result
