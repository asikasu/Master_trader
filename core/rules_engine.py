import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
import logging

logger = logging.getLogger(__name__)


class RulesEngine:
    """Post-filter rules to add Risk-Aware & sanity checks on ML signals."""

    def __init__(self, spread_bps=2.0, buy_threshold=0.80, sell_threshold=0.20,
                 stop_loss_pct=0.005, take_profit_pct=0.011):
        self.spread_bps = spread_bps
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct

    def compute_atr(self, df: pd.DataFrame, period: int = 14) -> float:
        high, low, close = df["HIGH"], df["LOW"], df["CLOSE"]
        tr = pd.concat([high - low,
                        (high - close.shift(1)).abs(),
                        (low - close.shift(1)).abs()], axis=1).max(axis=1)
        return tr.rolling(period).mean().iloc[-1]

    def calculate_sl_tp(self, side: str, entry: float):
        if side == "BUY":
            sl = entry * (1 - self.stop_loss_pct)
            tp = entry * (1 + self.take_profit_pct)
        else:
            sl = entry * (1 + self.stop_loss_pct)
            tp = entry * (1 - self.take_profit_pct)
        return sl, tp

    def validate_signal(self, row: pd.Series, prob: float, df: pd.DataFrame) -> dict:
        """
        Returns dict with signal, side, sl, tp or rejects with reason.
        """
        result = {"signal": "WAIT", "side": None, "sl": None, "tp": None, "reason": None}

        ema20 = row.get("EMA20", row.get("CLOSE"))
        ema50 = row.get("EMA50", row.get("CLOSE"))
        entry = row["CLOSE"]

        spread = 0
        if "SPREAD" in row:
            spread = row["SPREAD"]
        if spread > self.spread_bps:
            result["reason"] = f"Spread too high: {spread:.1f} bps"
            return result

        if prob >= self.buy_threshold and ema20 > ema50:
            side = "BUY"
            sl, tp = self.calculate_sl_tp(side, entry)
            result.update({"signal": "BUY", "side": "BUY", "sl": sl, "tp": tp})
        elif prob <= self.sell_threshold and ema20 < ema50:
            side = "SELL"
            sl, tp = self.calculate_sl_tp(side, entry)
            result.update({"signal": "SELL", "side": "SELL", "sl": sl, "tp": tp})
        else:
            result["reason"] = f"Probability={prob:.2f}, EMA20={ema20:.2f} vs EMA50={ema50:.2f}"

        return result

    def backtest_strategy(self, df: pd.DataFrame, probs: np.ndarray, threshold: float = 0.50) -> dict:
        signals = (probs > threshold).astype(int)
        close = df["CLOSE"].values
        atr = self.compute_atr(df)

        trades, wins, losses, total_pnl = 0, 0, 0, 0.0
        for i in range(1, len(signals)):
            if signals[i] == 1:
                trades += 1
                entry = close[i]
                sl = entry - atr * 1.5
                tp = entry + atr * 2.0
                for j in range(i + 1, min(i + 60, len(close))):
                    if close[j] <= sl:
                        total_pnl -= (entry - sl)
                        losses += 1
                        break
                    elif close[j] >= tp:
                        total_pnl += (tp - entry)
                        wins += 1
                        break
                else:
                    total_pnl += close[min(i + 59, len(close) - 1)] - entry
                    if total_pnl > 0:
                        wins += 1
                    else:
                        losses += 1

        win_rate = wins / trades if trades > 0 else 0
        return {"trades": trades, "wins": wins, "losses": losses,
                "win_rate": win_rate, "total_pnl": total_pnl}
