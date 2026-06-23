# core/backtester.py

import pandas as pd
import numpy as np


class Backtester:

    def __init__(self, data: pd.DataFrame):
        self.df = data.copy()
        self.results = None

    def run_simulation(
        self,
        buy_threshold=0.80,
        sell_threshold=0.20
    ):

        df = self.df.copy()

        # =====================
        # SIGNALS
        # =====================

        df["Signal"] = 0

        df.loc[
            df["Probability"] >= buy_threshold,
            "Signal"
        ] = 1

        df.loc[
            df["Probability"] <= sell_threshold,
            "Signal"
        ] = -1

        # =====================
        # RETURNS
        # =====================

        future_return = (
            df["CLOSE"].shift(-1)
            / df["CLOSE"]
            - 1
        )

        df["Returns"] = (
            future_return
            * df["Signal"]
        )

        df = df.dropna().copy()

        # =====================
        # EQUITY CURVE
        # =====================

        df["Equity"] = (
            1 +
            df["Returns"]
        ).cumprod()

        self.df = df
        self.results = df["Equity"]

        print("✅ Backtest finished")

        return self.results

    def calculate_metrics(self):

        trades = self.df[
            self.df["Signal"] != 0
        ].copy()

        trade_count = len(trades)

        if trade_count == 0:

            return {
                "Trades": 0,
                "Win Rate": 0,
                "Profit Factor": 0,
                "Total Return": 0,
                "Max Drawdown": 0
            }

        # =====================
        # WIN RATE
        # =====================

        win_rate = (
            trades["Returns"] > 0
        ).mean()

        # =====================
        # PROFIT FACTOR
        # =====================

        profits = (
            trades[
                trades["Returns"] > 0
            ]["Returns"]
            .sum()
        )

        losses = abs(
            trades[
                trades["Returns"] < 0
            ]["Returns"]
            .sum()
        )

        profit_factor = (
            profits / losses
            if losses > 0
            else float("inf")
        )

        # =====================
        # TOTAL RETURN
        # =====================

        total_return = (
            self.df["Equity"]
            .iloc[-1]
            - 1
        )

        # =====================
        # MAX DRAWDOWN
        # =====================

        peak = (
            self.df["Equity"]
            .cummax()
        )

        drawdown = (
            self.df["Equity"]
            - peak
        ) / peak

        max_drawdown = drawdown.min()

        # =====================
        # AVG WIN / LOSS
        # =====================

        avg_win = (
            trades[
                trades["Returns"] > 0
            ]["Returns"]
            .mean()
        )

        avg_loss = (
            trades[
                trades["Returns"] < 0
            ]["Returns"]
            .mean()
        )

        return {
            "Trades": trade_count,
            "Win Rate": win_rate,
            "Profit Factor": profit_factor,
            "Total Return": total_return,
            "Max Drawdown": max_drawdown,
            "Average Win": avg_win,
            "Average Loss": avg_loss
        }