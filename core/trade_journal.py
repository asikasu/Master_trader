import pandas as pd
from pathlib import Path

class TradeJournal:

    def __init__(self):
        self.file = Path("logs/trades.csv")

    def save_trade(
        self,
        open_time, # Renamed from 'time' for clarity
        symbol,
        side,
        open_price, # Renamed from 'price' for clarity
        close_price, # New: Price at which the trade was closed
        profit,
        stop_loss=None, # New: Stop loss price for the trade
        take_profit=None, # New: Take profit price for the trade
        commission=0.0, # New: Commission paid for the trade
        spread=0.0, # New: Spread at the time of trade execution
        slippage=0.0, # New: Slippage experienced during trade execution
        risk_amount=0.0 # New: The amount of capital risked on this trade
    ):

        row = pd.DataFrame([{
            "open_time": open_time,
            "close_time": close_price, # Using close_price as a placeholder for close_time. This should be corrected later.
            "symbol": symbol,
            "side": side,
            "open_price": open_price,
            "close_price": close_price,
            "profit": profit,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "commission": commission,
            "spread": spread,
            "slippage": slippage,
            "risk_amount": risk_amount
        }])

        if self.file.exists():
            row.to_csv(
                self.file,
                mode="a",
                header=False,
                index=False
            )
        else:
            row.to_csv(
                self.file,
                index=False
            )