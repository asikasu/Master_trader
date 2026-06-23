import pandas as pd
from pathlib import Path

class TradeJournal:

    def __init__(self):
        self.file = Path("logs/trades.csv")

    def save_trade(
        self,
        time,
        symbol,
        side,
        price,
        profit
    ):

        row = pd.DataFrame([{
            "time": time,
            "symbol": symbol,
            "side": side,
            "price": price,
            "profit": profit
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