# core/backtester.py
import pandas as pd

class Backtester:
    def __init__(self, data: pd.DataFrame):
        self.df = data
        self.results = None

    def run_simulation(self):
        """Энгийн стратегиар арилжааг дуурайх (Жишээ нь: EMA crossover)"""
        self.df['Signal'] = 0
        # Энгийн стратеги: EMA20 > EMA50 үед BUY (1)
        self.df.loc[self.df['EMA20'] > self.df['EMA50'], 'Signal'] = 1
        
        # Үр дүнг тооцоолох (Энгийн ашиг = үнийн зөрүү * дохио)
        self.df['Returns'] = self.df['CLOSE'].pct_change() * self.df['Signal'].shift(1)
        self.results = self.df['Returns'].cumsum()
        print("Backtest дууслаа.")
        return self.results

    def calculate_metrics(self):
        """Гол үзүүлэлтүүдийг тооцоолох"""
        win_rate = (self.df['Returns'] > 0).mean()
        profit_factor = self.df[self.df['Returns'] > 0]['Returns'].sum() / abs(self.df[self.df['Returns'] < 0]['Returns'].sum())
        return {"Win Rate": win_rate, "Profit Factor": profit_factor}

if __name__ == "__main__":
    print("Backtester модуль бэлэн байна.")