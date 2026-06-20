# core/feature_engine.py
import pandas as pd
import pandas_ta as ta

class FeatureEngine:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

    def add_all_indicators(self):
        """Бүх техникийн индикаторуудыг нэмэх"""
        # EMA
        self.df['EMA20'] = ta.ema(self.df['CLOSE'], length=20)
        self.df['EMA50'] = ta.ema(self.df['CLOSE'], length=50)
        
        # RSI
        self.df['RSI14'] = ta.rsi(self.df['CLOSE'], length=14)
        
        # ATR
        self.df['ATR14'] = ta.atr(self.df['HIGH'], self.df['LOW'], self.df['CLOSE'], length=14)
        
        # Momentum
        self.df['Momentum'] = self.df['CLOSE'].diff(3)
        
        return self.df.dropna()

# Хэрэглэх жишээ
if __name__ == "__main__":
    print("Feature Engine модуль бэлэн байна.")