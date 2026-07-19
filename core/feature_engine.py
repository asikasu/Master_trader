import pandas as pd
import numpy as np


class FeatureEngine:

    def add_features(self, df):

        df = df.copy()
        print(f"DEBUG add_features: input rows={len(df)}")

        # =====================
        # PRICE COLUMNS
        # =====================

        for col in ["OPEN", "HIGH", "LOW", "CLOSE"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        # =====================
        # DATE TIME
        # =====================

        if "DATE" in df.columns and "TIME" in df.columns:
            df["DATETIME"] = pd.to_datetime(
                df["DATE"].astype(str) + " " + df["TIME"].astype(str)
            )
            df["WEEKDAY"] = df["DATETIME"].dt.weekday
            df["HOUR"] = df["DATETIME"].dt.hour
            df["ASIA"] = ((df["HOUR"] >= 0) & (df["HOUR"] < 8)).astype(int)
            df["LONDON"] = ((df["HOUR"] >= 8) & (df["HOUR"] < 16)).astype(int)
            df["NEWYORK"] = ((df["HOUR"] >= 13) & (df["HOUR"] < 22)).astype(int)

        # =====================
        # EMA
        # =====================

        df["EMA20"] = df["CLOSE"].ewm(span=20).mean()
        df["EMA50"] = df["CLOSE"].ewm(span=50).mean()
        df["EMA200"] = df["CLOSE"].ewm(span=200).mean()
        df["EMA_DIFF"] = df["EMA20"] - df["EMA50"]
        df["EMA_SLOPE"] = df["EMA20"].diff(5)

        # =====================
        # MULTI TIMEFRAME EMA
        # =====================

        df["H1_EMA20"] = df["CLOSE"].ewm(span=80).mean()
        df["H1_EMA50"] = df["CLOSE"].ewm(span=200).mean()
        df["H4_EMA20"] = df["CLOSE"].ewm(span=320).mean()
        df["H4_EMA50"] = df["CLOSE"].ewm(span=800).mean()
        df["H1_TREND"] = (df["H1_EMA20"] > df["H1_EMA50"]).astype(int)
        df["H4_TREND"] = (df["H4_EMA20"] > df["H4_EMA50"]).astype(int)

        # =====================
        # RETURNS
        # =====================

        df["RET1"] = df["CLOSE"].pct_change(1)
        df["RET5"] = df["CLOSE"].pct_change(5)
        df["RET15"] = df["CLOSE"].pct_change(15)
        df["RET60"] = df["CLOSE"].pct_change(60)

        # =====================
        # MOMENTUM
        # =====================

        df["Momentum"] = df["CLOSE"].diff(3)
        df["Momentum10"] = df["CLOSE"].diff(10)
        df["Momentum30"] = df["CLOSE"].diff(30)

        # =====================
        # CANDLE FEATURES
        # =====================

        df["Body"] = df["CLOSE"] - df["OPEN"]
        df["Range"] = df["HIGH"] - df["LOW"]
        df["Range"] = df["Range"].replace(0, 1e-8)  # prevent div by zero
        df["BODY_PCT"] = df["Body"] / df["Range"]
        df["UPPER_WICK"] = df["HIGH"] - df[["OPEN", "CLOSE"]].max(axis=1)
        df["LOWER_WICK"] = df[["OPEN", "CLOSE"]].min(axis=1) - df["LOW"]

        # =====================
        # VOLATILITY
        # =====================

        df["VOL20"] = df["CLOSE"].pct_change().rolling(20).std()
        df["VOL60"] = df["CLOSE"].pct_change().rolling(60).std()

        # =====================
        # RSI
        # =====================

        delta = df["CLOSE"].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta).where(delta < 0, 0).rolling(14).mean()
        loss = loss.replace(0, 1e-8)  # prevent div by zero
        rs = gain / loss
        df["RSI14"] = 100 - (100 / (1 + rs))

        # =====================
        # MACD
        # =====================

        ema12 = df["CLOSE"].ewm(span=12).mean()
        ema26 = df["CLOSE"].ewm(span=26).mean()
        df["MACD"] = ema12 - ema26
        df["MACD_SIGNAL"] = df["MACD"].ewm(span=9).mean()
        df["MACD_HIST"] = df["MACD"] - df["MACD_SIGNAL"]

        # =====================
        # ATR
        # =====================

        tr1 = df["HIGH"] - df["LOW"]
        tr2 = abs(df["HIGH"] - df["CLOSE"].shift())
        tr3 = abs(df["LOW"] - df["CLOSE"].shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        df["ATR14"] = tr.rolling(14).mean()
        df["ATR14"] = df["ATR14"].replace(0, 1e-8)
        df["ATR_PCT"] = df["ATR14"] / df["CLOSE"].replace(0, 1e-8)

        # =====================
        # ADX
        # =====================

        up_move = df["HIGH"].diff()
        down_move = -df["LOW"].diff()
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
        tr14 = tr.rolling(14).sum().replace(0, 1e-8)
        plus_di = 100 * pd.Series(plus_dm).rolling(14).sum() / tr14
        minus_di = 100 * pd.Series(minus_dm).rolling(14).sum() / tr14
        dx = (abs(plus_di - minus_di) / (plus_di + minus_di + 1e-8)) * 100
        df["DI_PLUS"] = plus_di
        df["DI_MINUS"] = minus_di
        df["ADX14"] = dx.rolling(14).mean()

        # =====================
        # BREAKOUT
        # =====================

        df["HH20"] = df["HIGH"].rolling(20).max()
        df["LL20"] = df["LOW"].rolling(20).min()
        df["DIST_HH"] = df["HH20"] - df["CLOSE"]
        df["DIST_LL"] = df["CLOSE"] - df["LL20"]
        df["BREAK_HIGH"] = (df["CLOSE"] > df["HH20"].shift(1)).astype(int)
        df["BREAK_LOW"] = (df["CLOSE"] < df["LL20"].shift(1)).astype(int)

        # =====================
        # TREND STRENGTH
        # =====================

        df["TREND_UP"] = ((df["EMA20"] > df["EMA50"]) & (df["EMA50"] > df["EMA200"])).astype(int)
        df["TREND_DOWN"] = ((df["EMA20"] < df["EMA50"]) & (df["EMA50"] < df["EMA200"])).astype(int)

        # =====================
        # EXTRA FEATURES
        # =====================

        df["EMA20_50"] = df["EMA20"] - df["EMA50"]
        df["EMA50_200"] = df["EMA50"] - df["EMA200"]
        df["RSI_CHANGE"] = df["RSI14"].diff()
        df["ATR_CHANGE"] = df["ATR14"].pct_change()
        df["HH_BREAK_5"] = (df["HIGH"] > df["HIGH"].shift(5)).astype(int)
        df["LL_BREAK_5"] = (df["LOW"] < df["LOW"].shift(5)).astype(int)
        df["RSI_OVERBOUGHT"] = (df["RSI14"] > 70).astype(int)
        df["RSI_OVERSOLD"] = (df["RSI14"] < 30).astype(int)
        df["ATR_SPIKE"] = (df["ATR14"] > df["ATR14"].rolling(50).mean()).astype(int)

        # safe replacements
        df = df.replace([float('inf'), -float('inf')], float('nan'))

        before_drop = len(df)
        df = df.dropna()
        print(f"DEBUG add_features: dropped {before_drop - len(df)} rows (NaN). remaining={len(df)}")

        return df
