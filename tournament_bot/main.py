from pathlib import Path
from core.data_loader import DataLoader
from core.feature_engine import FeatureEngine
from core.ai_model import AIModel
from core.risk_manager import RiskManager
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)


class TournamentBot:

    def __init__(self):

        self.mode = "TRAIN"

        self.loader = DataLoader("data")
        self.features = FeatureEngine()
        self.ai = AIModel()
        self.risk = RiskManager()

        self.best_accuracy = 0.50

    def run(self):

        try:

            print(f"🚀 Tournament Bot {self.mode} mode")

            # =====================
            # LOAD DATA
            # =====================

            gold = self.loader.load_gold_data()

            print(f"Gold rows: {len(gold)}")

            # =====================
            # CREATE FEATURES
            # =====================

            print("⚙️ Creating features...")

            gold = self.features.add_features(gold)

            print(
                f"Rows after features: {len(gold)}"
            )

            # =====================
            # NOISE FILTER
            # =====================

          #  move = abs(
           #     gold["CLOSE"].shift(-15)
          #      - gold["CLOSE"]
          #  )

           # gold = gold[
               # move >
                #gold["ATR14"] * 0.30
            #]

            print(
                f"Rows after filter: {len(gold)}"
            )

            # =====================
            # CREATE TARGET
            # =====================

            future_move = (
                gold["CLOSE"].shift(-15)
                - gold["CLOSE"]
            )

            gold["Target"] = (
                future_move >
                gold["ATR14"] * 1.0
            ).astype(int)

            gold = gold.iloc[:-15]

            # =====================
            # FEATURE LIST
            # =====================

            features = [
            "EMA20",
            "EMA50",
            "EMA200",
            "EMA_DIFF",
            "EMA_SLOPE",
           
            "H1_EMA20",
            "H1_EMA50",
            "H1_TREND",

            "H4_EMA20",
            "H4_EMA50",
            "H4_TREND",

            "EMA20_50",
            "EMA50_200",

            "RSI14",
            "RSI_CHANGE",

            "ATR14",
            "ATR_PCT",
            "ATR_CHANGE",

            "Momentum",
            "Momentum10",
            "Momentum30",

            "Body",
            "Range",
            "BODY_PCT",
            "UPPER_WICK",
            "LOWER_WICK",

            "RET1",
            "RET5",
            "RET15",
            "RET60",

            "VOL20",
            "VOL60",

            "DIST_HH",
            "DIST_LL",

            "BREAK_HIGH",
            "BREAK_LOW",

            "HH_BREAK_5",
            "LL_BREAK_5",

            "TREND_UP",
            "TREND_DOWN",

            "WEEKDAY",
            "HOUR",

            "ASIA",
            "LONDON",
            "NEWYORK",

            "RSI_OVERBOUGHT",
            "RSI_OVERSOLD",
            "ATR_SPIKE",
            
            "MACD",
            "MACD_SIGNAL",
            "MACD_HIST",
           
            "ADX14",
            "DI_PLUS",
            "DI_MINUS"
            ]
            
            # =====================
            # PREPARE DATA
            # =====================

            data = gold.dropna().copy()

            X = data[features]
            y = data["Target"]

            print(
                f"Training rows: {len(X)}"
            )

            print(
                "\n===== TARGET DISTRIBUTION ====="
            )

            print(
                data["Target"]
                .value_counts(normalize=True)
            )

            # =====================
            # TRAIN TEST SPLIT
            # =====================

            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=0.20,
                shuffle=False
            )

            print(
                f"Train rows: {len(X_train)}"
            )

            print(
                f"Test rows: {len(X_test)}"
            )

            # =====================
            # TRAIN MODEL
            # =====================

            print("🧠 Training model...")

            self.ai.train(
                X_train,
                y_train
            )

            print("✅ Model trained")

            # =====================
            # TRAIN ACCURACY
            # =====================

            train_pred = self.ai.model.predict(
                X_train
            )

            train_acc = accuracy_score(
                y_train,
                train_pred
            )

            print(
                f"Train Accuracy: {train_acc:.2%}"
            )

            # =====================
            # TEST ACCURACY
            # =====================

            probs = (
                self.ai.model
                .predict_proba(X_test)[:, 1]
            )

            pred = (
                probs > 0.40
            ).astype(int)

            acc = accuracy_score(
                y_test,
                pred
            )

            print(
                f"Test Accuracy: {acc:.2%}"
            )

            print(
                "\n===== CONFUSION MATRIX ====="
            )

            print(
                confusion_matrix(
                    y_test,
                    pred
                )
            )

            print(
                "\n===== CLASSIFICATION REPORT ====="
            )

            print(
                classification_report(
                    y_test,
                    pred,
                    digits=4
                )
            )

            # =====================
            # CONFUSION MATRIX
            # =====================

            cm = confusion_matrix(
                y_test,
                pred
            )

            print(
                "\n===== CONFUSION MATRIX ====="
            )

            print(cm)

            # =====================
            # CLASSIFICATION REPORT
            # =====================

            print(
                "\n===== CLASSIFICATION REPORT ====="
            )

            print(
                classification_report(
                    y_test,
                    pred
                )
            )

            # =====================
            # FEATURE IMPORTANCE
            # =====================

            print(
                "\n===== FEATURE IMPORTANCE ====="
            )

            imp = (
                self.ai.model
                .feature_importances_
            )

            for name, value in sorted(
                zip(features, imp),
                key=lambda x: x[1],
                reverse=True
            ):
                print(
                    f"{name:20} {value:.4f}"
                )

            # =====================
            # SAVE BEST MODEL
            # =====================

            if acc > self.best_accuracy:

                self.best_accuracy = acc

                print(
                    "🔥 New best model"
                )

                self.ai.save()

            else:

                model_file = Path(
                    "models/best_model.pkl"
                )

                if model_file.exists():

                    print(
                        "📂 Loading previous best model..."
                    )

                    self.ai.load()

            # =====================
            # LATEST PREDICTION
            # =====================

            last_x = X.iloc[-1:].copy()

            prob = self.ai.predict_probability(
                last_x
            )

            print(
                f"Probability: {prob:.2%}"
            )

            # =====================
            # RISK MANAGEMENT
            # =====================

            lot = self.risk.get_lot_size(
                prob
            )

            print(
                f"Lot Size: {lot}"
            )

            # =====================
            # SIGNAL
            # =====================

            ema20 = data.iloc[-1]["EMA20"]
            ema50 = data.iloc[-1]["EMA50"]

            if (
                prob >= 0.80
                and ema20 > ema50
            ):
                print("🟢 BUY SIGNAL")

            elif (
                prob <= 0.20
                and ema20 < ema50
            ):
                print("🔴 SELL SIGNAL")

            else:
                print("🟡 WAIT")

            # =====================
            # SELF LEARNING CHECK
            # =====================

            if acc < 0.55:
                print(
                    "⚠ Accuracy low. Retraining recommended."
                )

        except Exception as e:

            print(
                f"❌ Error: {e}"
            )


if __name__ == "__main__":

    bot = TournamentBot()
    bot.run()