import sys
from pathlib import Path

# Fix encoding for Windows console
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.data_loader import DataLoader
from core.feature_engine import FeatureEngine
from core.ai_model import AIModel
from core.risk_manager import RiskManager
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    precision_score,
    recall_score,
    f1_score
)

class TournamentBot:

    def __init__(self):

        self.mode = "TRAIN"

        self.loader = DataLoader("data")
        self.features = FeatureEngine()
        self.ai = AIModel()
        self.risk = RiskManager()

        self.best_f1 = 0.50

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
                gold["CLOSE"].shift(-60)
                - gold["CLOSE"]
            )

            gold["Target"] = (
                future_move >
                gold["ATR14"] * 0.5
            ).astype(int)

            gold = gold.iloc[:-60]

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

            best_score = 0
            best_threshold = None

            probs = (
                self.ai.model
                .predict_proba(X_test)[:, 1]
            )

            print("\n===== THRESHOLD SEARCH =====")

            for threshold in [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]:
                pred = (
                    probs > threshold
                ).astype(int)

                acc = accuracy_score(
                    y_test,
                    pred
                )

                precision = precision_score(
                    y_test,
                    pred,
                    zero_division=0
                )

                recall = recall_score(
                    y_test,
                    pred,
                    zero_division=0
                )

                f1 = f1_score(
                    y_test,
                    pred,
                    zero_division=0
                )

                print(
                    f"T={threshold:.2f} "
                    f"ACC={acc:.4f} "
                    f"P={precision:.4f} "
                    f"R={recall:.4f} "
                    f"F1={f1:.4f}"
                )

                if f1 > best_score:
                    best_score = f1
                    best_threshold = threshold
            
            print(f"\nBEST THRESHOLD: {best_threshold}")
            print(f"BEST F1: {best_score:.4f}")

            pred = (
                probs > best_threshold
            ).astype(int)

            acc = accuracy_score(
                y_test,
                pred
            )
            # =====================
            # TRADING STATS (RISK-BASED)
            # =====================

            wins = 0
            losses = 0
            equity_curve = [0]
            pnl_trades = []
            
            # Risk management parameters
            RISK_PER_TRADE = 1.0  # 1 point risk
            REWARD_RATIO = 1.5    # 1:1.5 risk-reward

            for i in range(len(pred)):
                if pred[i] == 1:
                    atr = data.iloc[i]["ATR14"]
                    
                    # Calculate risk and reward
                    stop_loss_dist = atr * 2.0  # 2 ATR
                    risk = RISK_PER_TRADE if atr > 0 else 1.0
                    reward = risk * REWARD_RATIO
                    
                    # PnL calculation
                    if y_test.iloc[i] == 1:
                        pnl = reward
                        wins += 1
                    else:
                        pnl = -risk
                        losses += 1
                    
                    pnl_trades.append(pnl)
                    equity_curve.append(equity_curve[-1] + pnl)

            total_trades = wins + losses
            
            if total_trades > 0:
                win_rate = wins / total_trades
                total_pnl = sum(pnl_trades)
                avg_win = (
                    sum([p for p in pnl_trades if p > 0]) / wins
                    if wins > 0
                    else 0
                )
                avg_loss = (
                    abs(sum([p for p in pnl_trades if p < 0])) / losses
                    if losses > 0
                    else 0
                )
                profit_factor = (
                    avg_win * wins / (avg_loss * losses)
                    if (avg_loss * losses) > 0
                    else float("inf")
                )
                expected_value = (
                    win_rate * avg_win - (1 - win_rate) * avg_loss
                )
            else:
                win_rate = 0
                total_pnl = 0
                avg_win = 0
                avg_loss = 0
                profit_factor = 0
                expected_value = 0

            # Calculate drawdown
            if len(equity_curve) > 1:
                peak = equity_curve[0]
                max_dd = 0
                max_dd_pct = 0

                for value in equity_curve[1:]:
                    if value > peak:
                        peak = value

                    drawdown = peak - value
                    drawdown_pct = (
                        (drawdown / peak * 100)
                        if peak != 0
                        else 0
                    )

                    if drawdown > max_dd:
                        max_dd = drawdown
                        max_dd_pct = drawdown_pct
            else:
                max_dd = 0
                max_dd_pct = 0

            # Calculate additional metrics
            returns = [equity_curve[i+1] - equity_curve[i] 
                      for i in range(len(equity_curve)-1)]
            
            if len(returns) > 0:
                mean_return = sum(returns) / len(returns)
                variance = sum((r - mean_return) ** 2 for r in returns) / len(returns)
                std_dev = variance ** 0.5
                
                if std_dev > 0:
                    sharpe_ratio = (mean_return / std_dev) * (252 ** 0.5)
                else:
                    sharpe_ratio = 0
            else:
                std_dev = 0
                sharpe_ratio = 0

            print("\n===== TRADING STATS (RISK-BASED) =====")
            print(f"Total Trades: {total_trades}")
            print(f"Wins: {wins}")
            print(f"Losses: {losses}")
            print(f"Win Rate: {win_rate:.2%}")
            print(f"Avg Win: {avg_win:.2f}")
            print(f"Avg Loss: {avg_loss:.2f}")
            
            if profit_factor == float("inf"):
                print("Profit Factor: ∞")
            else:
                print(f"Profit Factor: {profit_factor:.2f}")
            
            print(f"Expected Value: {expected_value:.4f}")
            print(f"Total PnL: {total_pnl:.2f}")
            print(f"Max Drawdown: {max_dd:.2f} ({max_dd_pct:.2f}%)")
            print(f"Std Dev: {std_dev:.4f}")
            print(f"Sharpe Ratio: {sharpe_ratio:.4f}")
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

            if best_score > self.best_f1:

                self.best_f1 = best_score

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

            if best_score < 0.55:
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