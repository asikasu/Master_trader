import sys
import os
import time
import signal
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import logging
import threading
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score

from core.data_loader import DataLoader
from core.feature_engine import FeatureEngine
from core.ai_model import AIModel
from core.risk_manager import RiskManager
from core.mt5_executor import MT5Executor
from core.filters import NewsFilter, SpreadFilter, SelfHealingEngine
from core.rules_engine import RulesEngine
from core.mtf_filter import MTFFilter
from core.evolutionary.evolutionary_trainer import EvolutionaryTrainer, FEATURE_COLUMNS
from core.evolutionary.evolutionary_engine import EvolutionaryEngine
from core.evolutionary.persistence import load_best_params, load_state
from core.evolutionary.types import ParameterCombo, XGBoostConfig, TradingConfig
from tournament_bot.dashboard import Dashboard


class TournamentBot:
    def __init__(self):
        self.loader = DataLoader("data")
        self.features = FeatureEngine()
        self.ai = AIModel()
        self.risk = RiskManager()
        self.evo_trainer = EvolutionaryTrainer("data")
        self.spread_filter = SpreadFilter(max_spread_bps=2.0)
        self.news_filter = NewsFilter()
        self.mtf_filter = MTFFilter()
        self.executor = MT5Executor(spread_filter=self.spread_filter, news_filter=self.news_filter)
        self.rules = RulesEngine(spread_bps=2.0)
        self.best_f1 = 0.50
        self.feature_list = FEATURE_COLUMNS
        self.healing_engine = None
        self.dashboard = Dashboard()
        self._running = False
        self.scan_interval = 5

    def _prepare_data(self, df=None):
        if df is None:
            df = self.loader.load_gold_data()
        df = self.features.add_features(df)
        future_move = df["CLOSE"].shift(-60) - df["CLOSE"]
        df["Target"] = (future_move > df["ATR14"] * 0.5).astype(int)
        df = df.iloc[:-60].dropna().copy()
        return df

    def run_train(self):
        print("=== TRAIN MODE ===")
        gold = self._prepare_data()
        X = gold[self.feature_list]
        y = gold["Target"]
        print(f"Train rows: {len(X)}  (0: {sum(y==0)}, 1: {sum(y==1)})")

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, shuffle=False)
        print(f"Train: {len(X_train)}, Test: {len(X_test)}")

        self.ai.train(X_train, y_train, eval_set=[(X_test, y_test)], verbose=True)
        train_acc = accuracy_score(y_train, self.ai.model.predict(X_train))
        print(f"Train Accuracy: {train_acc:.2%}")

        probs = self.ai.model.predict_proba(X_test)[:, 1]
        best_score, best_threshold = 0, 0.5
        print("\nTHRESHOLD SEARCH:")
        for t in [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]:
            p = (probs > t).astype(int)
            f1 = f1_score(y_test, p, zero_division=0)
            acc = accuracy_score(y_test, p)
            print(f"  T={t:.2f} ACC={acc:.4f} F1={f1:.4f}")
            if f1 > best_score:
                best_score, best_threshold = f1, t
        print(f"Best: threshold={best_threshold}, F1={best_score:.4f}")

        pred = (probs > best_threshold).astype(int)
        wins = sum(1 for i in range(len(pred)) if pred[i] == 1 and y_test.iloc[i] == 1)
        losses = sum(1 for i in range(len(pred)) if pred[i] == 1 and y_test.iloc[i] == 0)
        total = wins + losses
        if total > 0:
            win_rate = wins / total
            print(f"\nTrading Stats: Trades={total}, WinRate={win_rate:.2%}")

        cm = confusion_matrix(y_test, pred)
        print(f"\nConfusion Matrix:\n{cm}")
        print(f"\n{classification_report(y_test, pred, zero_division=0)}")

        imp = self.ai.model.feature_importances_
        print("\nFEATURE IMPORTANCE (top 10):")
        for name, val in sorted(zip(self.feature_list, imp), key=lambda x: x[1], reverse=True)[:10]:
            print(f"  {name:20} {val:.4f}")

        if best_score > self.best_f1:
            self.best_f1 = best_score
            self.ai.save()
            print("[SAVE] New best model")

        prob = self.ai.predict_probability(X.iloc[-1:])
        print(f"\nProbability: {prob:.2%}")
        print(f"Lot Size: {self.risk.get_lot_size(prob)}")

        now = datetime.now()
        news_warning = f" (NEXT: {self.news_filter.next_news(now)})" if self.news_filter.is_news_event(now) else ""
        print(f"News Filter: {'BLOCKED' if self.news_filter.is_news_event(now) else 'OK'}{news_warning}")

        last_row = gold.iloc[-1]
        sig = self.rules.validate_signal(last_row, prob, gold)
        sl_str = f"SL={sig['sl']:.2f}" if sig['sl'] else ""
        tp_str = f"TP={sig['tp']:.2f}" if sig['tp'] else ""
        reason = f" | {sig['reason']}" if sig['reason'] else ""
        print(f"SIGNAL: {sig['signal']} | Prob={prob:.2%} {sl_str} {tp_str}{reason}")

    def run_evolve(self, generations=10, population=100, sample_size=5000):
        print("=== EVOLVE MODE ===")
        engine = self.evo_trainer.run_evolution(
            generations=generations,
            population=population,
            resume=True,
            state_dir=str(Path(".").resolve()),
            sample_size=sample_size,
        )
        state = engine.state
        if state and state.best_fitness:
            print(f"\nEvolution complete: {state.generation} generations")
            print(f"  Composite Score: {state.best_fitness.composite_score:.4f}")
            print(f"  Sharpe Ratio:    {state.best_fitness.sharpe_ratio:.4f}")
            print(f"  Total Profit:    {state.best_fitness.total_profit:.2f}")
            print(f"  Max Drawdown:    {state.best_fitness.max_drawdown_pct:.2f}%")

    def _init_healing(self):
        gold = self._prepare_data()
        evo = EvolutionaryEngine(
            data=gold,
            feature_columns=self.feature_list,
            target_column="Target",
            population_size=50,
            max_generations=5,
            state_dir=".",
            resume=True,
        )
        self.healing_engine = SelfHealingEngine(evo)

    def _check_open_positions(self, symbol):
        positions = self.executor.positions()
        return [p for p in positions if p.symbol == symbol] if positions else []

    def _manage_open_position(self, pos, current_price):
        """Check SL/TP for open position."""
        side = "BUY" if pos.type == 0 else "SELL"
        if side == "BUY" and current_price <= pos.sl:
            self.executor.close_position(pos.ticket)
            return "SL_HIT"
        if side == "SELL" and current_price >= pos.sl:
            self.executor.close_position(pos.ticket)
            return "SL_HIT"
        if side == "BUY" and current_price >= pos.tp:
            self.executor.close_position(pos.ticket)
            return "TP_HIT"
        if side == "SELL" and current_price <= pos.tp:
            self.executor.close_position(pos.ticket)
            return "TP_HIT"
        return "HOLDING"

    def run_live(self, symbol="XAUUSD"):
        print("=== LIVE TRADING MODE ===")
        mt5_ok = self.executor.initialize()
        if not mt5_ok:
            print("[WARNING] MT5 not connected. Running in SIMULATION mode.")

        gold = self._prepare_data()
        X = gold[self.feature_list]
        y = gold["Target"]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, shuffle=False)
        self.ai.train(X_train, y_train)
        train_acc = accuracy_score(y_train, self.ai.model.predict(X_train))
        print(f"Model ready, train acc: {train_acc:.2%}")

        self._init_healing()

        self.dashboard.update(
            mode="LIVE",
            mt5_status="connected" if mt5_ok else "simulation",
        )
        self._running = True

        def handle_stop(sig, frame):
            self._running = False
        signal.signal(signal.SIGINT, handle_stop)
        signal.signal(signal.SIGTERM, handle_stop)

        dashboard_thread = threading.Thread(target=self.dashboard.start, daemon=True)
        dashboard_thread.start()

        last_train_time = datetime.now()
        open_ticket = None
        signal_count = 0

        while self._running:
            try:
                now = datetime.now()

                self.dashboard.update(news_block=self.news_filter.is_news_event(now))

                if self.news_filter.is_news_event(now):
                    if open_ticket:
                        self.executor.close_position(open_ticket)
                        open_ticket = None
                    time.sleep(60)
                    continue

                if open_ticket:
                    tick = self.executor.mt5.symbol_info_tick(symbol) if self.executor.connected else None
                    if tick:
                        price = tick.bid
                        result = self._manage_open_position(open_ticket, price)
                        if result in ("SL_HIT", "TP_HIT"):
                            self.healing_engine.on_trade_result(-1 if result == "SL_HIT" else 1)
                            open_ticket = None
                    time.sleep(self.scan_interval)
                    continue

                if not self.executor.can_trade(symbol):
                    time.sleep(60)
                    continue

                if self.healing_engine.should_retrain(now):
                    print("Weekly retrain triggered")
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, shuffle=False)
                    self.ai.train(X_train, y_train)
                    last_train_time = now

                prob = self.ai.predict_probability(X.iloc[-1:])
                last_row = gold.iloc[-1]
                sig = self.rules.validate_signal(last_row, prob, gold)

                mtf_result = self.mtf_filter.confirm(gold, prob, sig["signal"])
                sig["signal"] = mtf_result["signal"]
                sig["reason"] = mtf_result["reason"]

                self.dashboard.update(
                    last_signal=sig["signal"],
                    last_prob=mtf_result["confidence"],
                    spread_bps=(gold.iloc[-1].get("SPREAD", 0) if "SPREAD" in gold.columns else 0),
                    ema20=last_row.get("EMA20", 0),
                    ema50=last_row.get("EMA50", 0),
                    atr=last_row.get("ATR14", 0),
                )

                if sig["signal"] in ("BUY", "SELL"):
                    order_type = 0 if sig["signal"] == "BUY" else 1
                    price = self.executor.mt5.symbol_info_tick(symbol).ask if (
                        self.executor.connected and order_type == 0
                    ) else (self.executor.mt5.symbol_info_tick(symbol).bid if self.executor.connected else last_row["CLOSE"])
                    if self.executor.connected and not price:
                        price = last_row["CLOSE"]
                    lot = self.risk.get_lot_size(mtf_result["confidence"])
                    ticket = self.executor.place_order(
                        symbol, order_type, lot, price,
                        sig["sl"], sig["tp"],
                        comment=f"gen{self.dashboard._data['generation']}",
                    )
                    if ticket:
                        open_ticket = ticket
                        signal_count += 1

                time.sleep(self.scan_interval)

            except Exception as e:
                logging.error("LIVE loop error: %s", e)
                time.sleep(10)

        self.executor.shutdown()
        print(f"LIVE stopped. Signals sent: {signal_count}")

    def run(self):
        print(f"Master Trader Bot - {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        self.run_train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Master Trader Tournament Bot")
    parser.add_argument("--mode", choices=["TRAIN", "EVOLVE", "LIVE", "BOTH"], default="TRAIN")
    parser.add_argument("--generations", type=int, default=10)
    parser.add_argument("--population", type=int, default=100)
    parser.add_argument("--symbol", default="XAUUSD")
    parser.add_argument("--sample", type=int, default=5000,
                        help="Evolution: use first N rows for speed (0 = all)")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level),
                        format="%(asctime)s [%(levelname)s] %(message)s")

    try:
        bot = TournamentBot()
        if args.mode in ("TRAIN", "BOTH"):
            bot.run_train()
        if args.mode in ("EVOLVE", "BOTH"):
            bot.run_evolve(generations=args.generations, population=args.population, sample_size=args.sample)
        if args.mode == "LIVE":
            bot.run_live(symbol=args.symbol)
    except Exception as e:
        import traceback
        print(f"!!! CRITICAL ERROR: {e}")
        traceback.print_exc()
        raise
