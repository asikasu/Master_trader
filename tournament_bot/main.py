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

import json

KELLY_FRACTION = 0.25
MAX_RISK_PER_TRADE = 0.02
TRAILING_ACTIVATE = 0.5
BREAK_EVEN_ACTIVATE = 0.3
PARTIAL_CLOSE_RATIO = 0.5
STATE_FILE = "live_state.json"


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
        self.scan_interval = 15
        self._kelly_win_rate = 0.55
        self._kelly_avg_win = 1.0
        self._kelly_avg_loss = 1.0
        self._raw_data_store = None

    def _load_state(self):
        try:
            with open(STATE_FILE) as f:
                s = json.load(f)
            self._kelly_avg_win = s.get("avg_win", 1.0)
            self._kelly_avg_loss = s.get("avg_loss", 1.0)
            return s.get("balance", 10000.0)
        except (FileNotFoundError, json.JSONDecodeError):
            return 10000.0

    def _save_state(self, balance):
        try:
            with open(STATE_FILE, "w") as f:
                json.dump({
                    "balance": balance,
                    "avg_win": self._kelly_avg_win,
                    "avg_loss": self._kelly_avg_loss,
                }, f)
        except Exception:
            pass

    def _prepare_data(self, df=None):
        if df is None:
            df = self.loader.load_gold_data()
        raw_count = len(df)
        df = self.features.add_features(df)
        future_move = df["CLOSE"].shift(-60) - df["CLOSE"]
        df["Target"] = (future_move > df["ATR14"] * 0.5).astype(int)
        df = df.iloc[:-60].dropna(subset=["Target", "CLOSE"]).copy()
        logging.info("Data: raw=%d, features=%d, trainable=%d", raw_count, raw_count, len(df))
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

    def _manage_open_position(self, pos, current_price, atr, entry_sl, entry_tp):
        """Modify SL/TP only, no close/reopen. Returns status string."""
        side = "BUY" if pos.type == 0 else "SELL"
        entry = pos.price_open
        direction = 1 if side == "BUY" else -1
        pnl_pct = (current_price - entry) / entry * direction

        if pnl_pct <= -0.02:
            return "SL_HIT"

        if pnl_pct >= 0.02:
            return "TP_HIT"

        new_sl = pos.sl
        new_tp = pos.tp

        if pnl_pct >= BREAK_EVEN_ACTIVATE * 0.01:
            be_sl = entry * (1 + 0.001 * direction)
            if (side == "BUY" and be_sl > pos.sl) or (side == "SELL" and be_sl < pos.sl):
                new_sl = be_sl

        if pnl_pct >= TRAILING_ACTIVATE * 0.01:
            trail_dist = atr * 1.5
            if side == "BUY":
                proposed = current_price - trail_dist
                if proposed > new_sl:
                    new_sl = proposed
            else:
                proposed = current_price + trail_dist
                if proposed < new_sl:
                    new_sl = proposed

        if (new_sl != pos.sl or new_tp != pos.tp):
            self.executor.modify_position(pos.ticket, new_sl, new_tp)

        return "HOLDING"

    def _kelly_lot(self, prob, balance, sl_points):
        if prob <= 0.5 or sl_points <= 0:
            return 0.01
        win_rate = prob
        loss_rate = 1 - prob
        avg_win = max(self._kelly_avg_win, 0.1)
        avg_loss = max(self._kelly_avg_loss, 0.1)
        b = avg_win / avg_loss
        kelly_pct = (win_rate - loss_rate / b) * KELLY_FRACTION
        risk_amount = balance * min(MAX_RISK_PER_TRADE, max(kelly_pct, 0.001))
        tick_value = 1.0
        lot = risk_amount / (sl_points * tick_value)
        return round(max(min(lot, 1.0), 0.01), 2)

    def _sync_open_positions(self, symbol):
        positions = self.executor.positions()
        if self.executor.connected and positions:
            for p in positions:
                if p.symbol == symbol:
                    return p
        return None

    def _reconnect_mt5(self):
        if self.executor.connected:
            return True
        logging.warning("MT5 disconnected. Checking terminal status...")
        for attempt in range(3):
            try:
                info = self.executor.mt5.terminal_info()
                if info is not None:
                    self.executor.connected = True
                    return True
            except Exception:
                pass
            logging.info("MT5 reconnect attempt %d...", attempt + 1)
            self.executor.initialize()
            time.sleep(2)
        return False

    def run_live(self, symbol="XAUUSD"):
        print("=== LIVE TRADING MODE ===")
        mt5_ok = self.executor.initialize()
        if not mt5_ok:
            print("[WARNING] MT5 not connected. Running in SIMULATION mode.")

        self.ai.load()

        gold = self._prepare_data()
        X = gold[self.feature_list]

        self._init_healing()

        balance = self._load_state()

        self.dashboard.update(
            mode="LIVE",
            mt5_status="connected" if mt5_ok else "simulation",
        )
        self._running = True

        def handle_stop(sig, frame):
            self._save_state(balance)
            self._running = False
        signal.signal(signal.SIGINT, handle_stop)
        signal.signal(signal.SIGTERM, handle_stop)

        dashboard_thread = threading.Thread(target=self.dashboard.start, daemon=True)
        dashboard_thread.start()

        last_train_time = datetime.now()
        signal_count = 0
        entry_sl = None
        entry_tp = None

        while self._running:
            try:
                now = datetime.now()

                if self.news_filter.is_news_event(now):
                    print("[NEWS] Medeenii uyed arijaag zasvarlaj baina.")
                    time.sleep(60)
                    continue

                gold = self._prepare_data()
                X = gold[self.feature_list]
                last_row = gold.iloc[-1]
                current_atr = last_row.get("ATR14", 10.0)

                open_pos = self._sync_open_positions(symbol)

                if open_pos:
                    tick = self.executor.mt5.symbol_info_tick(symbol) if self.executor.connected else None
                    if not tick and not self._reconnect_mt5():
                        time.sleep(self.scan_interval)
                        continue
                    if not tick:
                        tick = self.executor.mt5.symbol_info_tick(symbol)
                    price = tick.bid
                    result = self._manage_open_position(open_pos, price, current_atr, entry_sl, entry_tp)
                    real_pnl = open_pos.profit if self.executor.connected else 0.0

                    if result == "SL_HIT":
                        balance += real_pnl
                        self._kelly_avg_loss = (self._kelly_avg_loss * 9 + abs(real_pnl)) / 10
                        self.healing_engine.on_trade_result(-1)
                        self._save_state(balance)
                        print(f"[CLOSE] SL_HIT PnL={real_pnl:.2f} Bal={balance:.2f}")
                        entry_sl = entry_tp = None
                    elif result == "TP_HIT":
                        balance += real_pnl
                        self._kelly_avg_win = (self._kelly_avg_win * 9 + real_pnl) / 10
                        self.healing_engine.on_trade_result(1)
                        self._save_state(balance)
                        print(f"[CLOSE] TP_HIT PnL={real_pnl:.2f} Bal={balance:.2f}")
                        entry_sl = entry_tp = None
                    elif result == "HOLDING":
                        print(f"[POS] Price={price:.2f} PnL={real_pnl:.2f} Bal={balance:.2f}")
                    time.sleep(self.scan_interval)
                    continue

                if not self.executor.can_trade(symbol):
                    time.sleep(60)
                    continue

                if self.healing_engine.should_retrain(now):
                    print("[RETRAIN] Weekly retrain with fresh data...")
                    gold = self._prepare_data()
                    X = gold[self.feature_list]
                    y = gold["Target"]
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, shuffle=False)
                    self.ai.train(X_train, y_train)
                    self.ai.save()
                    last_train_time = now

                prob = self.ai.predict_probability(X.iloc[-1:])
                sig = self.rules.validate_signal(last_row, prob, gold)
                mtf_result = self.mtf_filter.confirm(gold, prob, sig["signal"])
                logging.info("Signal: prob=%.3f raw=%s mtf=%s/%s",
                             prob, sig["signal"], mtf_result["signal"], mtf_result.get("reason",""))
                sig["signal"] = mtf_result["signal"]
                sig["reason"] = mtf_result["reason"]

                self.dashboard.update(
                    last_signal=sig["signal"],
                    last_prob=mtf_result["confidence"],
                    spread_bps=last_row.get("SPREAD", 0),
                    ema20=last_row.get("EMA20", 0),
                    ema50=last_row.get("EMA50", 0),
                    atr=current_atr,
                )

                if sig["signal"] in ("BUY", "SELL"):
                    sl_points = abs(sig["sl"] - last_row["CLOSE"]) if sig["sl"] else current_atr
                    lot = self._kelly_lot(mtf_result["confidence"], balance, sl_points)
                    if lot > 0.01:
                        if self.executor.connected:
                            tick = self.executor.mt5.symbol_info_tick(symbol)
                            price = tick.ask if sig["signal"] == "BUY" else tick.bid
                        else:
                            price = last_row["CLOSE"]
                        order_type = 0 if sig["signal"] == "BUY" else 1
                        logging.info("Placing order: %s lot=%.2f price=%.2f sl=%.2f tp=%.2f",
                                      sig["signal"], lot, price, sig["sl"], sig["tp"])
                        ticket = self.executor.place_order(
                            symbol, order_type, lot, price,
                            sig["sl"], sig["tp"],
                            comment=f"kelly_{prob:.2f}",
                        )
                        if ticket:
                            entry_sl = sig["sl"]
                            entry_tp = sig["tp"]
                            signal_count += 1
                            print(f"[OPEN] {sig['signal']} Lot={lot:.2f} Price={price:.2f} SL={sig['sl']:.2f} TP={sig['tp']:.2f} Bal={balance:.2f}")
                        else:
                            logging.warning("Place order returned no ticket")
                    else:
                        logging.info("Lot too small: %.3f (prob=%.3f, balance=%.2f, sl_pts=%.1f)",
                                     lot, prob, balance, sl_points)

                time.sleep(self.scan_interval)

            except (ConnectionError, TimeoutError, ValueError, OSError) as e:
                logging.error("LIVE recoverable error: %s", e)
                self._save_state(balance)
                time.sleep(30)
            except Exception as e:
                logging.critical("LIVE unrecoverable error: %s", e, exc_info=True)
                self._save_state(balance)
                print(f"[CRITICAL] Bot crashed: {e}")
                self.executor.shutdown()
                sys.exit(1)

        self._save_state(balance)
        self.executor.shutdown()
        print(f"LIVE stopped. Signals: {signal_count}, Final Bal: {balance:.2f}")

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

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler("bot.log", encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )

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
