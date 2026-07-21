import sys
import os
import time
import signal
import concurrent.futures
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
from xgboost import XGBClassifier

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
        self.evo_trainer = EvolutionaryTrainer("data")
        self.spread_filter = SpreadFilter(max_spread_bps=2.0)
        self.news_filter = NewsFilter()
        self.mtf_filter = MTFFilter()
        self.executor = MT5Executor(spread_filter=self.spread_filter, news_filter=self.news_filter)
        self._best_xgb_config = None
        self._best_trading_config = None
        self._load_best_evolution_params()
        tc = self._best_trading_config
        if tc:
            self.rules = RulesEngine(spread_bps=2.0, buy_threshold=tc.buy_threshold,
                                     sell_threshold=tc.sell_threshold,
                                     stop_loss_pct=tc.stop_loss_pct,
                                     take_profit_pct=tc.take_profit_pct)
        else:
            self.rules = RulesEngine(spread_bps=2.0)
        self.best_f1 = 0.50
        self.feature_list = FEATURE_COLUMNS
        self.healing_engine = None
        self.dashboard = Dashboard()
        self._running = False
        self.scan_interval = 15
        self.ai = AIModel(xgb_config=self._best_xgb_config)

    def _load_best_evolution_params(self):
        try:
            best = load_best_params("best_params.json")
            if best:
                combo = best[0][0]
                self._best_xgb_config = combo.xgb
                self._best_trading_config = combo.trading
                logging.info("Loaded best params: combo_id=%d n_est=%d depth=%d lr=%.3f buy_thr=%.2f sl=%.3f tp=%.3f",
                             combo.combo_id, combo.xgb.n_estimators, combo.xgb.max_depth,
                             combo.xgb.learning_rate, combo.trading.buy_threshold,
                             combo.trading.stop_loss_pct, combo.trading.take_profit_pct)
                return
        except Exception:
            pass
        self._best_xgb_config = None
        self._best_trading_config = None
        logging.info("No best_params.json, using defaults")
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

    def _fetch_mtf_rates(self, sym, tf, label):
        """Нэг timeframe-аас 500 мөр татах, timeout 30s."""
        if not self.executor.connected:
            return None
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
                fut = ex.submit(self.executor.mt5.copy_rates_from_pos, sym, tf, 0, 500)
                rates = fut.result(timeout=30)
            if rates is None or len(rates) == 0:
                logging.warning("No %s data for %s", label, sym)
                return None
            df = pd.DataFrame(rates)
            df.rename(columns={"open": "OPEN", "high": "HIGH", "low": "LOW", "close": "CLOSE", "tick_volume": "VOLUME"}, inplace=True)
            dt = pd.to_datetime(df["time"], unit="s")
            df["DATETIME"] = dt
            return df
        except concurrent.futures.TimeoutError:
            logging.warning("MT5 %s TIMEOUT for %s", label, sym)
            return None
        except Exception as e:
            logging.warning("MT5 %s error: %s", label, e)
            return None

    def _prepare_live_data(self, symbol="XAUUSD"):
        """M15, H1, H4 - гурван timeframe-аас data татаж features бэлтгэнэ."""
        if not self.executor.connected:
            logging.warning("MT5 not connected, using cached data")
            return self._cached_data
        try:
            sym = self.executor._resolve_symbol(symbol)
            d15 = self._fetch_mtf_rates(sym, self.executor.mt5.TIMEFRAME_M15, "M15")
            h1 = self._fetch_mtf_rates(sym, self.executor.mt5.TIMEFRAME_H1, "H1")
            h4 = self._fetch_mtf_rates(sym, self.executor.mt5.TIMEFRAME_H4, "H4")
            main_df = d15 if d15 is not None else (h1 if h1 is not None else h4)
            if main_df is None:
                logging.warning("No data from any timeframe")
                return None
            df = main_df.copy()
            dt = df["DATETIME"]
            df["DATE"] = dt.dt.strftime("%Y.%m.%d")
            df["TIME"] = dt.dt.strftime("%H:%M:%S")
            df.sort_values("DATETIME", inplace=True)
            if h1 is not None:
                h1_s = h1[["DATETIME", "OPEN", "HIGH", "LOW", "CLOSE"]].copy()
                h1_s.rename(columns={"OPEN": "H1_OPEN", "HIGH": "H1_HIGH", "LOW": "H1_LOW", "CLOSE": "H1_CLOSE"}, inplace=True)
                h1_s.sort_values("DATETIME", inplace=True)
                df = pd.merge_asof(df, h1_s, on="DATETIME", direction="backward")
            if h4 is not None:
                h4_s = h4[["DATETIME", "OPEN", "HIGH", "LOW", "CLOSE"]].copy()
                h4_s.rename(columns={"OPEN": "H4_OPEN", "HIGH": "H4_HIGH", "LOW": "H4_LOW", "CLOSE": "H4_CLOSE"}, inplace=True)
                h4_s.sort_values("DATETIME", inplace=True)
                df = pd.merge_asof(df, h4_s, on="DATETIME", direction="backward")
            tick = self.executor.mt5.symbol_info_tick(sym)
            last_spread = (tick.ask - tick.bid) / tick.bid * 10000 if tick and tick.bid > 0 else 1.0
            df["SPREAD"] = last_spread
            logging.info("Live data: %d rows from %s", len(df), sym)
            logging.debug("MT5 raw last 3: %s", df[["DATE","TIME","OPEN","HIGH","LOW","CLOSE","VOLUME"]].tail(3).to_string())
            result = self._prepare_data(df)
            if result is not None and len(result) > 0:
                self._cached_data = result
            return result
        except Exception as e:
            logging.error("_prepare_live_data failed: %s", e, exc_info=True)
            return None

    def _prepare_data(self, df=None, n_rows=0, year_filter=None):
        if df is None:
            df = self.loader.load_gold_data(n_rows=n_rows)
        raw_count = len(df)
        df = self.features.add_features(df)
        if year_filter is not None:
            start_y, end_y = year_filter
            mask = (df["DATETIME"].dt.year >= start_y) & (df["DATETIME"].dt.year <= end_y)
            df = df[mask].copy()
        future_move = df["CLOSE"].shift(-4) - df["CLOSE"]
        df["Target"] = (future_move > df["ATR14"] * 0.3).astype(int)
        df = df.iloc[:-4].dropna(subset=["Target", "CLOSE"]).copy()
        logging.info("Data: raw=%d, features=%d, trainable=%d year=%s", raw_count, raw_count, len(df), year_filter)
        if len(df) > 0:
            logging.debug("Last row features: EMA20=%.2f EMA50=%.2f ATR14=%.4f SPREAD=%.2f CLOSE=%.2f",
                          df["EMA20"].iloc[-1], df["EMA50"].iloc[-1], df["ATR14"].iloc[-1],
                          df["SPREAD"].iloc[-1] if "SPREAD" in df.columns else 0,
                          df["CLOSE"].iloc[-1])
        return df

    def run_train(self, n_rows=0):
        print("=== TRAIN MODE (Yearly CV, 3 folds) ===")

        # Override XGB config for training speed
        class FastXGBConfig:
            n_estimators = 50
            max_depth = 6
            learning_rate = 0.05
        self.ai.xgb_config = FastXGBConfig()
        self.ai.model = None  # force rebuild on first train

        cv_pairs = [
            (2016, 2017, 2018),
            (2019, 2020, 2021),
            (2022, 2023, 2024),
        ]
        print(f"Folds: {cv_pairs}")
        print()

        best_overall_f1 = 0
        best_overall_threshold = 0.5
        fold_results = []

        for train_start, train_end, test_year in cv_pairs:
            train_years = (train_start, train_end)
            print(f"\n{'='*60}")
            print(f"FOLD: Train={train_years}  Test={test_year}")
            print(f"{'='*60}")

            gold = self._prepare_data(year_filter=train_years)
            X_train_all = gold[self.feature_list]
            y_train_all = gold["Target"]
            print(f"  Train: {len(X_train_all)} rows  (0: {sum(y_train_all==0)}, 1: {sum(y_train_all==1)})")

            test_df = self._prepare_data(year_filter=(test_year, test_year))
            X_test = test_df[self.feature_list]
            y_test = test_df["Target"]
            print(f"  Test:  {len(X_test)} rows  (0: {sum(y_test==0)}, 1: {sum(y_test==1)})")

            if len(X_train_all) < 100 or len(X_test) < 10:
                print(f"  SKIP: insufficient data")
                continue

            self.ai.train(X_train_all, y_train_all, eval_set=[(X_test, y_test)], verbose=True)

            probs = self.ai.model.predict_proba(X_test)[:, 1]
            best_f1, best_thr = 0, 0.5
            for t in [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]:
                p = (probs > t).astype(int)
                f1 = f1_score(y_test, p, zero_division=0)
                if f1 > best_f1:
                    best_f1, best_thr = f1, t
            print(f"  Best F1={best_f1:.4f} @ thr={best_thr:.2f}")

            pred = (probs > best_thr).astype(int)
            wins = sum(1 for i in range(len(pred)) if pred[i] == 1 and y_test.iloc[i] == 1)
            losses = sum(1 for i in range(len(pred)) if pred[i] == 1 and y_test.iloc[i] == 0)
            total_trades = wins + losses
            win_rate = wins / total_trades if total_trades > 0 else 0
            cm = confusion_matrix(y_test, pred)

            fold_results.append({
                "test_year": test_year,
                "train_years": train_years,
                "f1": best_f1,
                "threshold": best_thr,
                "win_rate": win_rate,
                "trades": total_trades,
                "confusion": cm.tolist(),
            })

            if best_f1 > best_overall_f1:
                best_overall_f1 = best_f1
                best_overall_threshold = best_thr
                self.ai.save()
                print(f"  [SAVE] New best model (F1={best_f1:.4f})")

        print(f"\n{'='*60}")
        print("YEARLY CV SUMMARY")
        print(f"{'='*60}")
        print(f"{'Train':>10} {'Test':>5} {'F1':>6} {'Thr':>5} {'WinRate':>8} {'Trades':>6}")
        for r in fold_results:
            train_str = f"{r['train_years'][0]}-{r['train_years'][1]}"
            print(f"{train_str:>10} {r['test_year']:>5} {r['f1']:>6.4f} {r['threshold']:>5.2f} {r['win_rate']:>7.2%} {r['trades']:>6}")
        print(f"\nBest overall: F1={best_overall_f1:.4f} @ threshold={best_overall_threshold:.2f}")

        now = datetime.now()
        news_warning = f" (NEXT: {self.news_filter.next_news(now)})" if self.news_filter.is_news_event(now) else ""
        print(f"News Filter: {'BLOCKED' if self.news_filter.is_news_event(now) else 'OK'}{news_warning}")

        last_row = gold.iloc[-1]
        sig = self.rules.validate_signal(last_row, prob, gold)
        sl_str = f"SL={sig['sl']:.2f}" if sig['sl'] else ""
        tp_str = f"TP={sig['tp']:.2f}" if sig['tp'] else ""
        reason = f" | {sig['reason']}" if sig['reason'] else ""
        print(f"SIGNAL: {sig['signal']} | Prob={prob:.2%} {sl_str} {tp_str}{reason}")

    def run_evolve(self, generations=10, population=100, sample_size=5000, year_filter=None):
        print("=== EVOLVE MODE (Walkforward Evolutionary CV) ===")
        generators = [
            (2016, 2017, 2018, "Gen1"),
            (2019, 2020, 2021, "Gen2"),
            (2022, 2023, 2024, "Gen3"),
        ]
        n_pop = 10
        pop = []
        results_summary = []

        for gen_i, (train_start, train_end, test_year, gen_name) in enumerate(generators):
            print(f"\n{'='*60}")
            print(f"  GENERATION {gen_name}: train={train_start}-{train_end}, test={test_year}")
            print(f"{'='*60}")

            # data
            gold = self._prepare_data(year_filter=(train_start, train_end))
            X_train = gold[self.feature_list]
            y_train = gold["Target"]
            test_gold = self._prepare_data(year_filter=(test_year, test_year))
            X_test = test_gold[self.feature_list]
            y_test = test_gold["Target"]
            print(f"  Train: {len(X_train)} rows, Test: {len(X_test)} rows")

            # first gen: create 10 bot combos
            if gen_i == 0:
                from core.evolutionary.mutation import mutate_combo
                from core.evolutionary.types import ParameterCombo, XGBoostConfig, TradingConfig
                base = ParameterCombo(
                    xgb=XGBoostConfig(n_estimators=50, max_depth=6, learning_rate=0.05),
                    trading=TradingConfig(buy_threshold=0.72, sell_threshold=0.28, stop_loss_pct=0.005, take_profit_pct=0.011),
                    combo_id=0,
                )
                pop = [mutate_combo(base, i) for i in range(1, n_pop + 1)]

            # train & evaluate each bot
            gen_results = []
            for idx, combo in enumerate(pop):
                est = min(combo.xgb.n_estimators, 100)
                model = XGBClassifier(
                    n_estimators=est,
                    max_depth=combo.xgb.max_depth,
                    learning_rate=combo.xgb.learning_rate,
                    subsample=combo.xgb.subsample,
                    colsample_bytree=combo.xgb.colsample_bytree,
                    min_child_weight=combo.xgb.min_child_weight,
                    gamma=combo.xgb.gamma,
                    scale_pos_weight=combo.xgb.scale_pos_weight,
                    random_state=42, n_jobs=-1,
                )
                model.fit(X_train, y_train, verbose=False)
                probs = model.predict_proba(X_test)[:, 1]

                # eval: F1 at thr=0.50
                pred = (probs > 0.50).astype(int)
                f1 = f1_score(y_test, pred, zero_division=0)
                acc = accuracy_score(y_test, pred)

                wins = sum(1 for i in range(len(pred)) if pred[i]==1 and y_test.iloc[i]==1)
                losses = sum(1 for i in range(len(pred)) if pred[i]==1 and y_test.iloc[i]==0)
                trades = wins + losses
                wr = wins/trades if trades>0 else 0

                score = f1 * 0.5 + wr * 0.3 + (acc-0.5) * 0.2
                gen_results.append((combo, score, f1, wr, trades))
                print(f"  Bot #{combo.combo_id:>3}: F1={f1:.4f}  WR={wr:.2%}  trades={trades}  score={score:.4f}")

            # rank & keep top 2
            gen_results.sort(key=lambda x: x[1], reverse=True)
            top2 = [r[0] for r in gen_results[:2]]
            top2_score = gen_results[0][1]
            print(f"\n  >> Top 2 bots: #{top2[0].combo_id} (score={top2_score:.4f}), #{top2[1].combo_id}")
            results_summary.append((gen_name, top2_score, top2[0].combo_id, top2[1].combo_id))

            # create next gen children from top 2
            from core.evolutionary.mutation import crossover_combos, mutate_combo
            next_id = max((c.combo_id for c in pop), default=0) + 1
            new_pop = list(top2)
            for i in range(2, n_pop):
                child = crossover_combos(top2[0], top2[1], next_id + i)
                child = mutate_combo(child, next_id + i)
                new_pop.append(child)
            pop = new_pop

        # final: train best bot on all years & save
        print(f"\n{'='*60}")
        print("  FINAL: training best bot on 2016-2024")
        gold = self._prepare_data(year_filter=(2016, 2024))
        X_all = gold[self.feature_list]
        y_all = gold["Target"]
        best_combo = pop[0]
        model = XGBClassifier(
            n_estimators=best_combo.xgb.n_estimators,
            max_depth=best_combo.xgb.max_depth,
            learning_rate=best_combo.xgb.learning_rate,
            subsample=best_combo.xgb.subsample,
            colsample_bytree=best_combo.xgb.colsample_bytree,
            min_child_weight=best_combo.xgb.min_child_weight,
            gamma=best_combo.xgb.gamma,
            scale_pos_weight=best_combo.xgb.scale_pos_weight,
            random_state=42, n_jobs=-1,
        )
        model.fit(X_all, y_all, verbose=False)
        self.ai.model = model
        self.ai.save()
        print(f"  Model saved. combo_id={best_combo.combo_id}, n_est={best_combo.xgb.n_estimators}, depth={best_combo.xgb.max_depth}, lr={best_combo.xgb.learning_rate}")

        # update best_params.json
        from core.evolutionary.persistence import save_best_params
        from core.evolutionary.fitness import FitnessScore
        final_combo = best_combo
        best_score = max((r[1] for r in results_summary), default=0)
        final_fitness = FitnessScore(composite_score=best_score, total_profit=best_score * 10)
        save_best_params([(final_combo, final_fitness)], "best_params.json")
        print(f"  best_params.json updated (score={best_score:.4f})")

        print(f"\n{'='*60}")
        print("  WALKFORWARD EVOLUTION SUMMARY")
        for gen_name, score, c1, c2 in results_summary:
            print(f"  {gen_name}: score={score:.4f}, top2={c1},{c2}")
        print(f"{'='*60}")

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
        logging.warning("MT5 not connected. Attempting reconnect...")
        for attempt in range(5):
            try:
                ok = self.executor.initialize()
                if ok:
                    logging.info("MT5 reconnected")
                    return True
            except Exception as ex:
                logging.warning("Reconnect attempt %d failed: %s", attempt + 1, ex)
            time.sleep(3)
        logging.error("MT5 reconnect failed after 5 attempts")
        return False

    def run_live(self, symbol="XAUUSD"):
        print("=== LIVE TRADING MODE ===")
        mt5_ok = self.executor.initialize()
        if not mt5_ok:
            print("[WARNING] MT5 not connected. Running in SIMULATION mode.")

        self.ai.load()

        self._cached_data = None
        gold = self._prepare_live_data()
        X = gold[self.feature_list]

        balance = self._load_state()

        self.dashboard.update(
            mode="LIVE",
            mt5_status="connected" if mt5_ok else "simulation",
        )

        def handle_stop(sig, frame):
            self._save_state(balance)
            sys.exit(0)
        signal.signal(signal.SIGINT, handle_stop)
        signal.signal(signal.SIGTERM, handle_stop)

        dashboard_thread = threading.Thread(target=self.dashboard.start, daemon=True)
        dashboard_thread.start()

        last_train_time = datetime.now()
        signal_count = 0
        entry_sl = None
        entry_tp = None

        logging.info("[STATUS] Bot main loop started. Fetching MT5 data...")

        def _wait_for_next_m15():
            """Дараагийн M15 лаа гарах хүртэл хүлээнэ."""
            now = datetime.now()
            # M15 лаа: 00, 15, 30, 45 минут
            minute = now.minute
            next_min = ((minute // 15) + 1) * 15
            if next_min >= 60:
                next_dt = now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
            else:
                next_dt = now.replace(minute=next_min, second=0, microsecond=0)
            sleep_sec = (next_dt - now).total_seconds()
            if sleep_sec > 0:
                logging.info("[WAIT] Next M15 candle at %s (%ds)", next_dt.strftime("%H:%M"), int(sleep_sec))
                time.sleep(sleep_sec)

        self._cached_mtf = None

        while True:
            try:
                _wait_for_next_m15()
                now = datetime.now()
                logging.info("[CYCLE] start")

                if self.news_filter.is_news_event(now):
                    print("[NEWS] Medeenii uyed arijaag zasvarlaj baina.")
                    time.sleep(60)
                    continue

                gold = self._prepare_live_data()
                if gold is None or len(gold) == 0:
                    logging.warning("[CYCLE] No data available, waiting 15s...")
                    time.sleep(15)
                    continue
                logging.info("[CYCLE] data_ready rows=%d", len(gold))
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
                    time.sleep(5)
                    continue

                if not self.executor.can_trade(symbol):
                    time.sleep(30)
                    continue

                if now - last_train_time > timedelta(days=7):
                    print("[RETRAIN] Weekly retrain with fresh MT5 data...")
                    gold = self._prepare_live_data()
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

                safe_row = last_row.fillna(0) if hasattr(last_row, 'fillna') else last_row
                self.dashboard.update(
                    last_signal=sig["signal"],
                    last_prob=mtf_result["confidence"] if mtf_result["confidence"] > 0 else prob,
                    spread_bps=safe_row.get("SPREAD", 0),
                    ema20=safe_row.get("EMA20", 0),
                    ema50=safe_row.get("EMA50", 0),
                    atr=current_atr if current_atr > 0 else safe_row.get("ATR14", 0),
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

                logging.info("[CYCLE] end prob=%.3f signal=%s", prob, sig["signal"])

            except (ConnectionError, TimeoutError, ValueError, OSError) as e:
                logging.error("[BEAT] Recoverable error: %s - retrying in 30s", e)
                self._save_state(balance)
                time.sleep(30)
            except Exception as e:
                logging.critical("LIVE unrecoverable error: %s", e, exc_info=True)
                self._save_state(balance)
                print(f"[CRITICAL] Bot crashed: {e}")
                self.executor.shutdown()
                raise  # гаднах while True loop барьж restart хийнэ

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
            bot.run_train(n_rows=args.sample)
        if args.mode in ("EVOLVE", "BOTH"):
            bot.run_evolve(generations=args.generations, population=args.population, sample_size=args.sample, year_filter=(2016, 2023))
        if args.mode == "LIVE":
            while True:
                try:
                    bot.run_live(symbol=args.symbol)
                    break
                except SystemExit as e:
                    if e.code == 0:
                        break
                    logging.error("LIVE crash code=%s, restarting in 5s", e.code)
                    time.sleep(5)
                except Exception as e:
                    logging.critical("LIVE error: %s", e)
                    logging.info("Reinitializing executor and retrying in 15s...")
                    try:
                        bot.executor.shutdown()
                    except Exception:
                        pass
                    time.sleep(15)
                    bot.executor.initialize()
    except Exception as e:
        import traceback
        print(f"!!! CRITICAL ERROR: {e}")
        traceback.print_exc()
        raise
