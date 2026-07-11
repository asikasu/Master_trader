import pandas as pd
import numpy as np
from core.trade_journal import TradeJournal # Import TradeJournal

class Backtester:

    def __init__(self, data: pd.DataFrame):
        self.df = data.copy()
        self.results = None
        self.trade_journal = TradeJournal() # Initialize TradeJournal

    def run_simulation(
        self,
        buy_threshold=0.80,
        sell_threshold=0.20,
        take_profit_ratio=0.005,  # 0.5% of price
        stop_loss_ratio=0.002,   # 0.2% of price
        spread_bps=1.0,          # 1.0 basis points (0.01%)
        commission_per_trade=0.0001, # e.g., $0.0001 per unit traded
        slippage_bps=0.5         # 0.5 basis points (0.005%)
    ):

        df = self.df.copy()
        df.index = pd.to_datetime(df.index) # Ensure datetime index

        # Convert bps to decimal
        spread_decimal = spread_bps / 10000
        slippage_decimal = slippage_bps / 10000

        # Initialize columns for trade tracking
        df["Signal"] = 0
        df["Position"] = 0 # 1 for Long, -1 for Short, 0 for Flat

        # Variables to track open trades
        current_position = 0
        open_trade = None # Stores details of an open trade

        trades_list = [] # To store details of completed trades

        for i in range(len(df)):
            current_bar = df.iloc[i]
            # Ensure we have a next bar for lookahead, otherwise skip trade logic for the last bar
            if i + 1 >= len(df):
                if open_trade: # Close any open trades at the end of data
                    # Simulate closing at current bar's close price
                    close_price = current_bar["CLOSE"]
                    profit = (close_price - open_trade["open_price"]) * open_trade["side"] - open_trade["commission"]
                    open_trade["close_time"] = current_bar.name
                    open_trade["close_price"] = close_price
                    open_trade["profit"] = profit
                    trades_list.append(open_trade)
                    self.trade_journal.save_trade(
                        open_time=open_trade["open_time"],
                        symbol="SYMBOL", # Placeholder
                        side=open_trade["side"],
                        open_price=open_trade["open_price"],
                        close_price=open_trade["close_price"],
                        profit=open_trade["profit"],
                        stop_loss=open_trade["stop_loss"],
                        take_profit=open_trade["take_profit"],
                        commission=open_trade["commission"],
                        spread=open_trade["spread"],
                        slippage=open_trade["slippage"],
                        risk_amount=open_trade["risk_amount"] # This will need to be calculated properly
                    )
                break

            next_bar = df.iloc[i + 1]

            # Determine signal
            if current_bar["Probability"] >= buy_threshold:
                df.loc[current_bar.name, "Signal"] = 1 # Buy signal
            elif current_bar["Probability"] <= sell_threshold:
                df.loc[current_bar.name, "Signal"] = -1 # Sell signal

            signal = df.loc[current_bar.name, "Signal"]

            # Manage open trades
            if open_trade:
                # Check for TP/SL hit in the current bar (High/Low)
                high = current_bar["HIGH"]
                low = current_bar["LOW"]
                close = current_bar["CLOSE"]

                tp_hit = False
                sl_hit = False

                if open_trade["side"] == 1: # Long position
                    if high >= open_trade["take_profit"]:
                        tp_hit = True
                    if low <= open_trade["stop_loss"]:
                        sl_hit = True
                else: # Short position
                    if low <= open_trade["take_profit"]: # TP for short is lower price
                        tp_hit = True
                    if high >= open_trade["stop_loss"]: # SL for short is higher price
                        sl_hit = True

                # Determine exit price and reason
                exit_price = None
                exit_reason = ""

                if sl_hit and tp_hit: # Both hit, prioritize SL (worst case)
                    exit_reason = "SL Hit"
                    exit_price = open_trade["stop_loss"]
                elif sl_hit:
                    exit_reason = "SL Hit"
                    exit_price = open_trade["stop_loss"]
                elif tp_hit:
                    exit_reason = "TP Hit"
                    exit_price = open_trade["take_profit"]
                elif (signal == -open_trade["side"]): # Reverse signal, close current trade
                    exit_reason = "Reverse Signal"
                    # Exit at next bar's open (adjusted for spread/slippage)
                    exit_price = next_bar["OPEN"] * (1 - spread_decimal - slippage_decimal) if open_trade["side"] == 1 else \
                                 next_bar["OPEN"] * (1 + spread_decimal + slippage_decimal)
                elif i == len(df) - 2: # Close at next bar's open if it's the second to last bar (to ensure next_bar exists)
                    exit_reason = "End of Data"
                    exit_price = next_bar["OPEN"] * (1 - spread_decimal - slippage_decimal) if open_trade["side"] == 1 else \
                                 next_bar["OPEN"] * (1 + spread_decimal + slippage_decimal)


                if exit_price is not None:
                    profit = (exit_price - open_trade["open_price"]) * open_trade["side"] - open_trade["commission"]
                    open_trade["close_time"] = current_bar.name
                    open_trade["close_price"] = exit_price
                    open_trade["profit"] = profit
                    open_trade["exit_reason"] = exit_reason
                    trades_list.append(open_trade)
                    self.trade_journal.save_trade(
                        open_time=open_trade["open_time"],
                        symbol="SYMBOL", # Placeholder
                        side=open_trade["side"],
                        open_price=open_trade["open_price"],
                        close_price=open_trade["close_price"],
                        profit=open_trade["profit"],
                        stop_loss=open_trade["stop_loss"],
                        take_profit=open_trade["take_profit"],
                        commission=open_trade["commission"],
                        spread=open_trade["spread"],
                        slippage=open_trade["slippage"],
                        risk_amount=open_trade["risk_amount"] # This will need to be calculated properly
                    )
                    open_trade = None # Close the trade

            # Open new trade if flat and signal exists
            if not open_trade and signal != 0:
                entry_time = next_bar.name # Entry on the open of the next bar
                entry_price = next_bar["OPEN"]

                # Apply spread and slippage
                if signal == 1: # Long
                    entry_price = entry_price * (1 + spread_decimal + slippage_decimal)
                    stop_loss = entry_price * (1 - stop_loss_ratio)
                    take_profit = entry_price * (1 + take_profit_ratio)
                else: # Short
                    entry_price = entry_price * (1 - spread_decimal - slippage_decimal)
                    stop_loss = entry_price * (1 + stop_loss_ratio)
                    take_profit = entry_price * (1 - take_profit_ratio)

                commission = commission_per_trade # Simple fixed commission per trade

                open_trade = {
                    "open_time": entry_time,
                    "side": signal,
                    "open_price": entry_price,
                    "stop_loss": stop_loss,
                    "take_profit": take_profit,
                    "commission": commission,
                    "spread": spread_decimal,
                    "slippage": slippage_decimal,
                    "risk_amount": abs(entry_price - stop_loss) # Simple risk per unit
                }

        # After the loop, ensure all trades are recorded in the trade_journal
        self.df = df
        self.results = pd.DataFrame(trades_list)
        if not self.results.empty:
            self.results["cum_profit"] = self.results["profit"].cumsum()
            self.results["equity"] = 1000 + self.results["cum_profit"] # Starting equity of 1000 for equity curve
            self.equity_curve = self.results.set_index("close_time")["equity"]
        else:
            self.equity_curve = pd.Series([1000], index=[df.index[-1]]) # If no trades, just flat equity

        print("✅ Backtest finished")
        return self.equity_curve

    def calculate_metrics(self):
        # Load trades from the journal
        try:
            trades_df = pd.read_csv(self.trade_journal.file, parse_dates=["open_time", "close_time"])
        except FileNotFoundError:
            return {
                "Trades": 0, "Win Rate": 0, "Profit Factor": 0, "Net Profit": 0,
                "Gross Profit": 0, "Gross Loss": 0, "Average Win": 0,
                "Average Loss": 0, "Expectancy": 0, "Sharpe Ratio": 0,
                "Sortino Ratio": 0, "Max Drawdown": 0, "Recovery Factor": 0,
                "Profit per Month": 0, "Profit per Year": 0,
                "Longest Win Streak": 0, "Longest Lose Streak": 0,
                "Average Holding Time": 0, "Average RR": 0,
                "Commission": 0, "Spread": 0, "Slippage": 0,
                "TP Hit Count": 0, "SL Hit Count": 0,
                "CAGR": 0, "Calmar Ratio": 0
            }

        if trades_df.empty:
            return {
                "Trades": 0, "Win Rate": 0, "Profit Factor": 0, "Net Profit": 0,
                "Gross Profit": 0, "Gross Loss": 0, "Average Win": 0,
                "Average Loss": 0, "Expectancy": 0, "Sharpe Ratio": 0,
                "Sortino Ratio": 0, "Max Drawdown": 0, "Recovery Factor": 0,
                "Profit per Month": 0, "Profit per Year": 0,
                "Longest Win Streak": 0, "Longest Lose Streak": 0,
                "Average Holding Time": 0, "Average RR": 0,
                "Commission": 0, "Spread": 0, "Slippage": 0,
                "TP Hit Count": 0, "SL Hit Count": 0,
                "CAGR": 0, "Calmar Ratio": 0
            }

        # Basic Trade Statistics
        trade_count = len(trades_df)
        net_profit = trades_df["profit"].sum()

        winning_trades = trades_df[trades_df["profit"] > 0]
        losing_trades = trades_df[trades_df["profit"] < 0]

        gross_profit = winning_trades["profit"].sum()
        gross_loss = losing_trades["profit"].sum()
        win_rate = len(winning_trades) / trade_count if trade_count > 0 else 0

        average_win = winning_trades["profit"].mean() if not winning_trades.empty else 0
        average_loss = losing_trades["profit"].mean() if not losing_trades.empty else 0

        profit_factor = abs(gross_profit / gross_loss) if gross_loss != 0 else float('inf')
        expectancy = trades_df["profit"].mean()

        # Commission, Spread, Slippage (Averaged or Summed)
        total_commission = trades_df["commission"].sum() * 2 # Entry and Exit
        avg_spread = trades_df["spread"].mean()
        avg_slippage = trades_df["slippage"].mean()

        # Holding Time
        trades_df["holding_time"] = (trades_df["close_time"] - trades_df["open_time"]).dt.total_seconds() / (60 * 60 * 24) # in days
        average_holding_time = trades_df["holding_time"].mean() if not trades_df.empty else 0

        # Equity Curve and Drawdown
        initial_capital = 1000 # Assuming initial capital
        equity_curve_values = [initial_capital] + (initial_capital + trades_df["profit"].cumsum()).tolist()
        equity_curve_series = pd.Series(equity_curve_values)

        peak = equity_curve_series.cummax()
        drawdown = (equity_curve_series - peak) / peak
        max_drawdown = drawdown.min()

        # Recovery Factor
        recovery_factor = net_profit / abs(max_drawdown * initial_capital) if max_drawdown != 0 else float('inf')

        # CAGR
        start_date = trades_df["open_time"].min()
        end_date = trades_df["close_time"].max()
        num_years = (end_date - start_date).days / 365.25 if start_date and end_date else 0.001 # Avoid division by zero
        total_return = (equity_curve_series.iloc[-1] / initial_capital) - 1
        cagr = ((1 + total_return)**(1/num_years)) - 1 if total_return > -1 and num_years > 0 else 0

        # Calmar Ratio
        calmar_ratio = cagr / abs(max_drawdown) if max_drawdown != 0 else float('inf')

        # Profit per Month / Year
        trades_df["year_month"] = trades_df["close_time"].dt.to_period('M')
        profit_per_month = trades_df.groupby("year_month")["profit"].sum().mean() if not trades_df.empty else 0
        profit_per_year = profit_per_month * 12

        # Win/Loss Streaks
        if not trades_df.empty:
            streaks = (trades_df["profit"] > 0).astype(int).groupby((trades_df["profit"] > 0).diff().fillna(False).cumsum()).cumsum()
            longest_win_streak = streaks.max() if not streaks.empty else 0

            streaks_loss = (trades_df["profit"] < 0).astype(int).groupby((trades_df["profit"] < 0).diff().fillna(False).cumsum()).cumsum()
            longest_lose_streak = streaks_loss.max() if not streaks_loss.empty else 0
        else:
            longest_win_streak = 0
            longest_lose_streak = 0


        # Average RR (Risk-Reward)
        average_rr = abs(average_win / average_loss) if average_loss != 0 else float('inf')

        # Sharpe Ratio (Requires daily returns and risk-free rate) - Simplistic for now
        # For a more accurate Sharpe, you'd need to resample equity_curve to daily returns
        if len(equity_curve_series) > 1:
            returns = equity_curve_series.pct_change().dropna()
            sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() != 0 else 0 # Annualized
        else:
            sharpe_ratio = 0

        # Sortino Ratio (Requires downside deviation) - Simplistic for now
        if len(equity_curve_series) > 1:
            returns = equity_curve_series.pct_change().dropna()
            downside_returns = returns[returns < 0]
            downside_std = downside_returns.std()
            sortino_ratio = (returns.mean() / downside_std) * np.sqrt(252) if downside_std != 0 else 0 # Annualized
        else:
            sortino_ratio = 0

        # TP/SL Hit Counts (Requires \'exit_reason\' in trade_journal)
        tp_hit_count = trades_df[trades_df["exit_reason"] == "TP Hit"].shape[0] if "exit_reason" in trades_df.columns else 0
        sl_hit_count = trades_df[trades_df["exit_reason"] == "SL Hit"].shape[0] if "exit_reason" in trades_df.columns else 0


        return {
            "Trades": trade_count,
            "Win Rate": win_rate,
            "Profit Factor": profit_factor,
            "Net Profit": net_profit,
            "Gross Profit": gross_profit,
            "Gross Loss": gross_loss,
            "Average Win": average_win,
            "Average Loss": average_loss,
            "Expectancy": expectancy,
            "Sharpe Ratio": sharpe_ratio,
            "Sortino Ratio": sortino_ratio,
            "Max Drawdown": max_drawdown,
            "Recovery Factor": recovery_factor,
            "Profit per Month": profit_per_month,
            "Profit per Year": profit_per_year,
            "Longest Win Streak": longest_win_streak,
            "Longest Lose Streak": longest_lose_streak,
            "Average Holding Time": average_holding_time,
            "Average RR": average_rr,
            "Commission": total_commission,
            "Spread": avg_spread,
            "Slippage": avg_slippage,
            "TP Hit Count": tp_hit_count,
            "SL Hit Count": sl_hit_count,
            "CAGR": cagr,
            "Calmar Ratio": calmar_ratio
        }

        self.results = df["Equity"]

        print("✅ Backtest finished")

        return self.results

    def calculate_metrics(self):

        trades = self.df[
            self.df["Signal"] != 0
        ].copy()

        trade_count = len(trades)

        if trade_count == 0:

            return {
                "Trades": 0,
                "Win Rate": 0,
                "Profit Factor": 0,
                "Total Return": 0,
                "Max Drawdown": 0
            }

        # =====================
        # WIN RATE
        # =====================

        win_rate = (
            trades["Returns"] > 0
        ).mean()

        # =====================
        # PROFIT FACTOR
        # =====================

        profits = (
            trades[
                trades["Returns"] > 0
            ]["Returns"]
            .sum()
        )

        losses = abs(
            trades[
                trades["Returns"] < 0
            ]["Returns"]
            .sum()
        )

        profit_factor = (
            profits / losses
            if losses > 0
            else float("inf")
        )

        # =====================
        # TOTAL RETURN
        # =====================

        total_return = (
            self.df["Equity"]
            .iloc[-1]
            - 1
        )

        # =====================
        # MAX DRAWDOWN
        # =====================

        peak = (
            self.df["Equity"]
            .cummax()
        )

        drawdown = (
            self.df["Equity"]
            - peak
        ) / peak

        max_drawdown = drawdown.min()

        # =====================
        # AVG WIN / LOSS
        # =====================

        avg_win = (
            trades[
                trades["Returns"] > 0
            ]["Returns"]
            .mean()
        )

        avg_loss = (
            trades[
                trades["Returns"] < 0
            ]["Returns"]
            .mean()
        )

        return {
            "Trades": trade_count,
            "Win Rate": win_rate,
            "Profit Factor": profit_factor,
            "Total Return": total_return,
            "Max Drawdown": max_drawdown,
            "Average Win": avg_win,
            "Average Loss": avg_loss
        }