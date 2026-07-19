import os
import time
import threading
import json
from datetime import datetime
from pathlib import Path

try:
    from rich.console import Console
    from rich.table import Table
    from rich.layout import Layout
    from rich.panel import Panel
    from rich.live import Live
    from rich import box
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

console = Console() if RICH_AVAILABLE else None

BEST_PARAMS_FILE = "best_params.json"
STATE_FILE = "evolution_state.json"
TRADE_LOG = "logs/trades.csv"


class Dashboard:
    """CLI Dashboard showing real-time bot status."""

    def __init__(self, update_interval: float = 2.0):
        self.update_interval = update_interval
        self._running = False
        self._thread = None
        self._data = {
            "mode": "IDLE",
            "generation": 0,
            "best_score": 0.0,
            "best_sharpe": 0.0,
            "best_profit": 0.0,
            "total_trades": 0,
            "wins": 0,
            "losses": 0,
            "balance": 1000.0,
            "last_signal": "WAIT",
            "last_prob": 0.0,
            "healing": False,
            "mt5_status": "disconnected",
            "news_block": False,
            "spread_bps": 0.0,
            "ema20": 0.0,
            "ema50": 0.0,
            "atr": 0.0,
        }

    def update(self, **kwargs):
        self._data.update(kwargs)

    def _read_best_params(self):
        path = BEST_PARAMS_FILE
        if os.path.exists(path):
            try:
                with open(path) as f:
                    data = json.load(f)
                if data:
                    entry = data[0]
                    self._data["best_score"] = entry.get("fitness", {}).get("composite_score", 0)
                    self._data["best_sharpe"] = entry.get("fitness", {}).get("sharpe_ratio", 0)
                    self._data["best_profit"] = entry.get("fitness", {}).get("total_profit", 0)
            except Exception:
                pass

    def _read_state(self):
        path = STATE_FILE
        if os.path.exists(path):
            try:
                with open(path) as f:
                    data = json.load(f)
                self._data["generation"] = data.get("generation", 0)
            except Exception:
                pass

    def _read_trades(self):
        path = TRADE_LOG
        if os.path.exists(path):
            import pandas as pd
            try:
                df = pd.read_csv(path)
                total = len(df)
                wins = len(df[df["profit"] > 0]) if "profit" in df else 0
                self._data["total_trades"] = total
                self._data["wins"] = wins
                self._data["losses"] = total - wins
            except Exception:
                pass

    def _render_rich(self):
        layout = Layout()
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="body"),
            Layout(name="footer", size=3),
        )

        header = Panel(
            f"[bold cyan]Master Trader Dashboard[/bold cyan]   "
            f"[yellow]{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}[/yellow]   "
            f"Mode: [bold]{self._data['mode']}[/bold]",
            box=box.ROUNDED,
        )
        layout["header"].update(header)

        stats = Table.grid(padding=1)
        stats.add_column()
        stats.add_column()

        g = self._data
        stats.add_row(
            f"Generation: [bold]{g['generation']}[/bold]",
            f"Best Score: [bold green]{g['best_score']:.4f}[/bold green]",
        )
        stats.add_row(
            f"Sharpe: [bold]{g['best_sharpe']:.2f}[/bold]",
            f"Profit: [bold]{g['best_profit']:.2f}[/bold]",
        )
        stats.add_row(
            f"Total Trades: [bold]{g['total_trades']}[/bold]",
            f"W/L: [green]{g['wins']}[/green]/[red]{g['losses']}[/red]",
        )
        stats.add_row(
            f"Balance: [bold yellow]${g['balance']:.2f}[/bold yellow]",
            f"Healing: {'[red]ACTIVE[/red]' if g['healing'] else '[green]OK[/green]'}",
        )

        left_panel = Panel(stats, title="[bold]Stats[/bold]", box=box.ROUNDED)

        signal = Table.grid(padding=1)
        signal.add_column()
        signal.add_column()
        signal.add_row("Last Signal:", f"[bold cyan]{g['last_signal']}[/bold cyan]")
        signal.add_row("Probability:", f"[bold]{g['last_prob']:.2%}[/bold]")
        signal.add_row("MT5:", f"{'[green]connected[/green]' if g['mt5_status'] == 'connected' else '[red]disconnected[/red]'}")
        signal.add_row("News Block:", f"{'[red]YES[/red]' if g['news_block'] else '[green]NO[/green]'}")
        signal.add_row("Spread (bps):", f"{g['spread_bps']:.1f}")
        signal.add_row("EMA20/50:", f"{g['ema20']:.2f} / {g['ema50']:.2f}")
        signal.add_row("ATR14:", f"{g['atr']:.2f}")

        right_panel = Panel(signal, title="[bold]Signal[/bold]", box=box.ROUNDED)

        body = Layout()
        body.split_row(left_panel, right_panel)
        layout["body"].update(body)

        footer = Panel(
            "[dim]Press Ctrl+C to stop[/dim]",
            box=box.ROUNDED,
        )
        layout["footer"].update(footer)

        return layout

    def _render_plain(self):
        g = self._data
        lines = [
            "=" * 60,
            f"MASTER TRADER DASHBOARD  |  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  |  Mode: {g['mode']}",
            "=" * 60,
            f"  Generation: {g['generation']}    Best Score: {g['best_score']:.4f}    Sharpe: {g['best_sharpe']:.2f}    Profit: {g['best_profit']:.2f}",
            f"  Trades: {g['total_trades']}    W/L: {g['wins']}/{g['losses']}    Balance: ${g['balance']:.2f}    Healing: {'YES' if g['healing'] else 'NO'}",
            f"  Signal: {g['last_signal']}    Prob: {g['last_prob']:.2%}    MT5: {g['mt5_status']}",
            f"  News: {'BLOCKED' if g['news_block'] else 'OK'}  Spread: {g['spread_bps']:.1f} bps  EMA20/50: {g['ema20']:.1f}/{g['ema50']:.1f}  ATR: {g['atr']:.2f}",
        ]
        return "\n".join(lines)

    def _loop(self):
        while self._running:
            self._read_best_params()
            self._read_state()
            self._read_trades()
            time.sleep(self.update_interval)

    def _render_loop(self):
        if RICH_AVAILABLE:
            with Live(self._render_rich(), refresh_per_second=1.0 / self.update_interval, console=console):
                while self._running:
                    time.sleep(self.update_interval)
        else:
            while self._running:
                os.system("cls" if os.name == "nt" else "clear")
                print(self._render_plain())
                time.sleep(self.update_interval)

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        self._render_loop()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=1)
