import time
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class MT5Executor:

    def __init__(self, spread_filter=None, news_filter=None):
        self.mt5 = None
        self.connected = False
        self.spread_filter = spread_filter
        self.news_filter = news_filter

    SYMBOL_MAP = {
        "XAUUSD": "GOLD",
        "XAUEUR": "XAUEUR",
        "XAUJPY": "XAUJPY",
        "XAUCNH": "XAUCNH",
    }

    def initialize(self):
        try:
            import MetaTrader5 as mt5
            self.mt5 = mt5
            self.connected = mt5.initialize()
            if self.connected:
                logger.info("MT5 initialized")
                info = mt5.terminal_info()
                if info:
                    logger.info("Terminal: connected=%s, trade_allowed=%s, name=%s",
                                info.connected, info.trade_allowed, info.name)
                account = mt5.account_info()
                if account:
                    logger.info("Account: login=%d, balance=%.2f, server=%s",
                                account.login, account.balance, account.server)
                symbols = [s.name for s in mt5.symbols_get() or []]
                xau = [s for s in symbols if "XAU" in s.upper() or "GOLD" in s.upper()]
                logger.info("Available XAU/GOLD symbols: %s", xau)
                self._resolve_symbol("XAUUSD")
            else:
                logger.error("MT5 init failed: %s", mt5.last_error())
            return self.connected
        except ImportError:
            logger.warning("MetaTrader5 not installed. Simulation mode.")
            self.connected = False
            return False

    def _resolve_symbol(self, name):
        """Брокерын жинхэнэ симбол нэрийг олно."""
        mapped = self.SYMBOL_MAP.get(name, name)
        if self.mt5.symbol_info(mapped):
            logger.info("Using symbol: %s (mapped from %s)", mapped, name)
            return mapped
        # exact match
        if self.mt5.symbol_info(name):
            return name
        # search
        for s in (self.mt5.symbols_get() or []):
            if s.name.upper() == name.upper():
                return s.name
        logger.warning("Symbol %s not found on broker, using fallback", name)
        return name

    def can_trade(self, symbol: str = "XAUUSD") -> bool:
        sym = self._resolve_symbol(symbol)
        if self.news_filter and self.news_filter.is_news_event(datetime.now()):
            logger.info("Blocked by news filter")
            return False
        if self.spread_filter and self.connected:
            tick = self.mt5.symbol_info_tick(sym) if self.connected else None
            if tick and not self.spread_filter.check_from_tick(tick):
                logger.info("Blocked by spread filter")
                return False
        return True

    def symbol_info(self, symbol):
        if not self.connected:
            return None
        sym = self._resolve_symbol(symbol)
        return self.mt5.symbol_info(sym)

    def positions(self):
        if not self.connected:
            return []
        return self.mt5.positions_get()

    def place_order(self, symbol, order_type, volume, price, sl, tp, comment=""):
        sym = self._resolve_symbol(symbol)
        if not self.can_trade(sym):
            logger.info("[BLOCKED] Trade skipped by filter")
            return None
        if not self.connected:
            logger.info("[SIM] %s %s vol=%.2f price=%.5f sl=%.5f tp=%.5f",
                        sym, "BUY" if order_type == 0 else "SELL", volume, price, sl, tp)
            return 1000000 + int(time.time() % 100000)
        request = {
            "action": self.mt5.TRADE_ACTION_DEAL,
            "symbol": sym,
            "volume": volume,
            "type": order_type,
            "price": price,
            "sl": sl,
            "tp": tp,
            "deviation": 10,
            "magic": 202407,
            "comment": comment,
            "type_time": self.mt5.ORDER_TIME_GTC,
            "type_filling": self.mt5.ORDER_FILLING_IOC,
        }
        result = self.mt5.order_send(request)
        if result.retcode != self.mt5.TRADE_RETCODE_DONE:
            logger.error("Order failed: %s (retcode=%d)", result.comment, result.retcode)
            return None
        logger.info("Order placed: ticket=%d", result.order)
        return result.order

    def modify_position(self, ticket, sl, tp):
        if not self.connected:
            logger.info("[SIM] Modify ticket=%d sl=%.5f tp=%.5f", ticket, sl, tp)
            return True
        request = {
            "action": self.mt5.TRADE_ACTION_SLTP,
            "position": ticket,
            "sl": sl,
            "tp": tp,
        }
        result = self.mt5.order_send(request)
        if result.retcode != self.mt5.TRADE_RETCODE_DONE:
            logger.error("Modify failed: %s (retcode=%d)", result.comment, result.retcode)
            return False
        logger.info("Position %d modified: sl=%.5f tp=%.5f", ticket, sl, tp)
        return True

    def close_position(self, ticket):
        if not self.connected:
            logger.info("[SIM] Close ticket=%d", ticket)
            return True
        positions = self.mt5.positions_get(ticket=ticket)
        if not positions:
            return False
        pos = positions[0]
        order_type = self.mt5.ORDER_TYPE_SELL if pos.type == 0 else self.mt5.ORDER_TYPE_BUY
        price = self.mt5.symbol_info_tick(pos.symbol).bid if order_type == self.mt5.ORDER_TYPE_SELL else self.mt5.symbol_info_tick(pos.symbol).ask
        request = {
            "action": self.mt5.TRADE_ACTION_DEAL,
            "symbol": pos.symbol,
            "volume": pos.volume,
            "type": order_type,
            "position": ticket,
            "price": price,
            "deviation": 10,
            "magic": 202407,
            "comment": "close",
            "type_time": self.mt5.ORDER_TIME_GTC,
            "type_filling": self.mt5.ORDER_FILLING_IOC,
        }
        result = self.mt5.order_send(request)
        return result.retcode == self.mt5.TRADE_RETCODE_DONE

    def shutdown(self):
        if self.connected and self.mt5:
            self.mt5.shutdown()
            self.connected = False
