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

    def initialize(self):
        try:
            import MetaTrader5 as mt5
            self.mt5 = mt5
            self.connected = mt5.initialize()
            if self.connected:
                logger.info("MT5 initialized")
            else:
                logger.error("MT5 init failed: %s", mt5.last_error())
            return self.connected
        except ImportError:
            logger.warning("MetaTrader5 not installed. Simulation mode.")
            self.connected = False
            return False

    def can_trade(self, symbol: str = "XAUUSD") -> bool:
        if self.news_filter and self.news_filter.is_news_event(datetime.now()):
            logger.info("Blocked by news filter")
            return False
        if self.spread_filter and self.connected:
            tick = self.mt5.symbol_info_tick(symbol) if self.connected else None
            if tick and not self.spread_filter.check_from_tick(tick):
                logger.info("Blocked by spread filter")
                return False
        return True

    def symbol_info(self, symbol):
        if not self.connected:
            return None
        return self.mt5.symbol_info(symbol)

    def positions(self):
        if not self.connected:
            return []
        return self.mt5.positions_get()

    def place_order(self, symbol, order_type, volume, price, sl, tp, comment=""):
        if not self.can_trade(symbol):
            logger.info("[BLOCKED] Trade skipped by filter")
            return None
        if not self.connected:
            logger.info("[SIM] %s %s vol=%.2f price=%.5f sl=%.5f tp=%.5f",
                        symbol, "BUY" if order_type == 0 else "SELL", volume, price, sl, tp)
            return 1000000 + int(time.time() % 100000)
        request = {
            "action": self.mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
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
