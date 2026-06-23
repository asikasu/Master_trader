import MetaTrader5 as mt5

class MT5Executor:

    def initialize(self):
        return mt5.initialize()

    def positions(self):
        return mt5.positions_get()

    def symbol_info(self, symbol):
        return mt5.symbol_info(symbol)