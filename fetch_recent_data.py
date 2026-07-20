import MetaTrader5 as mt5
import pandas as pd
import json
from datetime import datetime, timedelta

# 1. Config unshih
with open('config.json', 'r') as f:
    config = json.load(f)

mt5_config = config

# 2. MT5-d holbogdoh
if not mt5.initialize():
    print("MT5-tai holbogdoj chadsangui. MT5 app asaaltai bgaa eseh shalgana uu.")
    quit()

# 3. Dansand nevtreh
authorized = mt5.login(
    login=mt5_config['login'],
    password=mt5_config['password'],
    server=mt5_config['server']
)

if not authorized:
    print(f"Newtreh amjilttgui: {mt5.last_error()}")
    mt5.shutdown()
    quit()

print("Amjilttai nevtrelee!")

# 4. Data tatah
end_date = datetime.now()
start_date = end_date - timedelta(days=60)
symbol = mt5_config.get('symbol', 'XAUUSD')

rates = mt5.copy_rates_range(symbol, mt5.TIMEFRAME_H1, start_date, end_date)

if rates is None or len(rates) == 0:
    print(f"Data tataagdangui ({symbol}): {mt5.last_error()}")
else:
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    output = "recent_test_data.parquet"
    df.to_parquet(output)
    print(f"Amjilttai hadgallaa: {len(df)} mor {symbol} H1 data. Fail: {output}")

mt5.shutdown()
