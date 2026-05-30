import ccxt
import pandas as pd
import time
from datetime import datetime, timedelta

# -----------------------------
# 1. Initialize exchange
# -----------------------------
exchange = ccxt.binance({
    'enableRateLimit': True,  # important!
})

symbol = 'BTC/USDT'
timeframe = '5m'

# -----------------------------
# 2. Define time range
# -----------------------------
start_date = "2024-01-01"
since = int(datetime.strptime(start_date, "%Y-%m-%d").timestamp() * 1000)

# Binance limit per request (usually 500–1000)
limit = 1000

all_data = []

# -----------------------------
# 3. Pagination loop
# -----------------------------
while True:
    print(f"Fetching from {datetime.utcfromtimestamp(since/1000)}")

    ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=limit)

    if not ohlcv:
        break

    all_data.extend(ohlcv)

    # Move forward in time
    since = ohlcv[-1][0] + 1

    # Stop condition (optional: up to today)
    if since >= exchange.milliseconds():
        break

    # Respect rate limits
    time.sleep(exchange.rateLimit / 1000)

# -----------------------------
# 4. Convert to DataFrame
# -----------------------------
df = pd.DataFrame(all_data, columns=[
    'timestamp', 'open', 'high', 'low', 'close', 'volume'
])

# Convert timestamp to readable format
df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')

# Remove duplicates (important)
df = df.drop_duplicates(subset='timestamp')

# Sort properly
df = df.sort_values('timestamp')

# -----------------------------
# 5. Save to CSV
# -----------------------------
df.to_csv('btc_5m_ohlcv.csv', index=False)

print("Done. Saved to btc_5m_ohlcv.csv")