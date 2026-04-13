from binance.client import Client
import pandas as pd

client = Client()

klines = client.get_historical_klines(
    "ETHUSDT",
    Client.KLINE_INTERVAL_1DAY,
    "3 years ago UTC"
)

df = pd.DataFrame(klines, columns=[
    "time", "open", "high", "low", "close", "volume",
    "close_time", "qav", "num_trades",
    "taker_base_vol", "taker_quote_vol", "ignore"
])

df = df[["time", "open", "high", "low", "close", "volume"]]
df["time"] = pd.to_datetime(df["time"], unit="ms")

df.to_csv("eth_binance_3y.csv", index=False)

print("Saved Binance ETH data")