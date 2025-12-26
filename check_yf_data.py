import yfinance as yf
import json

tickers = ['BTC-USD', 'ETH-USD', 'DOGE-USD']
results = {}

for t in tickers:
    tick = yf.Ticker(t)
    try:
        info = tick.info
        # Filter for relevant keys
        relevant_keys = [
            'marketCap', 'circulatingSupply', 'maxSupply', 'totalRevenue', 
            'revenuePerShare', 'priceToSalesTrailing12Months', 'volume', 
            'averageVolume', 'previousClose', 'open', 'dayLow', 'dayHigh'
        ]
        results[t] = {k: info.get(k, 'N/A') for k in relevant_keys}
    except Exception as e:
        results[t] = str(e)

print(json.dumps(results, indent=2))
