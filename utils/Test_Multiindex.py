import pandas as pd
import numpy as np

# Let's simulate stock prices for 2 days and 3 tickers
arrays = [
    ['Close', 'Close', 'Close', 'Open', 'Open', 'Open'],
    ['AAPL', 'MSFT', 'GOOG', 'AAPL', 'MSFT', 'GOOG']
]

# Create a MultiIndex for columns
multi_cols = pd.MultiIndex.from_arrays(arrays, names=["PriceType", "Ticker"])

# Create some sample price data
data = [
    [170.5, 310.2, 2800.1, 169.8, 309.9, 2795.5],
    [171.2, 312.0, 2812.4, 170.0, 311.0, 2802.3],
]

# Create the DataFrame
df = pd.DataFrame(data, index=pd.date_range("2025-04-10", periods=2), columns=multi_cols)

print("📊 MultiIndex DataFrame:")
print(df)

# Access all 'Close' prices
print("\n🔍 Close prices only:")
print(df['Close'])

# Access only MSFT's 'Open' price
print("\n🔎 MSFT Open prices:")
print(df['Open']['MSFT'])

# Cross-section: all data for one ticker across price types
print("\n📈 All prices for AAPL (using .xs):")
print(df.xs('AAPL', axis=1, level='Ticker'))
