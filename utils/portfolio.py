import pandas as pd
import yfinance as yf

def load_portfolio(path):
    df = pd.read_csv(path)
    return df

def fetch_prices(tickers):
    data = yf.download(tickers, period='1d', group_by='ticker', auto_adjust=False)
    print(data)
    print(data.columns.get_level_values(0))
    price_types = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']

    # Create the MultiIndex
    columns = pd.MultiIndex.from_product([price_types, tickers], names=['Price', 'Ticker'])
    # Reassign the columns to your DataFrame
    data.columns = columns
    print(data.head())
    close_prices = data['Close']
    latest_prices = close_prices.iloc[-1]
    return latest_prices

def calculate_value(df, prices):
    df['price'] = df['ticker'].map(prices.to_dict())
    df['current_value'] = df['shares'] * df['price']
    df['gain_loss'] = df['current_value'] - (df['shares'] * df['cost_basis'])
    return df
