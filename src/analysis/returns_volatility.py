import os
import pandas as pd
import yaml

def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

CONFIG_PATH = os.path.join(os.path.dirname(__file__), '../../config/config.yaml')
config = load_config(CONFIG_PATH)

def daily_returns(series, days=1):
    return series.pct_change(periods=days)

def rolling_volatility(series, window=20):
    return series.pct_change().rolling(window=window).std()

def generate_returns_volatility():
    for symbol_entry in config['data']['symbols']:
        symbol = symbol_entry['symbol']
        folder = symbol.replace('.NS', '').replace('.BSE', '').upper()
        folder_path = os.path.join(os.path.dirname(__file__), f'../../data/{folder}/unclean_timedata')
        os.makedirs(folder_path, exist_ok=True)
        price_path = os.path.join(os.path.dirname(__file__), f'../../data/{folder}/unclean_timedata/{symbol}_price.csv')
        out_path = os.path.join(folder_path, f'{symbol}_returns_volatility.csv')
        if not os.path.exists(price_path):
            print(f"Price file not found for {symbol}, skipping.")
            continue
        df = pd.read_csv(price_path)
        # Ensure numeric column
        if 'Close' in df.columns:
            df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
        if 'Close' not in df.columns:
            print(f"No 'Close' column for {symbol}, skipping.")
            continue
        df['Daily_Return'] = daily_returns(df['Close'], 1)
        df['Return_5d'] = daily_returns(df['Close'], 5)
        df['Return_10d'] = daily_returns(df['Close'], 10)
        df['Rolling_Volatility_20'] = rolling_volatility(df['Close'], 20)
        cols = ['Date'] if 'Date' in df.columns else []
        cols += ['Daily_Return', 'Return_5d', 'Return_10d', 'Rolling_Volatility_20']
        df[cols].to_csv(out_path, index=False)
        print(f"Saved returns and volatility for {symbol} to {out_path}")

if __name__ == '__main__':
    generate_returns_volatility() 