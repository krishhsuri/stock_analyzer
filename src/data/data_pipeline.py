import os
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import yaml

# Load configuration
def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

config = load_config(os.path.join(os.path.dirname(__file__), '../../config/config.yaml'))

# Fetch historical data for each symbol with error handling
def fetch_price_data(symbol, start, end, adjust_splits=True, adjust_dividends=True):
    try:
        print(f"Attempting to fetch data for {symbol} from {start} to {end}...")
        df = yf.download(symbol, start=start, end=end, auto_adjust=(adjust_splits and adjust_dividends))
        if df is None or not isinstance(df, pd.DataFrame):
            print(f"ERROR: yfinance did not return a DataFrame for {symbol}.")
            return pd.DataFrame()
        if df.empty:
            print(f"WARNING: No data returned for {symbol} in range {start} to {end}.")
            return df
        # Only keep price and volume columns
        columns_to_keep = [col for col in ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume'] if col in df.columns]
        df = df[columns_to_keep]
        return df
    except Exception as e:
        print(f"ERROR fetching data for {symbol}: {e}")
        return pd.DataFrame()

# Main pipeline function
def run_pipeline():
    all_data = {}
    today = datetime.today().date()
    yesterday = today - timedelta(days=1)
    for symbol_entry in config['data']['symbols']:
        symbol = symbol_entry['symbol']
        symbol_folder = symbol.replace('.NS', '').replace('.BSE', '').upper()
        out_dir = os.path.join(os.path.dirname(__file__), f'../../data/{symbol_folder}/unclean_timedata')
        os.makedirs(out_dir, exist_ok=True)
        print(f"\n--- Fetching full range for {symbol} ---")
        df = fetch_price_data(
            symbol,
            datetime.strptime(config['data']['start_date'], '%Y-%m-%d').date(),
            yesterday,
            adjust_splits=config['data']['adjust_for_splits'],
            adjust_dividends=config['data']['adjust_for_dividends']
        )
        all_data[symbol] = df
        # Save to CSV if not empty and is a DataFrame
        if isinstance(df, pd.DataFrame) and not df.empty:
            # Save with date as a column for Excel compatibility
            df_reset = df.reset_index().rename(columns={'index': 'Date'})
            df_reset['Date'] = df_reset['Date'].astype(str)
            out_path = os.path.join(out_dir, f'{symbol}_price.csv')
            df_reset.to_csv(out_path, index=False)
            print(f"Saved {symbol} data to {out_path}")
        else:
            print(f"No data saved for {symbol}.")
    # Placeholder: Fetch and save fundamentals (Screener/TickerTape scraping)
    # ...
    print("Pipeline complete.")

if __name__ == '__main__':
    run_pipeline() 