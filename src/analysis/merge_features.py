import os
import pandas as pd
import yaml
from functools import reduce

def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

CONFIG_PATH = os.path.join(os.path.dirname(__file__), '../../config/config.yaml')
config = load_config(CONFIG_PATH)

FEATURE_FILES = [
    '{symbol}_price.csv',
    '{symbol}_technical_indicators.csv',
    '{symbol}_returns_volatility.csv',
    '{symbol}_volume_indicators.csv'
]

def merge_features_for_stock(symbol, folder):
    unclean_dir = os.path.join(os.path.dirname(__file__), f'../../data/{folder}/unclean_timedata')
    final_dir = os.path.join(os.path.dirname(__file__), f'../../data/{folder}/final_timedata')
    os.makedirs(final_dir, exist_ok=True)
    # Read all feature files
    dfs = []
    for fname in FEATURE_FILES:
        fpath = os.path.join(unclean_dir, fname.format(symbol=symbol))
        if not os.path.exists(fpath):
            print(f"File not found: {fpath}, skipping {symbol}.")
            return
        df = pd.read_csv(fpath)
        dfs.append(df)
    # Merge on 'Date'
    df_merged = reduce(lambda left, right: pd.merge(left, right, on='Date', how='inner'), dfs)
    # Drop rows with any NaN
    df_clean = df_merged.dropna()
    out_path = os.path.join(final_dir, f'{symbol}_features_clean.csv')
    df_clean.to_csv(out_path, index=False)
    print(f"Saved merged, cleaned features for {symbol} to {out_path}")

def merge_all_stocks():
    for symbol_entry in config['data']['symbols']:
        symbol = symbol_entry['symbol']
        folder = symbol.replace('.NS', '').replace('.BSE', '').upper()
        merge_features_for_stock(symbol, folder)

if __name__ == '__main__':
    merge_all_stocks() 