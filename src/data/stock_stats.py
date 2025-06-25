import os
import pandas as pd
import yaml
from scipy.stats import skew, kurtosis

# --- Utility to load config ---
def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

CONFIG_PATH = os.path.join(os.path.dirname(__file__), '../../config/config.yaml')
config = load_config(CONFIG_PATH)

# --- Stats to compute ---
def compute_stats(df, columns):
    stats = {}
    for col in columns:
        s = pd.Series(pd.to_numeric(df[col], errors='coerce'))
        stats[f'{col}_count'] = s.count()
        stats[f'{col}_mean'] = s.mean()
        stats[f'{col}_std'] = s.std()
        stats[f'{col}_min'] = s.min()
        stats[f'{col}_q25'] = s.quantile(0.25)
        stats[f'{col}_median'] = s.median()
        stats[f'{col}_q75'] = s.quantile(0.75)
        stats[f'{col}_max'] = s.max()
        stats[f'{col}_skew'] = skew(s.dropna()) if s.count() > 2 else float('nan')
        stats[f'{col}_kurtosis'] = kurtosis(s.dropna()) if s.count() > 3 else float('nan')
        stats[f'{col}_missing'] = s.isna().sum()
    return stats

# --- Main script ---
def generate_all_stock_stats():
    for symbol_entry in config['data']['symbols']:
        symbol = symbol_entry['symbol']
        folder = symbol.replace('.NS', '').replace('.BSE', '').upper()
        price_path = os.path.join(os.path.dirname(__file__), f'../../data/{folder}/unclean_timedata/{symbol}_price.csv')
        stats_dir = os.path.join(os.path.dirname(__file__), f'../../data/{folder}/stats')
        os.makedirs(stats_dir, exist_ok=True)
        stats_path = os.path.join(stats_dir, f'{folder}_stats.csv')
        if not os.path.exists(price_path):
            print(f"Price file not found for {symbol}, skipping.")
            continue
        df = pd.read_csv(price_path)
        # Clean columns
        numeric_cols = ['Close']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            else:
                df[col] = float('nan')
        stats = compute_stats(df, numeric_cols)
        # Save as single-row CSV
        stats_df = pd.DataFrame([stats])
        stats_df.to_csv(stats_path, index=False)
        print(f"Saved stats for {symbol} to {stats_path}")

if __name__ == '__main__':
    generate_all_stock_stats()
