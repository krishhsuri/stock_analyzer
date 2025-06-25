import os
import pandas as pd
import yaml
import numpy as np

def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

CONFIG_PATH = os.path.join(os.path.dirname(__file__), '../../config/config.yaml')
config = load_config(CONFIG_PATH)

def obv(close, volume):
    obv = [0]
    for i in range(1, len(close)):
        if close[i] > close[i-1]:
            obv.append(obv[-1] + volume[i])
        elif close[i] < close[i-1]:
            obv.append(obv[-1] - volume[i])
        else:
            obv.append(obv[-1])
    return pd.Series(obv, index=close.index)

def vwap(df):
    return (df['Close'] * df['Volume']).cumsum() / df['Volume'].cumsum()

def chaikin_money_flow(df, window=20):
    mf_multiplier = ((df['Close'] - df['Low']) - (df['High'] - df['Close'])) / (df['High'] - df['Low'])
    mf_multiplier = mf_multiplier.replace([np.inf, -np.inf], 0).fillna(0)
    mf_volume = mf_multiplier * df['Volume']
    cmf = mf_volume.rolling(window=window).sum() / df['Volume'].rolling(window=window).sum()
    return cmf

def add_lags(df, col_names, lags=[1,2,3]):
    for col in col_names:
        for lag in lags:
            df[f'{col}_lag{lag}'] = df[col].shift(lag)
    return df

def generate_volume_indicators():
    for symbol_entry in config['data']['symbols']:
        symbol = symbol_entry['symbol']
        folder = symbol.replace('.NS', '').replace('.BSE', '').upper()
        folder_path = os.path.join(os.path.dirname(__file__), f'../../data/{folder}/unclean_timedata')
        os.makedirs(folder_path, exist_ok=True)
        price_path = os.path.join(os.path.dirname(__file__), f'../../data/{folder}/unclean_timedata/{symbol}_price.csv')
        out_path = os.path.join(folder_path, f'{symbol}_volume_indicators.csv')
        if not os.path.exists(price_path):
            print(f"Price file not found for {symbol}, skipping.")
            continue
        df = pd.read_csv(price_path)
        # Ensure numeric columns
        for col in ['Close', 'High', 'Low', 'Volume']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        if not all(col in df.columns for col in ['Close','High','Low','Volume']):
            print(f"Required columns missing for {symbol}, skipping.")
            continue
        df['OBV'] = obv(df['Close'], df['Volume'])
        df['VWAP'] = vwap(df)
        df['CMF_20'] = chaikin_money_flow(df, 20)
        lag_cols = ['OBV','CMF_20']
        df = add_lags(df, lag_cols, lags=[1,2,3])
        cols = ['Date'] if 'Date' in df.columns else []
        cols += ['OBV','VWAP','CMF_20']
        for col in lag_cols:
            for lag in [1,2,3]:
                cols.append(f'{col}_lag{lag}')
        df[cols].to_csv(out_path, index=False)
        print(f"Saved volume indicators for {symbol} to {out_path}")

if __name__ == '__main__':
    generate_volume_indicators() 