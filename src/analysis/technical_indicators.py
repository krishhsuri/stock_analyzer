import os
import pandas as pd
import yaml
import numpy as np

def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

CONFIG_PATH = os.path.join(os.path.dirname(__file__), '../../config/config.yaml')
config = load_config(CONFIG_PATH)

# --- Technical Indicator Functions ---
def ema(series, window):
    return series.ewm(span=window, adjust=False).mean()

def rsi(series, window=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def bollinger_bands(series, window=20, num_std=2):
    sma_ = series.rolling(window=window).mean()
    std = series.rolling(window=window).std()
    upper = sma_ + num_std * std
    lower = sma_ - num_std * std
    return sma_, upper, lower

def atr(df, window=14):
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(window=window).mean()

def stochastic_oscillator(df, k_window=14, d_window=3):
    low_min = df['Low'].rolling(window=k_window).min()
    high_max = df['High'].rolling(window=k_window).max()
    percent_k = 100 * (df['Close'] - low_min) / (high_max - low_min)
    percent_d = percent_k.rolling(window=d_window).mean()
    return percent_k, percent_d

def macd(series, fast=12, slow=26, signal=9):
    ema_fast = ema(series, fast)
    ema_slow = ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line, signal_line

def add_lags(df, col_names, lags=[1,2,3]):
    for col in col_names:
        for lag in lags:
            df[f'{col}_lag{lag}'] = df[col].shift(lag)
    return df

def generate_technical_indicators():
    for symbol_entry in config['data']['symbols']:
        symbol = symbol_entry['symbol']
        folder = symbol.replace('.NS', '').replace('.BSE', '').upper()
        folder_path = os.path.join(os.path.dirname(__file__), f'../../data/{folder}/unclean_timedata')
        os.makedirs(folder_path, exist_ok=True)
        price_path = os.path.join(os.path.dirname(__file__), f'../../data/{folder}/unclean_timedata/{symbol}_price.csv')
        out_path = os.path.join(folder_path, f'{symbol}_technical_indicators.csv')
        if not os.path.exists(price_path):
            print(f"Price file not found for {symbol}, skipping.")
            continue
        df = pd.read_csv(price_path)
        # Ensure numeric columns
        for col in ['Close', 'High', 'Low']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        if not all(col in df.columns for col in ['Close','High','Low']):
            print(f"Required columns missing for {symbol}, skipping.")
            continue
        # EMAs
        for w in [9,21,50,200]:
            df[f'EMA_{w}'] = ema(df['Close'], w)
        # RSI
        df['RSI_14'] = rsi(df['Close'], 14)
        # Bollinger Bands
        bb_mid, bb_upper, bb_lower = bollinger_bands(df['Close'], 20, 2)
        df['BB_Mid'] = bb_mid
        df['BB_Upper'] = bb_upper
        df['BB_Lower'] = bb_lower
        # ATR
        df['ATR_14'] = atr(df, 14)
        # Stochastic Oscillator
        percent_k, percent_d = stochastic_oscillator(df, 14, 3)
        df['Stoch_%K_14_3'] = percent_k
        df['Stoch_%D_3'] = percent_d
        # MACD
        macd_line, signal_line = macd(df['Close'])
        df['MACD'] = macd_line
        df['MACD_Signal'] = signal_line
        # Lags
        lag_cols = [f'EMA_{w}' for w in [9,21,50,200]] + ['RSI_14','ATR_14','Stoch_%K_14_3','Stoch_%D_3']
        df = add_lags(df, lag_cols, lags=[1,2,3])
        # Save
        cols = ['Date'] if 'Date' in df.columns else []
        cols += [f'EMA_{w}' for w in [9,21,50,200]] + ['RSI_14','BB_Mid','BB_Upper','BB_Lower','ATR_14','Stoch_%K_14_3','Stoch_%D_3','MACD','MACD_Signal']
        for col in lag_cols:
            for lag in [1,2,3]:
                cols.append(f'{col}_lag{lag}')
        df[cols].to_csv(out_path, index=False)
        print(f"Saved technical indicators for {symbol} to {out_path}")

if __name__ == '__main__':
    generate_technical_indicators() 