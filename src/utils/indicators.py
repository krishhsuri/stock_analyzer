import pandas as pd
import numpy as np

# =====================
# Moving Averages
# =====================
def sma(series: pd.Series, window: int) -> pd.Series:
    """
    Simple Moving Average (SMA)
    Args:
        series: Price series
        window: Window length
    Returns:
        pd.Series of SMA values
    Example:
        signal = (close > sma(close, 20)).astype(int)
    """
    return pd.Series(series.rolling(window=window, min_periods=window).mean(), index=series.index)

def ema(series: pd.Series, window: int) -> pd.Series:
    """
    Exponential Moving Average (EMA)
    Args:
        series: Price series
        window: Window length
    Returns:
        pd.Series of EMA values
    Example:
        signal = (ema(close, 12) > ema(close, 26)).astype(int)
    """
    return pd.Series(series.ewm(span=window, adjust=False).mean(), index=series.index)

def wma(series: pd.Series, window: int) -> pd.Series:
    """
    Weighted Moving Average (WMA)
    Args:
        series: Price series
        window: Window length
    Returns:
        pd.Series of WMA values
    Example:
        signal = (close > wma(close, 20)).astype(int)
    """
    weights = np.arange(1, window + 1)
    return pd.Series(series.rolling(window).apply(lambda x: np.dot(x, weights)/weights.sum(), raw=True), index=series.index)

# =====================
# RSI
# =====================
def rsi(series: pd.Series, window: int = 14) -> pd.Series:
    """
    Relative Strength Index (RSI)
    Args:
        series: Price series
        window: Window length (default 14)
    Returns:
        pd.Series of RSI values (0-100)
    Example:
        signal = (rsi(close) < 30).astype(int)  # Buy when oversold
    """
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=window, min_periods=window).mean()
    avg_loss = loss.rolling(window=window, min_periods=window).mean()
    rs = avg_gain / avg_loss
    return pd.Series(100 - (100 / (1 + rs)), index=series.index)

# =====================
# MACD
# =====================
def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    """
    Moving Average Convergence Divergence (MACD)
    Args:
        series: Price series
        fast: Fast EMA window (default 12)
        slow: Slow EMA window (default 26)
        signal: Signal line EMA window (default 9)
    Returns:
        pd.DataFrame with columns ['macd', 'signal', 'histogram']
    Example:
        macd_df = macd(close)
        signal = (macd_df['macd'] > macd_df['signal']).astype(int)
    """
    ema_fast = ema(series, fast)
    ema_slow = ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return pd.DataFrame({
        'macd': macd_line, 
        'signal': signal_line,
        'histogram': histogram
    })

# =====================
# Bollinger Bands
# =====================
def bollinger_bands(series: pd.Series, window: int = 20, num_std: float = 2.0) -> pd.DataFrame:
    """
    Bollinger Bands
    Args:
        series: Price series
        window: Window length (default 20)
        num_std: Number of standard deviations (default 2)
    Returns:
        pd.DataFrame with columns ['mid', 'upper', 'lower']
    Example:
        bb = bollinger_bands(close)
        signal = (close < bb['lower']).astype(int)  # Buy when price below lower band
    """
    mid = series.rolling(window=window, min_periods=window).mean()
    std = series.rolling(window=window, min_periods=window).std()
    upper = mid + num_std * std
    lower = mid - num_std * std
    return pd.DataFrame({'mid': mid, 'upper': upper, 'lower': lower})

# =====================
# ATR
# =====================
def atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    """
    Average True Range (ATR)
    Args:
        df: DataFrame with columns ['High', 'Low', 'Close']
        window: Window length (default 14)
    Returns:
        pd.Series of ATR values
    Example:
        # ATR is used for volatility, not direct signals
        volatility_signal = (atr(df) > atr(df).rolling(20).mean()).astype(int)
    """
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return pd.Series(tr.rolling(window=window, min_periods=window).mean(), index=df.index)

# =====================
# Stochastic Oscillator
# =====================
def stochastic_oscillator(df: pd.DataFrame, k_window: int = 14, d_window: int = 3) -> pd.DataFrame:
    """
    Stochastic Oscillator
    Args:
        df: DataFrame with columns ['High', 'Low', 'Close']
        k_window: Window for %K (default 14)
        d_window: Window for %D (default 3)
    Returns:
        pd.DataFrame with columns ['%K', '%D']
    Example:
        stoch = stochastic_oscillator(df)
        signal = (stoch['%K'] < 20).astype(int)  # Buy when oversold
    """
    low_min = df['Low'].rolling(window=k_window, min_periods=k_window).min()
    high_max = df['High'].rolling(window=k_window, min_periods=k_window).max()
    range_ = high_max - low_min
    # Avoid division by zero
    range_ = range_.replace(0, np.nan)
    percent_k = 100 * (df['Close'] - low_min) / range_
    percent_d = percent_k.rolling(window=d_window, min_periods=d_window).mean()
    return pd.DataFrame({'%K': percent_k, '%D': percent_d}, index=df.index)

def hma(series: pd.Series, window: int) -> pd.Series:
    """
    Hull Moving Average (HMA)
    Args:
        series: Price series
        window: Window length
    Returns:
        pd.Series of HMA values
    Example:
        signal = (close > hma(close, 20)).astype(int)
    """
    half = int(window / 2)
    sqrt_win = int(np.sqrt(window))
    wma_half = wma(series, half)
    wma_full = wma(series, window)
    raw_hma = 2 * wma_half - wma_full
    return wma(raw_hma, sqrt_win)

def dema(series: pd.Series, window: int) -> pd.Series:
    """
    Double Exponential Moving Average (DEMA)
    Args:
        series: Price series
        window: Window length
    Returns:
        pd.Series of DEMA values
    Example:
        signal = (close > dema(close, 20)).astype(int)
    """
    ema1 = ema(series, window)
    ema2 = ema(ema1, window)
    return 2 * ema1 - ema2

def tema(series: pd.Series, window: int) -> pd.Series:
    """
    Triple Exponential Moving Average (TEMA)
    Args:
        series: Price series
        window: Window length
    Returns:
        pd.Series of TEMA values
    Example:
        signal = (close > tema(close, 20)).astype(int)
    """
    ema1 = ema(series, window)
    ema2 = ema(ema1, window)
    ema3 = ema(ema2, window)
    return 3 * (ema1 - ema2) + ema3

def kama(series: pd.Series, window: int = 10, pow1: int = 2, pow2: int = 30) -> pd.Series:
    """
    Kaufman Adaptive Moving Average (KAMA)
    Args:
        series: Price series
        window: Efficiency Ratio window (default 10)
        pow1: Fast EMA constant (default 2)
        pow2: Slow EMA constant (default 30)
    Returns:
        pd.Series of KAMA values
    Example:
        signal = (close > kama(close)).astype(int)
    """
    change = abs(series - series.shift(window))
    volatility = series.diff().abs().rolling(window=window).sum()
    er = change / volatility
    fast = 2 / (pow1 + 1)
    slow = 2 / (pow2 + 1)
    sc = (er * (fast - slow) + slow) ** 2
    kama_values = series.copy()
    for i in range(window, len(series)):
        if pd.notna(sc.iloc[i]):
            kama_values.iloc[i] = kama_values.iloc[i-1] + sc.iloc[i] * (series.iloc[i] - kama_values.iloc[i-1])
    return kama_values

def alma(series: pd.Series, window: int = 9, offset: float = 0.85, sigma: float = 6.0) -> pd.Series:
    """
    Arnaud Legoux Moving Average (ALMA)
    Args:
        series: Price series
        window: Window length (default 9)
        offset: Offset (default 0.85)
        sigma: Sigma (default 6.0)
    Returns:
        pd.Series of ALMA values
    Example:
        signal = (close > alma(close)).astype(int)
    """
    m = offset * (window - 1)
    s = window / sigma
    weights = np.exp(-((np.arange(window) - m) ** 2) / (2 * s * s))
    weights /= weights.sum()
    return pd.Series(series.rolling(window).apply(lambda x: np.dot(x, weights), raw=True), index=series.index)

def moving_average_ribbon(series: pd.Series, windows: list) -> pd.DataFrame:
    """
    Moving Average Ribbon
    Args:
        series: Price series
        windows: List of window lengths
    Returns:
        pd.DataFrame with each column as a moving average
    Example:
        ribbon = moving_average_ribbon(close, [10, 20, 30, 40, 50])
        # Signal when price above all MAs
        signal = (close > ribbon.min(axis=1)).astype(int)
    """
    data = {f'SMA_{w}': sma(series, w) for w in windows}
    return pd.DataFrame(data)

def parabolic_sar(df: pd.DataFrame, step: float = 0.02, max_step: float = 0.2) -> pd.Series:
    """
    Parabolic SAR (Stop and Reverse)
    Args:
        df: DataFrame with columns ['High', 'Low', 'Close']
        step: Acceleration factor step (default 0.02)
        max_step: Maximum acceleration factor (default 0.2)
    Returns:
        pd.Series of SAR values
    Example:
        sar = parabolic_sar(df)
        signal = (close > sar).astype(int)  # 1 when above SAR (uptrend)
    """
    high = df['High']
    low = df['Low']
    close = df['Close']
    sar = pd.Series(index=close.index, dtype='float64')
    long = True
    af = step
    ep = low.iloc[0]
    sar.iloc[0] = low.iloc[0]
    
    for i in range(1, len(df)):
        prev_sar = sar.iloc[i-1]
        if long:
            sar.iloc[i] = prev_sar + af * (ep - prev_sar)
            if low.iloc[i] < sar.iloc[i]:
                long = False
                sar.iloc[i] = ep
                af = step
                ep = high.iloc[i]
            else:
                if high.iloc[i] > ep:
                    ep = high.iloc[i]
                    af = min(af + step, max_step)
        else:
            sar.iloc[i] = prev_sar + af * (ep - prev_sar)
            if high.iloc[i] > sar.iloc[i]:
                long = True
                sar.iloc[i] = ep
                af = step
                ep = low.iloc[i]
            else:
                if low.iloc[i] < ep:
                    ep = low.iloc[i]
                    af = min(af + step, max_step)
    return sar

def supertrend(df: pd.DataFrame, atr_window: int = 10, multiplier: float = 3.0) -> pd.DataFrame:
    """
    SuperTrend Indicator
    Args:
        df: DataFrame with columns ['High', 'Low', 'Close']
        atr_window: ATR window (default 10)
        multiplier: Multiplier for ATR (default 3.0)
    Returns:
        pd.DataFrame with columns ['supertrend', 'direction'] where direction is 1 (up) or -1 (down)
    Example:
        st = supertrend(df)
        signal = st['direction']  # Already returns 1/-1 signals
    """
    atr_val = atr(df, atr_window)
    hl2 = (df['High'] + df['Low']) / 2
    upperband = hl2 + (multiplier * atr_val)
    lowerband = hl2 - (multiplier * atr_val)
    
    supertrend_val = pd.Series(index=df.index, dtype='float64')
    direction = pd.Series(index=df.index, dtype='int64')
    trend = 1  # 1 for uptrend, -1 for downtrend
    
    for i in range(len(df)):
        if i == 0:
            supertrend_val.iloc[i] = lowerband.iloc[i]
            direction.iloc[i] = 1
            continue
            
        if df['Close'].iloc[i] > upperband.iloc[i-1]:
            trend = 1
        elif df['Close'].iloc[i] < lowerband.iloc[i-1]:
            trend = -1
            
        if trend == 1:
            supertrend_val.iloc[i] = lowerband.iloc[i]
            direction.iloc[i] = 1
        else:
            supertrend_val.iloc[i] = upperband.iloc[i]
            direction.iloc[i] = -1
            
    return pd.DataFrame({'supertrend': supertrend_val, 'direction': direction})

def ichimoku_cloud(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ichimoku Cloud (Kumo)
    Args:
        df: DataFrame with columns ['High', 'Low', 'Close']
    Returns:
        pd.DataFrame with columns ['tenkan_sen', 'kijun_sen', 'senkou_span_a', 'senkou_span_b', 'chikou_span']
    Example:
        ichimoku = ichimoku_cloud(df)
        # Signal when price above cloud
        signal = (close > ichimoku[['senkou_span_a', 'senkou_span_b']].max(axis=1)).astype(int)
    """
    high = df['High']
    low = df['Low']
    tenkan_sen = (high.rolling(window=9).max() + low.rolling(window=9).min()) / 2
    kijun_sen = (high.rolling(window=26).max() + low.rolling(window=26).min()) / 2
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(26)
    senkou_span_b = ((high.rolling(window=52).max() + low.rolling(window=52).min()) / 2).shift(26)
    chikou_span = df['Close'].shift(-26)
    return pd.DataFrame({
        'tenkan_sen': tenkan_sen,
        'kijun_sen': kijun_sen,
        'senkou_span_a': senkou_span_a,
        'senkou_span_b': senkou_span_b,
        'chikou_span': chikou_span
    })

def donchian_channels(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """
    Donchian Channels
    Args:
        df: DataFrame with columns ['High', 'Low']
        window: Window length (default 20)
    Returns:
        pd.DataFrame with columns ['upper', 'lower', 'mid']
    Example:
        dc = donchian_channels(df)
        signal = (close > dc['upper']).astype(int)
    """
    upper = df['High'].rolling(window=window, min_periods=window).max()
    lower = df['Low'].rolling(window=window, min_periods=window).min()
    mid = (upper + lower) / 2
    return pd.DataFrame({'upper': upper, 'lower': lower, 'mid': mid})

# =====================
# Signal Generation Helper Functions
# =====================
def generate_crossover_signal(fast_series: pd.Series, slow_series: pd.Series) -> pd.Series:
    """
    Generate crossover signals (1, 0, -1)
    Args:
        fast_series: Fast moving indicator
        slow_series: Slow moving indicator
    Returns:
        pd.Series with 1 (bullish crossover), -1 (bearish crossover), 0 (no signal)
    """
    signal = pd.Series(0, index=fast_series.index)
    
    # Bullish crossover: fast crosses above slow
    bullish = (fast_series > slow_series) & (fast_series.shift(1) <= slow_series.shift(1))
    signal[bullish] = 1
    
    # Bearish crossover: fast crosses below slow
    bearish = (fast_series < slow_series) & (fast_series.shift(1) >= slow_series.shift(1))
    signal[bearish] = -1
    
    return signal

def generate_threshold_signal(series: pd.Series, buy_threshold: float, sell_threshold: float) -> pd.Series:
    """
    Generate threshold-based signals for oscillators like RSI
    Args:
        series: Oscillator values
        buy_threshold: Buy when series crosses below this (e.g., 30 for RSI)
        sell_threshold: Sell when series crosses above this (e.g., 70 for RSI)
    Returns:
        pd.Series with 1 (buy signal), -1 (sell signal), 0 (no signal)
    """
    signal = pd.Series(0, index=series.index)
    
    # Buy signal: crosses below buy_threshold (oversold)
    buy_signal = (series < buy_threshold) & (series.shift(1) >= buy_threshold)
    signal[buy_signal] = 1
    
    # Sell signal: crosses above sell_threshold (overbought)
    sell_signal = (series > sell_threshold) & (series.shift(1) <= sell_threshold)
    signal[sell_signal] = -1
    
    return signal

def stochrsi(series: pd.Series, window: int = 14) -> pd.Series:
    """
    Stochastic RSI (StochRSI)
    Args:
        series: Price series
        window: RSI window (default 14)
    Returns:
        pd.Series of StochRSI values (0-1)
    Example:
        signal = (stochrsi(close) < 0.2).astype(int)  # Buy when oversold
    """
    rsi_vals = rsi(series, window)
    min_rsi = rsi_vals.rolling(window=window, min_periods=window).min()
    max_rsi = rsi_vals.rolling(window=window, min_periods=window).max()
    stochrsi = (rsi_vals - min_rsi) / (max_rsi - min_rsi)
    return pd.Series(stochrsi, index=series.index)

def cci(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """
    Commodity Channel Index (CCI)
    Args:
        df: DataFrame with columns ['High', 'Low', 'Close']
        window: Window length (default 20)
    Returns:
        pd.Series of CCI values
    Example:
        signal = (cci(df) < -100).astype(int)  # Buy when oversold
    """
    tp = (df['High'] + df['Low'] + df['Close']) / 3
    sma_tp = tp.rolling(window=window, min_periods=window).mean()
    mad = tp.rolling(window=window, min_periods=window).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True)
    cci = (tp - sma_tp) / (0.015 * mad)
    return pd.Series(cci, index=df.index)

def cmo(series: pd.Series, window: int = 14) -> pd.Series:
    """
    Chande Momentum Oscillator (CMO)
    Args:
        series: Price series
        window: Window length (default 14)
    Returns:
        pd.Series of CMO values
    Example:
        signal = (cmo(close) < -50).astype(int)
    """
    diff = series.diff()
    up = diff.where(diff > 0, 0.0).rolling(window=window).sum()
    down = -diff.where(diff < 0, 0.0).rolling(window=window).sum()
    cmo = 100 * (up - down) / (up + down)
    return pd.Series(cmo, index=series.index)

def awesome_oscillator(df: pd.DataFrame, short_window: int = 5, long_window: int = 34) -> pd.Series:
    """
    Awesome Oscillator (AO)
    Args:
        df: DataFrame with columns ['High', 'Low']
        short_window: Short SMA window (default 5)
        long_window: Long SMA window (default 34)
    Returns:
        pd.Series of AO values
    Example:
        signal = (awesome_oscillator(df) > 0).astype(int)
    """
    median_price = (df['High'] + df['Low']) / 2
    ao = sma(median_price, short_window) - sma(median_price, long_window)
    return pd.Series(ao, index=df.index)

def ppo(series: pd.Series, fast: int = 12, slow: int = 26) -> pd.Series:
    """
    Percentage Price Oscillator (PPO)
    Args:
        series: Price series
        fast: Fast EMA window (default 12)
        slow: Slow EMA window (default 26)
    Returns:
        pd.Series of PPO values
    Example:
        signal = (ppo(close) > 0).astype(int)
    """
    ema_fast = ema(series, fast)
    ema_slow = ema(series, slow)
    ppo = 100 * (ema_fast - ema_slow) / ema_slow
    return pd.Series(ppo, index=series.index)

def rvi(df: pd.DataFrame, window: int = 10) -> pd.Series:
    """
    Relative Vigor Index (RVI)
    Args:
        df: DataFrame with columns ['Open', 'High', 'Low', 'Close']
        window: Window length (default 10)
    Returns:
        pd.Series of RVI values
    Example:
        signal = (rvi(df) > 0).astype(int)
    """
    num = (df['Close'] - df['Open']).rolling(window=window).mean()
    denom = (df['High'] - df['Low']).rolling(window=window).mean()
    rvi = num / denom
    return pd.Series(rvi, index=df.index)

def ultimate_oscillator(df: pd.DataFrame, s1: int = 7, s2: int = 14, s3: int = 28) -> pd.Series:
    """
    Ultimate Oscillator
    Args:
        df: DataFrame with columns ['High', 'Low', 'Close']
        s1, s2, s3: Short, medium, long periods (default 7, 14, 28)
    Returns:
        pd.Series of Ultimate Oscillator values
    Example:
        signal = (ultimate_oscillator(df) > 50).astype(int)
    """
    bp = df['Close'] - pd.concat([df['Low'], df['Close'].shift(1)], axis=1).min(axis=1)
    tr = pd.concat([
        df['High'] - df['Low'],
        (df['High'] - df['Close'].shift(1)).abs(),
        (df['Low'] - df['Close'].shift(1)).abs()
    ], axis=1).max(axis=1)
    avg7 = bp.rolling(s1).sum() / tr.rolling(s1).sum()
    avg14 = bp.rolling(s2).sum() / tr.rolling(s2).sum()
    avg28 = bp.rolling(s3).sum() / tr.rolling(s3).sum()
    uo = 100 * (4 * avg7 + 2 * avg14 + avg28) / 7
    return pd.Series(uo, index=df.index)

def williams_r(df: pd.DataFrame, window: int = 14) -> pd.Series:
    """
    Williams %R
    Args:
        df: DataFrame with columns ['High', 'Low', 'Close']
        window: Window length (default 14)
    Returns:
        pd.Series of Williams %R values
    Example:
        signal = (williams_r(df) < -80).astype(int)
    """
    highest_high = df['High'].rolling(window=window, min_periods=window).max()
    lowest_low = df['Low'].rolling(window=window, min_periods=window).min()
    wr = -100 * (highest_high - df['Close']) / (highest_high - lowest_low)
    return pd.Series(wr, index=df.index)

def trix(series: pd.Series, window: int = 15) -> pd.Series:
    """
    TRIX (Triple Exponential Average)
    Args:
        series: Price series
        window: Window length (default 15)
    Returns:
        pd.Series of TRIX values
    Example:
        signal = (trix(close) > 0).astype(int)
    """
    ema1 = ema(series, window)
    ema2 = ema(ema1, window)
    ema3 = ema(ema2, window)
    trix = ema3.pct_change() * 100
    return pd.Series(trix, index=series.index)

def dpo(series: pd.Series, window: int = 20) -> pd.Series:
    """
    Detrended Price Oscillator (DPO)
    Args:
        series: Price series
        window: Window length (default 20)
    Returns:
        pd.Series of DPO values
    Example:
        signal = (dpo(close) > 0).astype(int)
    """
    shift = int(window / 2 + 1)
    sma_ = sma(series, window)
    dpo = series.shift(-shift) - sma_
    return pd.Series(dpo, index=series.index)

def elder_ray(df: pd.DataFrame, ema_window: int = 13) -> pd.DataFrame:
    """
    Elder Ray Index (Bull Power / Bear Power)
    Args:
        df: DataFrame with columns ['High', 'Low', 'Close']
        ema_window: EMA window (default 13)
    Returns:
        pd.DataFrame with columns ['bull_power', 'bear_power']
    Example:
        er = elder_ray(df)
        signal = (er['bull_power'] > 0).astype(int)
    """
    ema_ = ema(pd.Series(df['Close'], index=df.index), ema_window)
    bull_power = df['High'] - ema_
    bear_power = df['Low'] - ema_
    return pd.DataFrame({'bull_power': bull_power, 'bear_power': bear_power}, index=df.index)

def keltner_channels(df: pd.DataFrame, window: int = 20, atr_mult: float = 2.0) -> pd.DataFrame:
    """
    Keltner Channels (KC)
    Args:
        df: DataFrame with columns ['High', 'Low', 'Close']
        window: Window length (default 20)
        atr_mult: ATR multiplier (default 2.0)
    Returns:
        pd.DataFrame with columns ['mid', 'upper', 'lower']
    Example:
        kc = keltner_channels(df)
        signal = (close > kc['upper']).astype(int)
    """
    mid = ema(pd.Series(df['Close'], index=df.index), window)
    atr_ = atr(df, window)
    upper = mid + atr_mult * atr_
    lower = mid - atr_mult * atr_
    return pd.DataFrame({'mid': mid, 'upper': upper, 'lower': lower}, index=df.index)

def chaikin_volatility(df: pd.DataFrame, window: int = 10) -> pd.Series:
    """
    Chaikin Volatility
    Args:
        df: DataFrame with columns ['High', 'Low']
        window: Window length (default 10)
    Returns:
        pd.Series of Chaikin Volatility values
    Example:
        signal = (chaikin_volatility(df) > 0).astype(int)
    """
    diff = df['High'] - df['Low']
    ema_diff = ema(diff, window)
    chaikin = 100 * (ema_diff - ema_diff.shift(window)) / ema_diff.shift(window)
    return pd.Series(chaikin, index=df.index)

def stddev_channel(series: pd.Series, window: int = 20, num_std: float = 2.0) -> pd.DataFrame:
    """
    Standard Deviation Channel
    Args:
        series: Price series
        window: Window length (default 20)
        num_std: Number of standard deviations (default 2)
    Returns:
        pd.DataFrame with columns ['mid', 'upper', 'lower']
    Example:
        sdc = stddev_channel(close)
        signal = (close > sdc['upper']).astype(int)
    """
    mid = sma(series, window)
    std = series.rolling(window=window, min_periods=window).std()
    upper = mid + num_std * std
    lower = mid - num_std * std
    return pd.DataFrame({'mid': mid, 'upper': upper, 'lower': lower}, index=series.index)

def historical_volatility(series: pd.Series, window: int = 20) -> pd.Series:
    """
    Historical Volatility (HV)
    Args:
        series: Price series
        window: Window length (default 20)
    Returns:
        pd.Series of annualized volatility values
    Example:
        signal = (historical_volatility(close) > 0.02).astype(int)
    """
    log_ret = np.log(series / series.shift(1))
    hv = log_ret.rolling(window=window).std() * np.sqrt(252)
    return pd.Series(hv, index=series.index)

def rvi_volatility(series: pd.Series, window: int = 10) -> pd.Series:
    """
    Relative Volatility Index (RVI)
    Args:
        series: Price series
        window: Window length (default 10)
    Returns:
        pd.Series of RVI values (0-100)
    Example:
        signal = (rvi_volatility(close) > 50).astype(int)
    """
    up = series.diff().where(lambda x: x > 0, 0.0)
    down = -series.diff().where(lambda x: x < 0, 0.0)
    std_up = up.rolling(window=window).std()
    std_down = down.rolling(window=window).std()
    rvi = 100 * std_up / (std_up + std_down)
    return pd.Series(rvi, index=series.index)

