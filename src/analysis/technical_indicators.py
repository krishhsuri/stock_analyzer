import pandas as pd
import numpy as np
import talib
from typing import Union, Optional

class TechnicalIndicators:
    """Class for calculating and analyzing technical indicators"""
    
    @staticmethod
    def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
        """
        Calculate Relative Strength Index (RSI)
        
        Args:
            prices (pd.Series): Price series
            period (int): RSI period
            
        Returns:
            pd.Series: RSI values
        """
        return pd.Series(talib.RSI(prices.values, timeperiod=period), index=prices.index)
    
    @staticmethod
    def calculate_macd(prices: pd.Series, 
                      fast_period: int = 12,
                      slow_period: int = 26,
                      signal_period: int = 9) -> tuple:
        """
        Calculate Moving Average Convergence Divergence (MACD)
        
        Args:
            prices (pd.Series): Price series
            fast_period (int): Fast period
            slow_period (int): Slow period
            signal_period (int): Signal period
            
        Returns:
            tuple: (MACD line, Signal line, Histogram)
        """
        macd, signal, hist = talib.MACD(prices.values,
                                      fastperiod=fast_period,
                                      slowperiod=slow_period,
                                      signalperiod=signal_period)
        return (pd.Series(macd, index=prices.index),
                pd.Series(signal, index=prices.index),
                pd.Series(hist, index=prices.index))
    
    @staticmethod
    def calculate_bollinger_bands(prices: pd.Series,
                                period: int = 20,
                                num_std: float = 2.0) -> tuple:
        """
        Calculate Bollinger Bands
        
        Args:
            prices (pd.Series): Price series
            period (int): Moving average period
            num_std (float): Number of standard deviations
            
        Returns:
            tuple: (Upper band, Middle band, Lower band)
        """
        upper, middle, lower = talib.BBANDS(prices.values,
                                          timeperiod=period,
                                          nbdevup=num_std,
                                          nbdevdn=num_std)
        return (pd.Series(upper, index=prices.index),
                pd.Series(middle, index=prices.index),
                pd.Series(lower, index=prices.index))
    
    @staticmethod
    def detect_patterns(prices: pd.Series,
                       high: pd.Series,
                       low: pd.Series,
                       volume: pd.Series) -> dict:
        """
        Detect common chart patterns
        
        Args:
            prices (pd.Series): Close prices
            high (pd.Series): High prices
            low (pd.Series): Low prices
            volume (pd.Series): Volume
            
        Returns:
            dict: Dictionary of detected patterns
        """
        patterns = {}
        
        # Example pattern detection (simplified)
        # TODO: Implement more sophisticated pattern detection
        return patterns

if __name__ == "__main__":
    # Example usage
    import yfinance as yf
    
    # Download sample data
    ticker = "AAPL"
    data = yf.download(ticker, start="2023-01-01", end="2024-01-01")
    
    # Calculate indicators
    ti = TechnicalIndicators()
    rsi = ti.calculate_rsi(data['Close'])
    macd, signal, hist = ti.calculate_macd(data['Close'])
    upper, middle, lower = ti.calculate_bollinger_bands(data['Close'])
    
    # Print results
    print(f"RSI for {ticker}:")
    print(rsi.tail()) 