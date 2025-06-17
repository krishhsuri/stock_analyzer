import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Tuple, List
import mplfinance as mpf

class TechnicalVisualizer:
    """Class for creating technical analysis visualizations"""
    
    def __init__(self, style: str = 'darkgrid'):
        """
        Initialize the visualizer
        
        Args:
            style (str): Seaborn style for plots
        """
        sns.set_style(style)
        plt.rcParams['figure.figsize'] = (12, 8)
    
    def plot_candlestick(self, data: pd.DataFrame,
                        title: str = "Candlestick Chart",
                        volume: bool = True) -> None:
        """
        Plot candlestick chart with optional volume
        
        Args:
            data (pd.DataFrame): OHLCV data
            title (str): Chart title
            volume (bool): Whether to show volume
        """
        mpf.plot(data,
                type='candle',
                title=title,
                volume=volume,
                style='charles')
    
    def plot_technical_indicators(self,
                                data: pd.DataFrame,
                                indicators: dict,
                                title: str = "Technical Indicators") -> None:
        """
        Plot price with technical indicators
        
        Args:
            data (pd.DataFrame): Price data
            indicators (dict): Dictionary of indicator series
            title (str): Chart title
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [3, 1]})
        
        # Plot price
        ax1.plot(data.index, data['Close'], label='Close Price')
        
        # Plot indicators
        for name, series in indicators.items():
            if name in ['RSI', 'MACD']:
                ax2.plot(series.index, series, label=name)
            else:
                ax1.plot(series.index, series, label=name)
        
        ax1.set_title(title)
        ax1.legend()
        ax2.legend()
        plt.tight_layout()
        plt.show()
    
    def plot_correlation_matrix(self,
                              data: pd.DataFrame,
                              title: str = "Correlation Matrix") -> None:
        """
        Plot correlation matrix of technical indicators
        
        Args:
            data (pd.DataFrame): DataFrame of indicators
            title (str): Chart title
        """
        corr = data.corr()
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr,
                   annot=True,
                   cmap='coolwarm',
                   center=0,
                   title=title)
        plt.tight_layout()
        plt.show()
    
    def plot_pattern(self,
                    data: pd.DataFrame,
                    pattern_points: List[Tuple],
                    title: str = "Chart Pattern") -> None:
        """
        Plot price with identified pattern
        
        Args:
            data (pd.DataFrame): Price data
            pattern_points (List[Tuple]): List of (x, y) coordinates for pattern
            title (str): Chart title
        """
        plt.figure(figsize=(12, 6))
        plt.plot(data.index, data['Close'], label='Close Price')
        
        # Plot pattern points
        x_coords, y_coords = zip(*pattern_points)
        plt.plot(x_coords, y_coords, 'r--', label='Pattern')
        
        plt.title(title)
        plt.legend()
        plt.grid(True)
        plt.show()

if __name__ == "__main__":
    # Example usage
    import yfinance as yf
    from technical_indicators import TechnicalIndicators
    
    # Download sample data
    ticker = "AAPL"
    data = yf.download(ticker, start="2023-01-01", end="2024-01-01")
    
    # Calculate indicators
    ti = TechnicalIndicators()
    rsi = ti.calculate_rsi(data['Close'])
    macd, signal, hist = ti.calculate_macd(data['Close'])
    upper, middle, lower = ti.calculate_bollinger_bands(data['Close'])
    
    # Create visualizations
    viz = TechnicalVisualizer()
    
    # Plot candlestick
    viz.plot_candlestick(data, title=f"{ticker} Candlestick Chart")
    
    # Plot indicators
    indicators = {
        'Upper BB': upper,
        'Middle BB': middle,
        'Lower BB': lower,
        'RSI': rsi,
        'MACD': macd
    }
    viz.plot_technical_indicators(data, indicators, title=f"{ticker} Technical Analysis") 