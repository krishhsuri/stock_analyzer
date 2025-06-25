import os
import sys
import pandas as pd
import mplfinance as mpf
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider, Cursor
import matplotlib.dates as mdates
from datetime import datetime

# --- Global Definitions for Timeframes ---
TIMEFRAMES = ['1m', 'ytd', '1y', '5y', 'all']
TIMEFRAME_WINDOWS = {
    '1m': 22,
    '1y': 252,
    '5y': 252 * 5,
    'ytd': None,
    'all': None
}

def get_timeframe_dates(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """Slices the DataFrame to a specific date range based on the selected timeframe."""
    if not isinstance(df.index, pd.DatetimeIndex):
        df = df.set_index('Date', drop=False)

    today = df.index.max()
    
    if not isinstance(today, pd.Timestamp):
        print("Warning: Could not determine the last trading day.")
        return df.copy()

    if timeframe == '1m':
        start = today - pd.DateOffset(months=1)
    elif timeframe == '1y':
        start = today - pd.DateOffset(years=1)
    elif timeframe == 'ytd':
        start = pd.Timestamp(year=today.year, month=1, day=1)
    elif timeframe == '5y':
        start = today - pd.DateOffset(years=5)
    else:  # 'all'
        start = df.index.min()

    return df[df.index >= start].copy()


def plot_candlestick_with_dividends(symbol: str, initial_timeframe: str = 'all'):
    """Loads and plots interactive candlestick chart with dividends."""
    try:
        script_dir = os.path.dirname(__file__)
    except NameError:
        script_dir = os.getcwd()

    symbol_folder = symbol.replace('.NS', '').replace('.BSE', '').upper()
    data_path = os.path.join(script_dir, f'../../data/{symbol_folder}/{symbol}_price.csv')

    if not os.path.exists(data_path):
        print(f"Data file not found for {symbol} at {data_path}")
        return

    # --- Data Loading and Cleaning ---
    df = pd.read_csv(data_path)
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df.dropna(subset=['Date'], inplace=True)
    df.set_index('Date', inplace=True)

    ohlcv_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    for col in ohlcv_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    df[ohlcv_cols] = df[ohlcv_cols].ffill().bfill()

    for col in ['Open', 'High', 'Low', 'Close']:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].mean())
    if df['Volume'].isnull().any():
        df['Volume'] = df['Volume'].fillna(0)

    if 'Dividend' in df.columns:
        df['Dividend'] = pd.to_numeric(df['Dividend'], errors='coerce').fillna(0)
    else:
        df['Dividend'] = 0

    # --- Interactive Plot State ---
    state = {'timeframe': initial_timeframe, 'start_idx': 0, 'full_df': df}

    def get_windowed_df():
        tf = state['timeframe']
        dff_timeframe = get_timeframe_dates(state['full_df'], tf)
        
        window = TIMEFRAME_WINDOWS.get(tf)
        if window is None or window > len(dff_timeframe):
            window = len(dff_timeframe)
        
        start = state['start_idx']
        end = start + window
        if end > len(dff_timeframe):
            end = len(dff_timeframe)
            start = max(0, end - window)
        
        dff_windowed = dff_timeframe.iloc[start:end]
        return dff_windowed, window, len(dff_timeframe)

    # --- Matplotlib Figure Setup ---
    fig = plt.figure(figsize=(15, 9))
    gs = fig.add_gridspec(6, 1, height_ratios=[6, 2, 1, 1, 1, 1])
    ax_main = fig.add_subplot(gs[0, 0])
    ax_vol = fig.add_subplot(gs[1, 0], sharex=ax_main)
    plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.15, hspace=0.0)

    ax_vol.grid(True, linestyle='--', alpha=0.6)
    ax_main.grid(True, linestyle='--', alpha=0.6)
    cursor = Cursor(ax_main, useblit=True, color='gray', linewidth=1, linestyle='dotted')

    def update(val=None):
        ax_main.clear()
        ax_vol.clear()
        dff, window, total_in_timeframe = get_windowed_df()

        # --- FIX: Check for sufficient data to prevent crash ---
        # Plotting 0 or 1 data points can cause Matplotlib to fail when drawing ticks.
        if len(dff) < 2:
            ax_main.text(0.5, 0.5, f"Not enough data to display for {symbol} ({state['timeframe'].upper()})",
                         ha='center', va='center', transform=ax_main.transAxes)
            ax_main.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
            ax_vol.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
            ax_vol.get_yaxis().set_visible(False) # Hide y-axis of volume plot
            fig.canvas.draw_idle()
            return
        
        ax_vol.get_yaxis().set_visible(True) # Ensure it's visible again if it was hidden

        mpf.plot(dff, type='candle', volume=ax_vol, ax=ax_main, style='yahoo', show_nontrading=False, warn_too_much_data=10000)

        divs = dff[dff['Dividend'] > 0]
        if not divs.empty:
            for date, row in divs.iterrows():
                x = float(mdates.date2num(date))
                y = float(row['Low'])
                
                ax_main.annotate(
                    'D', (x, y), textcoords="offset points", xytext=(0, -25),
                    ha='center', va='center', color='red', fontsize=12, fontweight='bold',
                    arrowprops=dict(arrowstyle='-|>', color='red', lw=1.5, shrinkA=5)
                )
                ax_main.annotate(
                    f"₹{float(row['Dividend']):.2f}", (x, y), textcoords="offset points", xytext=(0, -45),
                    ha='center', color='black', fontsize=9,
                    bbox=dict(boxstyle='round,pad=0.3', fc='yellow', alpha=0.8, ec='black', lw=0.5)
                )

        ax_main.set_title(f"{symbol} Price & Dividends ({state['timeframe'].upper()})")
        ax_vol.set_ylabel('Volume')
        # This parameter is set by mplfinance, but we ensure it's set for our main axis too
        ax_main.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
        
        max_scroll = max(0, total_in_timeframe - window)
        slider.ax.set_xlim(0, max_scroll if max_scroll > 0 else 1)
        if max_scroll > 0:
            slider.set_val(state['start_idx'])
            slider.valmax = max_scroll
            slider.ax.set_visible(True)
        else:
            slider.ax.set_visible(False)
        
        fig.canvas.draw_idle()

    def on_timeframe_clicked(label: str):
        state['timeframe'] = label
        state['start_idx'] = 0
        update()

    def on_slider_changed(val: float):
        state['start_idx'] = int(val)
        update()

    # --- Widgets Setup ---
    buttons = []
    total_button_width = 0.8
    button_width = total_button_width / len(TIMEFRAMES)
    start_x = 0.5 - total_button_width / 2

    for i, tf in enumerate(TIMEFRAMES):
        rect = (start_x + i * button_width, 0.05, button_width - 0.01, 0.04)
        bax = fig.add_axes(rect)
        btn = Button(bax, tf.upper())
        btn.on_clicked(lambda event, label=tf: on_timeframe_clicked(label))
        buttons.append(btn)

    slider_ax_rect = (0.15, 0.1, 0.7, 0.02)
    slider_ax = fig.add_axes(slider_ax_rect)
    dff, window, total = get_windowed_df()
    max_scroll_init = max(0, total - window)
    slider = Slider(
        slider_ax, 'Scroll', 0, max_scroll_init if max_scroll_init > 0 else 1,
        valinit=0, valstep=1, color='#cde'
    )
    slider.on_changed(on_slider_changed)

    update()
    plt.show()

def main():
    if len(sys.argv) > 1:
        symbol = sys.argv[1].strip().upper()
    else:
        symbol = input("Enter the stock symbol (e.g., RELIANCE.NS): ").strip().upper()
    
    if not symbol:
        print("No symbol provided. Exiting.")
        return
        
    plot_candlestick_with_dividends(symbol)

if __name__ == '__main__':
    main()