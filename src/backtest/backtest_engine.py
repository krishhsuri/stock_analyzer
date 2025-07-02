import pandas as pd
import numpy as np

class BacktestEngine:
    def __init__(self, data, signal_col, initial_capital=100000, position_size='all_in', transaction_cost=0.001):
        self.data = data.copy()
        self.signal_col = signal_col
        self.initial_capital = initial_capital
        self.position_size = position_size
        self.transaction_cost = transaction_cost
        self.results = None
        self.trade_log = []

    def run(self):
        df = self.data.copy()
        df['Position'] = 0
        df['Cash'] = self.initial_capital
        df['Holdings'] = 0
        df['Total'] = self.initial_capital
        in_position = False
        entry_price = 0
        shares = 0
        for i, row in df.iterrows():
            signal = row[self.signal_col]
            price = row['Close']
            if signal == 'buy' and not in_position:
                # Enter position
                if self.position_size == 'all_in':
                    shares = (df.at[i, 'Cash'] * (1 - self.transaction_cost)) // price
                else:
                    shares = ((df.at[i, 'Cash'] * self.position_size) * (1 - self.transaction_cost)) // price
                cost = shares * price * (1 + self.transaction_cost)
                df.at[i, 'Cash'] -= cost
                df.at[i, 'Holdings'] = shares * price
                in_position = True
                entry_price = price
                self.trade_log.append({'type': 'buy', 'date': row['Date'], 'price': price, 'shares': shares})
            elif signal == 'sell' and in_position:
                # Exit position
                proceeds = shares * price * (1 - self.transaction_cost)
                df.at[i, 'Cash'] += proceeds
                df.at[i, 'Holdings'] = 0
                in_position = False
                self.trade_log.append({'type': 'sell', 'date': row['Date'], 'price': price, 'shares': shares})
                shares = 0
            else:
                # Hold
                if in_position:
                    df.at[i, 'Holdings'] = shares * price
            df.at[i, 'Total'] = df.at[i, 'Cash'] + df.at[i, 'Holdings']
        self.results = df

    def get_metrics(self):
        df = self.results
        if df is None:
            return {}
        total_return = (df['Total'].iloc[-1] - self.initial_capital) / self.initial_capital
        years = (pd.to_datetime(df['Date'].iloc[-1]) - pd.to_datetime(df['Date'].iloc[0])).days / 365.25
        annualized_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else np.nan
        returns = df['Total'].pct_change().dropna()
        sharpe = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else np.nan
        max_drawdown = ((df['Total'].cummax() - df['Total']) / df['Total'].cummax()).max()
        wins = [t for t in self.trade_log if t['type'] == 'sell' and t['price'] > 0]
        win_trades = sum(1 for i in range(1, len(self.trade_log), 2)
                         if self.trade_log[i]['type'] == 'sell' and self.trade_log[i]['price'] > self.trade_log[i-1]['price'])
        win_rate = win_trades / (len(self.trade_log)//2) if len(self.trade_log) >= 2 else np.nan
        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate
        }

    def get_trade_log(self):
        return self.trade_log

    def get_results(self):
        return self.results 