import os
import pytest
import pandas as pd
from src.data.data_pipeline import load_config, fetch_price_data
from datetime import datetime

def test_load_config():
    config = load_config('config/config.yaml')
    assert 'data' in config
    assert 'symbols' in config['data']

def test_fetch_price_data():
    config = load_config('config/config.yaml')
    symbol = config['data']['symbols'][0]
    df = fetch_price_data(
        symbol,
        datetime.strptime(config['data']['start_date'], '%Y-%m-%d'),
        datetime.strptime(config['data']['end_date'], '%Y-%m-%d')
    )
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    # Check that CSV is saved
    out_path = f'data/{symbol}_price.csv'
    assert os.path.exists(out_path) 