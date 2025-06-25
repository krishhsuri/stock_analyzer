import os
import pandas as pd
import yaml
import time
import re
import requests
from bs4 import BeautifulSoup, Tag

def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

config = load_config(os.path.join(os.path.dirname(__file__), '../../config/config.yaml'))

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Accept-Language': 'en-US,en;q=0.9',
}

def get_dividends_trendlyne(url):
    print(f"Scraping dividends from Trendlyne: {url}")
    resp = requests.get(url, headers=HEADERS)
    if resp.status_code != 200:
        print(f"Failed to fetch data: {resp.status_code}")
        return pd.DataFrame()
    soup = BeautifulSoup(resp.text, 'html.parser')
    table = soup.find('table', class_='table')
    if not isinstance(table, Tag):
        print("No dividend table found.")
        return pd.DataFrame()
    tbody = table.find('tbody')
    if isinstance(tbody, Tag):
        rows = [row for row in tbody.find_all('tr') if isinstance(row, Tag)]
    else:
        rows = [row for row in table.find_all('tr') if isinstance(row, Tag)]
    records = []
    for row in rows:
        cols = [td.get_text(strip=True) for td in row.find_all('td')]
        if len(cols) < 2:
            continue
        ex_date = cols[0]
        amount = cols[1]
        try:
            amount = float(amount)
        except Exception:
            amount = 0.0
        records.append({'Date': ex_date, 'Dividend': amount})
    if not records:
        print("No dividends found in table.")
        return pd.DataFrame()
    df = pd.DataFrame(records)
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    return df[['Date', 'Dividend']]

def augment_with_dividends():
    for symbol_entry in config['data']['symbols']:
        symbol = symbol_entry['symbol']
        trendlyne_url = symbol_entry['trendlyne_dividend_url']
        symbol_folder = symbol.replace('.NS', '').replace('.BSE', '').upper()
        in_dir = os.path.join(os.path.dirname(__file__), f'../../data/{symbol_folder}/unclean_timedata')
        os.makedirs(in_dir, exist_ok=True)
        price_path = os.path.join(in_dir, f'{symbol}_price.csv')
        if not os.path.exists(price_path):
            print(f"Price file not found for {symbol}, skipping.")
            continue
        df = pd.read_csv(price_path)
        if 'Dividend' not in df.columns:
            df['Dividend'] = 0.0
        df['Date'] = pd.to_datetime(df['Date'])
        div_df = get_dividends_trendlyne(trendlyne_url)
        if not div_df.empty:
            div_map = dict(zip(div_df['Date'], div_df['Dividend']))
            df['Dividend'] = df['Date'].map(lambda d: div_map.get(d, 0.0))
        df.to_csv(price_path, index=False)
        print(f"Updated {symbol} with dividend info.")
        time.sleep(1)  # Be polite to Trendlyne
    print("Dividend augmentation complete.")

if __name__ == '__main__':
    augment_with_dividends() 