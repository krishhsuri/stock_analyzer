import os
import pandas as pd
import yaml
import requests
from bs4 import BeautifulSoup

# Parameters and their corresponding tables
PROFIT_LOSS_PARAMS = [
    'Sales',
    'Operating Profit',
    'OPM %',
    'Net Profit',
    'EPS in Rs',
    'Interest',
    'Depreciation',
]
BALANCE_SHEET_PARAMS = [
    'Borrowings',
    'Reserves',
    'Total Liabilities',
    'Total Assets',
]
CASH_FLOW_PARAMS = [
    'Cash from Operating Activity',
    'Cash from Investing Activity',
    'Cash from Financing Activity',
]

SECTIONS = [
    ('profit-loss', PROFIT_LOSS_PARAMS),
    ('balance-sheet', BALANCE_SHEET_PARAMS),
    ('cash-flow', CASH_FLOW_PARAMS),
]

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Accept-Language': 'en-US,en;q=0.9',
}

def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

CONFIG_PATH = os.path.join(os.path.dirname(__file__), '../../config/config.yaml')
config = load_config(CONFIG_PATH)

def scrape_table(soup, section_id, params):
    table = soup.select_one(f'#{section_id} > div.responsive-holder.fill-card-width > table')
    if not table:
        print(f"Table not found for section {section_id}")
        return {}, []
    # Get years from header
    header = [th.get_text(strip=True) for th in table.find_all('th')]
    years = header[1:]  # skip first column (parameter name)
    data = {}
    for row in table.find_all('tr'):
        cells = [td.get_text(strip=True) for td in row.find_all(['th', 'td'])]
        if not cells or len(cells) < 2:
            continue
        first_td = row.find(['th', 'td'])
        if first_td:
            button = first_td.find('button')
            if button:
                param = button.get_text(strip=True).replace('+', '').strip()
            else:
                param = first_td.get_text(strip=True)
        else:
            param = ''
        if param in params:
            data[param] = cells[1:len(years)+1]
    return data, years

def scrape_fundamentals(symbol, screener_url, out_path):
    resp = requests.get(screener_url, headers=HEADERS)
    if resp.status_code != 200:
        print(f"Failed to fetch {screener_url} for {symbol}")
        return
    soup = BeautifulSoup(resp.text, 'html.parser')
    all_data = {}
    all_years = None
    for section_id, params in SECTIONS:
        data, years = scrape_table(soup, section_id, params)
        if all_years is None and years:
            all_years = years
        all_data.update(data)
    # Only keep the parameters in the order specified
    all_params = PROFIT_LOSS_PARAMS + BALANCE_SHEET_PARAMS + CASH_FLOW_PARAMS
    filtered = {p: all_data.get(p, [''] * (len(all_years) if all_years else 0)) for p in all_params}
    if all_years is None:
        all_years = []
    df = pd.DataFrame(filtered).T
    df.columns = all_years
    # Save
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path)
    print(f"Saved fundamentals for {symbol} to {out_path}")

def run_scraper():
    for symbol_entry in config['data']['symbols']:
        symbol = symbol_entry['symbol']
        screener_url = symbol_entry.get('screener_url')
        if not screener_url:
            screener_url = f'https://www.screener.in/company/{symbol.replace(".NS", "")}/consolidated/'
        folder = symbol.replace('.NS', '').replace('.BSE', '').upper()
        out_path = os.path.join(os.path.dirname(__file__), f'../../data/{folder}/fundamental_data/fundamentals.csv')
        scrape_fundamentals(symbol, screener_url, out_path)

if __name__ == '__main__':
    run_scraper() 