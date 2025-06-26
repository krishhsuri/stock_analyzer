import subprocess
import sys

steps = [
    ("Fetching and saving price data...", "src/data/data_pipeline.py"),
    ("Augmenting with dividend data...", "src/data/augment_with_dividends.py"),
    ("Scraping fundamental data...", "src/data/fundamental_scraper.py"),
    ("Caculating stats ...", "src/data/stock_stats.py"),
    ("Calculating technical indicators...", "src/analysis/technical_indicators.py"),
    ("Calculating returns and volatility...", "src/analysis/returns_volatility.py"),
    ("Calculating volume indicators...", "src/analysis/volume_indicators.py"),
    ("Merging and cleaning all features...", "src/analysis/merge_features.py")
]

def run_pipeline():
    for msg, script in steps:
        print(f"\n=== {msg} ===")
        result = subprocess.run([sys.executable, script], capture_output=True, text=True)
        print(result.stdout)
        if result.returncode != 0:
            print(f"Error running {script}:")
            print(result.stderr)
            sys.exit(result.returncode)
    print("\nAll data pipelines completed successfully.")

if __name__ == "__main__":
    run_pipeline() 