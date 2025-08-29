#!/usr/bin/env python3
"""
Data Downloader for ORB Strategy
Downloads historical price data for various instruments
"""

import yfinance as yf
import pandas as pd
from datetime import datetime
import yaml
import os
from tqdm import tqdm
import time


class DataDownloader:
    def __init__(self, config_path='configs/config.yaml'):
        """Initialize data downloader with configuration"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # Ensure data directory exists
        self.data_dir = os.path.join(os.getcwd(), 'data')
        os.makedirs(self.data_dir, exist_ok=True)

        # Extract all symbols from config
        self.symbols = self._extract_symbols()

    def _extract_symbols(self):
        """Extract all trading symbols from config"""
        symbols = []
        for category in ['futures', 'forex', 'commodities', 'stocks']:
            if category in self.config.get('symbols', {}):
                for item in self.config['symbols'][category]:
                    symbols.append({
                        'symbol': item['symbol'],
                        'name': item['name'],
                        'category': category
                    })
        return symbols

    def _outfile(self, symbol, interval):
        """Build safe output filename"""
        safe_symbol = symbol.replace('=', '_').replace('^', '_')
        return os.path.join(self.data_dir, f"{safe_symbol}_{interval}.csv")

    def download_data(self, symbol, interval, start_date, end_date):
        """
        Download historical data for a single symbol.

        For intraday (5m/15m), Yahoo only provides ~60 days:
        use period="60d" to avoid start/end errors.
        """
        try:
            ticker = yf.Ticker(symbol)

            if interval in ("5m", "15m"):
                # Always fetch the latest 60 days for intraday
                print(f"[{symbol} {interval}] Using period='60d' (Yahoo intraday limit).")
                data = ticker.history(
                    period="60d",
                    interval=interval,
                    auto_adjust=True,
                    prepost=True
                )
            else:
                # For higher intervals, allow full start/end from config
                print(f"[{symbol} {interval}] Using start={start_date}, end={end_date}.")
                data = ticker.history(
                    start=start_date,
                    end=end_date,
                    interval=interval,
                    auto_adjust=True,
                    prepost=True
                )

            if data is None or data.empty:
                print(f"Warning: No data returned for {symbol} ({interval}).")
                return None

            # Normalize index & columns
            data = data.copy()
            data.index = pd.to_datetime(data.index, utc=True)  # tz-aware UTC
            data['Symbol'] = symbol
            data.reset_index(inplace=True)
            # yfinance index name can be Datetime/Date; normalize to 'Datetime'
            if 'Date' in data.columns and 'Datetime' not in data.columns:
                data.rename(columns={'Date': 'Datetime'}, inplace=True)
            else:
                data.rename(columns={'index': 'Datetime'}, inplace=True)
            # Standard OHLCV + Symbol
            rename_map = {
                'Open': 'Open', 'High': 'High', 'Low': 'Low',
                'Close': 'Close', 'Volume': 'Volume'
            }
            data.rename(columns=rename_map, inplace=True)
            data = data[['Datetime', 'Open', 'High', 'Low', 'Close', 'Volume', 'Symbol']]

            # Basic sanity
            if data['Datetime'].isna().all():
                print(f"Warning: Datetime column empty for {symbol} {interval}.")
                return None

            return data

        except Exception as e:
            print(f"Error downloading {symbol} ({interval}): {e}")
            return None

    def save_data(self, data, symbol, interval):
        """Save downloaded data to CSV file"""
        if data is None or data.empty:
            return False
        out_path = self._outfile(symbol, interval)
        data.to_csv(out_path, index=False)
        print(f"Saved {len(data)} records to {out_path}")
        return True

    def download_all(self):
        """Download data for all configured symbols and intervals"""
        cfg_data = self.config.get('data', {})
        start_date = cfg_data.get('start_date')
        end_date = cfg_data.get('end_date')
        intervals = cfg_data.get('intervals', ['5m'])

        print(f"Downloading data from {start_date} to {end_date}")
        print(f"Intervals: {intervals}")
        print(f"Total symbols: {len(self.symbols)}")
        print("-" * 50)

        successful = 0
        failed = []

        total_downloads = len(self.symbols) * len(intervals)
        pbar = tqdm(total=total_downloads, desc="Downloading")

        for symbol_info in self.symbols:
            symbol = symbol_info['symbol']
            name = symbol_info['name']

            for interval in intervals:
                pbar.set_description(f"Downloading {name} ({symbol}) - {interval}")

                data = self.download_data(symbol, interval, start_date, end_date)

                if self.save_data(data, symbol, interval):
                    successful += 1
                else:
                    failed.append(f"{symbol}_{interval}")

                # Gentle pacing
                time.sleep(0.4)
                pbar.update(1)

        pbar.close()

        print("\n" + "=" * 50)
        print("Download Summary:")
        print(f"Successful: {successful}/{total_downloads}")
        if failed:
            print(f"Failed downloads: {', '.join(failed)}")

        return successful, failed

    def verify_data(self):
        """Verify downloaded data files"""
        print("\nVerifying downloaded data...")
        intervals = self.config.get('data', {}).get('intervals', ['5m'])

        for symbol_info in self.symbols:
            symbol = symbol_info['symbol']
            for interval in intervals:
                path = self._outfile(symbol, interval)
                if os.path.exists(path):
                    try:
                        df = pd.read_csv(path)
                        if not df.empty and 'Datetime' in df.columns:
                            first = pd.to_datetime(df['Datetime'].iloc[0], utc=True, errors='coerce')
                            last = pd.to_datetime(df['Datetime'].iloc[-1], utc=True, errors='coerce')
                            print(f"✓ {symbol} ({interval}): {len(df)} records | {first} → {last}")
                        else:
                            print(f"⚠ {symbol} ({interval}): file exists but empty or malformed")
                    except Exception as e:
                        print(f"⚠ {symbol} ({interval}): error reading file ({e})")
                else:
                    print(f"✗ {symbol} ({interval}): File not found")

    def get_data(self, symbol, interval):
        """Load data for a specific symbol and interval"""
        path = self._outfile(symbol, interval)
        if not os.path.exists(path):
            print(f"Data file not found: {path}")
            return None

        df = pd.read_csv(path)
        df['Datetime'] = pd.to_datetime(df['Datetime'], utc=True, errors='coerce')
        df.set_index('Datetime', inplace=True)
        return df


def main():
    """Main function to run data downloader standalone"""
    print("=" * 50)
    print("ORB Strategy Data Downloader")
    print("=" * 50)

    downloader = DataDownloader()

    successful, failed = downloader.download_all()
    downloader.verify_data()

    print("\nData download complete!")

    # Test loading data
    print("\nTesting data loading...")
    test_data = downloader.get_data('AAPL', '5m')
    if test_data is not None:
        print(f"Successfully loaded AAPL 5m data: {len(test_data)} records")
        print(f"Date range: {test_data.index[0]} to {test_data.index[-1]}")


if __name__ == "__main__":
    main()
