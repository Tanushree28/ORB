#!/usr/bin/env python3
"""
Data Downloader for ORB Strategy
Downloads historical price data for various instruments
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz
import yaml
import os
from tqdm import tqdm
import time

class DataDownloader:
    def __init__(self, config_path='configs/config.yaml'):
        """Initialize data downloader with configuration"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.data_dir = 'data'
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Extract all symbols from config
        self.symbols = self._extract_symbols()
        
    def _extract_symbols(self):
        """Extract all trading symbols from config"""
        symbols = []
        
        for category in ['futures', 'forex', 'commodities', 'stocks']:
            if category in self.config['symbols']:
                for item in self.config['symbols'][category]:
                    symbols.append({
                        'symbol': item['symbol'],
                        'name': item['name'],
                        'category': category
                    })
        
        return symbols
    
    def download_data(self, symbol, interval, start_date, end_date):
        """Download historical data for a single symbol"""
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(
                start=start_date,
                end=end_date,
                interval=interval,
                auto_adjust=True,
                prepost=True
            )

            if data.empty:
                print(f"Warning: No data available for {symbol}")
                return None

            # Reset index to make datetime a column
            data.reset_index(inplace=True)

            # Only keep required columns
            expected_cols = ['Datetime', 'Open', 'High', 'Low', 'Close', 'Volume']
            data = data[expected_cols[:len(data.columns) if len(data.columns) < 6 else 6]]  # slice safe
            data['Symbol'] = symbol

            return data

        except Exception as e:
            print(f"Error downloading {symbol}: {str(e)}")
            return None

    
    def save_data(self, data, symbol, interval):
        """Save downloaded data to CSV file"""
        if data is None or data.empty:
            return False
        
        # Create filename
        safe_symbol = symbol.replace('=', '_').replace('^', '_')
        filename = f"{self.data_dir}/{safe_symbol}_{interval}.csv"
        
        # Save to CSV
        data.to_csv(filename, index=False)
        print(f"Saved {len(data)} records to {filename}")
        
        return True
    
    def download_all(self):
        """Download data for all configured symbols and intervals"""
        start_date = self.config['data']['start_date']
        end_date = self.config['data']['end_date']
        intervals = self.config['data']['intervals']
        
        print(f"Downloading data from {start_date} to {end_date}")
        print(f"Intervals: {intervals}")
        print(f"Total symbols: {len(self.symbols)}")
        print("-" * 50)
        
        successful = 0
        failed = []
        
        # Progress bar for all downloads
        total_downloads = len(self.symbols) * len(intervals)
        pbar = tqdm(total=total_downloads, desc="Downloading")
        
        for symbol_info in self.symbols:
            symbol = symbol_info['symbol']
            name = symbol_info['name']
            
            for interval in intervals:
                pbar.set_description(f"Downloading {name} ({symbol}) - {interval}")
                
                # Download data
                data = self.download_data(symbol, interval, start_date, end_date)
                
                # Save data
                if self.save_data(data, symbol, interval):
                    successful += 1
                else:
                    failed.append(f"{symbol}_{interval}")
                
                # Rate limiting to avoid API restrictions
                time.sleep(0.5)
                
                pbar.update(1)
        
        pbar.close()
        
        # Print summary
        print("\n" + "=" * 50)
        print(f"Download Summary:")
        print(f"Successful: {successful}/{total_downloads}")
        
        if failed:
            print(f"Failed downloads: {', '.join(failed)}")
        
        return successful, failed
    
    def verify_data(self):
        """Verify downloaded data files"""
        print("\nVerifying downloaded data...")
        
        for symbol_info in self.symbols:
            symbol = symbol_info['symbol']
            safe_symbol = symbol.replace('=', '_').replace('^', '_')
            
            for interval in self.config['data']['intervals']:
                filename = f"{self.data_dir}/{safe_symbol}_{interval}.csv"
                
                if os.path.exists(filename):
                    df = pd.read_csv(filename)
                    print(f"✓ {symbol} ({interval}): {len(df)} records")
                else:
                    print(f"✗ {symbol} ({interval}): File not found")
    
    def get_data(self, symbol, interval):
        """Load data for a specific symbol and interval"""
        safe_symbol = symbol.replace('=', '_').replace('^', '_')
        filename = f"{self.data_dir}/{safe_symbol}_{interval}.csv"
        
        if not os.path.exists(filename):
            print(f"Data file not found: {filename}")
            return None
        
        df = pd.read_csv(filename)
        df['Datetime'] = pd.to_datetime(df['Datetime'])
        df.set_index('Datetime', inplace=True)
        
        return df


def main():
    """Main function to run data downloader"""
    print("=" * 50)
    print("ORB Strategy Data Downloader")
    print("=" * 50)
    
    # Initialize downloader
    downloader = DataDownloader()
    
    # Download all data
    successful, failed = downloader.download_all()
    
    # Verify downloaded data
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