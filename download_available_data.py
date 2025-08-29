#!/usr/bin/env python3
"""
Download available historical data for ORB strategy
Saves data to the data/ folder
"""

import yfinance as yf
import pandas as pd
import os
from datetime import datetime, timedelta
from tqdm import tqdm
import time

def download_and_save_data():
    """Download available data for all symbols"""
    
    # Create data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)
    
    # Symbols to download
    symbols = {
        'AAPL': 'Apple',
        'MSFT': 'Microsoft', 
        'GOOGL': 'Google',
        'AMZN': 'Amazon',
        'TSLA': 'Tesla',
        'SPY': 'S&P 500 ETF',
        'QQQ': 'Nasdaq ETF',
        'GLD': 'Gold ETF',
        'META': 'Meta',
        'NVDA': 'NVIDIA',
        'JPM': 'JP Morgan',
        'BAC': 'Bank of America'
    }
    
    # Time periods and intervals to try
    download_configs = [
        {'period': '60d', 'interval': '5m', 'suffix': '5m'},
        {'period': '60d', 'interval': '15m', 'suffix': '15m'},
        {'period': '3mo', 'interval': '30m', 'suffix': '30m'},
        {'period': '3mo', 'interval': '1h', 'suffix': '1h'},
        {'period': '1y', 'interval': '1d', 'suffix': '1d'}
    ]
    
    print("=" * 80)
    print("DOWNLOADING MARKET DATA")
    print("=" * 80)
    print(f"\nDownloading data for {len(symbols)} symbols...")
    print("This may take a few minutes...\n")
    
    results = []
    total_files = 0
    
    # Progress bar for all downloads
    total_downloads = len(symbols) * len(download_configs)
    pbar = tqdm(total=total_downloads, desc="Overall Progress")
    
    for symbol, name in symbols.items():
        symbol_results = []
        
        for config in download_configs:
            try:
                # Create ticker object
                ticker = yf.Ticker(symbol)
                
                # Download data
                data = ticker.history(
                    period=config['period'],
                    interval=config['interval'],
                    auto_adjust=True,
                    prepost=True
                )
                
                if not data.empty:
                    # Save to CSV
                    filename = f"data/{symbol}_{config['suffix']}.csv"
                    
                    # Add symbol column
                    data['Symbol'] = symbol
                    
                    # Save with index (datetime)
                    data.to_csv(filename)
                    
                    symbol_results.append({
                        'interval': config['interval'],
                        'records': len(data),
                        'start': data.index[0],
                        'end': data.index[-1],
                        'filename': filename
                    })
                    
                    total_files += 1
                
                # Rate limiting
                time.sleep(0.1)
                
            except Exception as e:
                pass
            
            pbar.update(1)
        
        if symbol_results:
            results.append({
                'symbol': symbol,
                'name': name,
                'downloads': symbol_results
            })
    
    pbar.close()
    
    # Display results
    print("\n" + "=" * 80)
    print("DOWNLOAD SUMMARY")
    print("=" * 80)
    
    for result in results:
        print(f"\n{result['symbol']} ({result['name']}):")
        for download in result['downloads']:
            print(f"  ✓ {download['interval']:5} - {download['records']:5} records | "
                  f"{download['start'].strftime('%Y-%m-%d')} to {download['end'].strftime('%Y-%m-%d')}")
    
    print(f"\n" + "-" * 80)
    print(f"Total files saved: {total_files}")
    print(f"Location: /Users/kar/Desktop/ORB/data/")
    
    # Create a summary file
    summary_data = []
    for result in results:
        for download in result['downloads']:
            summary_data.append({
                'Symbol': result['symbol'],
                'Name': result['name'],
                'Interval': download['interval'],
                'Records': download['records'],
                'Start Date': download['start'].strftime('%Y-%m-%d %H:%M'),
                'End Date': download['end'].strftime('%Y-%m-%d %H:%M'),
                'Filename': download['filename']
            })
    
    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv('data/download_summary.csv', index=False)
        print("\n✓ Download summary saved to data/download_summary.csv")
    
    # List all files in data directory
    print("\n" + "=" * 80)
    print("FILES IN DATA FOLDER")
    print("=" * 80)
    
    data_files = os.listdir('data')
    data_files.sort()
    
    # Group by symbol
    symbol_files = {}
    for file in data_files:
        if file.endswith('.csv') and '_' in file:
            symbol = file.split('_')[0]
            if symbol not in symbol_files:
                symbol_files[symbol] = []
            symbol_files[symbol].append(file)
    
    for symbol, files in symbol_files.items():
        print(f"\n{symbol}:")
        for file in files:
            # Get file size
            file_path = f"data/{file}"
            file_size = os.path.getsize(file_path) / 1024  # KB
            
            # Get record count
            try:
                df = pd.read_csv(file_path)
                record_count = len(df)
                print(f"  • {file:20} ({record_count:5} records, {file_size:6.1f} KB)")
            except:
                print(f"  • {file:20} ({file_size:6.1f} KB)")
    
    return results

def verify_data_quality():
    """Verify the downloaded data quality"""
    print("\n" + "=" * 80)
    print("DATA QUALITY CHECK")
    print("=" * 80)
    
    # Check for best available intervals for ORB strategy
    ideal_intervals = ['5m', '15m', '30m', '1h']
    
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'SPY', 'QQQ', 'GLD']
    
    print("\nChecking for ORB-suitable data (intraday intervals):")
    print("-" * 60)
    
    orb_ready = []
    
    for symbol in symbols:
        available = []
        for interval in ideal_intervals:
            file_path = f"data/{symbol}_{interval}.csv"
            if os.path.exists(file_path):
                try:
                    df = pd.read_csv(file_path)
                    if len(df) > 100:  # Need sufficient data
                        available.append(interval)
                except:
                    pass
        
        if available:
            orb_ready.append(symbol)
            print(f"{symbol:6} ✓ Available intervals: {', '.join(available)}")
        else:
            print(f"{symbol:6} ✗ No intraday data available")
    
    print(f"\n{len(orb_ready)}/{len(symbols)} symbols have ORB-suitable data")
    
    if orb_ready:
        print("\nReady for ORB backtesting:", ", ".join(orb_ready))
    
    return orb_ready

def main():
    """Main function"""
    print("Starting data download...")
    print("Note: Some intervals may not be available for all symbols")
    print("-" * 80)
    
    # Download data
    results = download_and_save_data()
    
    # Verify data quality
    orb_ready = verify_data_quality()
    
    print("\n" + "=" * 80)
    print("DATA DOWNLOAD COMPLETE")
    print("=" * 80)
    print("\nNext steps:")
    print("1. Run backtesting with: python main.py --backtest")
    print("2. Or run optimization with: python optimize_and_retest.py")
    print("3. Check data/ folder for all downloaded files")
    
    return results, orb_ready

if __name__ == "__main__":
    main()