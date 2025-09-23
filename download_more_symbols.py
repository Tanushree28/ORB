#!/usr/bin/env python3
"""
Download additional symbols from Polygon with extended delays
"""

import time
import pandas as pd
from polygon import RESTClient
from datetime import datetime, timedelta
import os

def download_symbol(client, symbol, wait_time=90):
    """Download a single symbol with specified wait time"""
    
    # Check if already exists
    filename = f"data/polygon/{symbol}_5m.csv"
    if os.path.exists(filename):
        print(f"✓ {symbol} already exists, skipping...")
        return True
    
    print(f"\nWaiting {wait_time} seconds before downloading {symbol}...")
    time.sleep(wait_time)
    
    try:
        # Calculate date range - 1 year
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)
        
        print(f"Downloading {symbol}...")
        
        bars = []
        
        for agg in client.list_aggs(
            ticker=symbol,
            multiplier=5,
            timespan="minute",
            from_=start_date.strftime("%Y-%m-%d"),
            to=end_date.strftime("%Y-%m-%d"),
            limit=50000
        ):
            bars.append({
                'timestamp': agg.timestamp,
                'open': agg.open,
                'high': agg.high,
                'low': agg.low,
                'close': agg.close,
                'volume': agg.volume,
                'vwap': getattr(agg, 'vwap', None),
                'transactions': getattr(agg, 'transactions', None)
            })
        
        if bars:
            # Convert to DataFrame
            df = pd.DataFrame(bars)
            df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
            df = df.sort_values('datetime')
            df['symbol'] = symbol
            
            # Reorder columns
            columns = ['datetime', 'symbol', 'open', 'high', 'low', 'close', 'volume', 'vwap', 'transactions']
            df = df[columns]
            
            # Save to CSV
            df.to_csv(filename, index=False)
            print(f"✓ {symbol} downloaded: {len(df)} bars")
            return True
        else:
            print(f"✗ No data received for {symbol}")
            return False
            
    except Exception as e:
        error_msg = str(e)
        if '429' in error_msg or 'rate' in error_msg.lower():
            print(f"✗ Rate limit hit for {symbol}")
        else:
            print(f"✗ Error downloading {symbol}: {error_msg[:100]}")
        return False

def main():
    """Download priority symbols"""
    
    # Initialize client
    api_key = "eLixLfwP5E_K_XjqEieTlub6v32LY5Hv"
    client = RESTClient(api_key=api_key)
    
    # Priority symbols to download
    priority_symbols = [
        'SPY',    # Market benchmark
        'AAPL',   # Top tech stocks
        'MSFT',
        'GOOGL',
        'AMZN',
        'META',
        'NVDA',
        'TSLA'
    ]
    
    print("="*80)
    print("DOWNLOADING ADDITIONAL SYMBOLS FROM POLYGON")
    print("="*80)
    print(f"Attempting to download {len(priority_symbols)} priority symbols")
    print("Using 90-second delays between attempts to avoid rate limits")
    print("-"*80)
    
    os.makedirs('data/polygon', exist_ok=True)
    
    successful = 0
    failed = []
    
    for i, symbol in enumerate(priority_symbols, 1):
        print(f"\n[{i}/{len(priority_symbols)}] Processing {symbol}")
        
        if download_symbol(client, symbol, wait_time=90):
            successful += 1
        else:
            failed.append(symbol)
            # If we hit a rate limit, wait even longer
            print("Waiting extra 2 minutes after rate limit...")
            time.sleep(120)
    
    # Summary
    print("\n" + "="*80)
    print("DOWNLOAD SUMMARY")
    print("="*80)
    print(f"Successful: {successful}/{len(priority_symbols)}")
    
    if failed:
        print(f"\nFailed symbols: {', '.join(failed)}")
    
    # Show all available data
    print("\n" + "="*80)
    print("ALL AVAILABLE POLYGON DATA")
    print("="*80)
    
    files = [f for f in os.listdir('data/polygon') if f.endswith('_5m.csv')]
    print(f"Total symbols with data: {len(files)}")
    
    for file in sorted(files):
        symbol = file.replace('_5m.csv', '')
        df = pd.read_csv(f'data/polygon/{file}')
        print(f"  {symbol:8} - {len(df):6} bars - {df['datetime'].min()[:10]} to {df['datetime'].max()[:10]}")
    
    # Update available symbols
    available_symbols = [f.replace('_5m.csv', '') for f in files]
    with open('data/polygon/available_symbols.txt', 'w') as f:
        for symbol in available_symbols:
            f.write(f"{symbol}\n")
    
    print(f"\n✓ Updated available symbols list: {len(available_symbols)} symbols")

if __name__ == "__main__":
    main()