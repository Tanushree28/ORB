#!/usr/bin/env python3
"""
Download QQQ data from Polygon with extended delays
"""

import time
import pandas as pd
from polygon import RESTClient
from datetime import datetime, timedelta
import os

def download_qqq():
    """Download QQQ with very conservative rate limiting"""
    
    # Initialize client
    api_key = "eLixLfwP5E_K_XjqEieTlub6v32LY5Hv"
    client = RESTClient(api_key=api_key)
    
    symbol = "QQQ"
    
    print("="*80)
    print("DOWNLOADING QQQ FROM POLYGON")
    print("="*80)
    print("Waiting 2 minutes before starting to clear any rate limits...")
    time.sleep(120)  # Wait 2 minutes to clear rate limits
    
    try:
        # Calculate date range - try just 1 year first
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)  # 1 year for now
        
        print(f"\nDownloading QQQ from {start_date.date()} to {end_date.date()}...")
        print("This may take a few minutes due to rate limiting...")
        
        bars = []
        
        # Try to fetch data
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
            
            # Convert timestamp to datetime
            df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
            df = df.sort_values('datetime')
            
            # Add symbol column
            df['symbol'] = symbol
            
            # Reorder columns
            columns = ['datetime', 'symbol', 'open', 'high', 'low', 'close', 'volume', 'vwap', 'transactions']
            df = df[columns]
            
            print(f"\n✓ Downloaded {len(df)} bars for QQQ")
            
            # Save to CSV
            os.makedirs('data/polygon', exist_ok=True)
            filename = f"data/polygon/QQQ_5m.csv"
            df.to_csv(filename, index=False)
            print(f"✓ Saved to {filename}")
            
            # Show sample data
            print("\nSample data (first 5 rows):")
            print(df.head())
            
            print("\nDate range:")
            print(f"  From: {df['datetime'].min()}")
            print(f"  To: {df['datetime'].max()}")
            
            return df
        else:
            print("✗ No data received for QQQ")
            return None
            
    except Exception as e:
        print(f"✗ Error downloading QQQ: {str(e)}")
        
        # If rate limited, suggest alternative
        if '429' in str(e) or 'rate' in str(e).lower():
            print("\nRate limit detected. The free tier may have stricter limits.")
            print("Alternative: Try downloading from Yahoo Finance for 60-day data:")
            print("  python download_expanded_symbols.py")
        
        return None

def check_existing_data():
    """Check what data we already have"""
    polygon_dir = "data/polygon"
    
    print("\n" + "="*80)
    print("EXISTING POLYGON DATA")
    print("="*80)
    
    if os.path.exists(polygon_dir):
        files = [f for f in os.listdir(polygon_dir) if f.endswith('_5m.csv')]
        
        if files:
            print(f"Found {len(files)} symbols with data:")
            for file in sorted(files):
                symbol = file.replace('_5m.csv', '')
                filepath = os.path.join(polygon_dir, file)
                df = pd.read_csv(filepath)
                print(f"  {symbol:8} - {len(df):6} bars")
        else:
            print("No data files found")
    else:
        print("Polygon data directory not found")

if __name__ == "__main__":
    # Check existing data
    check_existing_data()
    
    # Check if QQQ already exists
    if os.path.exists("data/polygon/QQQ_5m.csv"):
        print("\n✓ QQQ data already exists!")
        df = pd.read_csv("data/polygon/QQQ_5m.csv")
        print(f"  Contains {len(df)} bars")
    else:
        # Try to download QQQ
        print("\nAttempting to download QQQ...")
        download_qqq()