#!/usr/bin/env python3
"""
Robust Polygon.io Data Downloader with Enhanced Rate Limiting
Handles 429 errors gracefully and downloads all available symbols
"""

import os
import yaml
import pandas as pd
import time
from datetime import datetime, timedelta
from polygon import RESTClient
from typing import List, Dict, Optional
import logging
import json

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class RobustPolygonDownloader:
    def __init__(self, config_path: str = "configs/polygon_config.yaml"):
        """Initialize Polygon downloader with configuration"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)['polygon']
        
        # Initialize client
        api_key = self.config['api_key']
        self.client = RESTClient(api_key=api_key)
        
        # Enhanced rate limiting settings
        self.base_delay = 15  # More conservative base delay
        self.retry_delay = 60  # Delay after rate limit error
        self.max_retries = 3  # Maximum retries per symbol
        
        # Create output directory
        self.output_dir = self.config['output']['directory']
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Track progress
        self.progress_file = f"{self.output_dir}/download_progress.json"
        self.completed_symbols = self.load_progress()
        
    def load_progress(self) -> set:
        """Load previously completed symbols"""
        if os.path.exists(self.progress_file):
            with open(self.progress_file, 'r') as f:
                data = json.load(f)
                return set(data.get('completed', []))
        return set()
    
    def save_progress(self):
        """Save download progress"""
        with open(self.progress_file, 'w') as f:
            json.dump({
                'completed': list(self.completed_symbols),
                'timestamp': datetime.now().isoformat()
            }, f, indent=2)
    
    def download_symbol_with_retry(self, symbol: str, years: int = 2) -> Optional[pd.DataFrame]:
        """
        Download historical data for a single symbol with retry logic
        
        Args:
            symbol: Stock ticker or forex pair (e.g., 'AAPL' or 'C:EURUSD')
            years: Number of years of history to download
            
        Returns:
            DataFrame with historical data or None if failed
        """
        # Skip if already downloaded
        safe_symbol = symbol.replace(':', '_')
        existing_file = f"{self.output_dir}/{safe_symbol}_5m.csv"
        if os.path.exists(existing_file):
            logger.info(f"✓ {symbol} already downloaded, skipping...")
            self.completed_symbols.add(symbol)
            return pd.read_csv(existing_file)
        
        for attempt in range(self.max_retries):
            try:
                # Calculate date range
                end_date = datetime.now()
                start_date = end_date - timedelta(days=years * 365)
                
                logger.info(f"Downloading {symbol} (attempt {attempt + 1}/{self.max_retries})...")
                
                # Wait before API call
                time.sleep(self.base_delay)
                
                # Fetch aggregates (bars)
                multiplier = self.config['download']['multiplier']
                timespan = self.config['download']['timespan']
                
                bars = []
                
                # Use the list_aggs method with proper error handling
                try:
                    for agg in self.client.list_aggs(
                        ticker=symbol,
                        multiplier=multiplier,
                        timespan=timespan,
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
                except Exception as api_error:
                    if '429' in str(api_error):
                        logger.warning(f"Rate limit hit for {symbol}, waiting {self.retry_delay}s...")
                        time.sleep(self.retry_delay)
                        continue
                    else:
                        raise api_error
                
                if not bars:
                    logger.warning(f"No data received for {symbol}")
                    return None
                
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
                
                logger.info(f"✓ Downloaded {len(df)} bars for {symbol}")
                
                # Save to CSV
                df.to_csv(existing_file, index=False)
                logger.info(f"✓ Saved to {existing_file}")
                
                # Mark as completed
                self.completed_symbols.add(symbol)
                self.save_progress()
                
                return df
                
            except Exception as e:
                error_msg = str(e)
                
                if '429' in error_msg or 'rate' in error_msg.lower():
                    logger.warning(f"Rate limit error for {symbol}, waiting {self.retry_delay}s before retry...")
                    time.sleep(self.retry_delay)
                elif attempt == self.max_retries - 1:
                    logger.error(f"✗ Failed to download {symbol} after {self.max_retries} attempts: {error_msg}")
                    return None
                else:
                    logger.warning(f"Error on attempt {attempt + 1} for {symbol}: {error_msg}")
                    time.sleep(self.base_delay)
        
        return None
    
    def download_priority_symbols(self) -> Dict[str, pd.DataFrame]:
        """Download symbols with priority order"""
        results = {}
        
        # Priority list - QQQ first, then other top performers
        priority_symbols = [
            'QQQ',    # Specifically requested
            'SPY',    # Market benchmark
            'AAPL',   # Top tech stocks
            'MSFT',
            'GOOGL',
            'AMZN',
            'META',
            'NVDA',
            'ADBE',   # High performers from 60-day test
            'PEP',
            'XLF',
            'AMD',
            'BAC',
            'UNH',
            'TSLA',
            'JPM',
            'V',
            'MA',
            'DIS',
            'WMT',
            'XOM',
            'CRM',
            'KO',
            'PG',
            'JNJ',
            'IWM',   # ETFs
            'EEM',
            'XLK',
            'VTI',
            'GLD'
        ]
        
        # Add remaining symbols from available list
        with open('data/available_symbols.txt', 'r') as f:
            all_symbols = [line.strip() for line in f.readlines() if line.strip()]
        
        # Combine lists, maintaining priority order
        symbols_to_download = []
        for symbol in priority_symbols:
            if symbol in all_symbols and symbol not in self.completed_symbols:
                symbols_to_download.append(symbol)
        
        # Add any remaining symbols
        for symbol in all_symbols:
            if symbol not in symbols_to_download and symbol not in self.completed_symbols:
                symbols_to_download.append(symbol)
        
        total = len(symbols_to_download)
        years = self.config['download']['years_of_history']
        
        print("\n" + "="*80)
        print("ROBUST POLYGON DATA DOWNLOAD")
        print("="*80)
        print(f"Downloading {years} years of 5-minute data")
        print(f"Symbols to download: {total}")
        print(f"Already completed: {len(self.completed_symbols)}")
        print(f"Estimated time: ~{total * self.base_delay / 60:.1f} minutes")
        print("-"*80)
        
        # Download each symbol
        successful = 0
        failed = []
        
        for i, symbol in enumerate(symbols_to_download, 1):
            print(f"\n[{i}/{total}] Processing {symbol}...")
            
            df = self.download_symbol_with_retry(symbol, years)
            if df is not None:
                results[symbol] = df
                successful += 1
                print(f"✓ {symbol} downloaded successfully ({len(df)} bars)")
            else:
                failed.append(symbol)
                print(f"✗ {symbol} failed to download")
        
        # Summary
        print("\n" + "="*80)
        print("DOWNLOAD SUMMARY")
        print("="*80)
        print(f"Successful: {successful}/{total}")
        print(f"Previously downloaded: {len(self.completed_symbols) - successful}")
        print(f"Total available: {len(self.completed_symbols)}")
        
        if failed:
            print(f"\nFailed symbols ({len(failed)}):")
            for symbol in failed[:10]:  # Show first 10
                print(f"  - {symbol}")
            if len(failed) > 10:
                print(f"  ... and {len(failed) - 10} more")
        
        return results

def verify_downloads():
    """Verify all downloaded files and create summary"""
    data_dir = "data/polygon"
    files = [f for f in os.listdir(data_dir) if f.endswith('_5m.csv')]
    
    print("\n" + "="*80)
    print("VERIFICATION")
    print("="*80)
    
    symbols_data = []
    for file in sorted(files):
        filepath = os.path.join(data_dir, file)
        df = pd.read_csv(filepath)
        symbol = file.replace('_5m.csv', '').replace('_', ':')
        
        if len(df) > 0:
            date_range = f"{df['datetime'].min()[:10]} to {df['datetime'].max()[:10]}"
            symbols_data.append({
                'Symbol': symbol,
                'Bars': len(df),
                'Date Range': date_range
            })
    
    print(f"\nSuccessfully downloaded: {len(symbols_data)} symbols")
    
    # Show summary
    for data in symbols_data[:10]:  # Show first 10
        print(f"  {data['Symbol']:8} - {data['Bars']:6} bars - {data['Date Range']}")
    
    if len(symbols_data) > 10:
        print(f"  ... and {len(symbols_data) - 10} more symbols")
    
    # Update available symbols file
    available_symbols = [d['Symbol'] for d in symbols_data]
    with open(f"{data_dir}/available_symbols.txt", 'w') as f:
        for symbol in available_symbols:
            f.write(f"{symbol}\n")
    
    print(f"\n✓ Updated available symbols list: {len(available_symbols)} symbols")
    
    return available_symbols

def main():
    """Main execution"""
    print("Starting robust Polygon.io data download...")
    print("This process will handle rate limiting gracefully.")
    
    try:
        # Initialize downloader
        downloader = RobustPolygonDownloader()
        
        # Download priority symbols
        results = downloader.download_priority_symbols()
        
        # Verify downloads
        available_symbols = verify_downloads()
        
        print("\n" + "="*80)
        print("DOWNLOAD COMPLETE")
        print("="*80)
        print(f"Total symbols available: {len(available_symbols)}")
        
        # Check if QQQ was downloaded
        if 'QQQ' in available_symbols:
            print("\n✓ QQQ successfully downloaded!")
        else:
            print("\n⚠️ QQQ download failed - may need manual retry")
        
        print("\nNext steps:")
        print("1. Run extended backtest: python backtest_extended_polygon.py")
        print("2. Analyze QQQ specifically: python analyze_qqq_polygon.py")
        
    except Exception as e:
        print(f"\n✗ Error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()