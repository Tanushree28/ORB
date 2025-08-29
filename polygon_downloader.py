#!/usr/bin/env python3
"""
Polygon.io Data Downloader for Extended ORB Backtesting
Downloads 2 years of 5-minute historical data with rate limiting for free tier
"""

import os
import yaml
import pandas as pd
import time
from datetime import datetime, timedelta
from polygon import RESTClient
from typing import List, Dict, Optional
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class PolygonDownloader:
    def __init__(self, config_path: str = "configs/polygon_config.yaml"):
        """Initialize Polygon downloader with configuration"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)['polygon']
        
        # Initialize client
        api_key = self.config['api_key']
        if api_key == "YOUR_POLYGON_API_KEY_HERE":
            raise ValueError(
                "\n" + "="*80 + "\n"
                "ERROR: Please set your Polygon API key in configs/polygon_config.yaml\n"
                "Get your free key at: https://polygon.io/dashboard/api-keys\n" +
                "="*80
            )
        
        self.client = RESTClient(api_key=api_key)
        self.rate_limit_delay = self.config['rate_limit']['delay_between_requests']
        
        # Create output directory
        self.output_dir = self.config['output']['directory']
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Track API calls for rate limiting
        self.last_api_call = None
        
    def _rate_limit(self):
        """Enforce rate limiting for free tier (5 requests per minute)"""
        if self.last_api_call:
            elapsed = time.time() - self.last_api_call
            if elapsed < self.rate_limit_delay:
                sleep_time = self.rate_limit_delay - elapsed
                logger.info(f"Rate limiting: sleeping for {sleep_time:.1f} seconds...")
                time.sleep(sleep_time)
        self.last_api_call = time.time()
    
    def download_symbol(self, symbol: str, years: int = 2) -> Optional[pd.DataFrame]:
        """
        Download historical data for a single symbol
        
        Args:
            symbol: Stock ticker or forex pair (e.g., 'AAPL' or 'C:EURUSD')
            years: Number of years of history to download
            
        Returns:
            DataFrame with historical data or None if failed
        """
        try:
            # Calculate date range
            end_date = datetime.now()
            start_date = end_date - timedelta(days=years * 365)
            
            logger.info(f"Downloading {symbol} from {start_date.date()} to {end_date.date()}...")
            
            # Rate limit before API call
            self._rate_limit()
            
            # Fetch aggregates (bars)
            multiplier = self.config['download']['multiplier']
            timespan = self.config['download']['timespan']
            
            bars = []
            for agg in self.client.list_aggs(
                ticker=symbol,
                multiplier=multiplier,
                timespan=timespan,
                from_=start_date.strftime("%Y-%m-%d"),
                to=end_date.strftime("%Y-%m-%d"),
                limit=50000  # Max limit per request
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
            safe_symbol = symbol.replace(':', '_')
            filename = f"{self.output_dir}/{safe_symbol}_5m.csv"
            df.to_csv(filename, index=False)
            logger.info(f"✓ Saved to {filename}")
            
            return df
            
        except Exception as e:
            logger.error(f"✗ Error downloading {symbol}: {str(e)}")
            return None
    
    def download_all_symbols(self) -> Dict[str, pd.DataFrame]:
        """Download data for all configured symbols"""
        results = {}
        
        # Combine all symbols
        all_symbols = []
        all_symbols.extend(self.config['symbols'].get('stocks', []))
        all_symbols.extend(self.config['symbols'].get('forex', []))
        
        total = len(all_symbols)
        years = self.config['download']['years_of_history']
        
        print("\n" + "="*80)
        print("POLYGON.IO DATA DOWNLOAD")
        print("="*80)
        print(f"Downloading {years} years of 5-minute data for {total} symbols")
        print(f"Rate limit: {self.config['rate_limit']['requests_per_minute']} requests/minute")
        print(f"Estimated time: ~{total * self.rate_limit_delay / 60:.1f} minutes")
        print("-"*80)
        
        # Download each symbol
        successful = 0
        failed = []
        
        for i, symbol in enumerate(all_symbols, 1):
            print(f"\n[{i}/{total}] Processing {symbol}...")
            
            df = self.download_symbol(symbol, years)
            if df is not None:
                results[symbol] = df
                successful += 1
            else:
                failed.append(symbol)
        
        # Summary
        print("\n" + "="*80)
        print("DOWNLOAD SUMMARY")
        print("="*80)
        print(f"Successful: {successful}/{total}")
        
        if failed:
            print(f"\nFailed symbols ({len(failed)}):")
            for symbol in failed:
                print(f"  - {symbol}")
        
        # Data statistics
        if results:
            print("\nData Statistics:")
            total_bars = sum(len(df) for df in results.values())
            print(f"  Total bars downloaded: {total_bars:,}")
            
            # Date range
            all_dates = []
            for df in results.values():
                all_dates.extend(df['datetime'].tolist())
            if all_dates:
                min_date = min(all_dates)
                max_date = max(all_dates)
                print(f"  Date range: {min_date.date()} to {max_date.date()}")
        
        return results
    
    def verify_downloads(self) -> List[str]:
        """Verify which symbols have been successfully downloaded"""
        files = os.listdir(self.output_dir)
        symbols = []
        
        for file in files:
            if file.endswith('_5m.csv'):
                # Extract symbol from filename
                symbol = file.replace('_5m.csv', '').replace('_', ':')
                symbols.append(symbol)
        
        return sorted(symbols)

def main():
    """Main execution"""
    print("Starting Polygon.io data download...")
    
    # Check if config exists
    if not os.path.exists("configs/polygon_config.yaml"):
        print("\n" + "="*80)
        print("ERROR: Configuration file not found!")
        print("Please ensure configs/polygon_config.yaml exists with your API key")
        print("="*80)
        return
    
    try:
        # Initialize downloader
        downloader = PolygonDownloader()
        
        # Download all configured symbols
        results = downloader.download_all_symbols()
        
        if results:
            # Save metadata
            metadata = {
                'download_date': datetime.now().isoformat(),
                'symbols': list(results.keys()),
                'total_symbols': len(results),
                'total_bars': sum(len(df) for df in results.values())
            }
            
            metadata_file = f"{downloader.output_dir}/download_metadata.yaml"
            with open(metadata_file, 'w') as f:
                yaml.dump(metadata, f)
            
            print(f"\n✓ Metadata saved to {metadata_file}")
            
            # Create symbol list for backtesting
            symbols_file = f"{downloader.output_dir}/available_symbols.txt"
            with open(symbols_file, 'w') as f:
                for symbol in sorted(results.keys()):
                    f.write(f"{symbol}\n")
            
            print(f"✓ Symbol list saved to {symbols_file}")
            
            print("\n" + "="*80)
            print("DOWNLOAD COMPLETE")
            print("="*80)
            print(f"Data saved to: {downloader.output_dir}/")
            print("\nNext steps:")
            print("1. Run extended backtest: python backtest_extended_polygon.py")
            print("2. Compare with 60-day results: python compare_results.py")
        else:
            print("\n⚠️ No data was downloaded. Please check your API key and connection.")
            
    except Exception as e:
        print(f"\n✗ Error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()