#!/usr/bin/env python3
"""
Download expanded symbol list for comprehensive ORB testing
Includes forex pairs, popular stocks, and ETFs
"""

import yfinance as yf
import pandas as pd
import os
from datetime import datetime, timedelta
from tqdm import tqdm
import time

def download_expanded_data():
    """Download data for expanded symbol list"""
    
    # Create data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)
    
    # Expanded symbol list
    symbols = {
        # Original symbols
        'AAPL': ('Apple', 'tech'),
        'MSFT': ('Microsoft', 'tech'),
        'GOOGL': ('Google', 'tech'),
        'AMZN': ('Amazon', 'tech'),
        'TSLA': ('Tesla', 'auto'),
        'SPY': ('S&P 500 ETF', 'etf'),
        'QQQ': ('Nasdaq ETF', 'etf'),
        'GLD': ('Gold ETF', 'commodity'),
        'META': ('Meta', 'tech'),
        'NVDA': ('NVIDIA', 'tech'),
        'JPM': ('JP Morgan', 'finance'),
        'BAC': ('Bank of America', 'finance'),
        
        # New tech stocks
        'NFLX': ('Netflix', 'tech'),
        'CRM': ('Salesforce', 'tech'),
        'ADBE': ('Adobe', 'tech'),
        'AMD': ('AMD', 'tech'),
        
        # Consumer stocks
        'DIS': ('Disney', 'consumer'),
        'WMT': ('Walmart', 'consumer'),
        'COST': ('Costco', 'consumer'),
        'KO': ('Coca-Cola', 'consumer'),
        'PEP': ('PepsiCo', 'consumer'),
        'PG': ('Procter & Gamble', 'consumer'),
        
        # Finance & Healthcare
        'V': ('Visa', 'finance'),
        'MA': ('Mastercard', 'finance'),
        'UNH': ('UnitedHealth', 'healthcare'),
        'JNJ': ('Johnson & Johnson', 'healthcare'),
        
        # Energy
        'XOM': ('Exxon Mobil', 'energy'),
        
        # Currency pairs (forex)
        'EURUSD=X': ('EUR/USD', 'forex'),
        'GBPUSD=X': ('GBP/USD', 'forex'),
        'USDJPY=X': ('USD/JPY', 'forex'),
        'AUDUSD=X': ('AUD/USD', 'forex'),
        'USDCAD=X': ('USD/CAD', 'forex'),
        'EURCHF=X': ('EUR/CHF', 'forex'),
        'USDCHF=X': ('USD/CHF', 'forex'),
        'NZDUSD=X': ('NZD/USD', 'forex'),
        
        # Additional ETFs
        'IWM': ('Russell 2000', 'etf'),
        'DIA': ('Dow Jones ETF', 'etf'),
        'VTI': ('Total Market ETF', 'etf'),
        'EEM': ('Emerging Markets', 'etf'),
        'XLF': ('Financial Sector', 'etf'),
        'XLK': ('Technology Sector', 'etf'),
    }
    
    # Download configurations
    download_configs = [
        {'period': '60d', 'interval': '5m', 'suffix': '5m'},
        {'period': '60d', 'interval': '15m', 'suffix': '15m'},
        {'period': '3mo', 'interval': '1h', 'suffix': '1h'},
        {'period': '1y', 'interval': '1d', 'suffix': '1d'}
    ]
    
    print("=" * 80)
    print("DOWNLOADING EXPANDED SYMBOL SET")
    print("=" * 80)
    print(f"\nTotal symbols to download: {len(symbols)}")
    print(f"Categories: Tech, Consumer, Finance, Healthcare, Energy, Forex, ETFs")
    print("-" * 80)
    
    results = []
    successful_downloads = 0
    failed_symbols = []
    
    # Progress bar
    total_downloads = len(symbols) * len(download_configs)
    pbar = tqdm(total=total_downloads, desc="Downloading")
    
    for symbol, (name, category) in symbols.items():
        symbol_success = False
        
        for config in download_configs:
            try:
                ticker = yf.Ticker(symbol)
                data = ticker.history(
                    period=config['period'],
                    interval=config['interval'],
                    auto_adjust=True,
                    prepost=True
                )
                
                if not data.empty:
                    # Save to CSV
                    safe_symbol = symbol.replace('=', '_')
                    filename = f"data/{safe_symbol}_{config['suffix']}.csv"
                    
                    # Add symbol column
                    data['Symbol'] = symbol
                    data.to_csv(filename)
                    
                    if not symbol_success:
                        symbol_success = True
                        successful_downloads += 1
                
                time.sleep(0.1)  # Rate limiting
                
            except Exception as e:
                pass
            
            pbar.update(1)
    
        if not symbol_success:
            failed_symbols.append(symbol)
    
    pbar.close()
    
    # Display summary
    print("\n" + "=" * 80)
    print("DOWNLOAD SUMMARY")
    print("=" * 80)
    
    print(f"\nSuccessful: {successful_downloads}/{len(symbols)} symbols")
    
    if failed_symbols:
        print(f"\nFailed symbols ({len(failed_symbols)}):")
        for symbol in failed_symbols:
            print(f"  - {symbol}")
    
    # Category summary
    categories = {}
    for symbol, (name, category) in symbols.items():
        if symbol not in failed_symbols:
            if category not in categories:
                categories[category] = 0
            categories[category] += 1
    
    print("\nSymbols by category:")
    for cat, count in sorted(categories.items()):
        print(f"  {cat.capitalize():12} {count:3} symbols")
    
    return successful_downloads, failed_symbols

def verify_downloads():
    """Verify which symbols have usable data"""
    print("\n" + "=" * 80)
    print("DATA VERIFICATION")
    print("=" * 80)
    
    # Check for 5m or 15m data (best for ORB)
    data_files = os.listdir('data')
    
    symbols_with_intraday = set()
    
    for file in data_files:
        if '_5m.csv' in file or '_15m.csv' in file:
            symbol = file.split('_')[0]
            # Handle forex symbols
            if 'USD' in symbol or 'EUR' in symbol or 'GBP' in symbol:
                symbol = symbol.replace('_', '=')
            symbols_with_intraday.add(symbol)
    
    print(f"\nSymbols with intraday data: {len(symbols_with_intraday)}")
    
    # Group by type
    stocks = []
    forex = []
    etfs = []
    
    for symbol in sorted(symbols_with_intraday):
        if '=' in symbol:
            forex.append(symbol)
        elif symbol in ['SPY', 'QQQ', 'GLD', 'IWM', 'DIA', 'VTI', 'EEM', 'XLF', 'XLK']:
            etfs.append(symbol)
        else:
            stocks.append(symbol)
    
    print(f"\nBreakdown:")
    print(f"  Stocks: {len(stocks)}")
    print(f"  ETFs: {len(etfs)}")
    print(f"  Forex: {len(forex)}")
    
    if forex:
        print(f"\nForex pairs available: {', '.join(forex)}")
    
    return list(symbols_with_intraday)

def main():
    """Main execution"""
    print("Starting expanded data download...")
    print("This will take a few minutes...")
    
    # Download data
    successful, failed = download_expanded_data()
    
    # Verify downloads
    available_symbols = verify_downloads()
    
    print("\n" + "=" * 80)
    print("DOWNLOAD COMPLETE")
    print("=" * 80)
    print(f"\nReady for backtesting with {len(available_symbols)} symbols")
    print("\nNext step: Run comprehensive backtest")
    print("Command: python backtest_expanded_2x.py")
    
    # Save symbol list
    with open('data/available_symbols.txt', 'w') as f:
        for symbol in available_symbols:
            f.write(f"{symbol}\n")
    
    print("\n✓ Symbol list saved to data/available_symbols.txt")

if __name__ == "__main__":
    main()