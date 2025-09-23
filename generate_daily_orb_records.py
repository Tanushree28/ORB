#!/usr/bin/env python3
"""
Generate Daily ORB Trading Records for Verification
Shows ORB levels, TP/SL, and trade outcomes for each day
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime, time
import warnings
warnings.filterwarnings('ignore')

class DailyORBAnalyzer:
    def __init__(self, data_dir="data/polygon", output_dir="reports/daily_orb_records"):
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.tp_multiplier = 2.0  # 2x reward/risk ratio
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
    def load_data(self, symbol):
        """Load 5-minute data for a symbol"""
        filepath = f"{self.data_dir}/{symbol}_5m.csv"
        
        if not os.path.exists(filepath):
            print(f"✗ Data not found for {symbol}")
            return None
            
        df = pd.read_csv(filepath)
        df['datetime'] = pd.to_datetime(df['datetime'])
        df.set_index('datetime', inplace=True)
        
        # Rename columns to match expected format
        df.rename(columns={
            'open': 'Open',
            'high': 'High', 
            'low': 'Low',
            'close': 'Close',
            'volume': 'Volume'
        }, inplace=True)
        
        return df
    
    def analyze_trading_day(self, data, date):
        """Analyze a single trading day"""
        # Get data for this specific date
        day_data = data[data.index.date == date.date()]
        
        if len(day_data) < 10:  # Need enough bars for ORB and trading
            return None
            
        # Filter for market hours (9:30 AM - 4:00 PM EST)
        market_hours = day_data.between_time('09:30', '16:00')
        
        if len(market_hours) < 10:
            return None
            
        # Calculate ORB (9:30-9:45 time range, not just first 3 bars)
        orb_bars = market_hours.between_time('09:30', '09:44')  # 09:44 to include 09:40 bar
        
        if len(orb_bars) == 0:
            return None  # No data in ORB period
            
        orb_high = orb_bars['High'].max()
        orb_low = orb_bars['Low'].min()
        orb_range = orb_high - orb_low
        
        if orb_range <= 0:
            return None
            
        # Calculate entry and exit levels
        long_entry = orb_high
        long_sl = orb_low
        long_tp = orb_high + (self.tp_multiplier * orb_range)
        
        short_entry = orb_low
        short_sl = orb_high
        short_tp = orb_low - (self.tp_multiplier * orb_range)
        
        # Get post-ORB trading data (after 9:45 AM)
        trading_bars = market_hours.between_time('09:45', '16:00')
        
        if len(trading_bars) == 0:
            return None
            
        # Track if trades were triggered and their outcomes
        long_triggered = False
        short_triggered = False
        long_result = "No Trade"
        short_result = "No Trade"
        
        # Track position status
        long_position = False
        short_position = False
        
        # Analyze each bar after ORB
        for idx, bar in trading_bars.iterrows():
            # Check for long entry
            if not long_position and not long_triggered:
                if bar['Close'] > long_entry and bar['Open'] <= long_entry:
                    long_triggered = True
                    long_position = True
            
            # Check for short entry
            if not short_position and not short_triggered:
                if bar['Close'] < short_entry and bar['Open'] >= short_entry:
                    short_triggered = True
                    short_position = True
            
            # Check long position exit
            if long_position:
                if bar['High'] >= long_tp:
                    long_result = "TP"
                    long_position = False
                elif bar['Low'] <= long_sl:
                    long_result = "SL"
                    long_position = False
            
            # Check short position exit
            if short_position:
                if bar['Low'] <= short_tp:
                    short_result = "TP"
                    short_position = False
                elif bar['High'] >= short_sl:
                    short_result = "SL"
                    short_position = False
        
        # If position was opened but not closed
        if long_position:
            long_result = "No Exit"
        if short_position:
            short_result = "No Exit"
            
        # Get day's high and low after ORB
        day_high = trading_bars['High'].max()
        day_low = trading_bars['Low'].min()
        
        return {
            'date': date.date(),
            'orb_high': round(orb_high, 2),
            'orb_low': round(orb_low, 2),
            'orb_range': round(orb_range, 2),
            'long_entry': round(long_entry, 2),
            'long_sl': round(long_sl, 2),
            'long_tp': round(long_tp, 2),
            'short_entry': round(short_entry, 2),
            'short_sl': round(short_sl, 2),
            'short_tp': round(short_tp, 2),
            'long_triggered': long_triggered,
            'short_triggered': short_triggered,
            'long_result': long_result,
            'short_result': short_result,
            'day_high': round(day_high, 2),
            'day_low': round(day_low, 2)
        }
    
    def analyze_symbol(self, symbol):
        """Analyze all trading days for a symbol"""
        print(f"\nAnalyzing {symbol}...")
        
        # Load data
        data = self.load_data(symbol)
        if data is None:
            return None
            
        # Get unique trading days
        trading_days = pd.to_datetime(data.index.date).unique()
        
        # Analyze each day
        daily_records = []
        for date in trading_days:
            record = self.analyze_trading_day(data, pd.Timestamp(date))
            if record:
                daily_records.append(record)
        
        if not daily_records:
            print(f"✗ No valid trading days found for {symbol}")
            return None
            
        # Create DataFrame
        df = pd.DataFrame(daily_records)
        
        # Save to CSV
        output_file = f"{self.output_dir}/{symbol}_daily_orb.csv"
        df.to_csv(output_file, index=False)
        
        # Calculate summary statistics
        total_days = len(df)
        long_trades = df[df['long_triggered'] == True]
        short_trades = df[df['short_triggered'] == True]
        
        long_wins = len(long_trades[long_trades['long_result'] == 'TP'])
        long_losses = len(long_trades[long_trades['long_result'] == 'SL'])
        short_wins = len(short_trades[short_trades['short_result'] == 'TP'])
        short_losses = len(short_trades[short_trades['short_result'] == 'SL'])
        
        total_wins = long_wins + short_wins
        total_losses = long_losses + short_losses
        total_trades = len(long_trades) + len(short_trades)
        
        win_rate = (total_wins / (total_wins + total_losses) * 100) if (total_wins + total_losses) > 0 else 0
        
        print(f"✓ {symbol} analyzed:")
        print(f"  Trading days: {total_days}")
        print(f"  Long trades: {len(long_trades)} (Wins: {long_wins}, Losses: {long_losses})")
        print(f"  Short trades: {len(short_trades)} (Wins: {short_wins}, Losses: {short_losses})")
        print(f"  Overall Win Rate: {win_rate:.1f}%")
        print(f"  Saved to: {output_file}")
        
        return {
            'symbol': symbol,
            'total_days': total_days,
            'long_trades': len(long_trades),
            'short_trades': len(short_trades),
            'long_wins': long_wins,
            'long_losses': long_losses,
            'short_wins': short_wins,
            'short_losses': short_losses,
            'total_wins': total_wins,
            'total_losses': total_losses,
            'win_rate': win_rate,
            'df': df
        }

def main():
    """Main execution"""
    print("="*80)
    print("GENERATING DAILY ORB TRADING RECORDS")
    print("="*80)
    print("\nThis will create detailed daily records for verification of:")
    print("- ORB high/low levels (9:30-9:45 AM)")
    print("- Take Profit and Stop Loss levels")
    print("- Whether trades were triggered")
    print("- Trade outcomes (TP hit, SL hit, or no exit)")
    print("-"*80)
    
    # Initialize analyzer
    analyzer = DailyORBAnalyzer()
    
    # List of symbols to analyze
    symbols = ['AAPL', 'AMZN', 'COST', 'DIA', 'GOOGL', 'META', 'MSFT', 'NFLX', 'QQQ', 'SPY']
    
    # Analyze each symbol
    results = []
    for symbol in symbols:
        result = analyzer.analyze_symbol(symbol)
        if result:
            results.append(result)
    
    # Create summary report
    print("\n" + "="*80)
    print("SUMMARY REPORT")
    print("="*80)
    
    summary_data = []
    for r in results:
        summary_data.append({
            'Symbol': r['symbol'],
            'Days': r['total_days'],
            'Long Trades': r['long_trades'],
            'Short Trades': r['short_trades'],
            'Total Trades': r['long_trades'] + r['short_trades'],
            'Wins': r['total_wins'],
            'Losses': r['total_losses'],
            'Win Rate': f"{r['win_rate']:.1f}%"
        })
    
    summary_df = pd.DataFrame(summary_data)
    print("\n" + summary_df.to_string(index=False))
    
    # Save summary
    summary_file = f"{analyzer.output_dir}/summary_statistics.csv"
    summary_df.to_csv(summary_file, index=False)
    print(f"\n✓ Summary saved to: {summary_file}")
    
    # Compare with backtest results
    print("\n" + "="*80)
    print("VERIFICATION NOTES")
    print("="*80)
    print("\n1. Check individual CSV files in reports/daily_orb_records/")
    print("2. Each file shows every trading day's ORB levels and outcomes")
    print("3. You can manually verify any specific day's trades")
    print("4. Compare win rates with backtest results:")
    
    # Load backtest results for comparison if available
    backtest_file = "reports/extended_backtest_results.csv"
    if os.path.exists(backtest_file):
        backtest_df = pd.read_csv(backtest_file)
        backtest_df = backtest_df[backtest_df['period'] == 'Full 2 Years']
        
        print("\n   Symbol | Daily Records | Backtest | Difference")
        print("   " + "-"*50)
        for r in results:
            symbol = r['symbol']
            backtest_row = backtest_df[backtest_df['symbol'] == symbol]
            if not backtest_row.empty:
                backtest_wr = backtest_row.iloc[0]['win_rate'] * 100
                diff = r['win_rate'] - backtest_wr
                print(f"   {symbol:6} |    {r['win_rate']:5.1f}%    |  {backtest_wr:5.1f}%  | {diff:+5.1f}%")
    
    print("\n✓ Daily ORB records generation complete!")
    print(f"✓ All files saved in: reports/daily_orb_records/")

if __name__ == "__main__":
    main()