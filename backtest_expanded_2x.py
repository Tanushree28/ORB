#!/usr/bin/env python3
"""
Comprehensive ORB Strategy Backtest with Expanded Symbol Set
Tests 40+ symbols including forex, stocks, and ETFs with 2x TP
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
import warnings
warnings.filterwarnings('ignore')

class ExpandedORBBacktest:
    def __init__(self, initial_capital=10000):
        self.initial_capital = initial_capital
        self.tp_multiplier = 2.0  # Fixed at 2x
        self.risk_per_trade = 0.01  # 1% risk
        
    def load_data(self, symbol, interval='5m'):
        """Load data from CSV files"""
        # Handle forex symbols
        safe_symbol = symbol.replace('=', '_')
        filepath = f'data/{safe_symbol}_{interval}.csv'
        
        if not os.path.exists(filepath):
            # Try 15m if 5m not available
            filepath = f'data/{safe_symbol}_15m.csv'
        if not os.path.exists(filepath):
            # Try 1h if others not available
            filepath = f'data/{safe_symbol}_1h.csv'
        if not os.path.exists(filepath):
            return None
            
        df = pd.read_csv(filepath)
        df['Datetime'] = pd.to_datetime(df['Datetime'])
        df.set_index('Datetime', inplace=True)
        return df
    
    def calculate_orb(self, data, date):
        """Calculate Opening Range for a specific date"""
        day_data = data[data.index.date == date.date()]
        
        if len(day_data) < 3:
            return None
            
        # Get first 3 bars for ORB
        orb_data = day_data.iloc[:3]
        
        orb_high = orb_data['High'].max()
        orb_low = orb_data['Low'].min()
        orb_range = orb_high - orb_low
        
        if orb_range <= 0:
            return None
            
        return {
            'high': orb_high,
            'low': orb_low,
            'range': orb_range,
            'date': date
        }
    
    def backtest_symbol(self, symbol):
        """Run backtest for a single symbol"""
        data = self.load_data(symbol)
        if data is None:
            return None
            
        trades = []
        capital = self.initial_capital
        
        # Get unique trading days
        trading_days = pd.to_datetime(data.index.date).unique()
        
        for date in trading_days:
            orb = self.calculate_orb(data, pd.Timestamp(date))
            if orb is None:
                continue
                
            day_data = data[data.index.date == date.date()]
            trading_data = day_data.iloc[3:]  # Skip ORB period
            
            if len(trading_data) == 0:
                continue
                
            position = None
            for idx, bar in trading_data.iterrows():
                # Check exit conditions for existing position
                if position is not None:
                    if position['type'] == 'LONG':
                        if bar['High'] >= position['tp']:
                            exit_price = position['tp']
                            pnl = position['shares'] * (exit_price - position['entry'])
                            capital += pnl
                            trades.append({
                                'date': idx,
                                'type': 'LONG',
                                'entry': position['entry'],
                                'exit': exit_price,
                                'pnl': pnl,
                                'exit_reason': 'TP'
                            })
                            position = None
                        elif bar['Low'] <= position['sl']:
                            exit_price = position['sl']
                            pnl = position['shares'] * (exit_price - position['entry'])
                            capital += pnl
                            trades.append({
                                'date': idx,
                                'type': 'LONG',
                                'entry': position['entry'],
                                'exit': exit_price,
                                'pnl': pnl,
                                'exit_reason': 'SL'
                            })
                            position = None
                    
                    elif position['type'] == 'SHORT':
                        if bar['Low'] <= position['tp']:
                            exit_price = position['tp']
                            pnl = position['shares'] * (position['entry'] - exit_price)
                            capital += pnl
                            trades.append({
                                'date': idx,
                                'type': 'SHORT',
                                'entry': position['entry'],
                                'exit': exit_price,
                                'pnl': pnl,
                                'exit_reason': 'TP'
                            })
                            position = None
                        elif bar['High'] >= position['sl']:
                            exit_price = position['sl']
                            pnl = position['shares'] * (position['entry'] - exit_price)
                            capital += pnl
                            trades.append({
                                'date': idx,
                                'type': 'SHORT',
                                'entry': position['entry'],
                                'exit': exit_price,
                                'pnl': pnl,
                                'exit_reason': 'SL'
                            })
                            position = None
                
                # Check for new entry signals
                if position is None:
                    # Long signal
                    if bar['Close'] > orb['high'] and bar['Open'] <= orb['high']:
                        entry = orb['high']
                        sl = orb['low']
                        tp = entry + (self.tp_multiplier * orb['range'])
                        
                        risk_amount = capital * self.risk_per_trade
                        risk_per_share = entry - sl
                        shares = risk_amount / risk_per_share if risk_per_share > 0 else 0
                        
                        if shares > 0:
                            position = {
                                'type': 'LONG',
                                'entry': entry,
                                'sl': sl,
                                'tp': tp,
                                'shares': shares
                            }
                    
                    # Short signal
                    elif bar['Close'] < orb['low'] and bar['Open'] >= orb['low']:
                        entry = orb['low']
                        sl = orb['high']
                        tp = entry - (self.tp_multiplier * orb['range'])
                        
                        risk_amount = capital * self.risk_per_trade
                        risk_per_share = sl - entry
                        shares = risk_amount / risk_per_share if risk_per_share > 0 else 0
                        
                        if shares > 0:
                            position = {
                                'type': 'SHORT',
                                'entry': entry,
                                'sl': sl,
                                'tp': tp,
                                'shares': shares
                            }
        
        # Calculate metrics
        if len(trades) > 0:
            trades_df = pd.DataFrame(trades)
            
            wins = len(trades_df[trades_df['pnl'] > 0])
            losses = len(trades_df[trades_df['pnl'] < 0])
            win_rate = wins / len(trades) if len(trades) > 0 else 0
            
            avg_win = trades_df[trades_df['pnl'] > 0]['pnl'].mean() if wins > 0 else 0
            avg_loss = abs(trades_df[trades_df['pnl'] < 0]['pnl'].mean()) if losses > 0 else 0
            
            gross_wins = trades_df[trades_df['pnl'] > 0]['pnl'].sum() if wins > 0 else 0
            gross_losses = abs(trades_df[trades_df['pnl'] < 0]['pnl'].sum()) if losses > 0 else 1
            profit_factor = gross_wins / gross_losses if gross_losses > 0 else 0
            
            return {
                'symbol': symbol,
                'total_trades': len(trades),
                'wins': wins,
                'losses': losses,
                'win_rate': win_rate,
                'total_pnl': trades_df['pnl'].sum(),
                'avg_win': avg_win,
                'avg_loss': avg_loss,
                'profit_factor': profit_factor,
                'final_capital': capital,
                'return_pct': ((capital - self.initial_capital) / self.initial_capital) * 100
            }
        
        return {
            'symbol': symbol,
            'total_trades': 0,
            'wins': 0,
            'losses': 0,
            'win_rate': 0,
            'total_pnl': 0,
            'avg_win': 0,
            'avg_loss': 0,
            'profit_factor': 0,
            'final_capital': capital,
            'return_pct': 0
        }

def categorize_symbols(results):
    """Categorize symbols by type"""
    categories = {
        'tech': [],
        'consumer': [],
        'finance': [],
        'healthcare': [],
        'energy': [],
        'forex': [],
        'etf': [],
        'other': []
    }
    
    for r in results:
        symbol = r['symbol']
        
        # Categorize
        if symbol in ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'NFLX', 'CRM', 'ADBE', 'AMD']:
            categories['tech'].append(r)
        elif symbol in ['TSLA', 'DIS', 'WMT', 'COST', 'KO', 'PEP', 'PG']:
            categories['consumer'].append(r)
        elif symbol in ['JPM', 'BAC', 'V', 'MA', 'XLF']:
            categories['finance'].append(r)
        elif symbol in ['UNH', 'JNJ']:
            categories['healthcare'].append(r)
        elif symbol in ['XOM']:
            categories['energy'].append(r)
        elif '=' in symbol or symbol in ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'USDCAD']:
            categories['forex'].append(r)
        elif symbol in ['SPY', 'QQQ', 'GLD', 'IWM', 'DIA', 'VTI', 'EEM', 'XLK']:
            categories['etf'].append(r)
        else:
            categories['other'].append(r)
    
    return categories

def create_comprehensive_charts(results, categories):
    """Create comprehensive charts for all results"""
    plt.style.use('seaborn-v0_8-darkgrid')
    
    # Create main figure
    fig = plt.figure(figsize=(24, 16))
    
    # 1. Top Performers Bar Chart
    ax1 = plt.subplot(3, 4, 1)
    top_10 = sorted(results, key=lambda x: x['return_pct'], reverse=True)[:10]
    symbols = [r['symbol'] for r in top_10]
    returns = [r['return_pct'] for r in top_10]
    colors = ['darkgreen' if r > 50 else 'green' if r > 20 else 'lightgreen' for r in returns]
    
    bars = ax1.barh(range(len(symbols)), returns, color=colors, alpha=0.8)
    ax1.set_yticks(range(len(symbols)))
    ax1.set_yticklabels(symbols)
    ax1.set_xlabel('Return (%)', fontsize=10)
    ax1.set_title('Top 10 Performers', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, returns)):
        ax1.text(val, i, f'{val:.1f}%', va='center', fontsize=8)
    
    # 2. Category Performance
    ax2 = plt.subplot(3, 4, 2)
    cat_names = []
    cat_returns = []
    cat_counts = []
    
    for cat_name, cat_results in categories.items():
        if cat_results:
            avg_return = np.mean([r['return_pct'] for r in cat_results])
            cat_names.append(cat_name.capitalize())
            cat_returns.append(avg_return)
            cat_counts.append(len(cat_results))
    
    colors2 = ['green' if r > 0 else 'red' for r in cat_returns]
    bars2 = ax2.bar(cat_names, cat_returns, color=colors2, alpha=0.7)
    ax2.set_title('Average Returns by Category', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Average Return (%)', fontsize=10)
    ax2.set_xticklabels(cat_names, rotation=45, ha='right')
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax2.grid(True, alpha=0.3)
    
    # Add count labels
    for bar, count in zip(bars2, cat_counts):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'n={count}', ha='center', va='bottom' if height > 0 else 'top',
                fontsize=8)
    
    # 3. Win Rate Distribution
    ax3 = plt.subplot(3, 4, 3)
    win_rates = [r['win_rate'] * 100 for r in results]
    ax3.hist(win_rates, bins=20, edgecolor='black', alpha=0.7, color='blue')
    ax3.axvline(x=33.33, color='red', linestyle='--', linewidth=2, label='Min for 2x TP')
    ax3.axvline(x=np.mean(win_rates), color='green', linestyle='-', linewidth=2, label=f'Mean: {np.mean(win_rates):.1f}%')
    ax3.set_title('Win Rate Distribution', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Win Rate (%)', fontsize=10)
    ax3.set_ylabel('Frequency', fontsize=10)
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)
    
    # 4. Profit Factor Scatter
    ax4 = plt.subplot(3, 4, 4)
    pf_values = [r['profit_factor'] for r in results]
    returns_all = [r['return_pct'] for r in results]
    
    scatter = ax4.scatter(pf_values, returns_all, alpha=0.6, c=returns_all, cmap='RdYlGn', s=50)
    ax4.axvline(x=1, color='black', linestyle='-', linewidth=1)
    ax4.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax4.set_title('Profit Factor vs Returns', fontsize=12, fontweight='bold')
    ax4.set_xlabel('Profit Factor', fontsize=10)
    ax4.set_ylabel('Return (%)', fontsize=10)
    ax4.grid(True, alpha=0.3)
    
    # 5. Returns by Symbol (All)
    ax5 = plt.subplot(3, 4, (5, 8))
    sorted_results = sorted(results, key=lambda x: x['return_pct'], reverse=True)
    all_symbols = [r['symbol'] for r in sorted_results]
    all_returns = [r['return_pct'] for r in sorted_results]
    colors5 = ['darkgreen' if r > 50 else 'green' if r > 0 else 'red' for r in all_returns]
    
    ax5.bar(range(len(all_symbols)), all_returns, color=colors5, alpha=0.7)
    ax5.set_xticks(range(len(all_symbols)))
    ax5.set_xticklabels(all_symbols, rotation=90, fontsize=8)
    ax5.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax5.set_title(f'Returns for All {len(results)} Symbols', fontsize=12, fontweight='bold')
    ax5.set_ylabel('Return (%)', fontsize=10)
    ax5.grid(True, alpha=0.3)
    
    # 6. Trade Count Analysis
    ax6 = plt.subplot(3, 4, 9)
    trade_counts = [r['total_trades'] for r in results]
    ax6.hist(trade_counts, bins=15, edgecolor='black', alpha=0.7, color='orange')
    ax6.set_title('Trade Frequency Distribution', fontsize=12, fontweight='bold')
    ax6.set_xlabel('Number of Trades', fontsize=10)
    ax6.set_ylabel('Number of Symbols', fontsize=10)
    ax6.grid(True, alpha=0.3)
    
    # 7. Win Rate vs Return (Categories)
    ax7 = plt.subplot(3, 4, 10)
    for cat_name, cat_results in categories.items():
        if cat_results:
            cat_wr = [r['win_rate'] * 100 for r in cat_results]
            cat_ret = [r['return_pct'] for r in cat_results]
            ax7.scatter(cat_wr, cat_ret, label=cat_name.capitalize(), alpha=0.6, s=50)
    
    ax7.axvline(x=33.33, color='red', linestyle='--', linewidth=1, alpha=0.5)
    ax7.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.5)
    ax7.set_title('Win Rate vs Return by Category', fontsize=12, fontweight='bold')
    ax7.set_xlabel('Win Rate (%)', fontsize=10)
    ax7.set_ylabel('Return (%)', fontsize=10)
    ax7.legend(fontsize=8, loc='best')
    ax7.grid(True, alpha=0.3)
    
    # 8. Summary Statistics Box
    ax8 = plt.subplot(3, 4, (11, 12))
    ax8.axis('off')
    
    # Calculate overall statistics
    total_symbols = len(results)
    profitable = sum(1 for r in results if r['return_pct'] > 0)
    avg_return = np.mean([r['return_pct'] for r in results])
    avg_win_rate = np.mean([r['win_rate'] for r in results]) * 100
    total_trades = sum(r['total_trades'] for r in results)
    
    # Best performing category
    best_cat = max(categories.items(), 
                   key=lambda x: np.mean([r['return_pct'] for r in x[1]]) if x[1] else 0)
    
    summary_text = f"""
    COMPREHENSIVE BACKTEST RESULTS
    {'=' * 50}
    
    Total Symbols Tested: {total_symbols}
    Profitable Symbols: {profitable}/{total_symbols} ({profitable/total_symbols*100:.1f}%)
    
    Average Return: {avg_return:.2f}%
    Average Win Rate: {avg_win_rate:.1f}%
    Total Trades: {total_trades}
    
    BEST PERFORMERS:
    1. {top_10[0]['symbol']}: {top_10[0]['return_pct']:.1f}%
    2. {top_10[1]['symbol']}: {top_10[1]['return_pct']:.1f}%
    3. {top_10[2]['symbol']}: {top_10[2]['return_pct']:.1f}%
    
    BEST CATEGORY: {best_cat[0].upper()}
    Average Return: {np.mean([r['return_pct'] for r in best_cat[1]]):.1f}%
    
    STRATEGY VALIDATION:
    ✓ Win Rate > 33.3%: {'YES' if avg_win_rate > 33.33 else 'NO'}
    ✓ Positive Expectancy: {'YES' if avg_return > 0 else 'NO'}
    ✓ Works Across Assets: {'YES' if profitable > total_symbols * 0.5 else 'NO'}
    """
    
    ax8.text(0.05, 0.95, summary_text, transform=ax8.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle('ORB Strategy - Comprehensive Backtest Results (2x TP)', 
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    # Save the figure
    plt.savefig('reports/comprehensive_backtest_2x.png', dpi=100, bbox_inches='tight')
    print("✓ Comprehensive charts saved to reports/comprehensive_backtest_2x.png")

def main():
    """Main execution"""
    print("=" * 80)
    print("COMPREHENSIVE ORB STRATEGY BACKTEST")
    print("=" * 80)
    print("\nConfiguration:")
    print("- Take Profit: 2x Opening Range")
    print("- Risk per Trade: 1%")
    print("- Testing 40+ symbols across multiple asset classes")
    print("-" * 80)
    
    # Load available symbols
    if os.path.exists('data/available_symbols.txt'):
        with open('data/available_symbols.txt', 'r') as f:
            symbols = [line.strip() for line in f.readlines()]
    else:
        # Default list
        symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'SPY', 'QQQ', 'GLD',
                  'META', 'NVDA', 'JPM', 'BAC', 'NFLX', 'DIS', 'V', 'MA', 
                  'UNH', 'JNJ', 'WMT', 'XOM', 'COST', 'AMD', 'CRM', 'ADBE',
                  'PG', 'KO', 'PEP', 'IWM', 'DIA', 'VTI', 'EEM', 'XLF', 'XLK']
    
    # Initialize backtester
    backtester = ExpandedORBBacktest()
    
    # Run backtests
    results = []
    print(f"\nTesting {len(symbols)} symbols...")
    
    for symbol in symbols:
        result = backtester.backtest_symbol(symbol)
        if result and result['total_trades'] > 0:
            results.append(result)
            status = "✓" if result['return_pct'] > 0 else "✗"
            print(f"{status} {symbol:8} - {result['total_trades']:3} trades, "
                  f"{result['return_pct']:7.1f}% return, "
                  f"{result['win_rate']*100:5.1f}% win rate")
    
    # Categorize results
    categories = categorize_symbols(results)
    
    # Display category summary
    print("\n" + "=" * 80)
    print("RESULTS BY CATEGORY")
    print("=" * 80)
    
    for cat_name, cat_results in categories.items():
        if cat_results:
            avg_return = np.mean([r['return_pct'] for r in cat_results])
            avg_wr = np.mean([r['win_rate'] for r in cat_results]) * 100
            profitable = sum(1 for r in cat_results if r['return_pct'] > 0)
            
            print(f"\n{cat_name.upper()} ({len(cat_results)} symbols):")
            print(f"  Average Return: {avg_return:.2f}%")
            print(f"  Average Win Rate: {avg_wr:.1f}%")
            print(f"  Profitable: {profitable}/{len(cat_results)}")
            
            # Top performer in category
            if cat_results:
                best = max(cat_results, key=lambda x: x['return_pct'])
                print(f"  Best: {best['symbol']} ({best['return_pct']:.1f}%)")
    
    # Create comprehensive charts
    print("\nGenerating comprehensive charts...")
    create_comprehensive_charts(results, categories)
    
    # Save detailed results
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('return_pct', ascending=False)
    results_df.to_csv('reports/comprehensive_results_2x.csv', index=False)
    print("✓ Detailed results saved to reports/comprehensive_results_2x.csv")
    
    # Final summary
    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)
    
    profitable_count = sum(1 for r in results if r['return_pct'] > 0)
    print(f"\nSuccess Rate: {profitable_count}/{len(results)} symbols profitable "
          f"({profitable_count/len(results)*100:.1f}%)")
    
    avg_return = np.mean([r['return_pct'] for r in results])
    print(f"Overall Average Return: {avg_return:.2f}%")
    
    if avg_return > 0:
        print("\n✅ STRATEGY VALIDATED: Positive expectancy across diverse assets")
    else:
        print("\n⚠️ Strategy needs refinement for consistent profitability")
    
    print("\n" + "=" * 80)
    print("BACKTEST COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    main()