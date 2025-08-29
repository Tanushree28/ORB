#!/usr/bin/env python3
"""
ORB Strategy Backtest with 2x Take Profit Multiplier
Proper implementation with strict SL/TP rules
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, time, timedelta
import os
import warnings
warnings.filterwarnings('ignore')

class ORBBacktest2x:
    def __init__(self, initial_capital=10000):
        self.initial_capital = initial_capital
        self.tp_multiplier = 2.0  # Fixed at 2x for all symbols
        self.risk_per_trade = 0.01  # 1% risk per trade
        
    def load_data(self, symbol, interval='5m'):
        """Load data from CSV files"""
        filepath = f'data/{symbol}_{interval}.csv'
        if not os.path.exists(filepath):
            print(f"Warning: {filepath} not found")
            return None
            
        df = pd.read_csv(filepath)
        df['Datetime'] = pd.to_datetime(df['Datetime'])
        df.set_index('Datetime', inplace=True)
        return df
    
    def calculate_orb(self, data, date):
        """Calculate Opening Range for a specific date"""
        # Filter data for the specific date
        day_data = data[data.index.date == date.date()]
        
        if len(day_data) < 3:
            return None
            
        # Get first 3 bars (15 minutes for 5-min data)
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
        # Try to load 5-minute data first
        data = self.load_data(symbol, '5m')
        if data is None:
            # Fallback to 15-minute data
            data = self.load_data(symbol, '15m')
        if data is None:
            # Fallback to hourly data
            data = self.load_data(symbol, '1h')
        if data is None:
            return None
            
        trades = []
        capital = self.initial_capital
        
        # Get unique trading days
        trading_days = pd.to_datetime(data.index.date).unique()
        
        for date in trading_days:
            # Calculate ORB for the day
            orb = self.calculate_orb(data, pd.Timestamp(date))
            if orb is None:
                continue
                
            # Get data after ORB period
            day_data = data[data.index.date == date.date()]
            # Skip first 3 bars (ORB period)
            trading_data = day_data.iloc[3:]
            
            if len(trading_data) == 0:
                continue
                
            position = None
            for idx, bar in trading_data.iterrows():
                # Check if we already have a position
                if position is not None:
                    # Check exit conditions
                    if position['type'] == 'LONG':
                        if bar['High'] >= position['tp']:
                            # Take profit hit
                            exit_price = position['tp']
                            pnl = position['shares'] * (exit_price - position['entry'])
                            capital += pnl
                            trades.append({
                                'date': idx,
                                'type': 'LONG',
                                'entry': position['entry'],
                                'exit': exit_price,
                                'shares': position['shares'],
                                'pnl': pnl,
                                'exit_reason': 'TP',
                                'return_pct': (pnl / (capital - pnl)) * 100
                            })
                            position = None
                        elif bar['Low'] <= position['sl']:
                            # Stop loss hit
                            exit_price = position['sl']
                            pnl = position['shares'] * (exit_price - position['entry'])
                            capital += pnl
                            trades.append({
                                'date': idx,
                                'type': 'LONG',
                                'entry': position['entry'],
                                'exit': exit_price,
                                'shares': position['shares'],
                                'pnl': pnl,
                                'exit_reason': 'SL',
                                'return_pct': (pnl / (capital - pnl)) * 100
                            })
                            position = None
                    
                    elif position['type'] == 'SHORT':
                        if bar['Low'] <= position['tp']:
                            # Take profit hit
                            exit_price = position['tp']
                            pnl = position['shares'] * (position['entry'] - exit_price)
                            capital += pnl
                            trades.append({
                                'date': idx,
                                'type': 'SHORT',
                                'entry': position['entry'],
                                'exit': exit_price,
                                'shares': position['shares'],
                                'pnl': pnl,
                                'exit_reason': 'TP',
                                'return_pct': (pnl / (capital - pnl)) * 100
                            })
                            position = None
                        elif bar['High'] >= position['sl']:
                            # Stop loss hit
                            exit_price = position['sl']
                            pnl = position['shares'] * (position['entry'] - exit_price)
                            capital += pnl
                            trades.append({
                                'date': idx,
                                'type': 'SHORT',
                                'entry': position['entry'],
                                'exit': exit_price,
                                'shares': position['shares'],
                                'pnl': pnl,
                                'exit_reason': 'SL',
                                'return_pct': (pnl / (capital - pnl)) * 100
                            })
                            position = None
                
                # Check for new entry signals (only if no position)
                if position is None:
                    # Long signal
                    if bar['Close'] > orb['high'] and bar['Open'] <= orb['high']:
                        entry = orb['high']
                        sl = orb['low']
                        tp = entry + (self.tp_multiplier * orb['range'])
                        
                        # Calculate position size
                        risk_amount = capital * self.risk_per_trade
                        risk_per_share = entry - sl
                        shares = risk_amount / risk_per_share if risk_per_share > 0 else 0
                        
                        if shares > 0:
                            position = {
                                'type': 'LONG',
                                'entry': entry,
                                'sl': sl,
                                'tp': tp,
                                'shares': shares,
                                'date': idx
                            }
                    
                    # Short signal
                    elif bar['Close'] < orb['low'] and bar['Open'] >= orb['low']:
                        entry = orb['low']
                        sl = orb['high']
                        tp = entry - (self.tp_multiplier * orb['range'])
                        
                        # Calculate position size
                        risk_amount = capital * self.risk_per_trade
                        risk_per_share = sl - entry
                        shares = risk_amount / risk_per_share if risk_per_share > 0 else 0
                        
                        if shares > 0:
                            position = {
                                'type': 'SHORT',
                                'entry': entry,
                                'sl': sl,
                                'tp': tp,
                                'shares': shares,
                                'date': idx
                            }
            
            # Close any open position at end of day
            if position is not None:
                last_bar = trading_data.iloc[-1]
                exit_price = last_bar['Close']
                
                if position['type'] == 'LONG':
                    pnl = position['shares'] * (exit_price - position['entry'])
                else:
                    pnl = position['shares'] * (position['entry'] - exit_price)
                
                capital += pnl
                trades.append({
                    'date': last_bar.name,
                    'type': position['type'],
                    'entry': position['entry'],
                    'exit': exit_price,
                    'shares': position['shares'],
                    'pnl': pnl,
                    'exit_reason': 'EOD',
                    'return_pct': (pnl / (capital - pnl)) * 100
                })
                position = None
        
        # Calculate metrics
        if len(trades) > 0:
            trades_df = pd.DataFrame(trades)
            
            wins = len(trades_df[trades_df['pnl'] > 0])
            losses = len(trades_df[trades_df['pnl'] < 0])
            
            win_rate = wins / len(trades) if len(trades) > 0 else 0
            
            # Calculate expectancy
            avg_win = trades_df[trades_df['pnl'] > 0]['pnl'].mean() if wins > 0 else 0
            avg_loss = abs(trades_df[trades_df['pnl'] < 0]['pnl'].mean()) if losses > 0 else 0
            expectancy = (win_rate * avg_win) - ((1 - win_rate) * avg_loss)
            
            # Profit factor
            gross_wins = trades_df[trades_df['pnl'] > 0]['pnl'].sum() if wins > 0 else 0
            gross_losses = abs(trades_df[trades_df['pnl'] < 0]['pnl'].sum()) if losses > 0 else 1
            profit_factor = gross_wins / gross_losses if gross_losses > 0 else 0
            
            # Calculate max drawdown
            trades_df['cumulative_pnl'] = trades_df['pnl'].cumsum()
            trades_df['running_max'] = trades_df['cumulative_pnl'].cummax()
            trades_df['drawdown'] = trades_df['cumulative_pnl'] - trades_df['running_max']
            max_drawdown = trades_df['drawdown'].min()
            
            return {
                'symbol': symbol,
                'total_trades': len(trades),
                'wins': wins,
                'losses': losses,
                'win_rate': win_rate,
                'total_pnl': trades_df['pnl'].sum(),
                'avg_win': avg_win,
                'avg_loss': avg_loss,
                'expectancy': expectancy,
                'profit_factor': profit_factor,
                'max_drawdown': max_drawdown,
                'final_capital': capital,
                'return_pct': ((capital - self.initial_capital) / self.initial_capital) * 100,
                'trades_df': trades_df
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
            'expectancy': 0,
            'profit_factor': 0,
            'max_drawdown': 0,
            'final_capital': capital,
            'return_pct': 0,
            'trades_df': pd.DataFrame()
        }

def create_comprehensive_charts(results):
    """Create comprehensive charts for backtest results"""
    # Set style
    plt.style.use('seaborn-v0_8-darkgrid')
    sns.set_palette("husl")
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 12))
    
    # 1. Returns by Symbol (Main Chart)
    ax1 = plt.subplot(2, 4, 1)
    symbols = [r['symbol'] for r in results]
    returns = [r['return_pct'] for r in results]
    colors = ['green' if r > 0 else 'red' for r in returns]
    bars = ax1.bar(symbols, returns, color=colors, alpha=0.7, edgecolor='black')
    ax1.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax1.set_title('Returns by Symbol (2x TP)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Return (%)', fontsize=12)
    ax1.set_xlabel('Symbol', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.set_xticklabels(symbols, rotation=45)
    
    # Add value labels on bars
    for bar, val in zip(bars, returns):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.1f}%', ha='center', va='bottom' if val > 0 else 'top',
                fontsize=9, fontweight='bold')
    
    # 2. Win Rate Analysis
    ax2 = plt.subplot(2, 4, 2)
    win_rates = [r['win_rate'] * 100 for r in results]
    bars2 = ax2.bar(symbols, win_rates, color='blue', alpha=0.7, edgecolor='black')
    ax2.axhline(y=33.33, color='red', linestyle='--', linewidth=1, label='Breakeven (33.3%)')
    ax2.set_title('Win Rates (Need >33.3% for 2x TP)', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Win Rate (%)', fontsize=12)
    ax2.set_xlabel('Symbol', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.set_xticklabels(symbols, rotation=45)
    
    # 3. Profit Factor
    ax3 = plt.subplot(2, 4, 3)
    profit_factors = [r['profit_factor'] for r in results]
    colors3 = ['green' if pf > 1 else 'red' for pf in profit_factors]
    bars3 = ax3.bar(symbols, profit_factors, color=colors3, alpha=0.7, edgecolor='black')
    ax3.axhline(y=1, color='black', linestyle='-', linewidth=2, label='Breakeven')
    ax3.set_title('Profit Factor (>1 is Profitable)', fontsize=14, fontweight='bold')
    ax3.set_ylabel('Profit Factor', fontsize=12)
    ax3.set_xlabel('Symbol', fontsize=12)
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    ax3.set_xticklabels(symbols, rotation=45)
    
    # 4. Total Trades
    ax4 = plt.subplot(2, 4, 4)
    total_trades = [r['total_trades'] for r in results]
    ax4.bar(symbols, total_trades, color='orange', alpha=0.7, edgecolor='black')
    ax4.set_title('Number of Trades Executed', fontsize=14, fontweight='bold')
    ax4.set_ylabel('Trade Count', fontsize=12)
    ax4.set_xlabel('Symbol', fontsize=12)
    ax4.grid(True, alpha=0.3)
    ax4.set_xticklabels(symbols, rotation=45)
    
    # 5. Risk-Reward Scatter
    ax5 = plt.subplot(2, 4, 5)
    ax5.scatter(win_rates, returns, s=100, alpha=0.6, c=returns, cmap='RdYlGn')
    for i, symbol in enumerate(symbols):
        ax5.annotate(symbol, (win_rates[i], returns[i]), fontsize=8)
    ax5.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax5.axvline(x=33.33, color='red', linestyle='--', linewidth=1)
    ax5.set_title('Win Rate vs Return Analysis', fontsize=14, fontweight='bold')
    ax5.set_xlabel('Win Rate (%)', fontsize=12)
    ax5.set_ylabel('Return (%)', fontsize=12)
    ax5.grid(True, alpha=0.3)
    
    # 6. Expectancy
    ax6 = plt.subplot(2, 4, 6)
    expectancy = [r['expectancy'] for r in results]
    colors6 = ['green' if e > 0 else 'red' for e in expectancy]
    ax6.bar(symbols, expectancy, color=colors6, alpha=0.7, edgecolor='black')
    ax6.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax6.set_title('Expectancy per Trade ($)', fontsize=14, fontweight='bold')
    ax6.set_ylabel('Expectancy ($)', fontsize=12)
    ax6.set_xlabel('Symbol', fontsize=12)
    ax6.grid(True, alpha=0.3)
    ax6.set_xticklabels(symbols, rotation=45)
    
    # 7. Max Drawdown
    ax7 = plt.subplot(2, 4, 7)
    max_dd = [abs(r['max_drawdown']) for r in results]
    ax7.bar(symbols, max_dd, color='darkred', alpha=0.7, edgecolor='black')
    ax7.set_title('Maximum Drawdown ($)', fontsize=14, fontweight='bold')
    ax7.set_ylabel('Max Drawdown ($)', fontsize=12)
    ax7.set_xlabel('Symbol', fontsize=12)
    ax7.grid(True, alpha=0.3)
    ax7.set_xticklabels(symbols, rotation=45)
    
    # 8. Summary Statistics Box
    ax8 = plt.subplot(2, 4, 8)
    ax8.axis('off')
    
    # Calculate overall statistics
    total_return = np.mean([r['return_pct'] for r in results])
    avg_win_rate = np.mean([r['win_rate'] for r in results])
    profitable_symbols = sum(1 for r in results if r['return_pct'] > 0)
    total_trades_all = sum(r['total_trades'] for r in results)
    
    summary_text = f"""
    OVERALL STATISTICS (2x TP Strategy)
    {'=' * 40}
    
    Average Return: {total_return:.2f}%
    Average Win Rate: {avg_win_rate*100:.1f}%
    Profitable Symbols: {profitable_symbols}/{len(results)}
    Total Trades: {total_trades_all}
    
    TOP PERFORMERS:
    """
    
    # Add top 3 performers
    sorted_results = sorted(results, key=lambda x: x['return_pct'], reverse=True)[:3]
    for i, r in enumerate(sorted_results, 1):
        summary_text += f"\n    {i}. {r['symbol']}: {r['return_pct']:.1f}% ({r['win_rate']*100:.0f}% WR)"
    
    summary_text += "\n\n    BOTTOM PERFORMERS:"
    sorted_results_bottom = sorted(results, key=lambda x: x['return_pct'])[:3]
    for i, r in enumerate(sorted_results_bottom, 1):
        summary_text += f"\n    {i}. {r['symbol']}: {r['return_pct']:.1f}% ({r['win_rate']*100:.0f}% WR)"
    
    ax8.text(0.1, 0.9, summary_text, transform=ax8.transAxes, fontsize=11,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle('ORB Strategy Backtest Results - 2x Take Profit Multiplier', 
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    # Save the figure
    plt.savefig('reports/backtest_2x_results.png', dpi=100, bbox_inches='tight')
    print("✓ Charts saved to reports/backtest_2x_results.png")
    
    return fig

def main():
    """Main execution function"""
    print("=" * 80)
    print("ORB STRATEGY BACKTEST - 2x TAKE PROFIT MULTIPLIER")
    print("=" * 80)
    print("\nConfiguration:")
    print("- Take Profit: 2x the Opening Range")
    print("- Stop Loss: Opposite side of Opening Range")
    print("- Risk per Trade: 1% of capital")
    print("- Initial Capital: $10,000")
    print("-" * 80)
    
    # List of symbols to test
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'SPY', 'QQQ', 'GLD', 
               'META', 'NVDA', 'JPM', 'BAC']
    
    # Initialize backtester
    backtester = ORBBacktest2x(initial_capital=10000)
    
    # Run backtests
    results = []
    print("\nRunning backtests...")
    
    for symbol in symbols:
        print(f"Testing {symbol}...", end=' ')
        result = backtester.backtest_symbol(symbol)
        if result:
            results.append(result)
            print(f"✓ {result['total_trades']} trades, {result['return_pct']:.1f}% return")
        else:
            print("✗ No data")
    
    # Display results table
    print("\n" + "=" * 80)
    print("BACKTEST RESULTS SUMMARY")
    print("=" * 80)
    
    # Create results DataFrame
    summary_data = []
    for r in results:
        summary_data.append({
            'Symbol': r['symbol'],
            'Trades': r['total_trades'],
            'Wins': r['wins'],
            'Losses': r['losses'],
            'Win Rate': f"{r['win_rate']*100:.1f}%",
            'Total PnL': f"${r['total_pnl']:.2f}",
            'Expectancy': f"${r['expectancy']:.2f}",
            'Profit Factor': f"{r['profit_factor']:.2f}",
            'Max DD': f"${r['max_drawdown']:.2f}",
            'Return': f"{r['return_pct']:.1f}%"
        })
    
    summary_df = pd.DataFrame(summary_data)
    print("\n" + summary_df.to_string(index=False))
    
    # Save results to CSV
    summary_df.to_csv('reports/backtest_2x_results.csv', index=False)
    print("\n✓ Results saved to reports/backtest_2x_results.csv")
    
    # Create comprehensive charts
    print("\nGenerating charts...")
    create_comprehensive_charts(results)
    
    # Print final analysis
    print("\n" + "=" * 80)
    print("FINAL ANALYSIS")
    print("=" * 80)
    
    profitable = [r for r in results if r['return_pct'] > 0]
    unprofitable = [r for r in results if r['return_pct'] <= 0]
    
    print(f"\n✅ Profitable Symbols ({len(profitable)}/{len(results)}):")
    for r in sorted(profitable, key=lambda x: x['return_pct'], reverse=True):
        print(f"   {r['symbol']:6} - Return: {r['return_pct']:6.1f}%, Win Rate: {r['win_rate']*100:5.1f}%, PF: {r['profit_factor']:.2f}")
    
    if unprofitable:
        print(f"\n❌ Unprofitable Symbols ({len(unprofitable)}/{len(results)}):")
        for r in sorted(unprofitable, key=lambda x: x['return_pct']):
            print(f"   {r['symbol']:6} - Return: {r['return_pct']:6.1f}%, Win Rate: {r['win_rate']*100:5.1f}%, PF: {r['profit_factor']:.2f}")
    
    # Overall statistics
    avg_return = np.mean([r['return_pct'] for r in results])
    avg_win_rate = np.mean([r['win_rate'] for r in results]) * 100
    
    print(f"\n📊 Overall Performance:")
    print(f"   Average Return: {avg_return:.2f}%")
    print(f"   Average Win Rate: {avg_win_rate:.1f}%")
    print(f"   Success Rate: {len(profitable)}/{len(results)} symbols profitable")
    
    if avg_win_rate > 33.33:
        print(f"   ✅ Win rate ({avg_win_rate:.1f}%) exceeds minimum (33.3%) for 2x TP")
    else:
        print(f"   ⚠️ Win rate ({avg_win_rate:.1f}%) below minimum (33.3%) for 2x TP")
    
    print("\n" + "=" * 80)
    print("BACKTEST COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    main()