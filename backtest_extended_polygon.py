#!/usr/bin/env python3
"""
Extended ORB Strategy Backtest using Polygon.io Data
Tests strategy over 2 years to validate long-term performance
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import os
import yaml
import warnings
warnings.filterwarnings('ignore')

class ExtendedORBBacktest:
    def __init__(self, initial_capital=10000, data_dir="data/polygon"):
        self.initial_capital = initial_capital
        self.tp_multiplier = 2.0  # Fixed at 2x
        self.risk_per_trade = 0.01  # 1% risk
        self.data_dir = data_dir
        
    def load_data(self, symbol):
        """Load data from Polygon CSV files"""
        safe_symbol = symbol.replace(':', '_')
        filepath = f'{self.data_dir}/{safe_symbol}_5m.csv'
        
        if not os.path.exists(filepath):
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
    
    def calculate_orb(self, data, date):
        """Calculate Opening Range for a specific date (9:30-9:45 AM)"""
        day_data = data[data.index.date == date.date()]
        
        if len(day_data) < 3:
            return None
        
        # Filter for market hours (9:30 AM - 4:00 PM EST)
        market_open = day_data.between_time('09:30', '16:00')
        
        if len(market_open) < 3:
            return None
            
        # Get first 3 bars (9:30-9:45) for ORB
        orb_data = market_open.iloc[:3]
        
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
    
    def backtest_symbol(self, symbol, start_date=None, end_date=None):
        """Run backtest for a single symbol"""
        data = self.load_data(symbol)
        if data is None:
            return None
        
        # Apply date filters if provided
        if start_date:
            data = data[data.index >= pd.to_datetime(start_date)]
        if end_date:
            data = data[data.index <= pd.to_datetime(end_date)]
            
        trades = []
        capital = self.initial_capital
        equity_curve = []
        
        # Get unique trading days
        trading_days = pd.to_datetime(data.index.date).unique()
        
        for date in trading_days:
            orb = self.calculate_orb(data, pd.Timestamp(date))
            if orb is None:
                continue
                
            day_data = data[data.index.date == date.date()]
            trading_data = day_data.between_time('09:45', '16:00')  # Post-ORB trading
            
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
                
                # Track equity curve
                equity_curve.append({
                    'date': idx,
                    'capital': capital
                })
        
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
            
            # Calculate max drawdown
            equity_df = pd.DataFrame(equity_curve)
            equity_df['peak'] = equity_df['capital'].cummax()
            equity_df['drawdown'] = (equity_df['capital'] - equity_df['peak']) / equity_df['peak']
            max_drawdown = equity_df['drawdown'].min() * 100
            
            # Calculate Sharpe ratio (annualized)
            if len(trades_df) > 1:
                trades_df['returns'] = trades_df['pnl'] / self.initial_capital
                sharpe = (trades_df['returns'].mean() / trades_df['returns'].std()) * np.sqrt(252) if trades_df['returns'].std() > 0 else 0
            else:
                sharpe = 0
            
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
                'return_pct': ((capital - self.initial_capital) / self.initial_capital) * 100,
                'max_drawdown': max_drawdown,
                'sharpe_ratio': sharpe,
                'trades_df': trades_df,
                'equity_curve': equity_df
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
            'return_pct': 0,
            'max_drawdown': 0,
            'sharpe_ratio': 0,
            'trades_df': pd.DataFrame(),
            'equity_curve': pd.DataFrame()
        }

def analyze_by_period(results_by_period):
    """Analyze performance across different market periods"""
    plt.style.use('seaborn-v0_8-darkgrid')
    
    fig = plt.figure(figsize=(20, 12))
    
    # 1. Returns by Period
    ax1 = plt.subplot(2, 3, 1)
    periods = list(results_by_period.keys())
    avg_returns = []
    for period, results in results_by_period.items():
        if results:
            avg_return = np.mean([r['return_pct'] for r in results])
            avg_returns.append(avg_return)
        else:
            avg_returns.append(0)
    
    colors = ['green' if r > 0 else 'red' for r in avg_returns]
    bars = ax1.bar(periods, avg_returns, color=colors, alpha=0.7)
    ax1.set_title('Average Returns by Market Period', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Average Return (%)', fontsize=12)
    ax1.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax1.grid(True, alpha=0.3)
    
    # Add value labels
    for bar, val in zip(bars, avg_returns):
        ax1.text(bar.get_x() + bar.get_width()/2., val,
                f'{val:.1f}%', ha='center', va='bottom' if val > 0 else 'top')
    
    # 2. Win Rate Comparison
    ax2 = plt.subplot(2, 3, 2)
    win_rates_by_period = {}
    for period, results in results_by_period.items():
        if results:
            win_rates = [r['win_rate'] * 100 for r in results]
            win_rates_by_period[period] = win_rates
    
    if win_rates_by_period:
        bp = ax2.boxplot(win_rates_by_period.values(), labels=win_rates_by_period.keys())
        ax2.axhline(y=33.33, color='red', linestyle='--', label='Min for 2x TP')
        ax2.set_title('Win Rate Distribution by Period', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Win Rate (%)', fontsize=12)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    # 3. Top Performers Consistency
    ax3 = plt.subplot(2, 3, 3)
    
    # Track performance of top symbols across periods
    top_symbols = ['COST', 'ADBE', 'NFLX', 'QQQ', 'PEP']
    symbol_performance = {symbol: [] for symbol in top_symbols}
    
    for period in periods:
        period_results = results_by_period[period]
        for symbol in top_symbols:
            result = next((r for r in period_results if r['symbol'] == symbol), None)
            if result:
                symbol_performance[symbol].append(result['return_pct'])
            else:
                symbol_performance[symbol].append(0)
    
    x = np.arange(len(periods))
    width = 0.15
    
    for i, (symbol, returns) in enumerate(symbol_performance.items()):
        ax3.bar(x + i * width, returns, width, label=symbol, alpha=0.8)
    
    ax3.set_title('Top Performers Across Periods', fontsize=14, fontweight='bold')
    ax3.set_ylabel('Return (%)', fontsize=12)
    ax3.set_xticks(x + width * 2)
    ax3.set_xticklabels(periods)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Max Drawdown Analysis
    ax4 = plt.subplot(2, 3, 4)
    drawdowns_by_period = {}
    for period, results in results_by_period.items():
        if results:
            drawdowns = [abs(r['max_drawdown']) for r in results if r['max_drawdown'] != 0]
            if drawdowns:
                drawdowns_by_period[period] = drawdowns
    
    if drawdowns_by_period:
        bp2 = ax4.boxplot(drawdowns_by_period.values(), labels=drawdowns_by_period.keys())
        ax4.set_title('Max Drawdown Distribution by Period', fontsize=14, fontweight='bold')
        ax4.set_ylabel('Max Drawdown (%)', fontsize=12)
        ax4.grid(True, alpha=0.3)
    
    # 5. Profit Factor Comparison
    ax5 = plt.subplot(2, 3, 5)
    pf_by_period = {}
    for period, results in results_by_period.items():
        if results:
            pfs = [r['profit_factor'] for r in results if r['profit_factor'] > 0]
            if pfs:
                pf_by_period[period] = pfs
    
    if pf_by_period:
        bp3 = ax5.boxplot(pf_by_period.values(), labels=pf_by_period.keys())
        ax5.axhline(y=1.0, color='red', linestyle='-', label='Breakeven')
        ax5.set_title('Profit Factor Distribution by Period', fontsize=14, fontweight='bold')
        ax5.set_ylabel('Profit Factor', fontsize=12)
        ax5.legend()
        ax5.grid(True, alpha=0.3)
    
    # 6. Summary Statistics
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')
    
    summary_text = "EXTENDED BACKTEST SUMMARY\n" + "="*40 + "\n\n"
    
    for period, results in results_by_period.items():
        if results:
            profitable = sum(1 for r in results if r['return_pct'] > 0)
            avg_return = np.mean([r['return_pct'] for r in results])
            avg_wr = np.mean([r['win_rate'] for r in results]) * 100
            total_trades = sum(r['total_trades'] for r in results)
            
            summary_text += f"{period}:\n"
            summary_text += f"  Profitable: {profitable}/{len(results)} ({profitable/len(results)*100:.0f}%)\n"
            summary_text += f"  Avg Return: {avg_return:.1f}%\n"
            summary_text += f"  Avg Win Rate: {avg_wr:.1f}%\n"
            summary_text += f"  Total Trades: {total_trades}\n\n"
    
    ax6.text(0.05, 0.95, summary_text, transform=ax6.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle('ORB Strategy - Extended Period Analysis (2 Years)', 
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    # Save the figure
    os.makedirs('reports', exist_ok=True)
    plt.savefig('reports/extended_period_analysis.png', dpi=100, bbox_inches='tight')
    print("✓ Period analysis saved to reports/extended_period_analysis.png")

def main():
    """Main execution"""
    print("="*80)
    print("EXTENDED ORB STRATEGY BACKTEST - POLYGON DATA")
    print("="*80)
    
    # Load configuration
    if os.path.exists('configs/polygon_config.yaml'):
        with open('configs/polygon_config.yaml', 'r') as f:
            config = yaml.safe_load(f)['polygon']
    else:
        print("✗ Configuration file not found")
        return
    
    # Check for available data
    data_dir = config['output']['directory']
    if not os.path.exists(data_dir):
        print(f"✗ Data directory not found: {data_dir}")
        print("Please run: python polygon_downloader.py")
        return
    
    # Load available symbols
    symbols_file = f"{data_dir}/available_symbols.txt"
    if os.path.exists(symbols_file):
        with open(symbols_file, 'r') as f:
            symbols = [line.strip() for line in f.readlines()]
    else:
        # Use configured symbols
        symbols = config['symbols']['stocks'] + [s for s in config['symbols'].get('forex', [])]
    
    print(f"\nTesting {len(symbols)} symbols with 2 years of data")
    print("-"*80)
    
    # Initialize backtester
    backtester = ExtendedORBBacktest(data_dir=data_dir)
    
    # Run backtests for different periods
    results_by_period = {}
    
    # Define analysis periods
    periods = [
        ("Full 2 Years", None, None),
        ("2023 Bull", "2023-01-01", "2023-12-31"),
        ("2022 Bear", "2022-01-01", "2022-12-31"),
        ("Recent 2024", "2024-01-01", None)
    ]
    
    for period_name, start_date, end_date in periods:
        print(f"\n{period_name}:")
        print("-"*40)
        
        period_results = []
        for symbol in symbols:
            result = backtester.backtest_symbol(symbol, start_date, end_date)
            if result and result['total_trades'] > 0:
                period_results.append(result)
                status = "✓" if result['return_pct'] > 0 else "✗"
                print(f"{status} {symbol:10} - {result['total_trades']:3} trades, "
                      f"{result['return_pct']:7.1f}% return, "
                      f"{result['win_rate']*100:5.1f}% WR, "
                      f"DD: {result['max_drawdown']:6.1f}%")
        
        results_by_period[period_name] = period_results
    
    # Analyze results by period
    print("\nGenerating period analysis charts...")
    analyze_by_period(results_by_period)
    
    # Save detailed results
    all_results = []
    for period_name, results in results_by_period.items():
        for r in results:
            r_copy = r.copy()
            r_copy['period'] = period_name
            # Remove non-serializable columns
            r_copy.pop('trades_df', None)
            r_copy.pop('equity_curve', None)
            all_results.append(r_copy)
    
    results_df = pd.DataFrame(all_results)
    results_df.to_csv('reports/extended_backtest_results.csv', index=False)
    print("✓ Detailed results saved to reports/extended_backtest_results.csv")
    
    # Final comparison with 60-day results
    print("\n" + "="*80)
    print("COMPARISON: 2 YEARS vs 60 DAYS")
    print("="*80)
    
    # Load 60-day results if available
    if os.path.exists('reports/comprehensive_results_2x.csv'):
        sixty_day_df = pd.read_csv('reports/comprehensive_results_2x.csv')
        
        # Compare full period with 60-day
        full_period_results = results_by_period.get("Full 2 Years", [])
        
        comparison = []
        for symbol in symbols[:20]:  # Top 20 symbols
            # Get 2-year result
            two_year = next((r for r in full_period_results if r['symbol'] == symbol), None)
            # Get 60-day result
            sixty_day = sixty_day_df[sixty_day_df['symbol'] == symbol].to_dict('records')
            sixty_day = sixty_day[0] if sixty_day else None
            
            if two_year and sixty_day:
                comparison.append({
                    'Symbol': symbol,
                    '60-Day Return': f"{sixty_day['return_pct']:.1f}%",
                    '2-Year Return': f"{two_year['return_pct']:.1f}%",
                    'Difference': f"{two_year['return_pct'] - sixty_day['return_pct']:+.1f}%",
                    '2Y Win Rate': f"{two_year['win_rate']*100:.1f}%",
                    '2Y Max DD': f"{two_year['max_drawdown']:.1f}%"
                })
        
        if comparison:
            comp_df = pd.DataFrame(comparison)
            print("\nTop Symbols Comparison:")
            print(comp_df.to_string(index=False))
    
    print("\n" + "="*80)
    print("EXTENDED BACKTEST COMPLETE")
    print("="*80)
    print("\nKey Insights:")
    print("- Strategy performance across different market cycles")
    print("- Maximum drawdown levels for risk management")
    print("- Consistency of top performers over 2 years")

if __name__ == "__main__":
    main()