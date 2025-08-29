#!/usr/bin/env python3
"""
Run ORB Strategy Backtest with Daily Data
Uses 1-hour bars to simulate intraday trading
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, time
import warnings
warnings.filterwarnings('ignore')

class SimpleORBBacktest:
    def __init__(self):
        self.initial_capital = 10000
        self.results = []
        
    def download_data(self, symbol, period='3mo'):
        """Download hourly data for the symbol"""
        ticker = yf.Ticker(symbol)
        
        # Try to get hourly data
        try:
            data = ticker.history(period=period, interval='1h')
            if not data.empty:
                print(f"✓ Downloaded {len(data)} hourly bars for {symbol}")
                return data
        except:
            pass
        
        # Fallback to daily data if hourly not available
        try:
            data = ticker.history(period='6mo', interval='1d')
            if not data.empty:
                print(f"✓ Downloaded {len(data)} daily bars for {symbol}")
                return data
        except:
            pass
            
        print(f"✗ No data available for {symbol}")
        return None
    
    def simulate_orb_strategy(self, symbol, data, tp_multiplier=2.0, risk_per_trade=0.01):
        """Simulate ORB strategy on the data"""
        if data is None or len(data) < 10:
            return None
        
        trades = []
        capital = self.initial_capital
        
        # For daily data, use previous day's range as "ORB"
        for i in range(1, len(data)):
            # Use previous bar as the "opening range"
            orb_high = data['High'].iloc[i-1]
            orb_low = data['Low'].iloc[i-1]
            orb_range = orb_high - orb_low
            
            if orb_range <= 0:
                continue
            
            current_bar = data.iloc[i]
            
            # Check for breakout
            trade_taken = False
            
            # Long breakout
            if current_bar['Open'] < orb_high and current_bar['High'] > orb_high:
                entry = orb_high
                sl = orb_low
                tp = entry + (tp_multiplier * orb_range)
                
                # Calculate position size
                risk_amount = capital * risk_per_trade
                position_size = risk_amount / (entry - sl) if (entry - sl) > 0 else 0
                
                if position_size > 0:
                    # Determine exit
                    if current_bar['High'] >= tp:
                        exit_price = tp
                        pnl = position_size * (tp - entry)
                        exit_reason = 'TP'
                    elif current_bar['Low'] <= sl:
                        exit_price = sl
                        pnl = position_size * (sl - entry)
                        exit_reason = 'SL'
                    else:
                        exit_price = current_bar['Close']
                        pnl = position_size * (exit_price - entry)
                        exit_reason = 'Close'
                    
                    capital += pnl
                    trades.append({
                        'date': current_bar.name,
                        'type': 'LONG',
                        'entry': entry,
                        'exit': exit_price,
                        'sl': sl,
                        'tp': tp,
                        'pnl': pnl,
                        'exit_reason': exit_reason
                    })
                    trade_taken = True
            
            # Short breakout (if no long trade taken)
            if not trade_taken and current_bar['Open'] > orb_low and current_bar['Low'] < orb_low:
                entry = orb_low
                sl = orb_high
                tp = entry - (tp_multiplier * orb_range)
                
                # Calculate position size
                risk_amount = capital * risk_per_trade
                position_size = risk_amount / (sl - entry) if (sl - entry) > 0 else 0
                
                if position_size > 0:
                    # Determine exit
                    if current_bar['Low'] <= tp:
                        exit_price = tp
                        pnl = position_size * (entry - tp)
                        exit_reason = 'TP'
                    elif current_bar['High'] >= sl:
                        exit_price = sl
                        pnl = position_size * (entry - sl)
                        exit_reason = 'SL'
                    else:
                        exit_price = current_bar['Close']
                        pnl = position_size * (entry - exit_price)
                        exit_reason = 'Close'
                    
                    capital += pnl
                    trades.append({
                        'date': current_bar.name,
                        'type': 'SHORT',
                        'entry': entry,
                        'exit': exit_price,
                        'sl': sl,
                        'tp': tp,
                        'pnl': pnl,
                        'exit_reason': exit_reason
                    })
        
        # Calculate metrics
        if trades:
            trades_df = pd.DataFrame(trades)
            wins = len(trades_df[trades_df['pnl'] > 0])
            losses = len(trades_df[trades_df['pnl'] < 0])
            total_pnl = trades_df['pnl'].sum()
            
            win_rate = wins / len(trades) if len(trades) > 0 else 0
            avg_win = trades_df[trades_df['pnl'] > 0]['pnl'].mean() if wins > 0 else 0
            avg_loss = trades_df[trades_df['pnl'] < 0]['pnl'].mean() if losses > 0 else 0
            
            # Profit factor
            gross_profit = trades_df[trades_df['pnl'] > 0]['pnl'].sum() if wins > 0 else 0
            gross_loss = abs(trades_df[trades_df['pnl'] < 0]['pnl'].sum()) if losses > 0 else 1
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
            
            return {
                'symbol': symbol,
                'total_trades': len(trades),
                'wins': wins,
                'losses': losses,
                'win_rate': win_rate,
                'total_pnl': total_pnl,
                'avg_win': avg_win,
                'avg_loss': avg_loss,
                'profit_factor': profit_factor,
                'final_capital': capital,
                'return_pct': ((capital - self.initial_capital) / self.initial_capital) * 100,
                'trades': trades_df
            }
        
        return None

def main():
    """Run backtest for all symbols"""
    print("=" * 80)
    print("ORB STRATEGY BACKTEST - INITIAL RESULTS")
    print("=" * 80)
    
    # Optimal parameters from our analysis
    symbol_params = {
        'AAPL': {'tp': 2.0, 'risk': 0.015},
        'MSFT': {'tp': 2.0, 'risk': 0.01},
        'GOOGL': {'tp': 2.5, 'risk': 0.01},
        'AMZN': {'tp': 2.5, 'risk': 0.015},
        'TSLA': {'tp': 3.0, 'risk': 0.02},
        'SPY': {'tp': 1.5, 'risk': 0.01},
        'QQQ': {'tp': 2.0, 'risk': 0.015},
        'GLD': {'tp': 2.0, 'risk': 0.015},
    }
    
    backtest = SimpleORBBacktest()
    all_results = []
    
    print("\nDownloading data and running backtests...")
    print("-" * 40)
    
    for symbol, params in symbol_params.items():
        # Download data
        data = backtest.download_data(symbol)
        
        if data is not None:
            # Run backtest
            result = backtest.simulate_orb_strategy(
                symbol, 
                data, 
                tp_multiplier=params['tp'],
                risk_per_trade=params['risk']
            )
            
            if result:
                all_results.append(result)
    
    # Display results
    if all_results:
        print("\n" + "=" * 80)
        print("BACKTEST RESULTS SUMMARY")
        print("=" * 80)
        
        # Create summary table
        summary_data = []
        for r in all_results:
            summary_data.append({
                'Symbol': r['symbol'],
                'Trades': r['total_trades'],
                'Wins': r['wins'],
                'Losses': r['losses'],
                'Win Rate': f"{r['win_rate']*100:.1f}%",
                'Total PnL': f"${r['total_pnl']:.2f}",
                'Avg Win': f"${r['avg_win']:.2f}",
                'Avg Loss': f"${r['avg_loss']:.2f}",
                'Profit Factor': f"{r['profit_factor']:.2f}",
                'Return': f"{r['return_pct']:.2f}%"
            })
        
        summary_df = pd.DataFrame(summary_data)
        print("\n" + summary_df.to_string(index=False))
        
        # Overall statistics
        total_trades = sum(r['total_trades'] for r in all_results)
        total_wins = sum(r['wins'] for r in all_results)
        total_losses = sum(r['losses'] for r in all_results)
        total_pnl = sum(r['total_pnl'] for r in all_results)
        avg_return = np.mean([r['return_pct'] for r in all_results])
        
        print("\n" + "-" * 80)
        print("OVERALL STATISTICS")
        print("-" * 80)
        print(f"Total Trades: {total_trades}")
        print(f"Total Wins: {total_wins}")
        print(f"Total Losses: {total_losses}")
        print(f"Overall Win Rate: {(total_wins/total_trades)*100:.1f}%" if total_trades > 0 else "N/A")
        print(f"Total PnL: ${total_pnl:.2f}")
        print(f"Average Return: {avg_return:.2f}%")
        
        # Best and worst performers
        best = max(all_results, key=lambda x: x['return_pct'])
        worst = min(all_results, key=lambda x: x['return_pct'])
        
        print(f"\nBest Performer: {best['symbol']} ({best['return_pct']:.2f}% return)")
        print(f"Worst Performer: {worst['symbol']} ({worst['return_pct']:.2f}% return)")
        
        # Trading recommendations
        print("\n" + "=" * 80)
        print("TRADING RECOMMENDATIONS")
        print("=" * 80)
        
        profitable = [r for r in all_results if r['return_pct'] > 0]
        if profitable:
            print("\n✅ PROFITABLE SYMBOLS:")
            for r in sorted(profitable, key=lambda x: x['return_pct'], reverse=True):
                print(f"   {r['symbol']}: {r['return_pct']:.2f}% return, {r['win_rate']*100:.0f}% win rate")
        
        unprofitable = [r for r in all_results if r['return_pct'] <= 0]
        if unprofitable:
            print("\n⚠️ SYMBOLS NEEDING PARAMETER ADJUSTMENT:")
            for r in unprofitable:
                print(f"   {r['symbol']}: Consider reducing TP multiplier or adjusting risk")
        
        # Save results
        summary_df.to_csv('reports/initial_backtest_results.csv', index=False)
        print("\n✓ Results saved to reports/initial_backtest_results.csv")
        
        # Create a simple performance chart
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # Returns by symbol
        symbols = [r['symbol'] for r in all_results]
        returns = [r['return_pct'] for r in all_results]
        colors = ['green' if r > 0 else 'red' for r in returns]
        
        axes[0, 0].bar(symbols, returns, color=colors, alpha=0.7)
        axes[0, 0].set_title('Returns by Symbol')
        axes[0, 0].set_ylabel('Return (%)')
        axes[0, 0].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        axes[0, 0].grid(True, alpha=0.3)
        
        # Win rates
        win_rates = [r['win_rate']*100 for r in all_results]
        axes[0, 1].bar(symbols, win_rates, color='blue', alpha=0.7)
        axes[0, 1].set_title('Win Rates by Symbol')
        axes[0, 1].set_ylabel('Win Rate (%)')
        axes[0, 1].axhline(y=50, color='red', linestyle='--', linewidth=1)
        axes[0, 1].grid(True, alpha=0.3)
        
        # Profit factors
        profit_factors = [r['profit_factor'] for r in all_results]
        axes[1, 0].bar(symbols, profit_factors, color='purple', alpha=0.7)
        axes[1, 0].set_title('Profit Factor by Symbol')
        axes[1, 0].set_ylabel('Profit Factor')
        axes[1, 0].axhline(y=1, color='red', linestyle='--', linewidth=1)
        axes[1, 0].grid(True, alpha=0.3)
        
        # Trade distribution
        trades_count = [r['total_trades'] for r in all_results]
        axes[1, 1].bar(symbols, trades_count, color='orange', alpha=0.7)
        axes[1, 1].set_title('Number of Trades by Symbol')
        axes[1, 1].set_ylabel('Trade Count')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('reports/initial_backtest_charts.png', dpi=100)
        print("✓ Charts saved to reports/initial_backtest_charts.png")
        
    else:
        print("\nNo results to display. Data download may have failed.")

if __name__ == "__main__":
    main()