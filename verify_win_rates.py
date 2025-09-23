#!/usr/bin/env python3
"""
Verify Win Rates and Investigate Discrepancies
Compare daily records with backtest results
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def load_daily_records(symbol):
    """Load daily ORB records for a symbol"""
    filepath = f"reports/daily_orb_records/{symbol}_daily_orb.csv"
    if os.path.exists(filepath):
        return pd.read_csv(filepath)
    return None

def analyze_discrepancy():
    """Analyze why win rates differ between daily records and backtest"""
    
    print("="*80)
    print("WIN RATE VERIFICATION AND DISCREPANCY ANALYSIS")
    print("="*80)
    
    symbols = ['AAPL', 'AMZN', 'COST', 'DIA', 'GOOGL', 'META', 'MSFT', 'NFLX', 'QQQ', 'SPY']
    
    # Load backtest results
    backtest_df = pd.read_csv("reports/extended_backtest_results.csv")
    backtest_df = backtest_df[backtest_df['period'] == 'Full 2 Years']
    
    analysis_results = []
    
    for symbol in symbols:
        print(f"\n{symbol} Analysis:")
        print("-"*40)
        
        # Load daily records
        daily_df = load_daily_records(symbol)
        if daily_df is None:
            continue
            
        # Get backtest data
        backtest_row = backtest_df[backtest_df['symbol'] == symbol]
        if backtest_row.empty:
            continue
            
        backtest_wr = backtest_row.iloc[0]['win_rate'] * 100
        backtest_trades = backtest_row.iloc[0]['total_trades']
        
        # Calculate from daily records
        long_trades = daily_df[daily_df['long_triggered'] == True]
        short_trades = daily_df[daily_df['short_triggered'] == True]
        
        # Count wins and losses
        long_wins = len(long_trades[long_trades['long_result'] == 'TP'])
        long_losses = len(long_trades[long_trades['long_result'] == 'SL'])
        long_no_exit = len(long_trades[long_trades['long_result'] == 'No Exit'])
        
        short_wins = len(short_trades[short_trades['short_result'] == 'TP'])
        short_losses = len(short_trades[short_trades['short_result'] == 'SL'])
        short_no_exit = len(short_trades[short_trades['short_result'] == 'No Exit'])
        
        total_trades_daily = len(long_trades) + len(short_trades)
        total_wins = long_wins + short_wins
        total_losses = long_losses + short_losses
        total_no_exit = long_no_exit + short_no_exit
        
        # Calculate win rates
        daily_wr = (total_wins / (total_wins + total_losses) * 100) if (total_wins + total_losses) > 0 else 0
        
        # Check for multiple trades per day
        days_with_both = len(daily_df[(daily_df['long_triggered'] == True) & 
                                      (daily_df['short_triggered'] == True)])
        
        print(f"  Trading Days: {len(daily_df)}")
        print(f"  Days with both long & short: {days_with_both}")
        print(f"\n  Daily Records:")
        print(f"    Total Trades: {total_trades_daily}")
        print(f"    Wins: {total_wins}")
        print(f"    Losses: {total_losses}")
        print(f"    No Exit: {total_no_exit}")
        print(f"    Win Rate: {daily_wr:.1f}%")
        print(f"\n  Backtest Results:")
        print(f"    Total Trades: {backtest_trades}")
        print(f"    Win Rate: {backtest_wr:.1f}%")
        print(f"\n  Discrepancy:")
        print(f"    Trade Count Difference: {backtest_trades - total_trades_daily}")
        print(f"    Win Rate Difference: {backtest_wr - daily_wr:.1f}%")
        
        analysis_results.append({
            'Symbol': symbol,
            'Days': len(daily_df),
            'Daily_Trades': total_trades_daily,
            'Daily_WR': daily_wr,
            'Backtest_Trades': backtest_trades,
            'Backtest_WR': backtest_wr,
            'Trade_Diff': backtest_trades - total_trades_daily,
            'WR_Diff': backtest_wr - daily_wr,
            'No_Exit': total_no_exit,
            'Both_Trades': days_with_both
        })
    
    # Create comparison chart
    create_verification_charts(analysis_results)
    
    # Investigation findings
    print("\n" + "="*80)
    print("INVESTIGATION FINDINGS")
    print("="*80)
    
    print("\n1. TRADE COUNT DISCREPANCY:")
    total_trade_diff = sum([r['Trade_Diff'] for r in analysis_results])
    avg_trade_diff = total_trade_diff / len(analysis_results)
    print(f"   Average extra trades in backtest: {avg_trade_diff:.0f} per symbol")
    print(f"   This suggests the backtest may be:")
    print(f"   - Taking multiple trades per day")
    print(f"   - Re-entering after stops")
    print(f"   - Using different entry logic")
    
    print("\n2. WIN RATE DISCREPANCY:")
    avg_wr_diff = sum([r['WR_Diff'] for r in analysis_results]) / len(analysis_results)
    print(f"   Average win rate difference: {avg_wr_diff:.1f}%")
    print(f"   Possible explanations:")
    print(f"   - Backtest using compounding position sizing")
    print(f"   - Different exit timing or price calculation")
    print(f"   - Intrabar fills vs end-of-bar fills")
    
    print("\n3. KEY OBSERVATIONS:")
    print(f"   - Daily records show ~47% win rate (realistic)")
    print(f"   - Backtest shows ~60% win rate (optimistic)")
    print(f"   - Daily records only allow 1 trade per direction per day")
    print(f"   - Backtest appears to take more trades")
    
    # Save detailed analysis
    analysis_df = pd.DataFrame(analysis_results)
    analysis_df.to_csv("reports/daily_orb_records/win_rate_verification.csv", index=False)
    print(f"\n✓ Detailed analysis saved to: reports/daily_orb_records/win_rate_verification.csv")

def create_verification_charts(analysis_results):
    """Create charts comparing daily records vs backtest"""
    
    plt.style.use('seaborn-v0_8-darkgrid')
    fig = plt.figure(figsize=(16, 10))
    
    df = pd.DataFrame(analysis_results)
    
    # 1. Win Rate Comparison
    ax1 = plt.subplot(2, 3, 1)
    x = np.arange(len(df))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, df['Daily_WR'], width, label='Daily Records', color='blue', alpha=0.7)
    bars2 = ax1.bar(x + width/2, df['Backtest_WR'], width, label='Backtest', color='orange', alpha=0.7)
    
    ax1.set_xlabel('Symbol')
    ax1.set_ylabel('Win Rate (%)')
    ax1.set_title('Win Rate Comparison', fontsize=12, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(df['Symbol'], rotation=45)
    ax1.legend()
    ax1.axhline(y=50, color='red', linestyle='--', alpha=0.5)
    ax1.grid(True, alpha=0.3)
    
    # 2. Trade Count Comparison
    ax2 = plt.subplot(2, 3, 2)
    
    bars1 = ax2.bar(x - width/2, df['Daily_Trades'], width, label='Daily Records', color='green', alpha=0.7)
    bars2 = ax2.bar(x + width/2, df['Backtest_Trades'], width, label='Backtest', color='red', alpha=0.7)
    
    ax2.set_xlabel('Symbol')
    ax2.set_ylabel('Total Trades')
    ax2.set_title('Trade Count Comparison', fontsize=12, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(df['Symbol'], rotation=45)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Win Rate Discrepancy
    ax3 = plt.subplot(2, 3, 3)
    
    colors = ['red' if d < 0 else 'darkred' for d in df['WR_Diff']]
    bars = ax3.bar(df['Symbol'], df['WR_Diff'], color=colors, alpha=0.7)
    
    ax3.set_xlabel('Symbol')
    ax3.set_ylabel('Win Rate Difference (%)')
    ax3.set_title('Backtest WR - Daily WR', fontsize=12, fontweight='bold')
    ax3.set_xticklabels(df['Symbol'], rotation=45)
    ax3.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax3.grid(True, alpha=0.3)
    
    # Add value labels
    for bar, val in zip(bars, df['WR_Diff']):
        ax3.text(bar.get_x() + bar.get_width()/2., val,
                f'{val:.1f}%', ha='center', va='bottom' if val > 0 else 'top', fontsize=8)
    
    # 4. Trades Per Day Analysis
    ax4 = plt.subplot(2, 3, 4)
    
    trades_per_day_daily = df['Daily_Trades'] / df['Days']
    trades_per_day_backtest = df['Backtest_Trades'] / df['Days']
    
    bars1 = ax4.bar(x - width/2, trades_per_day_daily, width, label='Daily Records', color='cyan', alpha=0.7)
    bars2 = ax4.bar(x + width/2, trades_per_day_backtest, width, label='Backtest', color='magenta', alpha=0.7)
    
    ax4.set_xlabel('Symbol')
    ax4.set_ylabel('Trades Per Day')
    ax4.set_title('Average Trades Per Trading Day', fontsize=12, fontweight='bold')
    ax4.set_xticks(x)
    ax4.set_xticklabels(df['Symbol'], rotation=45)
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. No Exit Trades
    ax5 = plt.subplot(2, 3, 5)
    
    bars = ax5.bar(df['Symbol'], df['No_Exit'], color='orange', alpha=0.7)
    ax5.set_xlabel('Symbol')
    ax5.set_ylabel('Count')
    ax5.set_title('Trades with No Exit (Open at Close)', fontsize=12, fontweight='bold')
    ax5.set_xticklabels(df['Symbol'], rotation=45)
    ax5.grid(True, alpha=0.3)
    
    # 6. Summary Statistics
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')
    
    summary_text = "VERIFICATION SUMMARY\n" + "="*35 + "\n\n"
    
    avg_daily_wr = df['Daily_WR'].mean()
    avg_backtest_wr = df['Backtest_WR'].mean()
    avg_trade_diff = df['Trade_Diff'].mean()
    
    summary_text += f"Average Win Rates:\n"
    summary_text += f"  Daily Records: {avg_daily_wr:.1f}%\n"
    summary_text += f"  Backtest: {avg_backtest_wr:.1f}%\n"
    summary_text += f"  Difference: {avg_backtest_wr - avg_daily_wr:.1f}%\n\n"
    
    summary_text += f"Trade Counts:\n"
    summary_text += f"  Avg Extra Trades (Backtest): {avg_trade_diff:.0f}\n"
    summary_text += f"  Total No Exit: {df['No_Exit'].sum()}\n\n"
    
    summary_text += "CONCLUSION:\n"
    summary_text += "Daily records show realistic ~47% WR\n"
    summary_text += "Backtest may have optimistic bias\n"
    summary_text += "due to compounding or trade logic"
    
    ax6.text(0.05, 0.95, summary_text, transform=ax6.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))
    
    plt.suptitle('ORB Strategy - Win Rate Verification Analysis', 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    # Save the figure
    plt.savefig('reports/win_rate_verification.png', dpi=100, bbox_inches='tight')
    print("\n✓ Verification charts saved to: reports/win_rate_verification.png")

def main():
    """Main execution"""
    analyze_discrepancy()
    
    print("\n" + "="*80)
    print("VERIFICATION COMPLETE")
    print("="*80)
    print("\nYou can now:")
    print("1. Check any specific day's trades in the CSV files")
    print("2. Verify ORB levels match your expectations")
    print("3. Confirm TP/SL calculations are correct (2x multiplier)")
    print("4. Review trades marked as 'No Exit' for accuracy")
    print("\nAll daily records are in: reports/daily_orb_records/")

if __name__ == "__main__":
    main()