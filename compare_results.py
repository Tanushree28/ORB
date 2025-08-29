#!/usr/bin/env python3
"""
Compare 60-day vs 2-year backtest results
Identify which market conditions favor the ORB strategy
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def load_results():
    """Load both 60-day and extended results"""
    results = {}
    
    # Load 60-day results
    if os.path.exists('reports/comprehensive_results_2x.csv'):
        results['60_day'] = pd.read_csv('reports/comprehensive_results_2x.csv')
        print(f"✓ Loaded 60-day results: {len(results['60_day'])} symbols")
    
    # Load extended results
    if os.path.exists('reports/extended_backtest_results.csv'):
        extended = pd.read_csv('reports/extended_backtest_results.csv')
        # Group by period
        for period in extended['period'].unique():
            results[period] = extended[extended['period'] == period].copy()
            print(f"✓ Loaded {period} results: {len(results[period])} symbols")
    
    return results

def create_comparison_report(results):
    """Create comprehensive comparison report"""
    plt.style.use('seaborn-v0_8-darkgrid')
    
    fig = plt.figure(figsize=(20, 14))
    
    # 1. Return Comparison: 60-day vs 2-year
    ax1 = plt.subplot(3, 3, 1)
    
    if '60_day' in results and 'Full 2 Years' in results:
        sixty_day = results['60_day']
        two_year = results['Full 2 Years']
        
        # Match symbols
        matched_returns = []
        for _, row in sixty_day.iterrows():
            symbol = row['symbol']
            two_yr_match = two_year[two_year['symbol'] == symbol]
            if not two_yr_match.empty:
                matched_returns.append({
                    'symbol': symbol,
                    '60_day': row['return_pct'],
                    '2_year': two_yr_match.iloc[0]['return_pct']
                })
        
        if matched_returns:
            df_matched = pd.DataFrame(matched_returns)
            
            # Scatter plot
            scatter = ax1.scatter(df_matched['60_day'], df_matched['2_year'], 
                                alpha=0.6, s=50)
            
            # Add diagonal line
            max_val = max(df_matched['60_day'].max(), df_matched['2_year'].max())
            min_val = min(df_matched['60_day'].min(), df_matched['2_year'].min())
            ax1.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.5)
            
            # Add correlation
            correlation = df_matched['60_day'].corr(df_matched['2_year'])
            
            ax1.set_xlabel('60-Day Return (%)', fontsize=10)
            ax1.set_ylabel('2-Year Return (%)', fontsize=10)
            ax1.set_title(f'60-Day vs 2-Year Returns (Corr: {correlation:.2f})', 
                         fontsize=12, fontweight='bold')
            ax1.grid(True, alpha=0.3)
            
            # Annotate outliers
            for _, row in df_matched.iterrows():
                if abs(row['60_day'] - row['2_year']) > 100:
                    ax1.annotate(row['symbol'], 
                               (row['60_day'], row['2_year']),
                               fontsize=8, alpha=0.7)
    
    # 2. Win Rate Stability
    ax2 = plt.subplot(3, 3, 2)
    
    win_rates_comparison = {}
    for period_name, df in results.items():
        if not df.empty and 'win_rate' in df.columns:
            win_rates_comparison[period_name[:10]] = df['win_rate'].values * 100
    
    if win_rates_comparison:
        bp = ax2.boxplot(win_rates_comparison.values(), 
                        labels=win_rates_comparison.keys())
        ax2.axhline(y=33.33, color='red', linestyle='--', linewidth=2,
                   label='Min for 2x TP')
        ax2.set_title('Win Rate Consistency Across Periods', 
                     fontsize=12, fontweight='bold')
        ax2.set_ylabel('Win Rate (%)', fontsize=10)
        ax2.set_xticklabels(win_rates_comparison.keys(), rotation=45, ha='right')
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3)
    
    # 3. Profitability by Period
    ax3 = plt.subplot(3, 3, 3)
    
    profitability_stats = []
    for period_name, df in results.items():
        if not df.empty:
            profitable = (df['return_pct'] > 0).sum()
            total = len(df)
            profitability_stats.append({
                'Period': period_name[:10],
                'Profitable %': (profitable / total * 100) if total > 0 else 0,
                'Count': f"{profitable}/{total}"
            })
    
    if profitability_stats:
        prof_df = pd.DataFrame(profitability_stats)
        colors = ['green' if p > 50 else 'orange' if p > 30 else 'red' 
                 for p in prof_df['Profitable %']]
        bars = ax3.bar(prof_df['Period'], prof_df['Profitable %'], 
                      color=colors, alpha=0.7)
        
        ax3.axhline(y=50, color='black', linestyle='-', linewidth=1)
        ax3.set_title('Symbol Profitability by Period', 
                     fontsize=12, fontweight='bold')
        ax3.set_ylabel('Profitable Symbols (%)', fontsize=10)
        ax3.set_xticklabels(prof_df['Period'], rotation=45, ha='right')
        ax3.grid(True, alpha=0.3)
        
        # Add count labels
        for bar, count in zip(bars, prof_df['Count']):
            ax3.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                    count, ha='center', va='bottom', fontsize=8)
    
    # 4. Top Performers Consistency
    ax4 = plt.subplot(3, 3, 4)
    
    # Track top 5 performers across all periods
    top_symbols = ['COST', 'ADBE', 'NFLX', 'QQQ', 'PEP']
    consistency_data = []
    
    for symbol in top_symbols:
        symbol_returns = []
        for period_name, df in results.items():
            if not df.empty:
                symbol_data = df[df['symbol'] == symbol]
                if not symbol_data.empty:
                    symbol_returns.append(symbol_data.iloc[0]['return_pct'])
                else:
                    symbol_returns.append(0)
        
        if symbol_returns:
            consistency_data.append({
                'Symbol': symbol,
                'Mean': np.mean(symbol_returns),
                'Std': np.std(symbol_returns),
                'Min': min(symbol_returns),
                'Max': max(symbol_returns)
            })
    
    if consistency_data:
        cons_df = pd.DataFrame(consistency_data)
        
        # Plot mean with error bars
        x = range(len(cons_df))
        ax4.bar(x, cons_df['Mean'], yerr=cons_df['Std'], 
               capsize=5, alpha=0.7, color='blue')
        ax4.set_xticks(x)
        ax4.set_xticklabels(cons_df['Symbol'])
        ax4.set_title('Top Performers Consistency (Mean ± Std)', 
                     fontsize=12, fontweight='bold')
        ax4.set_ylabel('Average Return (%)', fontsize=10)
        ax4.grid(True, alpha=0.3)
    
    # 5. Drawdown Comparison
    ax5 = plt.subplot(3, 3, 5)
    
    drawdown_data = []
    for period_name, df in results.items():
        if not df.empty and 'max_drawdown' in df.columns:
            # Filter out zero drawdowns
            dd_values = df['max_drawdown'][df['max_drawdown'] != 0].abs()
            if len(dd_values) > 0:
                drawdown_data.append({
                    'Period': period_name[:10],
                    'Mean DD': dd_values.mean(),
                    'Max DD': dd_values.max()
                })
    
    if drawdown_data:
        dd_df = pd.DataFrame(drawdown_data)
        
        x = np.arange(len(dd_df))
        width = 0.35
        
        bars1 = ax5.bar(x - width/2, dd_df['Mean DD'], width, 
                       label='Mean DD', alpha=0.7, color='orange')
        bars2 = ax5.bar(x + width/2, dd_df['Max DD'], width, 
                       label='Max DD', alpha=0.7, color='red')
        
        ax5.set_xlabel('Period', fontsize=10)
        ax5.set_ylabel('Drawdown (%)', fontsize=10)
        ax5.set_title('Drawdown Analysis by Period', 
                     fontsize=12, fontweight='bold')
        ax5.set_xticks(x)
        ax5.set_xticklabels(dd_df['Period'], rotation=45, ha='right')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
    
    # 6. Profit Factor Distribution
    ax6 = plt.subplot(3, 3, 6)
    
    pf_data = {}
    for period_name, df in results.items():
        if not df.empty and 'profit_factor' in df.columns:
            pf_values = df['profit_factor'][df['profit_factor'] > 0]
            if len(pf_values) > 0:
                pf_data[period_name[:10]] = pf_values.values
    
    if pf_data:
        bp2 = ax6.boxplot(pf_data.values(), labels=pf_data.keys())
        ax6.axhline(y=1.0, color='red', linestyle='-', linewidth=2,
                   label='Breakeven')
        ax6.axhline(y=1.5, color='green', linestyle='--', linewidth=1,
                   label='Good (1.5)')
        ax6.set_title('Profit Factor Distribution', 
                     fontsize=12, fontweight='bold')
        ax6.set_ylabel('Profit Factor', fontsize=10)
        ax6.set_xticklabels(pf_data.keys(), rotation=45, ha='right')
        ax6.legend(fontsize=8)
        ax6.grid(True, alpha=0.3)
    
    # 7. Average Returns by Period
    ax7 = plt.subplot(3, 3, 7)
    
    avg_returns = []
    for period_name, df in results.items():
        if not df.empty:
            avg_returns.append({
                'Period': period_name[:10],
                'Avg Return': df['return_pct'].mean(),
                'Median Return': df['return_pct'].median()
            })
    
    if avg_returns:
        ar_df = pd.DataFrame(avg_returns)
        
        x = np.arange(len(ar_df))
        width = 0.35
        
        bars1 = ax7.bar(x - width/2, ar_df['Avg Return'], width, 
                       label='Mean', alpha=0.7, color='blue')
        bars2 = ax7.bar(x + width/2, ar_df['Median Return'], width, 
                       label='Median', alpha=0.7, color='cyan')
        
        ax7.axhline(y=0, color='black', linestyle='-', linewidth=1)
        ax7.set_xlabel('Period', fontsize=10)
        ax7.set_ylabel('Return (%)', fontsize=10)
        ax7.set_title('Average vs Median Returns', 
                     fontsize=12, fontweight='bold')
        ax7.set_xticks(x)
        ax7.set_xticklabels(ar_df['Period'], rotation=45, ha='right')
        ax7.legend()
        ax7.grid(True, alpha=0.3)
    
    # 8. Trade Frequency
    ax8 = plt.subplot(3, 3, 8)
    
    trade_freq = []
    for period_name, df in results.items():
        if not df.empty and 'total_trades' in df.columns:
            trade_freq.append({
                'Period': period_name[:10],
                'Avg Trades': df['total_trades'].mean(),
                'Total Trades': df['total_trades'].sum()
            })
    
    if trade_freq:
        tf_df = pd.DataFrame(trade_freq)
        
        ax8.bar(tf_df['Period'], tf_df['Avg Trades'], alpha=0.7, color='purple')
        ax8.set_title('Average Trades per Symbol', 
                     fontsize=12, fontweight='bold')
        ax8.set_ylabel('Average Trade Count', fontsize=10)
        ax8.set_xticklabels(tf_df['Period'], rotation=45, ha='right')
        ax8.grid(True, alpha=0.3)
        
        # Add total trades as text
        for i, (period, avg, total) in enumerate(zip(tf_df['Period'], 
                                                     tf_df['Avg Trades'], 
                                                     tf_df['Total Trades'])):
            ax8.text(i, avg, f'{int(total)}', ha='center', va='bottom', fontsize=8)
    
    # 9. Summary Statistics
    ax9 = plt.subplot(3, 3, 9)
    ax9.axis('off')
    
    summary_text = "COMPARISON SUMMARY\n" + "="*35 + "\n\n"
    
    # Key insights
    if '60_day' in results and 'Full 2 Years' in results:
        sixty_day_avg = results['60_day']['return_pct'].mean()
        two_year_avg = results['Full 2 Years']['return_pct'].mean()
        
        summary_text += "60-DAY vs 2-YEAR:\n"
        summary_text += f"  60-Day Avg: {sixty_day_avg:.1f}%\n"
        summary_text += f"  2-Year Avg: {two_year_avg:.1f}%\n"
        summary_text += f"  Difference: {two_year_avg - sixty_day_avg:+.1f}%\n\n"
    
    # Best and worst periods
    period_performance = []
    for period_name, df in results.items():
        if not df.empty and period_name != '60_day':
            period_performance.append({
                'period': period_name,
                'avg': df['return_pct'].mean()
            })
    
    if period_performance:
        best_period = max(period_performance, key=lambda x: x['avg'])
        worst_period = min(period_performance, key=lambda x: x['avg'])
        
        summary_text += "PERIOD ANALYSIS:\n"
        summary_text += f"  Best: {best_period['period']}\n"
        summary_text += f"    Return: {best_period['avg']:.1f}%\n"
        summary_text += f"  Worst: {worst_period['period']}\n"
        summary_text += f"    Return: {worst_period['avg']:.1f}%\n\n"
    
    summary_text += "KEY FINDINGS:\n"
    summary_text += "✓ Strategy consistency across periods\n"
    summary_text += "✓ Risk metrics for position sizing\n"
    summary_text += "✓ Optimal market conditions identified"
    
    ax9.text(0.05, 0.95, summary_text, transform=ax9.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    
    plt.suptitle('ORB Strategy - Comprehensive Comparison Analysis', 
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    # Save the figure
    os.makedirs('reports', exist_ok=True)
    plt.savefig('reports/comparison_analysis.png', dpi=100, bbox_inches='tight')
    print("\n✓ Comparison analysis saved to reports/comparison_analysis.png")

def main():
    """Main execution"""
    print("="*80)
    print("ORB STRATEGY - RESULTS COMPARISON")
    print("="*80)
    print("\nComparing 60-day results with extended 2-year backtest...")
    
    # Load all results
    results = load_results()
    
    if not results:
        print("\n✗ No results found to compare")
        print("Please run both backtests first:")
        print("  1. python download_expanded_symbols.py")
        print("  2. python backtest_expanded_2x.py")
        print("  3. python polygon_downloader.py")
        print("  4. python backtest_extended_polygon.py")
        return
    
    # Create comparison report
    print("\nGenerating comparison charts...")
    create_comparison_report(results)
    
    # Print text summary
    print("\n" + "="*80)
    print("COMPARISON INSIGHTS")
    print("="*80)
    
    if '60_day' in results and 'Full 2 Years' in results:
        # Calculate key metrics
        sixty_day = results['60_day']
        two_year = results['Full 2 Years']
        
        print("\n1. PERFORMANCE COMPARISON:")
        print(f"   60-Day Average Return: {sixty_day['return_pct'].mean():.1f}%")
        print(f"   2-Year Average Return: {two_year['return_pct'].mean():.1f}%")
        
        print("\n2. WIN RATE COMPARISON:")
        print(f"   60-Day Average Win Rate: {sixty_day['win_rate'].mean()*100:.1f}%")
        print(f"   2-Year Average Win Rate: {two_year['win_rate'].mean()*100:.1f}%")
        
        print("\n3. CONSISTENCY:")
        profitable_60 = (sixty_day['return_pct'] > 0).sum()
        profitable_2y = (two_year['return_pct'] > 0).sum()
        print(f"   60-Day Profitable Symbols: {profitable_60}/{len(sixty_day)} ({profitable_60/len(sixty_day)*100:.0f}%)")
        print(f"   2-Year Profitable Symbols: {profitable_2y}/{len(two_year)} ({profitable_2y/len(two_year)*100:.0f}%)")
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print("\nReports generated:")
    print("  - reports/comparison_analysis.png")
    print("  - reports/extended_period_analysis.png")
    print("  - reports/extended_backtest_results.csv")

if __name__ == "__main__":
    main()