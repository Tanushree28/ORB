#!/usr/bin/env python3
"""
Backtesting Engine for ORB Strategy
Runs comprehensive backtests and generates performance reports
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from datetime import datetime
import yaml
from tqdm import tqdm
import warnings
from typing import Dict, List, Optional, Tuple
warnings.filterwarnings('ignore')

from data_downloader import DataDownloader
from strategy.orb_strategy import ORBStrategy

class BacktestEngine:
    def __init__(self, config_path='configs/config.yaml'):
        """Initialize backtest engine"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.data_downloader = DataDownloader(config_path)
        self.strategy = ORBStrategy(config_path)
        
        self.results = {}
        self.all_trades = []
        
    def run_single_backtest(self, symbol: str, interval: str = '5m') -> Dict:
        """
        Run backtest for a single symbol
        
        Args:
            symbol: Trading symbol
            interval: Time interval for data
            
        Returns:
            Dictionary with backtest results
        """
        print(f"\nRunning backtest for {symbol} ({interval})...")
        
        # Load data
        data = self.data_downloader.get_data(symbol, interval)
        
        if data is None or data.empty:
            print(f"No data available for {symbol}")
            return None
        
        # Run strategy backtest
        trades_df = self.strategy.backtest(data, symbol)
        
        # Calculate metrics
        metrics = self.strategy.calculate_metrics(trades_df)
        
        # Store results
        result = {
            'symbol': symbol,
            'interval': interval,
            'trades': trades_df,
            'metrics': metrics,
            'data_points': len(data),
            'date_range': f"{data.index[0]} to {data.index[-1]}"
        }
        
        return result
    
    def run_all_backtests(self, interval: str = '5m'):
        """
        Run backtests for all configured symbols

        Args:
            interval: Time interval for data
        """
        print("=" * 60)
        print(f"Starting Comprehensive Backtest ({interval} timeframe)")
        print("=" * 60)

        # reset containers for this run to avoid cross-run accumulation
        all_results = []
        self.all_trades = []
        self.results[interval] = []

        for symbol_info in self.data_downloader.symbols:
            symbol = symbol_info['symbol']
            name = symbol_info['name']
            category = symbol_info['category']

            # (Optional) you don't need to re-instantiate the strategy because
            # ORBStrategy.backtest() resets internal state at the start.
            # If you prefer to recreate it each time, keep the next line.
            self.strategy = ORBStrategy('configs/config.yaml')

            try:
                # Run backtest
                result = self.run_single_backtest(symbol, interval)
            except Exception as e:
                print(f"[WARN] Skipping {symbol} due to error: {e}")
                continue

            if result:
                result['name'] = name
                result['category'] = category
                all_results.append(result)

                # Collect all trades
                trades_df = result.get('trades')
                if trades_df is not None and not trades_df.empty:
                    trades = trades_df.copy()
                    trades['symbol_name'] = name
                    trades['category'] = category
                    self.all_trades.append(trades)

        # store per-interval results
        self.results[interval] = all_results

        # combine all trades for this run
        self.all_trades_df = (
            pd.concat(self.all_trades, ignore_index=True)
            if self.all_trades else pd.DataFrame()
        )

        return all_results

    
    def generate_summary_report(self, interval: str = '5m'):
        """Generate summary performance report"""
        if interval not in self.results:
            print(f"No results available for {interval}")
            return None
        
        results = self.results[interval]
        
        print("\n" + "=" * 80)
        print(f"BACKTEST SUMMARY REPORT ({interval} timeframe)")
        print("=" * 80)
        
        # Create summary DataFrame
        summary_data = []
        
        for result in results:
            if result and 'metrics' in result:
                metrics = result['metrics']
                summary_data.append({
                    'Symbol': result['symbol'],
                    'Name': result['name'],
                    'Category': result['category'],
                    'Total Trades': metrics['total_trades'],
                    'Win Rate': f"{metrics['win_rate']*100:.1f}%",
                    'Total PnL': f"${metrics['total_pnl']:.2f}",
                    'Avg Win': f"${metrics['average_win']:.2f}",
                    'Avg Loss': f"${metrics['average_loss']:.2f}",
                    'Profit Factor': f"{metrics['profit_factor']:.2f}",
                    'Max DD': f"${metrics['max_drawdown']:.2f}",
                    'Sharpe': f"{metrics['sharpe_ratio']:.2f}",
                    'Return %': f"{metrics['return_pct']:.1f}%"
                })
        
        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            
            # Sort by return percentage
            summary_df['Return_Sort'] = summary_df['Return %'].str.rstrip('%').astype(float)
            summary_df = summary_df.sort_values('Return_Sort', ascending=False)
            summary_df = summary_df.drop('Return_Sort', axis=1)
            
            print("\nPerformance by Symbol:")
            print(summary_df.to_string(index=False))
            
            # Category summary
            print("\n" + "-" * 80)
            print("Performance by Category:")
            print("-" * 80)
            
            category_summary = []
            for category in summary_df['Category'].unique():
                cat_data = summary_df[summary_df['Category'] == category]
                
                # Parse numeric values for aggregation
                total_trades = cat_data['Total Trades'].sum()
                avg_win_rate = cat_data['Win Rate'].str.rstrip('%').astype(float).mean()
                total_pnl = cat_data['Total PnL'].str.lstrip('$').astype(float).sum()
                avg_return = cat_data['Return %'].str.rstrip('%').astype(float).mean()
                
                category_summary.append({
                    'Category': category.capitalize(),
                    'Symbols': len(cat_data),
                    'Total Trades': total_trades,
                    'Avg Win Rate': f"{avg_win_rate:.1f}%",
                    'Total PnL': f"${total_pnl:.2f}",
                    'Avg Return': f"{avg_return:.1f}%"
                })
            
            category_df = pd.DataFrame(category_summary)
            print(category_df.to_string(index=False))
            
            # Overall statistics
            print("\n" + "-" * 80)
            print("Overall Statistics:")
            print("-" * 80)
            
            if not self.all_trades_df.empty:
                total_trades = len(self.all_trades_df)
                total_wins = len(self.all_trades_df[self.all_trades_df['pnl'] > 0])
                total_losses = len(self.all_trades_df[self.all_trades_df['pnl'] < 0])
                overall_win_rate = (total_wins / total_trades) * 100 if total_trades > 0 else 0
                total_pnl = self.all_trades_df['pnl'].sum()
                
                print(f"Total Trades Executed: {total_trades}")
                print(f"Total Winning Trades: {total_wins}")
                print(f"Total Losing Trades: {total_losses}")
                print(f"Overall Win Rate: {overall_win_rate:.1f}%")
                print(f"Total PnL: ${total_pnl:.2f}")
                print(f"Average Trade PnL: ${total_pnl/total_trades:.2f}" if total_trades > 0 else "N/A")
                
                # Best and worst trades
                best_trade = self.all_trades_df.loc[self.all_trades_df['pnl'].idxmax()]
                worst_trade = self.all_trades_df.loc[self.all_trades_df['pnl'].idxmin()]
                
                print(f"\nBest Trade:")
                print(f"  Symbol: {best_trade['symbol']}")
                print(f"  Type: {best_trade['type']}")
                print(f"  PnL: ${best_trade['pnl']:.2f}")
                print(f"  Date: {best_trade['entry_time']}")
                
                print(f"\nWorst Trade:")
                print(f"  Symbol: {worst_trade['symbol']}")
                print(f"  Type: {worst_trade['type']}")
                print(f"  PnL: ${worst_trade['pnl']:.2f}")
                print(f"  Date: {worst_trade['entry_time']}")
            
            # Save summary to CSV
            summary_df.to_csv('reports/backtest_summary.csv', index=False)
            print(f"\nSummary saved to reports/backtest_summary.csv")
            
            return summary_df
        
        return None
    
    def save_trades_to_csv(self):
        """Save all trades to CSV file"""
        if not self.all_trades_df.empty:
            # Sort by entry time
            self.all_trades_df = self.all_trades_df.sort_values('entry_time')
            
            # Save to CSV
            filename = 'reports/all_trades.csv'
            self.all_trades_df.to_csv(filename, index=False)
            print(f"All trades saved to {filename}")
            
            return self.all_trades_df
        else:
            print("No trades to save")
            return None
    
    def optimize_parameters(self, symbol: str, param_ranges: Dict):
        """
        Optimize strategy parameters for a specific symbol
        
        Args:
            symbol: Trading symbol
            param_ranges: Dictionary with parameter ranges to test
        """
        print(f"\nOptimizing parameters for {symbol}...")
        
        # Load data once
        data = self.data_downloader.get_data(symbol, '5m')
        
        if data is None:
            print(f"No data available for {symbol}")
            return None
        
        best_result = None
        best_params = None
        best_return = -float('inf')
        
        results = []
        
        # Default parameter ranges
        if param_ranges is None:
            param_ranges = {
                'tp_multiplier': [1.5, 2.0, 2.5, 3.0],
                'orb_duration': [15, 30, 45],
                'risk_per_trade': [0.005, 0.01, 0.015, 0.02]
            }
        
        total_combinations = 1
        for values in param_ranges.values():
            total_combinations *= len(values)
        
        print(f"Testing {total_combinations} parameter combinations...")
        
        # Test all combinations
        from itertools import product
        
        param_names = list(param_ranges.keys())
        param_values = list(param_ranges.values())
        
        for combination in tqdm(product(*param_values), total=total_combinations):
            params = dict(zip(param_names, combination))
            
            # Create new strategy with modified parameters
            strategy = ORBStrategy('configs/config.yaml')
            
            # Update parameters
            if 'tp_multiplier' in params:
                strategy.tp_multiplier = params['tp_multiplier']
            if 'orb_duration' in params:
                strategy.orb_duration = params['orb_duration']
            if 'risk_per_trade' in params:
                strategy.risk_per_trade = params['risk_per_trade']
            
            # Run backtest
            trades_df = strategy.backtest(data, symbol)
            metrics = strategy.calculate_metrics(trades_df)
            
            # Store result
            result = {
                'params': params,
                'metrics': metrics,
                'return_pct': metrics['return_pct']
            }
            results.append(result)
            
            # Check if this is the best result
            if metrics['return_pct'] > best_return:
                best_return = metrics['return_pct']
                best_params = params
                best_result = result
        
        # Print optimization results
        print("\n" + "=" * 60)
        print(f"OPTIMIZATION RESULTS FOR {symbol}")
        print("=" * 60)
        
        print(f"\nBest Parameters:")
        for param, value in best_params.items():
            print(f"  {param}: {value}")
        
        print(f"\nBest Performance:")
        print(f"  Return: {best_result['metrics']['return_pct']:.2f}%")
        print(f"  Win Rate: {best_result['metrics']['win_rate']*100:.1f}%")
        print(f"  Profit Factor: {best_result['metrics']['profit_factor']:.2f}")
        print(f"  Sharpe Ratio: {best_result['metrics']['sharpe_ratio']:.2f}")
        
        # Save optimization results
        opt_df = pd.DataFrame(results)
        opt_df = opt_df.sort_values('return_pct', ascending=False)
        
        filename = f'reports/optimization_{symbol.replace("=", "_")}.csv'
        opt_df.to_csv(filename, index=False)
        print(f"\nOptimization results saved to {filename}")
        
        return best_params, best_result


def main():
    """Main function to run backtesting"""
    print("=" * 60)
    print("ORB Strategy Backtesting Engine")
    print("=" * 60)
    
    # Initialize backtest engine
    engine = BacktestEngine()
    
    # Run backtests for all symbols
    print("\nRunning backtests for 5-minute timeframe...")
    engine.run_all_backtests('5m')
    
    # Generate summary report
    engine.generate_summary_report('5m')
    
    # Save all trades
    engine.save_trades_to_csv()
    
    # Run optimization for a sample symbol
    print("\n" + "=" * 60)
    print("Parameter Optimization Example")
    print("=" * 60)
    
    # Optimize for AAPL
    param_ranges = {
        'tp_multiplier': [1.5, 2.0, 2.5, 3.0],
        'risk_per_trade': [0.01, 0.015, 0.02]
    }
    
    best_params, best_result = engine.optimize_parameters('AAPL', param_ranges)
    
    print("\nBacktesting complete!")


if __name__ == "__main__":
    main()