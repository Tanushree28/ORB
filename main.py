#!/usr/bin/env python3
"""
Main Script for ORB Strategy
Orchestrates data download, backtesting, and reporting
"""

import os
import sys
import yaml
import argparse
from datetime import datetime
import pandas as pd

# Import modules
from data_downloader import DataDownloader
from strategy.orb_strategy import ORBStrategy
from backtesting.backtest import BacktestEngine
from reports.visualizer import StrategyVisualizer

class ORBTradingSystem:
    def __init__(self, config_path='configs/config.yaml'):
        """Initialize the ORB Trading System"""
        self.config_path = config_path
        
        # Load configuration
        with open(config_path, 'r') as f:
            self.config =  yaml.safe_load(f)
        
        # Initialize components
        self.downloader = DataDownloader(config_path)
        self.strategy = ORBStrategy(config_path)
        self.backtest_engine = BacktestEngine(config_path)
        self.visualizer = StrategyVisualizer()
        
        print("=" * 60)
        print("Opening Range Breakout (ORB) Trading System")
        print("=" * 60)
        print(f"Configuration loaded from: {config_path}")
    
    def download_data(self, force_download=False):
        """Download historical data for all symbols"""
        print("\n" + "=" * 60)
        print("Step 1: Data Download")
        print("=" * 60)
        
        # Check if data already exists
        data_files = os.listdir('data') if os.path.exists('data') else []
        
        if data_files and not force_download:
            print(f"Found {len(data_files)} existing data files.")
            response = input("Data already exists. Re-download? (y/n): ")
            if response.lower() != 'y':
                print("Using existing data.")
                return
        
        # Download data
        successful, failed = self.downloader.download_all()
        
        print(f"\nDownload complete: {successful} successful, {len(failed)} failed")
        
        # Verify data
        self.downloader.verify_data()
    
    def run_backtests(self, interval='5m'):
        """Run backtests for all configured symbols"""
        print("\n" + "=" * 60) 
        print("Step 2: Backtesting")
        print("=" * 60)
        
        # Run backtests
        results = self.backtest_engine.run_all_backtests(interval)
        
        # Generate summary
        summary_df = self.backtest_engine.generate_summary_report(interval)
        
        # Save trades
        trades_df = self.backtest_engine.save_trades_to_csv()
        
        return summary_df, trades_df
    
    def optimize_strategy(self, symbol='AAPL'):
        """Run parameter optimization for a specific symbol"""
        print("\n" + "=" * 60)
        print(f"Step 3: Parameter Optimization for {symbol}")
        print("=" * 60)
        
        # Define parameter ranges
        param_ranges = {
            'tp_multiplier': [1.0, 1.5, 2.0, 2.5, 3.0],
            'risk_per_trade': [0.005, 0.01, 0.015, 0.02]
        }
        
        # Run optimization
        best_params, best_result = self.backtest_engine.optimize_parameters(symbol, param_ranges)
        
        return best_params, best_result
    
    def generate_reports(self, summary_df=None, trades_df=None):
        """Generate visual reports and charts"""
        print("\n" + "=" * 60)
        print("Step 4: Report Generation")
        print("=" * 60)
        
        if trades_df is None or trades_df.empty:
            print("No trades to visualize")
            return
        
        # Create equity curve
        print("Creating equity curve...")
        self.visualizer.plot_equity_curve(
            trades_df, 
            self.config['strategy']['risk_management']['initial_capital'],
            'reports/equity_curve.png'
        )
        
        # Create trade distribution charts
        print("Creating trade distribution charts...")
        self.visualizer.plot_trade_distribution(
            trades_df,
            'reports/trade_distribution.png'
        )
        
        # Create performance heatmap
        print("Creating performance heatmap...")
        self.visualizer.plot_performance_heatmap(
            trades_df,
            'reports/performance_heatmap.png'
        )
        
        # Calculate overall metrics
        overall_metrics = self.strategy.calculate_metrics(trades_df)
        
        # Create HTML report
        print("Creating HTML report...")
        self.visualizer.create_html_report(
            summary_df if summary_df is not None else pd.DataFrame(),
            trades_df,
            overall_metrics,
            'reports/strategy_report.html'
        )
        
        print("\nAll reports generated successfully!")
        print("Check the 'reports' folder for output files.")
    
    def run_live_simulation(self, symbol='AAPL'):
        """Run a live simulation with real-time-like updates"""
        print("\n" + "=" * 60)
        print(f"Live Simulation for {symbol}")
        print("=" * 60)
        
        # Load data
        data = self.downloader.get_data(symbol, '5m')
        
        if data is None:
            print(f"No data available for {symbol}")
            return
        
        # Get last trading day
        last_date = data.index[-1].date()
        day_data = data[data.index.date == last_date]
        
        if len(day_data) < 10:
            # Get previous day if last day doesn't have enough data
            dates = pd.to_datetime(data.index.date).unique()
            if len(dates) > 1:
                last_date = dates[-2].date()
                day_data = data[data.index.date == last_date]
        
        print(f"\nSimulating trading day: {last_date}")
        
        # Calculate ORB for the day
        orb = self.strategy.calculate_opening_range(data, pd.Timestamp(last_date))
        
        if orb:
            print(f"\nOpening Range established:")
            print(f"  ORB High: ${orb['orb_high']:.2f}")
            print(f"  ORB Low: ${orb['orb_low']:.2f}")
            print(f"  Range: ${orb['orb_range']:.2f}")
            
            # Create ORB chart
            print("\nGenerating ORB chart...")
            self.visualizer.plot_orb_chart(
                day_data,
                orb,
                None,
                symbol,
                f'reports/{symbol}_orb_chart.html'
            )
        else:
            print("Could not establish opening range")
    
    def print_summary(self):
        """Print strategy configuration summary"""
        print("\n" + "=" * 60)
        print("Strategy Configuration Summary")
        print("=" * 60)
        
        print("\nOpening Range:")
        print(f"  Start: {self.config['strategy']['opening_range']['start_time']}")
        print(f"  End: {self.config['strategy']['opening_range']['end_time']}")
        print(f"  Duration: {self.config['strategy']['opening_range']['duration_minutes']} minutes")
        
        print("\nRisk Management:")
        print(f"  Risk per trade: {self.config['strategy']['risk_management']['risk_per_trade']*100}%")
        print(f"  Initial capital: ${self.config['strategy']['risk_management']['initial_capital']}")
        
        print("\nTrade Parameters:")
        print(f"  TP Multiplier: {self.config['strategy']['trade_parameters']['tp_multiplier']}x")
        print(f"  Max trades per day: {self.config['strategy']['trade_parameters']['max_trades_per_day']}")
        
        print("\nSymbols to trade:")
        for category in ['futures', 'forex', 'commodities', 'stocks']:
            if category in self.config['symbols']:
                symbols = [s['symbol'] for s in self.config['symbols'][category]]
                print(f"  {category.capitalize()}: {', '.join(symbols[:3])}{'...' if len(symbols) > 3 else ''}")


def main():
    """Main function with command line interface"""
    parser = argparse.ArgumentParser(description='ORB Trading System')
    parser.add_argument('--download', action='store_true', help='Download historical data')
    parser.add_argument('--backtest', action='store_true', help='Run backtesting')
    parser.add_argument('--optimize', type=str, help='Optimize parameters for symbol')
    parser.add_argument('--simulate', type=str, help='Run live simulation for symbol')
    parser.add_argument('--report', action='store_true', help='Generate reports')
    parser.add_argument('--all', action='store_true', help='Run complete workflow')
    parser.add_argument('--interval', type=str, default='5m', help='Data interval (5m or 15m)')
    
    args = parser.parse_args()
    
    # Initialize system
    system = ORBTradingSystem()
    
    # Print configuration summary
    system.print_summary()
    
    summary_df = None
    trades_df = None
    
    # Execute requested operations
    if args.all or args.download:
        system.download_data(force_download=args.download)
    
    if args.all or args.backtest:
        summary_df, trades_df = system.run_backtests(args.interval)
    
    if args.optimize:
        system.optimize_strategy(args.optimize)
    
    if args.simulate:
        system.run_live_simulation(args.simulate)
    
    if args.all or args.report:
        if trades_df is None:
            # Try to load existing trades
            if os.path.exists('reports/all_trades.csv'):
                trades_df = pd.read_csv('reports/all_trades.csv')
            if os.path.exists('reports/backtest_summary.csv'):
                summary_df = pd.read_csv('reports/backtest_summary.csv')
        
        system.generate_reports(summary_df, trades_df)
    
    if not any([args.download, args.backtest, args.optimize, args.simulate, args.report, args.all]):
        # No specific operation requested, show help
        parser.print_help()
        print("\n" + "=" * 60)
        print("Quick Start Examples:")
        print("=" * 60)
        print("\n1. Run complete workflow:")
        print("   python main.py --all")
        print("\n2. Download data only:")
        print("   python main.py --download")
        print("\n3. Run backtest with existing data:")
        print("   python main.py --backtest")
        print("\n4. Optimize parameters for AAPL:")
        print("   python main.py --optimize AAPL")
        print("\n5. Run simulation for TSLA:")
        print("   python main.py --simulate TSLA")
        print("\n6. Generate reports from existing results:")
        print("   python main.py --report")
    
    print("\n" + "=" * 60)
    print("ORB Trading System - Complete")
    print("=" * 60)


if __name__ == "__main__":
    main()