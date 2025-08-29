# Opening Range Breakout (ORB) Trading Strategy

## Overview
This is a complete implementation of the Opening Range Breakout (ORB) trading strategy with backtesting capabilities, parameter optimization, and comprehensive reporting.

## Strategy Description

### Core Concept
The ORB strategy identifies the price range established during the first 15 minutes of regular trading hours (9:30-9:45 AM EST) and trades breakouts from this range.

### Entry Rules
- **Long Entry**: When price breaks above the ORB high
- **Short Entry**: When price breaks below the ORB low

### Risk Management
- **Position Sizing**: Risk 1% of capital per trade
- **Stop Loss (Long)**: ORB Low
- **Stop Loss (Short)**: ORB High
- **Take Profit**: 2x the ORB range (configurable)
- **Max Trades**: 3 trades per symbol per day

### Configurable Parameters
The strategy includes a configurable TP multiplier (default: 2) that can be adjusted in `configs/config.yaml`.

## Installation

### 1. Navigate to the ORB directory
```bash
cd ~/Desktop/ORB
```

### 2. Install Python dependencies
```bash
pip install -r requirements.txt
```

## Quick Start

### Run Complete Workflow (Recommended for first time)
```bash
python main.py --all
```

This will:
1. Download historical data for all configured symbols
2. Run backtests on 5-minute data
3. Generate performance reports
4. Create visualization charts

## Usage Examples

### 1. Download Historical Data
```bash
python data_downloader.py
```
Or using main script:
```bash
python main.py --download
```

### 2. Run Backtesting
```bash
python main.py --backtest
```

For 15-minute timeframe:
```bash
python main.py --backtest --interval 15m
```

### 3. Optimize Parameters
Optimize for specific symbol (e.g., AAPL):
```bash
python main.py --optimize AAPL
```

### 4. Run Live Simulation
Simulate trading for a specific symbol:
```bash
python main.py --simulate TSLA
```

### 5. Generate Reports Only
```bash
python main.py --report
```

## Project Structure

```
ORB/
├── configs/
│   └── config.yaml          # Strategy configuration
├── data/                    # Downloaded historical data (CSV files)
├── strategy/
│   └── orb_strategy.py      # Core strategy logic
├── backtesting/
│   └── backtest.py          # Backtesting engine
├── reports/
│   ├── visualizer.py        # Visualization module
│   ├── backtest_summary.csv # Summary results
│   ├── all_trades.csv       # All executed trades
│   ├── equity_curve.png     # Equity curve chart
│   ├── trade_distribution.png # Trade analysis
│   ├── performance_heatmap.png # Performance by time
│   └── strategy_report.html # Comprehensive HTML report
├── data_downloader.py       # Data download module
├── main.py                  # Main orchestration script
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

## Configuration

Edit `configs/config.yaml` to customize:

### Strategy Parameters
- **Opening Range Time**: Default 9:30-9:45 AM EST
- **Risk Per Trade**: Default 1% (0.01)
- **TP Multiplier**: Default 2.0 (adjustable)
- **Max Trades Per Day**: Default 3

### Symbols
The system is configured to trade:
- **Futures**: MNQ (Micro Nasdaq), NQ (E-mini Nasdaq)
- **Forex**: EURUSD
- **Commodities**: Gold (GC=F, GLD)
- **Stocks**: AAPL, MSFT, GOOGL, AMZN, TSLA, SPY, QQQ

### Data Settings
- **Intervals**: 5m and 15m
- **Date Range**: 2023-01-01 to 2024-12-31

## Output Reports

After running backtests, check the `reports/` folder for:

1. **backtest_summary.csv**: Performance metrics for each symbol
2. **all_trades.csv**: Detailed trade log with entry/exit prices and PnL
3. **equity_curve.png**: Visual representation of account growth
4. **trade_distribution.png**: Analysis of win/loss distribution
5. **performance_heatmap.png**: Performance by day and hour
6. **strategy_report.html**: Comprehensive HTML report (open in browser)
7. **optimization_*.csv**: Parameter optimization results

## Performance Metrics

The system calculates and reports:
- Total trades executed
- Win rate percentage
- Total and average PnL
- Profit factor
- Maximum drawdown
- Sharpe ratio
- Return percentage

## Important Notes

1. **Data Quality**: The strategy uses Yahoo Finance data. Some futures and forex symbols may have limited data availability.

2. **Market Hours**: The strategy is designed for US market hours. Adjust the opening range times in config for other markets.

3. **Backtesting Limitations**: 
   - Past performance doesn't guarantee future results
   - Backtesting doesn't account for slippage, market impact, or partial fills
   - Commission and slippage are simplified in the model

4. **Risk Warning**: This is for educational purposes. Always test thoroughly and understand the risks before live trading.

## Customization

### Changing the TP Multiplier
Edit `configs/config.yaml`:
```yaml
trade_parameters:
  tp_multiplier: 2.5  # Change from 2.0 to 2.5
```

### Adding New Symbols
Add to the appropriate section in `configs/config.yaml`:
```yaml
stocks:
  - symbol: "META"
    name: "Meta Platforms"
```

### Adjusting Risk
```yaml
risk_management:
  risk_per_trade: 0.02  # Change to 2% risk
```

## Troubleshooting

### No Data Available
- Some symbols (especially futures) may not be available through Yahoo Finance
- Try using ETF alternatives (e.g., GLD instead of GC=F for gold)

### Import Errors
Ensure all dependencies are installed:
```bash
pip install -r requirements.txt
```

### Permission Errors
Make scripts executable:
```bash
chmod +x *.py
```

## Next Steps

1. **Review Results**: Open `reports/strategy_report.html` in your browser
2. **Analyze Trades**: Examine `reports/all_trades.csv` for detailed trade analysis
3. **Optimize Parameters**: Use the optimization feature to find best parameters for each symbol
4. **Paper Trade**: Test the strategy with paper trading before risking real capital
5. **Customize**: Adjust parameters in `config.yaml` based on your risk tolerance

## Support

For questions or issues, review the code comments in each module. The strategy is fully documented with docstrings explaining each function.

## Disclaimer

This trading strategy is provided for educational purposes only. Trading involves substantial risk of loss. Past performance is not indicative of future results. Always do your own research and consider your risk tolerance before trading.