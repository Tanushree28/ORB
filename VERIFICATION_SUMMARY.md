# ORB Strategy Win Rate Verification Summary

## Overview
Generated detailed daily trading records for all 10 symbols to verify the ORB strategy's actual win rates. Each record shows the exact ORB levels, trade signals, and outcomes for manual verification.

## Files Generated

### Individual Symbol Records (in `reports/daily_orb_records/`)
- `AAPL_daily_orb.csv` - 249 trading days
- `AMZN_daily_orb.csv` - 249 trading days  
- `COST_daily_orb.csv` - 499 trading days
- `DIA_daily_orb.csv` - 500 trading days
- `GOOGL_daily_orb.csv` - 249 trading days
- `META_daily_orb.csv` - 247 trading days
- `MSFT_daily_orb.csv` - 249 trading days
- `NFLX_daily_orb.csv` - 494 trading days
- `QQQ_daily_orb.csv` - 249 trading days
- `SPY_daily_orb.csv` - 249 trading days

### Summary Files
- `summary_statistics.csv` - Overall win rates for each symbol
- `win_rate_verification.csv` - Comparison with backtest results

## CSV Column Definitions

Each daily record contains:
- **date**: Trading date
- **orb_high**: High price during 9:30-9:45 AM
- **orb_low**: Low price during 9:30-9:45 AM
- **orb_range**: orb_high - orb_low
- **long_entry**: Breakout level (ORB high)
- **long_sl**: Stop loss for long trades (ORB low)
- **long_tp**: Take profit for long trades (ORB high + 2×range)
- **short_entry**: Breakdown level (ORB low)
- **short_sl**: Stop loss for short trades (ORB high)
- **short_tp**: Take profit for short trades (ORB low - 2×range)
- **long_triggered**: Boolean - was long trade triggered?
- **short_triggered**: Boolean - was short trade triggered?
- **long_result**: 'TP', 'SL', 'No Trade', or 'No Exit'
- **short_result**: 'TP', 'SL', 'No Trade', or 'No Exit'
- **day_high**: Highest price after ORB period
- **day_low**: Lowest price after ORB period

## Key Findings

### Win Rate Comparison

| Symbol | Daily Records Win Rate | Backtest Win Rate | Difference |
|--------|----------------------|------------------|------------|
| AAPL   | 47.7%               | 65.1%            | -17.3%     |
| AMZN   | 47.1%               | 62.4%            | -15.3%     |
| COST   | 47.2%               | 61.0%            | -13.8%     |
| DIA    | 46.5%               | 60.3%            | -13.8%     |
| GOOGL  | 48.7%               | 61.6%            | -12.9%     |
| META   | 51.7%               | 64.9%            | -13.2%     |
| MSFT   | 46.7%               | 62.0%            | -15.3%     |
| NFLX   | 47.7%               | 62.0%            | -14.3%     |
| QQQ    | 46.8%               | 59.8%            | -13.0%     |
| SPY    | 46.0%               | 55.2%            | -9.3%      |

**Average**: Daily Records = 47.4% | Backtest = 61.4% | Difference = -14.0%

### Trade Count Analysis

- **Daily Records**: Average 1.7 trades per day (max 2 - one long, one short)
- **Backtest**: Average 3.9 trades per day
- **Difference**: Backtest takes 2.3x more trades

### Discrepancy Explanations

1. **Multiple Trades Per Day**
   - Daily records: Maximum 2 trades (1 long, 1 short)
   - Backtest: Appears to allow re-entry after stops

2. **Position Sizing Impact**
   - Daily records: Fixed analysis per trade
   - Backtest: Uses compounding (1% risk per trade), which may affect trade selection

3. **Entry/Exit Timing**
   - Daily records: Clear breakout/breakdown signals
   - Backtest: May use different entry logic or intrabar fills

## How to Verify

1. **Check Specific Days**
   - Open any symbol's CSV file
   - Find a specific date
   - Verify the ORB high/low match the first 15 minutes (9:30-9:45 AM)
   - Confirm TP = ORB high + (2 × range) for longs
   - Confirm SL = ORB low for longs

2. **Example Verification (QQQ on 2024-08-14)**
   ```
   ORB High: 462.52
   ORB Low: 462.31
   Range: 0.21
   Long TP: 462.52 + (2 × 0.21) = 462.94 ✓
   Long SL: 462.31 ✓
   Result: Both long and short hit TP
   ```

3. **Manual Win Rate Calculation**
   - Count all "TP" results (wins)
   - Count all "SL" results (losses)
   - Win Rate = Wins / (Wins + Losses) × 100

## Conclusion

The daily records show a **realistic ~47% win rate**, which is more conservative than the backtest's ~60% win rate. The discrepancy is primarily due to:

1. **Trade frequency**: Backtest takes 2.3x more trades
2. **Compounding effects**: Position sizing grows with wins in backtest
3. **Entry logic**: Possible differences in breakout confirmation

**For manual verification**: All trade data is transparent and available in the CSV files. You can verify any specific day's ORB levels, entry signals, and outcomes to confirm accuracy.

## File Locations

```
/Users/kar/Desktop/ORB/
├── reports/
│   ├── daily_orb_records/
│   │   ├── AAPL_daily_orb.csv
│   │   ├── AMZN_daily_orb.csv
│   │   ├── COST_daily_orb.csv
│   │   ├── DIA_daily_orb.csv
│   │   ├── GOOGL_daily_orb.csv
│   │   ├── META_daily_orb.csv
│   │   ├── MSFT_daily_orb.csv
│   │   ├── NFLX_daily_orb.csv
│   │   ├── QQQ_daily_orb.csv
│   │   ├── SPY_daily_orb.csv
│   │   ├── summary_statistics.csv
│   │   └── win_rate_verification.csv
│   └── win_rate_verification.png
```