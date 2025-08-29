# Polygon.io Integration for Extended ORB Backtesting

## Overview
This integration allows you to download 2 years of historical 5-minute data from Polygon.io to validate the ORB strategy over longer timeframes and different market cycles.

## Setup Instructions

### 1. Get Your Free Polygon.io API Key
1. Visit https://polygon.io
2. Click "Get your Free API Key"
3. Sign up for a free account
4. Go to your dashboard: https://polygon.io/dashboard/api-keys
5. Copy your API key

### 2. Configure Your API Key
Edit `configs/polygon_config.yaml` and replace `YOUR_POLYGON_API_KEY_HERE` with your actual API key:

```yaml
polygon:
  api_key: "your_actual_api_key_here"
```

### 3. Install Required Package
```bash
pip3 install polygon-api-client
```

## Usage

### Step 1: Download Historical Data
```bash
python polygon_downloader.py
```

This will:
- Download 2 years of 5-minute data for all configured symbols
- Respect the free tier rate limit (5 requests/minute)
- Save data to `data/polygon/` directory
- Take approximately 10-15 minutes for 20 symbols

### Step 2: Run Extended Backtest
```bash
python backtest_extended_polygon.py
```

This will:
- Test the ORB strategy over the full 2-year period
- Analyze performance by market periods (2022 bear, 2023 bull, 2024 recent)
- Calculate max drawdowns and risk metrics
- Generate detailed performance reports

### Step 3: Compare Results
```bash
python compare_results.py
```

This will:
- Compare 60-day results with 2-year results
- Identify which periods favor the ORB strategy
- Show consistency of top performers
- Generate comprehensive comparison charts

## Free Tier Limitations

### What's Included (Free)
- ✅ 2 years of historical data
- ✅ All US stock tickers
- ✅ Minute-level data granularity
- ✅ End-of-day data
- ✅ 5 API calls per minute

### What's NOT Included (Free)
- ❌ Real-time data
- ❌ WebSocket streaming
- ❌ More than 5 requests/minute
- ❌ Data older than 2 years

### Upgrade Options
If you need more capabilities:
- **Stocks Starter ($29/month)**: 5 years history, unlimited API calls
- **Stocks Developer ($79/month)**: 10 years history, unlimited API calls

## Data Format

The downloaded data includes:
- **Timestamp**: Unix timestamp in milliseconds
- **Open, High, Low, Close**: Price data
- **Volume**: Trading volume
- **VWAP**: Volume-weighted average price
- **Transactions**: Number of transactions

## Symbol Configuration

Edit `configs/polygon_config.yaml` to customize symbols:

```yaml
symbols:
  stocks:
    - AAPL
    - MSFT
    - GOOGL
    # Add more stocks...
  
  forex:
    - "C:EURUSD"  # Note: Forex pairs use C: prefix
    - "C:GBPUSD"
    # Add more forex pairs...
```

## Top Performers from Initial Testing

Based on 60-day backtests, these symbols showed the best returns:
1. **COST**: 1867% return
2. **ADBE**: 490% return
3. **NFLX**: 383% return
4. **QQQ**: 301% return
5. **PEP**: 216% return

The extended 2-year backtest will validate if these returns are sustainable over longer periods.

## Troubleshooting

### API Key Issues
- Make sure your API key is correctly set in `configs/polygon_config.yaml`
- Verify your key at: https://polygon.io/dashboard/api-keys

### Rate Limiting
- Free tier is limited to 5 requests per minute
- The downloader automatically handles rate limiting
- If you get rate limit errors, wait 60 seconds and try again

### Missing Data
- Some symbols may not have complete 2-year history
- Forex pairs require the "C:" prefix (e.g., "C:EURUSD")
- Check `data/polygon/download_metadata.yaml` for download statistics

## Expected Results

With 2 years of data, you can:
1. **Validate strategy consistency** - Does it work in different market cycles?
2. **Calculate realistic drawdowns** - What's the worst-case scenario?
3. **Identify optimal conditions** - When does ORB perform best?
4. **Test robustness** - Is the strategy reliable long-term?

## Next Steps

After running the extended backtest:
1. Review `reports/extended_period_analysis.png` for period-specific performance
2. Check `reports/comparison_analysis.png` for 60-day vs 2-year comparison
3. Analyze `reports/extended_backtest_results.csv` for detailed metrics
4. Consider testing additional symbols or adjusting parameters based on findings