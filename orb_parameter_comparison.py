"""
orb_parameter_comparison.py
=================================

This script runs a series of backtests for the opening-range breakout (ORB)
strategy across a set of futures symbols (MNQ=F and NQ=F) using varying
risk parameters and take-profit/stop-loss ratios.  The goal is to compare
performance across combinations of the following settings:

* Opening range interval: 5-minute and 15-minute windows
* Take-profit to stop-loss ratio (tp_multiplier): 2.0, 1.0, 0.5
* Risk per trade: 1 % (0.01) and 2 % (0.02)
* Maximum one long and one short trade per day (max 2 trades/day)

For each symbol, interval and parameter combination the backtest computes
basic metrics such as total number of trades, win rate, total PnL and
return percentage.  Results are saved to an Excel workbook and a bar
chart summarising returns is generated for quick comparison.

Assumptions:
-----------
* Historical intraday data for each symbol is stored in CSV files named
  "<symbol>_<interval>.csv" under a "data/" directory.  For example
  "MNQ=F_5m.csv" contains 5-minute bars for MNQ=F.
* Each CSV must contain OHLC columns; the datetime can be in a column
  named one of: datetime, Datetime, timestamp, time, date, or the first
  (unnamed) column. If the datetime is already the index, that works too.

To execute the comparison:
-------------------------
1. Place this script in the root of your project (or adjust paths).
2. Ensure that the "data/" folder contains the required CSVs.
3. Run:  python orb_parameter_comparison.py
4. Outputs: "comparison_results.xlsx" and "returns_comparison.png"
"""

import os
import itertools
import yaml
import pandas as pd
import matplotlib.pyplot as plt


# -------------------------- Config utilities --------------------------


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# -------------------------- Data loading --------------------------

_POSSIBLE_DT_COLS = [
    "datetime",
    "Datetime",
    "date_time",
    "timestamp",
    "time",
    "date",
    "Date",
    "Timestamp",
    "DateTime",
]


def _coerce_datetime_index(df: pd.DataFrame, file_path: str) -> pd.DataFrame:
    """Make the DataFrame indexed by a proper pandas datetime index.

    Tries, in order:
      1) If index already looks like datetime -> parse it.
      2) Known datetime column names (_POSSIBLE_DT_COLS).
      3) An unnamed first column that looks like a saved index.
    """
    # 1) Already datetime-like index?
    if not isinstance(df.index, pd.RangeIndex):
        # Try converting index directly
        try:
            idx = pd.to_datetime(df.index, errors="raise", utc=False)
            df = df.copy()
            df.index = idx
            return df
        except Exception:
            pass  # fall through

    # 2) Search for a known datetime column
    for col in _POSSIBLE_DT_COLS:
        if col in df.columns:
            out = df.copy()
            out[col] = pd.to_datetime(out[col], errors="raise", utc=False)
            out = out.set_index(col)
            return out

    # 3) Unnamed first column (common when CSV saved with index=True)
    if df.columns.size > 0 and df.columns[0].startswith("Unnamed"):
        try:
            out = df.copy()
            out[out.columns[0]] = pd.to_datetime(
                out.iloc[:, 0], errors="raise", utc=False
            )
            out = out.set_index(out.columns[0]).drop(columns=[out.columns[0]])
            return out
        except Exception:
            pass

    # If we reach here, we couldn't find/parse a datetime index
    raise ValueError(
        f"Could not find/parse a datetime column or index in file: {file_path}\n"
        f"Provide a column named one of {_POSSIBLE_DT_COLS} or save the datetime as the index."
    )


def load_symbol_data(
    symbol: str, interval: str, data_dir: str = "data"
) -> pd.DataFrame:
    """Load intraday data for a given symbol and interval.

    Accepts typical variations: datetime in index, 'Datetime', 'datetime',
    'timestamp', 'date', or an unnamed first column.
    Renames lower-case ohlc to capitalised OHLC.
    """
    safe_symbol = symbol.replace("=", "_")
    filename = f"{safe_symbol}_{interval}.csv"
    file_path = os.path.join(data_dir, filename)
    if not os.path.exists(file_path):
        print(f"✗ Data file not found: {file_path}")
        return pd.DataFrame()

    df = pd.read_csv(file_path)

    # Standardise column names (only if needed)
    rename_map = {
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close",
        "volume": "Volume",
    }
    for k, v in rename_map.items():
        if k in df.columns and v not in df.columns:
            df.rename(columns={k: v}, inplace=True)

    # Coerce datetime index robustly
    df = _coerce_datetime_index(df, file_path)

    # Ensure required OHLC columns exist
    req = ["Open", "High", "Low", "Close"]
    missing = [c for c in req if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns {missing} in {file_path}")

    keep_cols = ["Open", "High", "Low", "Close"]
    if "Volume" in df.columns:
        keep_cols.append("Volume")
    return df[keep_cols].sort_index()


# -------------------------- ORB backtest --------------------------


def backtest_orb(
    data: pd.DataFrame,
    start_time: str,
    end_time: str,
    tp_multiplier: float,
    risk_per_trade: float,
    max_trades_per_day: int = 2,
) -> dict:
    """Run an ORB day backtest (max 1 long + 1 short)."""
    if data.empty:
        return {"trades": 0, "wins": 0, "losses": 0, "total_pnl": 0.0}

    opening_data = data.between_time(start_time, end_time, inclusive="both")
    if opening_data.empty:
        return {"trades": 0, "wins": 0, "losses": 0, "total_pnl": 0.0}

    orb_high = opening_data["High"].max()
    orb_low = opening_data["Low"].min()
    orb_range = orb_high - orb_low
    if orb_range <= 0:
        return {"trades": 0, "wins": 0, "losses": 0, "total_pnl": 0.0}

    long_entry = orb_high
    short_entry = orb_low
    long_sl = orb_low
    short_sl = orb_high
    long_tp = orb_high + tp_multiplier * orb_range
    short_tp = orb_low - tp_multiplier * orb_range

    # After the OR window
    trading_data = data.between_time(end_time, "23:59", inclusive="right")

    long_taken = False
    short_taken = False
    pnl = 0.0
    wins = 0
    losses = 0

    # itertuples returns a namedtuple: (Index, Open, High, Low, Close, Volume?)
    for row in trading_data.itertuples():
        # Long entry: close breaks above OR high with open at/below it
        if (not long_taken) and (row.Close > long_entry) and (row.Open <= long_entry):
            # scan forward for exit
            for row2 in trading_data.loc[row.Index :].itertuples():
                if row2.High >= long_tp:
                    pnl += long_tp - long_entry
                    wins += 1
                    break
                if row2.Low <= long_sl:
                    pnl += long_sl - long_entry
                    losses += 1
                    break
            long_taken = True

        # Short entry: close breaks below OR low with open at/above it
        if (
            (not short_taken)
            and (row.Close < short_entry)
            and (row.Open >= short_entry)
        ):
            for row2 in trading_data.loc[row.Index :].itertuples():
                if row2.Low <= short_tp:
                    pnl += short_entry - short_tp
                    wins += 1
                    break
                if row2.High >= short_sl:
                    pnl += short_entry - short_sl
                    losses += 1
                    break
            short_taken = True

        if long_taken and short_taken:
            break

    return {"trades": wins + losses, "wins": wins, "losses": losses, "total_pnl": pnl}


# -------------------------- Runner --------------------------


def run_comparison(
    config_path: str = "configs/config.yaml",
    output_excel: str = "comparison_results.xlsx",
) -> None:
    cfg = load_config(config_path)

    # Only MNQ and NQ from the futures set
    futures_list = cfg.get("symbols", {}).get("futures", [])
    symbols = [
        f["symbol"] for f in futures_list if f.get("symbol") in {"MNQ=F", "NQ=F"}
    ]
    if not symbols:
        print("✗ No MNQ/NQ symbols found in configuration.")
        return

    intervals = cfg.get("data", {}).get("intervals", ["5m", "15m"])
    start_time = cfg["strategy"]["opening_range"]["start_time"]  # e.g. 09:30
    end_time = cfg["strategy"]["opening_range"]["end_time"]  # e.g. 09:45

    tp_multipliers = [2.0, 1.0, 0.5]
    risk_per_trades = [0.01, 0.02]
    max_trades_per_day = 2

    results = []
    for symbol, interval, tp_mult, risk in itertools.product(
        symbols, intervals, tp_multipliers, risk_per_trades
    ):
        df = load_symbol_data(symbol, interval)
        if df.empty:
            continue

        grouped = df.groupby(df.index.date)
        total_pnl = 0.0
        total_trades = 0
        total_wins = 0
        total_losses = 0

        for _, day_data in grouped:
            perf = backtest_orb(
                day_data,
                start_time=start_time,
                end_time=end_time,
                tp_multiplier=tp_mult,
                risk_per_trade=risk,
                max_trades_per_day=max_trades_per_day,
            )
            total_pnl += perf["total_pnl"]
            total_trades += perf["trades"]
            total_wins += perf["wins"]
            total_losses += perf["losses"]

        win_rate = (total_wins / total_trades) if total_trades > 0 else 0.0
        results.append(
            {
                "Symbol": symbol,
                "Interval": interval,
                "TP_Multiplier": tp_mult,
                "Risk_Per_Trade": risk,
                "Total_Trades": total_trades,
                "Wins": total_wins,
                "Losses": total_losses,
                "Win_Rate": win_rate,
                "Total_PnL": total_pnl,
            }
        )

    if not results:
        print("✗ No results computed. Check your data directory and symbols.")
        return

    results_df = pd.DataFrame(results).sort_values(
        ["Symbol", "Interval", "TP_Multiplier", "Risk_Per_Trade"]
    )

    # Save Excel
    with pd.ExcelWriter(output_excel, engine="openpyxl") as writer:
        results_df.to_excel(writer, index=False, sheet_name="ORB Comparison")
    print(f"✓ Comparison results saved to {output_excel}")

    # Plot
    pivot_df = results_df.pivot_table(
        index=["TP_Multiplier", "Risk_Per_Trade"],
        columns="Symbol",
        values="Total_PnL",
        aggfunc="mean",
    ).sort_index()
    ax = pivot_df.plot(kind="bar", figsize=(11, 6))
    ax.set_title("Mean PnL by TP Multiplier & Risk Level (per Symbol)")
    ax.set_xlabel("TP Multiplier & Risk Per Trade")
    ax.set_ylabel("Mean PnL (points)")
    plt.tight_layout()
    plt.savefig("returns_comparison.png", dpi=150)
    print("✓ Returns comparison chart saved to returns_comparison.png")


if __name__ == "__main__":
    run_comparison()
