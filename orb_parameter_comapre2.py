"""
orb_parameter_comparison_all.py
Compare ORB performance across ALL symbols in config.yaml vs MNQ/NQ baseline.

- Symbols: all from config (futures, forex, commodities, stocks)
- Intervals: from config (e.g., 5m, 15m)
- ORB windows tested: 5 minutes and 15 minutes (derived from start_time)
- TP/SL ratios: 2.0, 1.0, 0.5
- Risk per trade: 1% (0.01), 2% (0.02)
- Max per day: ONE long + ONE short

Outputs:
- Excel: comparison_all_results.xlsx
  * Sheet 'Summary'                 -> per symbol/interval/orb_window/tp/risk
  * Sheet 'Relative_to_Baseline'    -> delta vs baseline (avg of MNQ=F & NQ=F)
- PNG  : returns_by_symbol.png      -> best Total_PnL per symbol across grid
"""

import os
import itertools
import yaml
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# -------------------------- Config I/O --------------------------


def load_config(config_path: str = "configs/config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# -------------------------- CSV loading (robust) --------------------------

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
    # If index looks like datetime already, try parsing it
    if not isinstance(df.index, pd.RangeIndex):
        try:
            idx = pd.to_datetime(df.index, errors="raise", utc=False)
            out = df.copy()
            out.index = idx
            return out
        except Exception:
            pass

    # Try known datetime columns
    for col in _POSSIBLE_DT_COLS:
        if col in df.columns:
            out = df.copy()
            out[col] = pd.to_datetime(out[col], errors="raise", utc=False)
            out = out.set_index(col)
            return out

    # Try unnamed first column as prior index
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

    raise ValueError(
        f"Could not parse datetime index in file: {file_path}\n"
        f"Provide a column named one of {_POSSIBLE_DT_COLS} or save the datetime as the index."
    )


def load_symbol_data(
    symbol: str, interval: str, data_dir: str = "data"
) -> pd.DataFrame:
    safe_symbol = symbol.replace("=", "_")
    filename = f"{safe_symbol}_{interval}.csv"
    file_path = os.path.join(data_dir, filename)
    if not os.path.exists(file_path):
        print(f"✗ Data file not found: {file_path}")
        return pd.DataFrame()

    df = pd.read_csv(file_path)

    # Standardise common OHLC names
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

    df = _coerce_datetime_index(df, file_path)

    req = ["Open", "High", "Low", "Close"]
    missing = [c for c in req if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns {missing} in {file_path}")

    keep = ["Open", "High", "Low", "Close"]
    if "Volume" in df.columns:
        keep.append("Volume")
    return df[keep].sort_index()


# -------------------------- ORB logic --------------------------


def compute_orb_end(start_str: str, duration_minutes: int) -> str:
    # start_str like "09:30" -> end is start + duration
    dt = datetime.strptime(start_str, "%H:%M")
    end_dt = dt + timedelta(minutes=duration_minutes)
    return end_dt.strftime("%H:%M")


def backtest_orb_day(
    data: pd.DataFrame,
    start_time: str,
    orb_duration_minutes: int,
    tp_multiplier: float,
    # risk_per_trade included for signature parity; sizing not used for points PnL here
    risk_per_trade: float,
) -> dict:
    """One-day ORB with at most one long + one short."""
    if data.empty:
        return {"trades": 0, "wins": 0, "losses": 0, "total_pnl": 0.0}

    end_time = compute_orb_end(start_time, orb_duration_minutes)

    opening = data.between_time(start_time, end_time, inclusive="both")
    if opening.empty:
        return {"trades": 0, "wins": 0, "losses": 0, "total_pnl": 0.0}

    orb_high = opening["High"].max()
    orb_low = opening["Low"].min()
    orb_range = orb_high - orb_low
    if orb_range <= 0:
        return {"trades": 0, "wins": 0, "losses": 0, "total_pnl": 0.0}

    long_entry = orb_high
    short_entry = orb_low
    long_sl = orb_low
    short_sl = orb_high
    long_tp = orb_high + tp_multiplier * orb_range
    short_tp = orb_low - tp_multiplier * orb_range

    trading = data.between_time(end_time, "23:59", inclusive="right")

    long_taken = False
    short_taken = False
    pnl = 0.0
    wins = 0
    losses = 0

    # scan bars after ORB window
    for row in trading.itertuples():
        # long trigger
        if (not long_taken) and (row.Close > long_entry) and (row.Open <= long_entry):
            for row2 in trading.loc[row.Index :].itertuples():
                if row2.High >= long_tp:
                    pnl += long_tp - long_entry
                    wins += 1
                    break
                if row2.Low <= long_sl:
                    pnl += long_sl - long_entry
                    losses += 1
                    break
            long_taken = True

        # short trigger
        if (
            (not short_taken)
            and (row.Close < short_entry)
            and (row.Open >= short_entry)
        ):
            for row2 in trading.loc[row.Index :].itertuples():
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


def run_all(
    config_path: str = "configs/config.yaml",
    output_excel: str = "comparison_all_results.xlsx",
) -> None:
    cfg = load_config(config_path)

    # symbols: gather all categories
    symbols = []
    for cat in ("futures", "forex", "commodities", "stocks"):
        items = cfg.get("symbols", {}).get(cat, [])
        for it in items:
            s = it.get("symbol")
            if s:
                symbols.append(s)

    intervals = cfg.get("data", {}).get("intervals", ["5m", "15m"])
    start_time = cfg["strategy"]["opening_range"]["start_time"]  # e.g., "09:30"

    # parameter grid
    orb_windows = [5, 15]  # minutes
    tp_multipliers = [2.0, 1.0, 0.5]
    risk_per_trades = [0.01, 0.02]  # 1% and 2%
    # one long + one short per day is enforced in logic (no extra flag needed)

    results = []
    for symbol, interval, orb_dur, tp_mult, risk in itertools.product(
        symbols, intervals, orb_windows, tp_multipliers, risk_per_trades
    ):
        df = load_symbol_data(symbol, interval)
        if df.empty:
            continue

        total_pnl = 0.0
        total_trades = 0
        total_wins = 0
        total_losses = 0

        for _, day_df in df.groupby(df.index.date):
            perf = backtest_orb_day(
                day_df,
                start_time=start_time,
                orb_duration_minutes=orb_dur,
                tp_multiplier=tp_mult,
                risk_per_trade=risk,
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
                "ORB_Window_Min": orb_dur,
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
        print("✗ No results computed (check data files / names).")
        return

    summary = pd.DataFrame(results)

    # -------- Baseline comparison (MNQ=F & NQ=F averaged) --------
    base_syms = {"MNQ=F", "NQ=F"}
    base = (
        summary[summary["Symbol"].isin(base_syms)]
        .groupby(
            ["Interval", "ORB_Window_Min", "TP_Multiplier", "Risk_Per_Trade"],
            as_index=False,
        )
        .agg({"Total_PnL": "mean", "Win_Rate": "mean"})
        .rename(
            columns={"Total_PnL": "Baseline_Total_PnL", "Win_Rate": "Baseline_Win_Rate"}
        )
    )

    # Merge baseline back onto all rows (on same params/interval/window)
    merged = pd.merge(
        summary,
        base,
        on=["Interval", "ORB_Window_Min", "TP_Multiplier", "Risk_Per_Trade"],
        how="left",
    )
    merged["Delta_PnL_vs_Baseline"] = merged["Total_PnL"] - merged["Baseline_Total_PnL"]
    merged["Delta_WinRate_vs_Baseline"] = (
        merged["Win_Rate"] - merged["Baseline_Win_Rate"]
    )

    # -------- Save Excel --------
    with pd.ExcelWriter(output_excel, engine="openpyxl") as writer:
        summary.sort_values(
            ["Symbol", "Interval", "ORB_Window_Min", "TP_Multiplier", "Risk_Per_Trade"]
        ).to_excel(writer, index=False, sheet_name="Summary")

        merged.sort_values(
            ["Symbol", "Interval", "ORB_Window_Min", "TP_Multiplier", "Risk_Per_Trade"]
        ).to_excel(writer, index=False, sheet_name="Relative_to_Baseline")

    print(f"✓ Saved results to {output_excel}")

    # -------- Plot: best total PnL per symbol across all combos --------
    best = summary.loc[summary.groupby("Symbol")["Total_PnL"].idxmax()].sort_values(
        "Total_PnL", ascending=False
    )
    plt.figure(figsize=(12, 6))
    best.plot(x="Symbol", y="Total_PnL", kind="bar", legend=False)
    plt.title("Best Total PnL per Symbol (across intervals, ORB windows, TP, risk)")
    plt.xlabel("Symbol")
    plt.ylabel("Total PnL (points)")
    plt.tight_layout()
    plt.savefig("returns_by_symbol.png", dpi=150)
    print("✓ Saved bar chart to returns_by_symbol.png")


if __name__ == "__main__":
    run_all()
