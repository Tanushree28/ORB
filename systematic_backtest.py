#!/usr/bin/env python3
"""Run systematic ORB backtests across parameter combinations."""

from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Dict, Iterable, List

import pandas as pd

from backtesting.backtest import BacktestEngine


TP_MULTIPLIERS: Iterable[float] = [2.0, 1.0, 0.5]
RISK_LEVELS: Iterable[float] = [0.01, 0.02]
ORB_DURATIONS: Iterable[int] = [5, 15]
INTERVALS: Iterable[str] = ["5m", "15m"]


def _safe_to_csv(df: pd.DataFrame, path: str, description: str) -> Path:
    """Persist *df* to *path*, falling back if the target file is locked."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)

    try:
        df.to_csv(target, index=False)
        print(f"Saved {description} to {target}")
        return target
    except PermissionError:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        fallback = target.with_name(f"{target.stem}_{timestamp}{target.suffix}")
        df.to_csv(fallback, index=False)
        print(
            "Permission denied when writing %s to %s. Saved to %s instead."
            % (description, target, fallback)
        )
        return fallback


def run_backtests() -> pd.DataFrame:
    """Execute backtests for each combination and return consolidated results."""

    all_results: List[Dict] = []

    for interval in INTERVALS:
        for orb_duration, tp_multiplier, risk in product(
            ORB_DURATIONS, TP_MULTIPLIERS, RISK_LEVELS
        ):
            # Skip incompatible combinations (e.g., 5-minute ORB on 15-minute data)
            if interval == "15m" and orb_duration < 15:
                continue

            overrides = {
                "orb_duration": orb_duration,
                "tp_multiplier": tp_multiplier,
                "risk_per_trade": risk,
                "max_trades_per_day": 2,
                "max_long_trades_per_day": 1,
                "max_short_trades_per_day": 1,
            }

            print(
                "\n=== Running scenario | Interval: %s | ORB: %sm | TP: %.1fx | Risk: %.0f%% ==="
                % (interval, orb_duration, tp_multiplier, risk * 100)
            )

            engine = BacktestEngine(strategy_overrides=overrides)
            scenario_results = engine.run_all_backtests(interval=interval)

            for result in scenario_results:
                metrics = result.get("metrics", {})
                all_results.append(
                    {
                        "symbol": result["symbol"],
                        "name": result["name"],
                        "category": result["category"],
                        "interval": interval,
                        "orb_duration": orb_duration,
                        "tp_multiplier": tp_multiplier,
                        "risk_per_trade": risk,
                        "total_trades": metrics.get("total_trades", 0),
                        "win_rate": metrics.get("win_rate", 0.0),
                        "profit_factor": metrics.get("profit_factor", 0.0),
                        "return_pct": metrics.get("return_pct", 0.0),
                        "max_drawdown": metrics.get("max_drawdown", 0.0),
                        "total_pnl": metrics.get("total_pnl", 0.0),
                    }
                )

    results_df = pd.DataFrame(all_results)

    if results_df.empty:
        print("No trades were generated for any scenario.")
        return results_df

    # Normalise numeric columns
    for column in ["win_rate", "return_pct", "profit_factor", "max_drawdown", "total_pnl"]:
        results_df[column] = pd.to_numeric(results_df[column], errors="coerce")

    print()
    _safe_to_csv(results_df, "reports/systematic_backtest_results.csv", "detailed results")

    return results_df


def summarise_results(results_df: pd.DataFrame) -> None:
    """Generate summary CSVs to highlight robust parameter sets."""

    if results_df.empty:
        return

    combo_summary = (
        results_df.groupby(["interval", "orb_duration", "tp_multiplier", "risk_per_trade"])
        .agg(
            avg_return=("return_pct", "mean"),
            median_return=("return_pct", "median"),
            avg_profit_factor=("profit_factor", "mean"),
            positive_symbols=("return_pct", lambda x: (x > 0).sum()),
            total_symbols=("symbol", "nunique"),
        )
        .reset_index()
    )
    combo_summary["positive_ratio"] = combo_summary["positive_symbols"] / combo_summary["total_symbols"].replace(0, pd.NA)

    combo_path = _safe_to_csv(
        combo_summary,
        "reports/systematic_backtest_combo_summary.csv",
        "combination summary",
    )

    symbol_combo_summary = (
        results_df.groupby(
            ["symbol", "interval", "orb_duration", "tp_multiplier", "risk_per_trade"]
        )
        .agg(
            avg_return=("return_pct", "mean"),
            median_return=("return_pct", "median"),
            avg_profit_factor=("profit_factor", "mean"),
            total_trades=("total_trades", "sum"),
        )
        .reset_index()
    )

    symbol_combo_path = _safe_to_csv(
        symbol_combo_summary,
        "reports/systematic_backtest_symbol_combo_summary.csv",
        "symbol combination summary",
    )

    top_by_symbol = (
        results_df.sort_values(["symbol", "return_pct"], ascending=[True, False])
        .groupby(["symbol", "interval"])
        .head(3)
    )
    top_path = _safe_to_csv(
        top_by_symbol,
        "reports/systematic_backtest_top3_by_symbol.csv",
        "per-symbol top combinations",
    )

    print("\nTop parameter combinations by average return:")
    print(
        combo_summary.sort_values("avg_return", ascending=False)
        .head(10)
        .to_string(index=False, formatters={"avg_return": "{:.2f}".format})
    )


def main() -> None:
    """Entry point."""

    results_df = run_backtests()
    summarise_results(results_df)


if __name__ == "__main__":
    main()
