#!/usr/bin/env python3
"""Run systematic ORB backtests on Polygon.io CSV data."""

from __future__ import annotations

import sys
from pathlib import Path


def _ensure_repo_root_on_path() -> Path:
    """Locate the repository root and prepend it to ``sys.path`` if needed."""

    script_path = Path(__file__).resolve()
    candidate_dirs = (script_path.parent,) + tuple(script_path.parents)

    for candidate in candidate_dirs:
        if (candidate / "strategy").is_dir():
            candidate_str = str(candidate)
            if candidate_str not in sys.path:
                sys.path.insert(0, candidate_str)
            return candidate

    # Fallback: use the script directory itself to avoid returning ``None``.
    fallback = script_path.parent
    if str(fallback) not in sys.path:
        sys.path.insert(0, str(fallback))
    return fallback


REPO_ROOT = _ensure_repo_root_on_path()

from datetime import datetime
from itertools import product
from typing import Dict, Iterable, List, Optional

import pandas as pd
import yaml

from strategy.orb_strategy import ORBStrategy

TP_MULTIPLIERS: Iterable[float] = [2.0, 1.0, 0.5]
RISK_LEVELS: Iterable[float] = [0.01, 0.02]
ORB_DURATIONS: Iterable[int] = [5, 15]
INTERVALS: Iterable[str] = ["5m", "15m"]
DATA_DIR = Path("data/polygon")
REPORTS_DIR = Path("reports")


def _safe_to_csv(df: pd.DataFrame, path: Path, description: str) -> Optional[Path]:
    """Persist *df* to *path*, falling back if the target file is locked."""

    if df.empty:
        return None

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


def load_polygon_symbols(config_path: Path = Path("configs/polygon_config.yaml")) -> List[Dict]:
    """Extract Polygon symbol metadata from the repo configuration."""

    if not config_path.exists():
        raise FileNotFoundError(
            "Polygon configuration not found. Expected configs/polygon_config.yaml"
        )

    with config_path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)

    polygon_cfg = config.get("polygon", {})
    symbols_cfg = polygon_cfg.get("symbols", {})

    symbols: List[Dict] = []
    for category, entries in symbols_cfg.items():
        for entry in entries:
            if isinstance(entry, dict):
                symbol = entry.get("symbol")
                name = entry.get("name", symbol)
            else:
                symbol = str(entry)
                name = symbol

            if not symbol:
                continue

            symbols.append({
                "symbol": symbol,
                "name": name,
                "category": category,
            })

    return symbols


def _safe_symbol(symbol: str) -> str:
    """Convert Polygon symbols into filesystem-friendly names."""

    return symbol.replace(":", "_")


def load_polygon_data(symbol: str, interval: str) -> Optional[pd.DataFrame]:
    """Load Polygon CSV data for *symbol* and resample to *interval* if needed."""

    source_file = DATA_DIR / f"{_safe_symbol(symbol)}_5m.csv"
    if not source_file.exists():
        print(f"[WARN] Missing Polygon data for {symbol}: {source_file} not found")
        return None

    df = pd.read_csv(source_file)
    datetime_col = "datetime" if "datetime" in df.columns else "Datetime"

    if datetime_col not in df.columns:
        print(f"[WARN] {source_file} has no datetime column; skipping {symbol}")
        return None

    df[datetime_col] = pd.to_datetime(df[datetime_col], utc=True, errors="coerce")
    df = df.dropna(subset=[datetime_col])
    df = df.sort_values(datetime_col)
    df.set_index(datetime_col, inplace=True)

    rename_map = {
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close",
        "volume": "Volume",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    ohlcv_cols = ["Open", "High", "Low", "Close", "Volume"]
    missing_cols = [col for col in ohlcv_cols if col not in df.columns]
    if missing_cols:
        print(
            f"[WARN] {source_file} missing required columns {missing_cols}; skipping {symbol}"
        )
        return None

    df = df[ohlcv_cols]

    if interval == "15m":
        df = (
            df.resample("15min", label="right", closed="right")
            .agg({
                "Open": "first",
                "High": "max",
                "Low": "min",
                "Close": "last",
                "Volume": "sum",
            })
            .dropna(subset=["Open", "High", "Low", "Close"])
        )

    df = df[~df.index.duplicated(keep="first")]

    if df.empty:
        print(f"[WARN] No usable data for {symbol} at interval {interval}")
        return None

    return df


def run_backtests(symbols: List[Dict]) -> pd.DataFrame:
    """Execute backtests for each parameter combination across *symbols*."""

    all_results: List[Dict] = []

    for interval in INTERVALS:
        for orb_duration, tp_multiplier, risk in product(
            ORB_DURATIONS, TP_MULTIPLIERS, RISK_LEVELS
        ):
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
                "\n=== Polygon scenario | Interval: %s | ORB: %sm | TP: %.1fx | Risk: %.0f%% ==="
                % (interval, orb_duration, tp_multiplier, risk * 100)
            )

            for symbol_info in symbols:
                symbol = symbol_info["symbol"]
                name = symbol_info["name"]
                category = symbol_info["category"]

                data = load_polygon_data(symbol, interval)
                if data is None:
                    continue

                strategy = ORBStrategy("configs/config.yaml")
                strategy.apply_parameter_overrides(overrides)

                try:
                    trades = strategy.backtest(data, symbol)
                except Exception as exc:  # pragma: no cover - defensive guard
                    print(f"[WARN] Backtest failed for {symbol}: {exc}")
                    continue

                metrics = strategy.calculate_metrics(trades)

                all_results.append(
                    {
                        "symbol": symbol,
                        "name": name,
                        "category": category,
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
                        "data_points": len(data),
                        "start": data.index.min(),
                        "end": data.index.max(),
                    }
                )

    results_df = pd.DataFrame(all_results)

    if results_df.empty:
        print("No Polygon trades were generated for any scenario.")
        return results_df

    for column in ["win_rate", "return_pct", "profit_factor", "max_drawdown", "total_pnl"]:
        results_df[column] = pd.to_numeric(results_df[column], errors="coerce")

    _safe_to_csv(
        results_df,
        REPORTS_DIR / "polygon_systematic_backtest_results.csv",
        "Polygon detailed results",
    )

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
    combo_summary["positive_ratio"] = combo_summary["positive_symbols"] / combo_summary[
        "total_symbols"
    ].replace(0, pd.NA)

    combo_path = _safe_to_csv(
        combo_summary,
        REPORTS_DIR / "polygon_systematic_combo_summary.csv",
        "Polygon combination summary",
    )

    symbol_combo_summary = (
        results_df.groupby(
            ["symbol", "name", "category", "interval", "orb_duration", "tp_multiplier", "risk_per_trade"]
        )
        .agg(
            avg_return=("return_pct", "mean"),
            median_return=("return_pct", "median"),
            avg_profit_factor=("profit_factor", "mean"),
            total_trades=("total_trades", "sum"),
        )
        .reset_index()
    )

    _safe_to_csv(
        symbol_combo_summary,
        REPORTS_DIR / "polygon_systematic_symbol_combo_summary.csv",
        "Polygon symbol combination summary",
    )

    symbol_summary = (
        results_df.groupby(["symbol", "name", "category"])
        .agg(
            avg_return=("return_pct", "mean"),
            median_return=("return_pct", "median"),
            best_return=("return_pct", "max"),
            worst_return=("return_pct", "min"),
            avg_profit_factor=("profit_factor", "mean"),
            total_trades=("total_trades", "sum"),
        )
        .reset_index()
    )

    symbol_positive_ratio = (
        results_df.assign(is_positive=results_df["return_pct"] > 0)
        .groupby(["symbol", "name", "category"])["is_positive"]
        .mean()
        .reset_index(name="positive_ratio")
    )
    symbol_summary = symbol_summary.merge(
        symbol_positive_ratio, on=["symbol", "name", "category"], how="left"
    )

    symbol_path = _safe_to_csv(
        symbol_summary,
        REPORTS_DIR / "polygon_systematic_symbol_summary.csv",
        "Polygon per-symbol aggregate performance",
    )

    top_by_symbol = (
        results_df.sort_values(["symbol", "return_pct"], ascending=[True, False])
        .groupby(["symbol", "interval"])
        .head(3)
    )
    top_path = _safe_to_csv(
        top_by_symbol,
        REPORTS_DIR / "polygon_systematic_top3_by_symbol.csv",
        "Polygon per-symbol top combinations",
    )

    robust_combos = combo_summary[
        (combo_summary["avg_return"] > 0)
        & (combo_summary["avg_profit_factor"] > 1)
        & (combo_summary["positive_ratio"].fillna(0) >= 0.6)
    ].copy()
    robust_path = _safe_to_csv(
        robust_combos,
        REPORTS_DIR / "polygon_systematic_robust_combos.csv",
        "Polygon robust combination summary",
    )

    if combo_path:
        print(f"Saved Polygon combination summary to {combo_path}")
    if symbol_path:
        print(f"Saved Polygon per-symbol summary to {symbol_path}")
    if top_path:
        print(f"Saved Polygon top combinations to {top_path}")
    if robust_path and not robust_combos.empty:
        print(f"Saved Polygon robust combination summary to {robust_path}")

    print("\nTop Polygon parameter combinations by average return:")
    print(
        combo_summary.sort_values("avg_return", ascending=False)
        .head(10)
        .to_string(index=False, formatters={"avg_return": "{:.2f}".format})
    )

    print("\nPolygon symbols with highest average returns across all scenarios:")
    print(
        symbol_summary.sort_values("avg_return", ascending=False)
        .head(10)
        .to_string(
            index=False,
            formatters={
                "avg_return": "{:.2f}".format,
                "median_return": "{:.2f}".format,
                "best_return": "{:.2f}".format,
                "worst_return": "{:.2f}".format,
                "avg_profit_factor": "{:.2f}".format,
                "positive_ratio": "{:.2f}".format,
            },
        )
    )

    if not robust_combos.empty:
        print("\nPolygon robust parameter sets (avg_return>0, PF>1, >=60% symbols profitable):")
        print(
            robust_combos.sort_values("avg_return", ascending=False)
            .to_string(
                index=False,
                formatters={
                    "avg_return": "{:.2f}".format,
                    "median_return": "{:.2f}".format,
                    "avg_profit_factor": "{:.2f}".format,
                    "positive_ratio": "{:.2f}".format,
                },
            )
        )
    else:
        print(
            "\nNo Polygon parameter sets met the robustness filter (avg_return>0, PF>1, >=60% positive symbols)."
        )


def main() -> None:
    """Entry point."""

    symbols = load_polygon_symbols()
    if not symbols:
        print("No Polygon symbols configured. Nothing to backtest.")
        return

    results_df = run_backtests(symbols)
    summarise_results(results_df)


if __name__ == "__main__":
    main()
