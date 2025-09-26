#!/usr/bin/env python3
"""Generate visualisations for the Yahoo-based systematic ORB backtests."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

REPORTS_DIR = Path("reports")
RESULTS_CSV = REPORTS_DIR / "systematic_backtest_results.csv"


def _ensure_reports_dir() -> None:
    """Create the reports directory if it does not already exist."""

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def load_results(path: Path = RESULTS_CSV) -> pd.DataFrame:
    """Load the detailed systematic backtest results from *path*."""

    if not path.exists():
        raise FileNotFoundError(
            "Systematic backtest results not found. Expected to load "
            f"{path} – run systematic_backtest.py first."
        )

    df = pd.read_csv(path)
    numeric_cols: Iterable[str] = [
        "orb_duration",
        "tp_multiplier",
        "risk_per_trade",
        "win_rate",
        "profit_factor",
        "return_pct",
        "max_drawdown",
        "total_trades",
        "total_pnl",
    ]

    for column in numeric_cols:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    return df


def build_combo_summary(results_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate results by parameter combination."""

    if results_df.empty:
        return pd.DataFrame()

    combo_summary = (
        results_df.groupby(
            ["interval", "orb_duration", "tp_multiplier", "risk_per_trade"]
        )
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

    return combo_summary


def build_symbol_summary(results_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate performance by symbol across all parameter combinations."""

    if results_df.empty:
        return pd.DataFrame()

    symbol_summary = (
        results_df.groupby(["symbol", "name", "category"])
        .agg(
            avg_return=("return_pct", "mean"),
            median_return=("return_pct", "median"),
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

    return symbol_summary.merge(
        symbol_positive_ratio, on=["symbol", "name", "category"], how="left"
    )


def plot_combo_heatmaps(
    results_df: pd.DataFrame, save_path: Path
) -> Optional[Path]:
    """Create heatmaps of average returns for each interval/risk combination."""

    intervals = sorted(results_df["interval"].dropna().unique())
    risks = sorted(results_df["risk_per_trade"].dropna().unique())

    if not intervals or not risks:
        return None

    fig, axes = plt.subplots(
        len(intervals),
        len(risks),
        figsize=(4 * len(risks), 3.2 * len(intervals)),
        squeeze=False,
    )

    for row_idx, interval in enumerate(intervals):
        for col_idx, risk in enumerate(risks):
            ax = axes[row_idx][col_idx]
            subset = results_df[
                (results_df["interval"] == interval)
                & (results_df["risk_per_trade"] == risk)
            ]
            pivot = (
                subset.pivot_table(
                    index="orb_duration",
                    columns="tp_multiplier",
                    values="return_pct",
                    aggfunc="mean",
                )
                .sort_index()
                .sort_index(axis=1)
            )

            if pivot.empty:
                ax.axis("off")
                ax.text(0.5, 0.5, "No data", ha="center", va="center")
                continue

            sns.heatmap(
                pivot,
                annot=True,
                fmt=".1f",
                cmap="RdYlGn",
                center=0,
                cbar=False,
                ax=ax,
            )
            ax.set_title(f"{interval} | Risk {risk * 100:.0f}%\nAvg Return (%)")
            ax.set_xlabel("TP Multiplier (x)")
            ax.set_ylabel("ORB Duration (min)")

    fig.suptitle("Systematic ORB Performance by Parameter", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(save_path, dpi=200)
    plt.close(fig)

    return save_path


def plot_positive_ratio_bars(
    combo_summary: pd.DataFrame, save_path: Path, top_n: int = 10
) -> Optional[Path]:
    """Plot the parameter sets with the highest fraction of winning symbols."""

    if combo_summary.empty:
        return None

    ranked = combo_summary.sort_values(
        ["positive_ratio", "avg_return"], ascending=[False, False]
    ).head(top_n)

    if ranked.empty:
        return None

    labels = [
        f"{row.interval} | {int(row.orb_duration)}m | TP {row.tp_multiplier:.1f}x | {row.risk_per_trade*100:.0f}%"
        for row in ranked.itertuples()
    ]

    fig, ax = plt.subplots(figsize=(12, max(6, 0.6 * len(ranked))))
    bars = ax.barh(
        range(len(ranked)),
        ranked["positive_ratio"],
        color="teal",
        alpha=0.8,
    )
    ax.set_yticks(range(len(ranked)))
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("Positive Symbol Ratio")
    ax.set_title("Top Parameter Sets by Positive Symbol Coverage")
    ax.set_xlim(0, 1)

    for bar, avg_return in zip(bars, ranked["avg_return"]):
        ax.text(
            bar.get_width() + 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"Avg Return: {avg_return:.1f}%",
            va="center",
        )

    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    return save_path


def plot_top_symbols(
    symbol_summary: pd.DataFrame, save_path: Path, top_n: int = 12
) -> Optional[Path]:
    """Plot the symbols with the highest average returns."""

    if symbol_summary.empty:
        return None

    top_symbols = symbol_summary.sort_values("avg_return", ascending=False).head(top_n)
    if top_symbols.empty:
        return None

    fig, ax = plt.subplots(figsize=(12, max(6, 0.6 * len(top_symbols))))
    bars = ax.barh(
        range(len(top_symbols)),
        top_symbols["avg_return"],
        color="steelblue",
        alpha=0.85,
    )
    ax.set_yticks(range(len(top_symbols)))
    labels = [
        f"{row.symbol} ({row.category})" if isinstance(row.category, str) else row.symbol
        for row in top_symbols.itertuples()
    ]
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("Average Return (%)")
    ax.set_title("Top Symbols by Average Return")

    for bar, win_ratio in zip(bars, top_symbols["positive_ratio"]):
        ax.text(
            bar.get_width() + 0.5,
            bar.get_y() + bar.get_height() / 2,
            f"Positive Days: {win_ratio * 100:.0f}%",
            va="center",
        )

    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    return save_path


def main() -> None:
    """Entrypoint for generating systematic backtest visualisations."""

    _ensure_reports_dir()

    results_df = load_results()
    combo_summary = build_combo_summary(results_df)
    symbol_summary = build_symbol_summary(results_df)

    heatmap_path = plot_combo_heatmaps(
        results_df, REPORTS_DIR / "systematic_backtest_heatmaps.png"
    )
    positive_path = plot_positive_ratio_bars(
        combo_summary, REPORTS_DIR / "systematic_backtest_positive_ratio.png"
    )
    top_symbols_path = plot_top_symbols(
        symbol_summary, REPORTS_DIR / "systematic_backtest_top_symbols.png"
    )

    if heatmap_path:
        print(f"Saved heatmaps to {heatmap_path}")
    if positive_path:
        print(f"Saved positive-ratio chart to {positive_path}")
    if top_symbols_path:
        print(f"Saved symbol performance chart to {top_symbols_path}")


if __name__ == "__main__":
    main()
