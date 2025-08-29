#!/usr/bin/env python3
"""
Visualization Module for ORB Strategy
Creates charts and visual reports for strategy performance
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
from typing import Dict, List, Optional
import warnings
from string import Template

warnings.filterwarnings('ignore')

class StrategyVisualizer:
    def __init__(self):
        """Initialize visualizer with style settings"""
        # Set style
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")
        
        # Create reports directory if it doesn't exist
        os.makedirs('reports', exist_ok=True)
    
    def plot_orb_chart(self, data: pd.DataFrame, orb_levels: Dict, trades: pd.DataFrame = None,
                       symbol: str = "Symbol", save_path: str = None):
        """
        Create candlestick chart with ORB levels and trades
        
        Args:
            data: Price data
            orb_levels: Dictionary with ORB high and low
            trades: DataFrame with trade entries and exits
            symbol: Symbol name
            save_path: Path to save the chart
        """
        fig = make_subplots(rows=2, cols=1, 
                           shared_xaxes=True,
                           vertical_spacing=0.03,
                           row_heights=[0.7, 0.3],
                           subplot_titles=(f'{symbol} - ORB Strategy', 'Volume'))
        
        # Candlestick chart
        fig.add_trace(go.Candlestick(
            x=data.index,
            open=data['Open'],
            high=data['High'],
            low=data['Low'],
            close=data['Close'],
            name='Price'
        ), row=1, col=1)
        
        # Add ORB levels
        if orb_levels:
            # ORB High
            fig.add_hline(y=orb_levels['orb_high'], 
                         line_color="green", 
                         line_width=2,
                         line_dash="dash",
                         annotation_text=f"ORB High: {orb_levels['orb_high']:.2f}",
                         row=1, col=1)
            
            # ORB Low
            fig.add_hline(y=orb_levels['orb_low'], 
                         line_color="red", 
                         line_width=2,
                         line_dash="dash",
                         annotation_text=f"ORB Low: {orb_levels['orb_low']:.2f}",
                         row=1, col=1)
            
            # Shade ORB period
            fig.add_vrect(x0=orb_levels['orb_start'], 
                         x1=orb_levels['orb_end'],
                         fillcolor="yellow", 
                         opacity=0.2,
                         line_width=0,
                         annotation_text="ORB Period",
                         row=1, col=1)
        
        # Add trades if provided
        if trades is not None and not trades.empty:
            # Long entries
            long_trades = trades[trades['type'] == 'LONG']
            if not long_trades.empty:
                fig.add_trace(go.Scatter(
                    x=long_trades['entry_time'],
                    y=long_trades['entry_price'],
                    mode='markers',
                    marker=dict(symbol='triangle-up', size=12, color='green'),
                    name='Long Entry'
                ), row=1, col=1)
            
            # Short entries
            short_trades = trades[trades['type'] == 'SHORT']
            if not short_trades.empty:
                fig.add_trace(go.Scatter(
                    x=short_trades['entry_time'],
                    y=short_trades['entry_price'],
                    mode='markers',
                    marker=dict(symbol='triangle-down', size=12, color='red'),
                    name='Short Entry'
                ), row=1, col=1)
            
            # Exit points
            if 'exit_time' in trades.columns:
                fig.add_trace(go.Scatter(
                    x=trades['exit_time'],
                    y=trades['exit_price'],
                    mode='markers',
                    marker=dict(symbol='x', size=10, color='blue'),
                    name='Exit'
                ), row=1, col=1)
        
        # Volume chart
        fig.add_trace(go.Bar(
            x=data.index,
            y=data['Volume'],
            name='Volume',
            marker_color='lightblue'
        ), row=2, col=1)
        
        # Update layout
        fig.update_layout(
            title=f'{symbol} - Opening Range Breakout Strategy',
            xaxis_title='Time',
            yaxis_title='Price',
            height=800,
            showlegend=True,
            xaxis_rangeslider_visible=False
        )
        
        # Save or show
        if save_path:
            fig.write_html(save_path)
            print(f"Chart saved to {save_path}")
        else:
            fig.show()
        
        return fig
    
    def plot_equity_curve(self, trades_df: pd.DataFrame, initial_capital: float = 10000,
                         save_path: str = None):
        """
        Plot equity curve from trades
        
        Args:
            trades_df: DataFrame with trade results
            initial_capital: Starting capital
            save_path: Path to save the plot
        """
        if trades_df.empty:
            print("No trades to plot")
            return None
        
        # Calculate cumulative PnL
        trades_df = trades_df.sort_values('entry_time')
        trades_df['cumulative_pnl'] = trades_df['pnl'].cumsum()
        trades_df['equity'] = initial_capital + trades_df['cumulative_pnl']
        
        # Create figure
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
        
        # Equity curve
        axes[0].plot(trades_df['exit_time'], trades_df['equity'], 
                    linewidth=2, color='blue', label='Equity')
        axes[0].axhline(y=initial_capital, color='gray', linestyle='--', 
                       alpha=0.5, label='Initial Capital')
        axes[0].fill_between(trades_df['exit_time'], initial_capital, trades_df['equity'],
                            where=(trades_df['equity'] >= initial_capital), 
                            color='green', alpha=0.3, label='Profit')
        axes[0].fill_between(trades_df['exit_time'], initial_capital, trades_df['equity'],
                            where=(trades_df['equity'] < initial_capital), 
                            color='red', alpha=0.3, label='Loss')
        
        axes[0].set_title('Equity Curve', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Date')
        axes[0].set_ylabel('Equity ($)')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Drawdown
        trades_df['peak'] = trades_df['equity'].cummax()
        trades_df['drawdown'] = (trades_df['equity'] - trades_df['peak']) / trades_df['peak'] * 100
        
        axes[1].fill_between(trades_df['exit_time'], 0, trades_df['drawdown'],
                            color='red', alpha=0.5)
        axes[1].set_title('Drawdown (%)', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Date')
        axes[1].set_ylabel('Drawdown (%)')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
            print(f"Equity curve saved to {save_path}")
        else:
            plt.show()
        
        return fig
    
    def plot_trade_distribution(self, trades_df: pd.DataFrame, save_path: str = None):
        """
        Plot trade PnL distribution
        
        Args:
            trades_df: DataFrame with trade results
            save_path: Path to save the plot
        """
        if trades_df.empty:
            print("No trades to plot")
            return None
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # PnL Distribution
        axes[0, 0].hist(trades_df['pnl'], bins=30, edgecolor='black', alpha=0.7)
        axes[0, 0].axvline(x=0, color='red', linestyle='--', linewidth=2)
        axes[0, 0].set_title('PnL Distribution', fontsize=12, fontweight='bold')
        axes[0, 0].set_xlabel('PnL ($)')
        axes[0, 0].set_ylabel('Frequency')
        
        # Win/Loss pie chart
        wins = len(trades_df[trades_df['pnl'] > 0])
        losses = len(trades_df[trades_df['pnl'] < 0])
        breakeven = len(trades_df[trades_df['pnl'] == 0])
        
        colors = ['green', 'red', 'gray']
        labels = [f'Wins ({wins})', f'Losses ({losses})', f'Breakeven ({breakeven})']
        sizes = [wins, losses, breakeven]
        
        axes[0, 1].pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
                      startangle=90)
        axes[0, 1].set_title('Win/Loss Distribution', fontsize=12, fontweight='bold')
        
        # Trade duration distribution
        if 'trade_duration' in trades_df.columns:
            axes[1, 0].hist(trades_df['trade_duration'], bins=20, 
                          edgecolor='black', alpha=0.7, color='purple')
            axes[1, 0].set_title('Trade Duration Distribution', fontsize=12, fontweight='bold')
            axes[1, 0].set_xlabel('Duration (minutes)')
            axes[1, 0].set_ylabel('Frequency')
        
        # PnL by trade type
        if 'type' in trades_df.columns:
            trade_types = trades_df.groupby('type')['pnl'].agg(['mean', 'sum', 'count'])
            
            x = range(len(trade_types))
            width = 0.35
            
            axes[1, 1].bar([i - width/2 for i in x], trade_types['mean'], 
                          width, label='Avg PnL', color='blue', alpha=0.7)
            axes[1, 1].bar([i + width/2 for i in x], trade_types['count'], 
                          width, label='Count', color='orange', alpha=0.7)
            
            axes[1, 1].set_xlabel('Trade Type')
            axes[1, 1].set_ylabel('Value')
            axes[1, 1].set_title('Performance by Trade Type', fontsize=12, fontweight='bold')
            axes[1, 1].set_xticks(x)
            axes[1, 1].set_xticklabels(trade_types.index)
            axes[1, 1].legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
            print(f"Trade distribution saved to {save_path}")
        else:
            plt.show()
        
        return fig
    
    def plot_performance_heatmap(self, trades_df: pd.DataFrame, save_path: str = None):
        """
        Create performance heatmap by day and hour
        
        Args:
            trades_df: DataFrame with trade results
            save_path: Path to save the plot
        """
        if trades_df.empty or 'entry_time' not in trades_df.columns:
            print("No trades to plot")
            return None
        
        # Extract day of week and hour
        trades_df['day_of_week'] = pd.to_datetime(trades_df['entry_time']).dt.dayofweek
        trades_df['hour'] = pd.to_datetime(trades_df['entry_time']).dt.hour
        
        # Create pivot table
        heatmap_data = trades_df.pivot_table(
            values='pnl', 
            index='hour', 
            columns='day_of_week', 
            aggfunc='sum',
            fill_value=0
        )
        
        # Rename columns
        days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
        heatmap_data.columns = [days[i] if i < len(days) else f'Day {i}' 
                                for i in heatmap_data.columns]
        
        # Create heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(heatmap_data, annot=True, fmt='.0f', cmap='RdYlGn', 
                   center=0, cbar_kws={'label': 'Total PnL ($)'})
        
        plt.title('Performance Heatmap by Day and Hour', fontsize=14, fontweight='bold')
        plt.xlabel('Day of Week')
        plt.ylabel('Hour of Day')
        
        if save_path:
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
            print(f"Heatmap saved to {save_path}")
        else:
            plt.show()
        
        return plt.gcf()
    
    def create_html_report(self, summary_df: pd.DataFrame, trades_df: pd.DataFrame,
                        metrics: Dict, save_path: str = 'reports/strategy_report.html'):
        """
        Create comprehensive HTML report (using string.Template to avoid {.format} VS CSS braces).
        """
        html_tpl = Template("""<!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <title>ORB Strategy Report</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                margin: 20px;
                background-color: #f5f5f5;
            }
            h1 {
                color: #333;
                border-bottom: 3px solid #4CAF50;
                padding-bottom: 10px;
            }
            h2 {
                color: #555;
                margin-top: 30px;
            }
            table {
                border-collapse: collapse;
                width: 100%;
                background-color: white;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
            th {
                background-color: #4CAF50;
                color: white;
                padding: 12px;
                text-align: left;
            }
            td {
                padding: 10px;
                border-bottom: 1px solid #ddd;
            }
            tr:hover {
                background-color: #f5f5f5;
            }
            .metric-box {
                display: inline-block;
                background-color: white;
                border: 1px solid #ddd;
                border-radius: 5px;
                padding: 15px;
                margin: 10px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
            .metric-value {
                font-size: 24px;
                font-weight: bold;
                color: #4CAF50;
            }
            .metric-label {
                font-size: 14px;
                color: #666;
            }
            .positive { color: green; }
            .negative { color: red; }
        </style>
    </head>
    <body>
        <h1>Opening Range Breakout Strategy - Performance Report</h1>
        <p>Report Generated: ${timestamp}</p>

        <h2>Key Performance Metrics</h2>
        <div>
            ${metrics_boxes}
        </div>

        <h2>Performance by Symbol</h2>
        ${summary_table}

        <h2>Recent Trades (Last 20)</h2>
        ${trades_table}

        <h2>Strategy Configuration</h2>
        <ul>
            <li>Opening Range: 9:30 AM - 9:45 AM EST</li>
            <li>Risk per Trade: 1%</li>
            <li>Take Profit Multiplier: 2x Range</li>
            <li>Maximum Trades per Day: 3</li>
        </ul>
    </body>
    </html>""")

        # Metrics boxes
        metrics_boxes = ""
        if metrics:
            key_metrics = [
                ('Total Return', f"{metrics.get('return_pct', 0):.1f}%"),
                ('Win Rate', f"{metrics.get('win_rate', 0)*100:.1f}%"),
                ('Profit Factor', f"{metrics.get('profit_factor', 0):.2f}"),
                ('Sharpe Ratio', f"{metrics.get('sharpe_ratio', 0):.2f}"),
                ('Max Drawdown', f"${metrics.get('max_drawdown', 0):.2f}"),
                ('Total PnL', f"${metrics.get('total_pnl', 0):.2f}")
            ]
            for label, value in key_metrics:
                # green for positives (except drawdown), else default color
                try:
                    num = float(value.replace('$', '').replace('%', ''))
                except Exception:
                    num = 0.0
                color_class = 'positive' if ('drawdown' not in label.lower() and num > 0) else ''
                metrics_boxes += (
                    f'<div class="metric-box">'
                    f'  <div class="metric-label">{label}</div>'
                    f'  <div class="metric-value {color_class}">{value}</div>'
                    f'</div>'
                )

        # Robustly handle None/empty DataFrames
        summary_df = summary_df if isinstance(summary_df, pd.DataFrame) else pd.DataFrame()
        trades_df = trades_df if isinstance(trades_df, pd.DataFrame) else pd.DataFrame()

        # Tables
        summary_table = (
            summary_df.to_html(index=False, classes='table', escape=False)
            if not summary_df.empty else "<p>No data available</p>"
        )

        if not trades_df.empty:
            recent_trades = trades_df.tail(20).copy()
            cols = [c for c in ['symbol', 'type', 'entry_time', 'entry_price',
                                'exit_price', 'exit_reason', 'pnl'] if c in recent_trades.columns]
            recent_trades = recent_trades[cols]
            trades_table = recent_trades.to_html(index=False, classes='table', escape=False)
        else:
            trades_table = "<p>No trades available</p>"

        # Substitute variables safely (CSS braces are not touched)
        html_out = html_tpl.safe_substitute(
            timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            metrics_boxes=metrics_boxes,
            summary_table=summary_table,
            trades_table=trades_table
        )

        # Write file
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(html_out)

        print(f"HTML report saved to {save_path}")
        return save_path


def main():
    """Test visualization functions"""
    print("ORB Strategy Visualizer Module")
    print("=" * 50)
    
    visualizer = StrategyVisualizer()
    print("Visualizer initialized successfully")
    print("Ready to create charts and reports")


if __name__ == "__main__":
    main()