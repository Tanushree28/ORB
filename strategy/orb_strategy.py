#!/usr/bin/env python3
"""
Opening Range Breakout (ORB) Strategy Implementation
Core strategy logic with configurable parameters
"""

import pandas as pd
import numpy as np
from datetime import datetime, time, timedelta
import pytz
import yaml
from typing import Dict, List, Tuple, Optional

class ORBStrategy:
    def __init__(self, config_path='configs/config.yaml'):
        """Initialize ORB Strategy with configuration"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Extract strategy parameters
        self.orb_start = self.config['strategy']['opening_range']['start_time']
        self.orb_end = self.config['strategy']['opening_range']['end_time']
        self.orb_duration = self.config['strategy']['opening_range']['duration_minutes']
        
        self.risk_per_trade = self.config['strategy']['risk_management']['risk_per_trade']
        self.initial_capital = self.config['strategy']['risk_management']['initial_capital']
        
        self.tp_multiplier = self.config['strategy']['trade_parameters']['tp_multiplier']
        self.sl_buffer = self.config['strategy']['trade_parameters']['sl_buffer']
        self.max_trades_per_day = self.config['strategy']['trade_parameters']['max_trades_per_day']
        
        # Trading state
        self.current_capital = self.initial_capital
        self.trades = []
        self.daily_trades = {}
        

    def calculate_opening_range(self, data: pd.DataFrame, session_date: pd.Timestamp):
        """
        Minimal, tz-safe ORB window:
        - Ensure data.index is tz-aware UTC
        - Build 09:30 -> 09:30+duration in America/New_York, then convert to UTC
        - Return your original keys (date, orb_high, orb_low, orb_range, orb_start, orb_end)
        """

        # 1) Make sure data index is tz-aware UTC
        if data.index.tz is None:
            data.index = data.index.tz_localize('UTC')
        else:
            data = data.tz_convert('UTC')

        # 2) Parse start time from self.orb_start (e.g., "09:30")
        start_h, start_m = map(int, str(self.orb_start).split(':'))
        dur_min = int(self.orb_duration)

        # 3) Build the window in market tz, then convert to UTC
        market_tz = 'America/New_York'
        d = pd.Timestamp(session_date)
        d = d.tz_localize(market_tz) if d.tzinfo is None else d.tz_convert(market_tz)

        start_local = d.replace(hour=start_h, minute=start_m, second=0, microsecond=0)
        end_local = start_local + pd.Timedelta(minutes=dur_min)

        # Convert window to UTC to match data.index
        start_dt = start_local.tz_convert('UTC')
        end_dt   = end_local.tz_convert('UTC')

        # 4) Filter
        orb_data = data[(data.index >= start_dt) & (data.index <= end_dt)]
        if orb_data.empty:
            return {
                'date': d.date(),
                'orb_high': None,
                'orb_low': None,
                'orb_range': None,
                'orb_start': start_dt,
                'orb_end': end_dt
            }

        # 5) Stats + return (keys unchanged)
        orb_high = float(orb_data['High'].max())
        orb_low  = float(orb_data['Low'].min())
        orb_range = orb_high - orb_low

        return {
            'date': d.date(),
            'orb_high': orb_high,
            'orb_low': orb_low,
            'orb_range': orb_range,
            'orb_start': start_dt,
            'orb_end': end_dt
        }

    
    def calculate_position_size(self, entry_price: float, stop_loss: float) -> int:
        """
        Calculate position size based on risk management rules
        
        Args:
            entry_price: Entry price for the trade
            stop_loss: Stop loss price
            
        Returns:
            Number of shares/contracts to trade
        """
        # Calculate risk amount
        risk_amount = self.current_capital * self.risk_per_trade
        
        # Calculate price risk per share
        price_risk = abs(entry_price - stop_loss)
        
        if price_risk == 0:
            return 0
        
        # Calculate position size
        position_size = int(risk_amount / price_risk)
        
        # Ensure we don't exceed available capital
        max_position = int(self.current_capital / entry_price)
        position_size = min(position_size, max_position)
        
        return max(1, position_size)  # At least 1 share
    
    def check_entry_signals(self, current_bar: pd.Series, orb: Dict, 
                           previous_bar: pd.Series = None) -> Optional[Dict]:
        """
        Check for entry signals based on ORB breakout
        
        Args:
            current_bar: Current price bar
            orb: Opening range data
            previous_bar: Previous price bar (for breakout confirmation)
            
        Returns:
            Trade signal dictionary or None
        """
        if orb is None:
            return None
        
        # Check if we're past the opening range period
        if current_bar.name <= orb['orb_end']:
            return None
        
        # Check daily trade limit
        date_str = current_bar.name.date()
        if date_str in self.daily_trades:
            if self.daily_trades[date_str] >= self.max_trades_per_day:
                return None
        
        signal = None
        
        # Check for breakout above ORB high (Long signal)
        if previous_bar is not None:
            if previous_bar['Close'] <= orb['orb_high'] and current_bar['Close'] > orb['orb_high']:
                # Long entry signal
                entry_price = orb['orb_high'] + self.sl_buffer
                stop_loss = orb['orb_low'] - self.sl_buffer
                take_profit = entry_price + (self.tp_multiplier * orb['orb_range'])
                
                signal = {
                    'type': 'LONG',
                    'entry_time': current_bar.name,
                    'entry_price': entry_price,
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'orb_high': orb['orb_high'],
                    'orb_low': orb['orb_low'],
                    'orb_range': orb['orb_range'],
                    'position_size': self.calculate_position_size(entry_price, stop_loss)
                }
            
            # Check for breakout below ORB low (Short signal)
            elif previous_bar['Close'] >= orb['orb_low'] and current_bar['Close'] < orb['orb_low']:
                # Short entry signal
                entry_price = orb['orb_low'] - self.sl_buffer
                stop_loss = orb['orb_high'] + self.sl_buffer
                take_profit = entry_price - (self.tp_multiplier * orb['orb_range'])
                
                signal = {
                    'type': 'SHORT',
                    'entry_time': current_bar.name,
                    'entry_price': entry_price,
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'orb_high': orb['orb_high'],
                    'orb_low': orb['orb_low'],
                    'orb_range': orb['orb_range'],
                    'position_size': self.calculate_position_size(entry_price, stop_loss)
                }
        
        return signal
    
    def check_exit_signals(self, current_bar: pd.Series, open_trade: Dict) -> Optional[Dict]:
        """
        Check for exit signals (SL or TP hit)
        
        Args:
            current_bar: Current price bar
            open_trade: Open trade dictionary
            
        Returns:
            Exit signal dictionary or None
        """
        if open_trade is None:
            return None
        
        exit_signal = None
        
        if open_trade['type'] == 'LONG':
            # Check stop loss
            if current_bar['Low'] <= open_trade['stop_loss']:
                exit_signal = {
                    'exit_time': current_bar.name,
                    'exit_price': open_trade['stop_loss'],
                    'exit_reason': 'STOP_LOSS',
                    'pnl': (open_trade['stop_loss'] - open_trade['entry_price']) * open_trade['position_size']
                }
            # Check take profit
            elif current_bar['High'] >= open_trade['take_profit']:
                exit_signal = {
                    'exit_time': current_bar.name,
                    'exit_price': open_trade['take_profit'],
                    'exit_reason': 'TAKE_PROFIT',
                    'pnl': (open_trade['take_profit'] - open_trade['entry_price']) * open_trade['position_size']
                }
        
        elif open_trade['type'] == 'SHORT':
            # Check stop loss
            if current_bar['High'] >= open_trade['stop_loss']:
                exit_signal = {
                    'exit_time': current_bar.name,
                    'exit_price': open_trade['stop_loss'],
                    'exit_reason': 'STOP_LOSS',
                    'pnl': (open_trade['entry_price'] - open_trade['stop_loss']) * open_trade['position_size']
                }
            # Check take profit
            elif current_bar['Low'] <= open_trade['take_profit']:
                exit_signal = {
                    'exit_time': current_bar.name,
                    'exit_price': open_trade['take_profit'],
                    'exit_reason': 'TAKE_PROFIT',
                    'pnl': (open_trade['entry_price'] - open_trade['take_profit']) * open_trade['position_size']
                }
        
        return exit_signal
    
    def backtest(self, data: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """
        Run backtest on historical data
        
        Args:
            data: Historical price data
            symbol: Trading symbol
            
        Returns:
            DataFrame with trade results
        """
        # Reset trading state
        self.current_capital = self.initial_capital
        self.trades = []
        self.daily_trades = {}
        
        # Group data by date
        dates = pd.to_datetime(data.index.date).unique()
        
        open_trade = None
        previous_bar = None
        
        print(f"Backtesting {symbol} from {dates[0]} to {dates[-1]}")
        
        for date in dates:
            # Calculate ORB for the day
            orb = self.calculate_opening_range(data, pd.Timestamp(date))
            
            if orb is None:
                continue
            
            # Get data for the day after ORB period
            day_data = data[data.index.date == date.date()]
            day_data = day_data[day_data.index > orb['orb_end']]
            
            for idx, current_bar in day_data.iterrows():
                # Check for exit signals first if we have an open trade
                if open_trade is not None:
                    exit_signal = self.check_exit_signals(current_bar, open_trade)
                    
                    if exit_signal:
                        # Close the trade
                        trade_result = {**open_trade, **exit_signal}
                        trade_result['symbol'] = symbol
                        trade_result['trade_duration'] = (exit_signal['exit_time'] - open_trade['entry_time']).total_seconds() / 60
                        
                        # Update capital
                        self.current_capital += exit_signal['pnl']
                        trade_result['capital_after'] = self.current_capital
                        
                        self.trades.append(trade_result)
                        open_trade = None
                
                # Check for entry signals if no open trade
                if open_trade is None and previous_bar is not None:
                    entry_signal = self.check_entry_signals(current_bar, orb, previous_bar)
                    
                    if entry_signal:
                        # Open new trade
                        open_trade = entry_signal
                        
                        # Update daily trade counter
                        date_str = current_bar.name.date()
                        if date_str not in self.daily_trades:
                            self.daily_trades[date_str] = 0
                        self.daily_trades[date_str] += 1
                
                previous_bar = current_bar
        
        # Close any remaining open trade at the end
        if open_trade is not None and len(day_data) > 0:
            last_bar = day_data.iloc[-1]
            exit_price = last_bar['Close']
            
            if open_trade['type'] == 'LONG':
                pnl = (exit_price - open_trade['entry_price']) * open_trade['position_size']
            else:
                pnl = (open_trade['entry_price'] - exit_price) * open_trade['position_size']
            
            trade_result = {
                **open_trade,
                'exit_time': last_bar.name,
                'exit_price': exit_price,
                'exit_reason': 'END_OF_DATA',
                'pnl': pnl,
                'symbol': symbol,
                'trade_duration': (last_bar.name - open_trade['entry_time']).total_seconds() / 60
            }
            
            self.current_capital += pnl
            trade_result['capital_after'] = self.current_capital
            
            self.trades.append(trade_result)
        
        # Create trades DataFrame
        if self.trades:
            trades_df = pd.DataFrame(self.trades)
            return trades_df
        else:
            return pd.DataFrame()
    
    def calculate_metrics(self, trades_df: pd.DataFrame) -> Dict:
        """
        Calculate performance metrics from trades
        
        Args:
            trades_df: DataFrame with trade results
            
        Returns:
            Dictionary with performance metrics
        """
        if trades_df.empty:
            return {
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0,
                'total_pnl': 0,
                'average_win': 0,
                'average_loss': 0,
                'profit_factor': 0,
                'max_drawdown': 0,
                'sharpe_ratio': 0,
                'final_capital': self.initial_capital,
                'return_pct': 0
            }
        
        # Basic metrics
        total_trades = len(trades_df)
        winning_trades = len(trades_df[trades_df['pnl'] > 0])
        losing_trades = len(trades_df[trades_df['pnl'] < 0])
        
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        # PnL metrics
        total_pnl = trades_df['pnl'].sum()
        
        wins = trades_df[trades_df['pnl'] > 0]['pnl']
        losses = trades_df[trades_df['pnl'] < 0]['pnl']
        
        average_win = wins.mean() if len(wins) > 0 else 0
        average_loss = losses.mean() if len(losses) > 0 else 0
        
        # Profit factor
        gross_profit = wins.sum() if len(wins) > 0 else 0
        gross_loss = abs(losses.sum()) if len(losses) > 0 else 1
        profit_factor = gross_profit / gross_loss if gross_loss != 0 else 0
        
        # Max drawdown
        trades_df['cumulative_pnl'] = trades_df['pnl'].cumsum()
        trades_df['running_max'] = trades_df['cumulative_pnl'].cummax()
        trades_df['drawdown'] = trades_df['cumulative_pnl'] - trades_df['running_max']
        max_drawdown = trades_df['drawdown'].min()
        
        # Sharpe ratio (simplified)
        if len(trades_df) > 1:
            returns = trades_df['pnl'] / self.initial_capital
            sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() != 0 else 0
        else:
            sharpe_ratio = 0
        
        # Final metrics
        final_capital = self.current_capital
        return_pct = ((final_capital - self.initial_capital) / self.initial_capital) * 100
        
        return {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'average_win': average_win,
            'average_loss': average_loss,
            'profit_factor': profit_factor,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'final_capital': final_capital,
            'return_pct': return_pct
        }


def main():
    """Test the ORB strategy"""
    print("ORB Strategy Module Loaded Successfully")
    
    # Initialize strategy
    strategy = ORBStrategy('../configs/config.yaml')
    print(f"Strategy initialized with:")
    print(f"  ORB Period: {strategy.orb_start} - {strategy.orb_end}")
    print(f"  Risk per trade: {strategy.risk_per_trade * 100}%")
    print(f"  TP Multiplier: {strategy.tp_multiplier}")


if __name__ == "__main__":
    main()