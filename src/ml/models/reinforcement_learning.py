"""
Reinforcement Learning for Options Strategy Optimization
Uses RL to optimize strategy parameters and position sizing for maximum risk-adjusted returns
"""

import numpy as np
import pandas as pd
import gym
from gym import spaces
import torch
import torch.nn as nn
import torch.optim as optim
from stable_baselines3 import PPO, A2C, SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
import optuna
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
from dataclasses import dataclass
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

@dataclass
class StrategyParams:
    """Strategy parameters to be optimized"""
    delta_target: float = 0.3
    days_to_expiry: int = 30
    profit_target: float = 0.5
    stop_loss: float = 0.3
    position_size: float = 0.1
    volatility_threshold: float = 0.25

@dataclass
class PortfolioState:
    """Current portfolio state"""
    cash: float
    positions: List[Dict]
    unrealized_pnl: float
    realized_pnl: float
    total_value: float
    max_drawdown: float
    sharpe_ratio: float
    win_rate: float

class OptionsStrategyEnv(gym.Env):
    """
    Gym environment for options strategy optimization
    """
    
    def __init__(self, market_data: pd.DataFrame, strategy_type: str = 'covered_call',
                 initial_capital: float = 100000, transaction_cost: float = 1.0):
        super(OptionsStrategyEnv, self).__init__()
        
        self.market_data = market_data.reset_index(drop=True)
        self.strategy_type = strategy_type
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        
        # Current state
        self.current_step = 0
        self.portfolio = PortfolioState(
            cash=initial_capital,
            positions=[],
            unrealized_pnl=0,
            realized_pnl=0,
            total_value=initial_capital,
            max_drawdown=0,
            sharpe_ratio=0,
            win_rate=0
        )
        
        # Action space: [delta_target, days_to_expiry, profit_target, stop_loss, position_size]
        self.action_space = spaces.Box(
            low=np.array([0.1, 7, 0.1, 0.1, 0.01]),
            high=np.array([0.9, 90, 1.0, 0.8, 0.5]),
            dtype=np.float32
        )
        
        # Observation space: market features + portfolio state
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(20,),  # Market features + portfolio state
            dtype=np.float32
        )
        
        # Track performance metrics
        self.equity_curve = [initial_capital]
        self.returns = []
        self.trade_history = []
        
        self.logger = logging.getLogger(__name__)
    
    def reset(self):
        """Reset environment to initial state"""
        
        self.current_step = 0
        self.portfolio = PortfolioState(
            cash=self.initial_capital,
            positions=[],
            unrealized_pnl=0,
            realized_pnl=0,
            total_value=self.initial_capital,
            max_drawdown=0,
            sharpe_ratio=0,
            win_rate=0
        )
        
        self.equity_curve = [self.initial_capital]
        self.returns = []
        self.trade_history = []
        
        return self._get_observation()
    
    def step(self, action):
        """Execute one step in the environment"""
        
        # Decode action
        strategy_params = StrategyParams(
            delta_target=action[0],
            days_to_expiry=int(action[1]),
            profit_target=action[2],
            stop_loss=action[3],
            position_size=action[4]
        )
        
        # Execute strategy with current parameters
        reward = self._execute_strategy(strategy_params)
        
        # Update portfolio state
        self._update_portfolio_state()
        
        # Check if episode is done
        self.current_step += 1
        done = (self.current_step >= len(self.market_data) - 30) or (self.portfolio.total_value <= 0.5 * self.initial_capital)
        
        # Get next observation
        obs = self._get_observation()
        
        # Additional info
        info = {
            'total_value': self.portfolio.total_value,
            'sharpe_ratio': self.portfolio.sharpe_ratio,
            'max_drawdown': self.portfolio.max_drawdown,
            'num_trades': len(self.trade_history)
        }
        
        return obs, reward, done, info
    
    def _get_observation(self):
        """Get current observation state"""
        
        if self.current_step >= len(self.market_data):
            self.current_step = len(self.market_data) - 1
        
        current_data = self.market_data.iloc[self.current_step]
        
        # Market features
        market_features = [
            current_data.get('Close', 100) / 100,  # Normalized price
            current_data.get('Volume', 1000000) / 1000000,  # Normalized volume
            current_data.get('returns', 0),
            current_data.get('realized_vol', 0.2),
            current_data.get('vix_level', 20) / 100,
            current_data.get('rsi_14', 50) / 100,
            current_data.get('bb_position', 0.5),
            current_data.get('momentum_10', 0),
            current_data.get('high_vol_regime', 0),
            current_data.get('put_call_ratio', 1),
        ]
        
        # Portfolio features
        portfolio_features = [
            self.portfolio.cash / self.initial_capital,
            len(self.portfolio.positions) / 10,  # Normalized position count
            self.portfolio.unrealized_pnl / self.initial_capital,
            self.portfolio.total_value / self.initial_capital,
            self.portfolio.max_drawdown,
            self.portfolio.sharpe_ratio / 3,  # Normalized Sharpe
            self.portfolio.win_rate,
            len(self.trade_history) / 100,  # Normalized trade count
            self._get_portfolio_delta(),
            self._get_portfolio_vega()
        ]
        
        # Combine features
        observation = np.array(market_features + portfolio_features, dtype=np.float32)
        
        # Handle any NaN or infinite values
        observation = np.nan_to_num(observation, nan=0.0, posinf=1.0, neginf=-1.0)
        
        return observation
    
    def _execute_strategy(self, params: StrategyParams):
        """Execute strategy with given parameters and return reward"""
        
        if self.current_step >= len(self.market_data) - params.days_to_expiry:
            return 0  # Not enough data for strategy execution
        
        current_price = self.market_data.iloc[self.current_step]['Close']
        current_vol = self.market_data.iloc[self.current_step].get('realized_vol', 0.2)
        
        # Strategy-specific execution
        if self.strategy_type == 'covered_call':
            return self._execute_covered_call(params, current_price, current_vol)
        elif self.strategy_type == 'straddle':
            return self._execute_straddle(params, current_price, current_vol)
        elif self.strategy_type == 'iron_condor':
            return self._execute_iron_condor(params, current_price, current_vol)
        else:
            return 0
    
    def _execute_covered_call(self, params: StrategyParams, current_price: float, current_vol: float):
        """Execute covered call strategy"""
        
        # Position sizing
        position_value = self.portfolio.cash * params.position_size
        num_shares = int(position_value / current_price / 100) * 100  # Round to lots
        
        if num_shares < 100:  # Need at least 100 shares for covered call
            return -0.1  # Penalty for invalid position size
        
        # Calculate option strike (slightly OTM)
        strike_price = current_price * (1 + params.delta_target * 0.1)
        
        # Estimate option premium using Black-Scholes approximation
        time_to_expiry = params.days_to_expiry / 365
        option_premium = self._estimate_option_price(current_price, strike_price, time_to_expiry, current_vol, 'call')
        
        # Execute trade
        stock_cost = num_shares * current_price
        option_premium_received = option_premium * (num_shares / 100)  # Premium per contract
        total_cost = stock_cost - option_premium_received + self.transaction_cost
        
        if total_cost > self.portfolio.cash:
            return -0.1  # Penalty for insufficient cash
        
        # Create position
        position = {
            'type': 'covered_call',
            'entry_date': self.current_step,
            'expiry_date': self.current_step + params.days_to_expiry,
            'shares': num_shares,
            'stock_price': current_price,
            'strike_price': strike_price,
            'option_premium': option_premium,
            'profit_target': params.profit_target,
            'stop_loss': params.stop_loss,
            'status': 'open'
        }
        
        self.portfolio.positions.append(position)
        self.portfolio.cash -= total_cost
        
        # Calculate expected reward based on historical performance
        reward = self._calculate_strategy_reward(position, params)
        
        return reward
    
    def _execute_straddle(self, params: StrategyParams, current_price: float, current_vol: float):
        """Execute long straddle strategy"""
        
        position_value = self.portfolio.cash * params.position_size
        
        # ATM straddle
        strike_price = current_price
        time_to_expiry = params.days_to_expiry / 365
        
        call_premium = self._estimate_option_price(current_price, strike_price, time_to_expiry, current_vol, 'call')
        put_premium = self._estimate_option_price(current_price, strike_price, time_to_expiry, current_vol, 'put')
        
        straddle_cost = call_premium + put_premium
        num_contracts = int(position_value / (straddle_cost * 100))
        
        if num_contracts < 1:
            return -0.1
        
        total_cost = num_contracts * straddle_cost * 100 + self.transaction_cost
        
        if total_cost > self.portfolio.cash:
            return -0.1
        
        position = {
            'type': 'straddle',
            'entry_date': self.current_step,
            'expiry_date': self.current_step + params.days_to_expiry,
            'contracts': num_contracts,
            'strike_price': strike_price,
            'entry_price': current_price,
            'call_premium': call_premium,
            'put_premium': put_premium,
            'profit_target': params.profit_target,
            'stop_loss': params.stop_loss,
            'status': 'open'
        }
        
        self.portfolio.positions.append(position)
        self.portfolio.cash -= total_cost
        
        reward = self._calculate_strategy_reward(position, params)
        return reward
    
    def _execute_iron_condor(self, params: StrategyParams, current_price: float, current_vol: float):
        """Execute iron condor strategy"""
        
        position_value = self.portfolio.cash * params.position_size
        
        # Iron condor strikes (simplified)
        strike_width = current_price * 0.05  # 5% width
        call_strike_short = current_price + strike_width
        call_strike_long = current_price + 2 * strike_width
        put_strike_short = current_price - strike_width
        put_strike_long = current_price - 2 * strike_width
        
        time_to_expiry = params.days_to_expiry / 365
        
        # Estimate net premium received
        premium_received = self._estimate_iron_condor_premium(
            current_price, put_strike_long, put_strike_short,
            call_strike_short, call_strike_long, time_to_expiry, current_vol
        )
        
        num_contracts = int(position_value / (strike_width * 100))  # Risk-based sizing
        
        if num_contracts < 1:
            return -0.1
        
        total_premium = num_contracts * premium_received * 100
        
        position = {
            'type': 'iron_condor',
            'entry_date': self.current_step,
            'expiry_date': self.current_step + params.days_to_expiry,
            'contracts': num_contracts,
            'entry_price': current_price,
            'premium_received': premium_received,
            'max_profit': total_premium,
            'max_loss': num_contracts * strike_width * 100 - total_premium,
            'profit_target': params.profit_target,
            'stop_loss': params.stop_loss,
            'status': 'open'
        }
        
        self.portfolio.positions.append(position)
        self.portfolio.cash += total_premium - self.transaction_cost
        
        reward = self._calculate_strategy_reward(position, params)
        return reward
    
    def _estimate_option_price(self, S: float, K: float, T: float, vol: float, option_type: str):
        """Simplified Black-Scholes option pricing"""
        
        from scipy.stats import norm
        import math
        
        r = 0.02  # Risk-free rate
        
        d1 = (math.log(S / K) + (r + 0.5 * vol**2) * T) / (vol * math.sqrt(T))
        d2 = d1 - vol * math.sqrt(T)
        
        if option_type == 'call':
            price = S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
        else:  # put
            price = K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        
        return max(price, 0.01)
    
    def _estimate_iron_condor_premium(self, S, K1, K2, K3, K4, T, vol):
        """Estimate iron condor net premium"""
        
        put_short = self._estimate_option_price(S, K2, T, vol, 'put')
        put_long = self._estimate_option_price(S, K1, T, vol, 'put')
        call_short = self._estimate_option_price(S, K3, T, vol, 'call')
        call_long = self._estimate_option_price(S, K4, T, vol, 'call')
        
        net_premium = put_short - put_long + call_short - call_long
        return max(net_premium, 0.01)
    
    def _calculate_strategy_reward(self, position: Dict, params: StrategyParams):
        """Calculate reward based on strategy performance"""
        
        # Simulate strategy outcome based on market conditions
        if self.current_step + params.days_to_expiry >= len(self.market_data):
            return 0
        
        entry_price = position.get('entry_price', position.get('stock_price', 100))
        future_prices = self.market_data['Close'].iloc[
            self.current_step:self.current_step + params.days_to_expiry
        ]
        
        if len(future_prices) == 0:
            return 0
        
        final_price = future_prices.iloc[-1]
        max_price = future_prices.max()
        min_price = future_prices.min()
        
        # Strategy-specific reward calculation
        if position['type'] == 'covered_call':
            # Covered call profits from sideways/slightly bullish movement
            price_change = (final_price - entry_price) / entry_price
            if price_change > params.profit_target:
                return -0.5  # Loss from called away shares
            elif price_change < -params.stop_loss:
                return -1.0  # Large loss on underlying
            else:
                return 1.0 + position['option_premium'] / entry_price  # Premium collected
                
        elif position['type'] == 'straddle':
            # Straddle profits from large moves
            max_move = max(abs(max_price - entry_price), abs(min_price - entry_price)) / entry_price
            if max_move > 0.05:  # 5% move
                return 2.0 * max_move  # Reward large moves
            else:
                return -0.5  # Time decay penalty
                
        elif position['type'] == 'iron_condor':
            # Iron condor profits from low volatility
            price_range = (max_price - min_price) / entry_price
            if price_range < 0.03:  # Low volatility
                return 1.5
            else:
                return -1.0 * price_range  # Penalty for high volatility
        
        return 0
    
    def _update_portfolio_state(self):
        """Update portfolio state and performance metrics"""
        
        # Close expired positions
        self._close_expired_positions()
        
        # Calculate current portfolio value
        current_value = self.portfolio.cash
        for position in self.portfolio.positions:
            if position['status'] == 'open':
                current_value += self._estimate_position_value(position)
        
        self.portfolio.total_value = current_value
        self.equity_curve.append(current_value)
        
        # Calculate returns
        if len(self.equity_curve) > 1:
            daily_return = (current_value - self.equity_curve[-2]) / self.equity_curve[-2]
            self.returns.append(daily_return)
        
        # Update performance metrics
        self._update_performance_metrics()
    
    def _close_expired_positions(self):
        """Close positions that have reached expiry"""
        
        for position in self.portfolio.positions:
            if position['status'] == 'open' and self.current_step >= position['expiry_date']:
                pnl = self._calculate_position_pnl(position)
                self.portfolio.realized_pnl += pnl
                position['status'] = 'closed'
                position['exit_pnl'] = pnl
                
                self.trade_history.append({
                    'entry_date': position['entry_date'],
                    'exit_date': self.current_step,
                    'strategy': position['type'],
                    'pnl': pnl,
                    'success': pnl > 0
                })
    
    def _estimate_position_value(self, position: Dict):
        """Estimate current value of open position"""
        
        current_price = self.market_data.iloc[self.current_step]['Close']
        
        if position['type'] == 'covered_call':
            stock_value = position['shares'] * current_price
            # Simplified option value estimation
            option_value = max(current_price - position['strike_price'], 0) * (position['shares'] / 100)
            return stock_value - option_value  # Short option position
            
        elif position['type'] == 'straddle':
            strike = position['strike_price']
            intrinsic_value = max(abs(current_price - strike) - position['call_premium'] - position['put_premium'], 0)
            return intrinsic_value * position['contracts'] * 100
            
        elif position['type'] == 'iron_condor':
            # Simplified: assume max profit if price is within strikes
            return position['max_profit'] if abs(current_price - position['entry_price']) < 0.05 * position['entry_price'] else -position['max_loss']
        
        return 0
    
    def _calculate_position_pnl(self, position: Dict):
        """Calculate P&L for a position at expiry"""
        
        final_price = self.market_data.iloc[min(position['expiry_date'], len(self.market_data)-1)]['Close']
        
        if position['type'] == 'covered_call':
            stock_pnl = position['shares'] * (final_price - position['stock_price'])
            option_pnl = position['option_premium'] * (position['shares'] / 100)
            if final_price > position['strike_price']:
                option_pnl -= (final_price - position['strike_price']) * (position['shares'] / 100)
            return stock_pnl + option_pnl
            
        elif position['type'] == 'straddle':
            intrinsic = max(abs(final_price - position['strike_price']) - position['call_premium'] - position['put_premium'], 0)
            return intrinsic * position['contracts'] * 100 - self.transaction_cost
            
        elif position['type'] == 'iron_condor':
            if abs(final_price - position['entry_price']) < 0.05 * position['entry_price']:
                return position['max_profit']
            else:
                return -position['max_loss']
        
        return 0
    
    def _update_performance_metrics(self):
        """Update Sharpe ratio, drawdown, and win rate"""
        
        if len(self.returns) > 1:
            # Sharpe ratio
            mean_return = np.mean(self.returns)
            std_return = np.std(self.returns)
            self.portfolio.sharpe_ratio = (mean_return / std_return * np.sqrt(252)) if std_return > 0 else 0
            
            # Max drawdown
            peak = max(self.equity_curve)
            current_dd = (peak - self.portfolio.total_value) / peak
            self.portfolio.max_drawdown = max(self.portfolio.max_drawdown, current_dd)
            
            # Win rate
            if self.trade_history:
                wins = sum(1 for trade in self.trade_history if trade['success'])
                self.portfolio.win_rate = wins / len(self.trade_history)
    
    def _get_portfolio_delta(self):
        """Estimate portfolio delta exposure"""
        
        total_delta = 0
        for position in self.portfolio.positions:
            if position['status'] == 'open':
                if position['type'] == 'covered_call':
                    total_delta += position['shares'] * 0.5  # Simplified delta
                elif position['type'] == 'straddle':
                    total_delta += 0  # Delta neutral
                elif position['type'] == 'iron_condor':
                    total_delta += 0  # Delta neutral
        
        return total_delta / self.initial_capital
    
    def _get_portfolio_vega(self):
        """Estimate portfolio vega exposure"""
        
        total_vega = 0
        for position in self.portfolio.positions:
            if position['status'] == 'open':
                if position['type'] == 'covered_call':
                    total_vega -= 100  # Short vega
                elif position['type'] == 'straddle':
                    total_vega += 200  # Long vega
                elif position['type'] == 'iron_condor':
                    total_vega -= 50  # Short vega
        
        return total_vega / self.initial_capital

class RLStrategyOptimizer:
    """
    Reinforcement Learning optimizer for options strategies
    """
    
    def __init__(self, market_data: pd.DataFrame, strategy_type: str = 'covered_call'):
        self.market_data = market_data
        self.strategy_type = strategy_type
        self.env = None
        self.model = None
        self.logger = logging.getLogger(__name__)
        
        # Results tracking
        self.optimization_results = {}
        self.best_params = None
        self.training_rewards = []
        
    def create_environment(self):
        """Create the RL environment"""
        
        self.env = OptionsStrategyEnv(
            market_data=self.market_data,
            strategy_type=self.strategy_type,
            initial_capital=100000
        )
        
        return self.env
    
    def train_ppo_agent(self, total_timesteps: int = 50000):
        """Train PPO agent for strategy optimization"""
        
        if self.env is None:
            self.create_environment()
        
        # Create vectorized environment
        vec_env = DummyVecEnv([lambda: self.env])
        
        # Initialize PPO model
        self.model = PPO(
            'MlpPolicy',
            vec_env,
            verbose=1,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01
        )
        
        # Training callback
        eval_callback = EvalCallback(
            vec_env,
            best_model_save_path='./logs/',
            log_path='./logs/',
            eval_freq=5000,
            deterministic=True,
            render=False
        )
        
        # Train the model
        self.logger.info(f"Training PPO agent for {total_timesteps} timesteps...")
        self.model.learn(total_timesteps=total_timesteps, callback=eval_callback)
        
        return self.model
    
    def optimize_with_optuna(self, n_trials: int = 100):
        """Use Optuna for hyperparameter optimization"""
        
        def objective(trial):
            # Define hyperparameter search space
            learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-2)
            n_steps = trial.suggest_int('n_steps', 512, 4096, step=512)
            batch_size = trial.suggest_int('batch_size', 32, 256, step=32)
            ent_coef = trial.suggest_loguniform('ent_coef', 1e-4, 1e-1)
            
            # Create environment
            env = OptionsStrategyEnv(self.market_data, self.strategy_type)
            vec_env = DummyVecEnv([lambda: env])
            
            # Create model with trial parameters
            model = PPO(
                'MlpPolicy',
                vec_env,
                learning_rate=learning_rate,
                n_steps=n_steps,
                batch_size=batch_size,
                ent_coef=ent_coef,
                verbose=0
            )
            
            # Train model
            model.learn(total_timesteps=10000)
            
            # Evaluate performance
            obs = vec_env.reset()
            total_reward = 0
            for _ in range(100):
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, info = vec_env.step(action)
                total_reward += reward[0]
                if done[0]:
                    break
            
            return total_reward
        
        # Run optimization
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials)
        
        self.optimization_results = {
            'best_params': study.best_params,
            'best_value': study.best_value,
            'trials': study.trials_dataframe()
        }
        
        self.logger.info(f"Best hyperparameters: {study.best_params}")
        self.logger.info(f"Best value: {study.best_value}")
        
        return study.best_params
    
    def evaluate_strategy(self, model, num_episodes: int = 10):
        """Evaluate trained strategy"""
        
        if self.env is None:
            self.create_environment()
        
        evaluation_results = []
        
        for episode in range(num_episodes):
            obs = self.env.reset()
            episode_reward = 0
            episode_info = []
            
            while True:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, info = self.env.step(action)
                episode_reward += reward
                episode_info.append(info)
                
                if done:
                    break
            
            evaluation_results.append({
                'episode': episode,
                'total_reward': episode_reward,
                'final_value': info['total_value'],
                'sharpe_ratio': info['sharpe_ratio'],
                'max_drawdown': info['max_drawdown'],
                'num_trades': info['num_trades']
            })
        
        # Calculate aggregate metrics
        aggregate_results = {
            'mean_reward': np.mean([r['total_reward'] for r in evaluation_results]),
            'mean_final_value': np.mean([r['final_value'] for r in evaluation_results]),
            'mean_sharpe': np.mean([r['sharpe_ratio'] for r in evaluation_results]),
            'mean_max_drawdown': np.mean([r['max_drawdown'] for r in evaluation_results]),
            'mean_num_trades': np.mean([r['num_trades'] for r in evaluation_results]),
            'std_reward': np.std([r['total_reward'] for r in evaluation_results])
        }
        
        return evaluation_results, aggregate_results
    
    def extract_optimal_strategy(self, model):
        """Extract optimal strategy parameters from trained model"""
        
        if self.env is None:
            self.create_environment()
        
        # Sample multiple states and extract actions
        optimal_actions = []
        
        for _ in range(100):
            obs = self.env.reset()
            action, _ = model.predict(obs, deterministic=True)
            optimal_actions.append(action)
        
        # Average the actions to get stable strategy parameters
        mean_action = np.mean(optimal_actions, axis=0)
        
        self.best_params = StrategyParams(
            delta_target=mean_action[0],
            days_to_expiry=int(mean_action[1]),
            profit_target=mean_action[2],
            stop_loss=mean_action[3],
            position_size=mean_action[4]
        )
        
        return self.best_params
    
    def save_results(self, filepath: str):
        """Save optimization results"""
        
        results = {
            'strategy_type': self.strategy_type,
            'optimization_results': self.optimization_results,
            'best_params': self.best_params.__dict__ if self.best_params else None,
            'training_rewards': self.training_rewards
        }
        
        import json
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2, default=str)
    
    def load_results(self, filepath: str):
        """Load optimization results"""
        
        import json
        with open(filepath, 'r') as f:
            results = json.load(f)
        
        self.optimization_results = results.get('optimization_results', {})
        self.training_rewards = results.get('training_rewards', [])
        
        if results.get('best_params'):
            self.best_params = StrategyParams(**results['best_params'])

# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Create sample market data
    dates = pd.date_range('2020-01-01', '2024-01-01', freq='D')
    np.random.seed(42)
    
    returns = np.random.normal(0.0005, 0.02, len(dates))
    prices = 100 * np.exp(np.cumsum(returns))
    
    market_data = pd.DataFrame({
        'Close': prices,
        'Volume': np.random.randint(1000000, 10000000, len(dates)),
        'returns': returns,
        'realized_vol': np.abs(returns).rolling(20).mean() * np.sqrt(252),
        'vix_level': 20 + 10 * np.random.randn(len(dates)),
        'rsi_14': 50 + 20 * np.random.randn(len(dates)),
        'bb_position': np.random.uniform(0, 1, len(dates)),
        'momentum_10': returns * 10,
        'high_vol_regime': np.random.choice([0, 1], len(dates)),
        'put_call_ratio': 1 + 0.5 * np.random.randn(len(dates))
    }, index=dates)
    
    # Test RL optimizer
    optimizer = RLStrategyOptimizer(market_data, 'covered_call')
    
    # Train agent
    model = optimizer.train_ppo_agent(total_timesteps=10000)
    
    # Evaluate strategy
    eval_results, agg_results = optimizer.evaluate_strategy(model, num_episodes=5)
    print("Evaluation results:", agg_results)
    
    # Extract optimal parameters
    best_params = optimizer.extract_optimal_strategy(model)
    print(f"Optimal parameters: {best_params}")