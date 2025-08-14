"""
ML-Based Strategy Signal Generation
Uses machine learning to predict optimal entry/exit signals for options strategies
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score, TimeSeriesSplit
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import xgboost as xgb
import lightgbm as lgb
from typing import Dict, List, Tuple, Optional, Any
import joblib
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
import ta  # Technical Analysis library
import warnings
warnings.filterwarnings('ignore')

@dataclass
class StrategySignal:
    """Signal output for strategy decisions"""
    signal: int  # 1: Enter Long, -1: Enter Short, 0: Stay Out
    confidence: float  # 0-1
    strategy_type: str
    timestamp: str
    features_used: Dict[str, float]
    expected_return: Optional[float] = None
    expected_risk: Optional[float] = None

class StrategySignalGenerator:
    """
    Generate ML-based trading signals for options strategies
    """
    
    def __init__(self, strategy_type: str = 'straddle', model_type: str = 'xgboost'):
        self.strategy_type = strategy_type
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = []
        self.logger = logging.getLogger(__name__)
        
        # Strategy-specific parameters
        self.strategy_configs = {
            'straddle': {
                'lookback_period': 20,
                'vol_threshold': 0.3,
                'profit_target': 0.2,
                'stop_loss': 0.4
            },
            'covered_call': {
                'lookback_period': 30,
                'delta_target': 0.3,
                'profit_target': 0.5,
                'stop_loss': 0.2
            },
            'iron_condor': {
                'lookback_period': 15,
                'vol_threshold': 0.25,
                'profit_target': 0.3,
                'stop_loss': 0.5
            }
        }
        
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the ML model based on type"""
        
        if self.model_type == 'xgboost':
            self.model = xgb.XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                eval_metric='logloss'
            )
            
        elif self.model_type == 'lightgbm':
            self.model = lgb.LGBMClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                verbose=-1
            )
            
        elif self.model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                random_state=42
            )
            
        elif self.model_type == 'logistic':
            self.model = LogisticRegression(
                random_state=42,
                max_iter=1000
            )
            
        elif self.model_type == 'svm':
            self.model = SVC(
                kernel='rbf',
                probability=True,
                random_state=42
            )
        
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
    
    def prepare_features(self, market_data: pd.DataFrame, 
                        options_data: Optional[pd.DataFrame] = None,
                        vix_data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Prepare comprehensive features for strategy signal generation
        """
        
        features_df = pd.DataFrame(index=market_data.index)
        
        # Price-based features
        features_df = self._add_price_features(features_df, market_data)
        
        # Volatility features
        features_df = self._add_volatility_features(features_df, market_data, vix_data)
        
        # Technical indicators
        features_df = self._add_technical_features(features_df, market_data)
        
        # Options-specific features
        if options_data is not None:
            features_df = self._add_options_features(features_df, options_data)
        
        # Strategy-specific features
        features_df = self._add_strategy_features(features_df, market_data)
        
        # Market regime features
        features_df = self._add_regime_features(features_df, market_data)
        
        # Clean and fill missing values
        features_df = self._clean_features(features_df)
        
        self.feature_names = features_df.columns.tolist()
        return features_df
    
    def _add_price_features(self, features_df: pd.DataFrame, 
                           market_data: pd.DataFrame) -> pd.DataFrame:
        """Add price-based features"""
        
        # Returns at different horizons
        for period in [1, 3, 5, 10, 20]:
            features_df[f'return_{period}d'] = market_data['Close'].pct_change(period)
        
        # Moving averages
        for window in [5, 10, 20, 50]:
            features_df[f'ma_{window}'] = market_data['Close'].rolling(window).mean()
            features_df[f'price_ma_ratio_{window}'] = market_data['Close'] / features_df[f'ma_{window}']
        
        # Price momentum
        features_df['momentum_10'] = market_data['Close'] / market_data['Close'].shift(10) - 1
        features_df['momentum_20'] = market_data['Close'] / market_data['Close'].shift(20) - 1
        
        # High-Low ratios
        features_df['hl_ratio'] = (market_data['High'] - market_data['Low']) / market_data['Close']
        features_df['close_range'] = (market_data['Close'] - market_data['Low']) / (market_data['High'] - market_data['Low'])
        
        return features_df
    
    def _add_volatility_features(self, features_df: pd.DataFrame,
                               market_data: pd.DataFrame,
                               vix_data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Add volatility-based features"""
        
        # Realized volatility at different windows
        returns = market_data['Close'].pct_change()
        
        for window in [5, 10, 20, 30]:
            features_df[f'realized_vol_{window}'] = returns.rolling(window).std() * np.sqrt(252)
        
        # Volatility of volatility
        vol_20 = returns.rolling(20).std()
        features_df['vol_of_vol'] = vol_20.rolling(20).std()
        
        # VIX features
        if vix_data is not None:
            features_df['vix_level'] = vix_data['Close']
            features_df['vix_change'] = vix_data['Close'].pct_change()
            features_df['vix_ma_10'] = vix_data['Close'].rolling(10).mean()
            features_df['vix_percentile'] = vix_data['Close'].rolling(252).apply(
                lambda x: (x.iloc[-1] - x.min()) / (x.max() - x.min()) if x.max() != x.min() else 0.5
            )
            
            # VIX term structure (if available)
            features_df['vix_term_structure'] = features_df['vix_level'] - features_df['vix_ma_10']
        
        # Parkinson estimator (using high-low)
        features_df['parkinson_vol'] = np.sqrt(
            0.361 * (np.log(market_data['High'] / market_data['Low']) ** 2).rolling(20).mean() * 252
        )
        
        return features_df
    
    def _add_technical_features(self, features_df: pd.DataFrame,
                              market_data: pd.DataFrame) -> pd.DataFrame:
        """Add technical analysis features"""
        
        close = market_data['Close']
        high = market_data['High']
        low = market_data['Low']
        volume = market_data['Volume']
        
        # RSI
        features_df['rsi_14'] = ta.momentum.RSIIndicator(close, window=14).rsi()
        features_df['rsi_30'] = ta.momentum.RSIIndicator(close, window=30).rsi()
        
        # MACD
        macd = ta.trend.MACD(close)
        features_df['macd'] = macd.macd()
        features_df['macd_signal'] = macd.macd_signal()
        features_df['macd_histogram'] = macd.macd_diff()
        
        # Bollinger Bands
        bb = ta.volatility.BollingerBands(close, window=20, window_dev=2)
        features_df['bb_upper'] = bb.bollinger_hband()
        features_df['bb_lower'] = bb.bollinger_lband()
        features_df['bb_position'] = (close - bb.bollinger_lband()) / (bb.bollinger_hband() - bb.bollinger_lband())
        features_df['bb_width'] = (bb.bollinger_hband() - bb.bollinger_lband()) / bb.bollinger_mavg()
        
        # Stochastic oscillator
        stoch = ta.momentum.StochasticOscillator(high, low, close)
        features_df['stoch_k'] = stoch.stoch()
        features_df['stoch_d'] = stoch.stoch_signal()
        
        # Williams %R
        features_df['williams_r'] = ta.momentum.WilliamsRIndicator(high, low, close).williams_r()
        
        # ADX (trend strength)
        features_df['adx'] = ta.trend.ADXIndicator(high, low, close).adx()
        
        # Volume indicators
        features_df['volume_sma'] = volume.rolling(20).mean()
        features_df['volume_ratio'] = volume / features_df['volume_sma']
        
        # On Balance Volume
        features_df['obv'] = ta.volume.OnBalanceVolumeIndicator(close, volume).on_balance_volume()
        
        return features_df
    
    def _add_options_features(self, features_df: pd.DataFrame,
                            options_data: pd.DataFrame) -> pd.DataFrame:
        """Add options market features"""
        
        # Group options data by date
        daily_options = options_data.groupby('date').agg({
            'volume': 'sum',
            'open_interest': 'sum',
            'iv': 'mean'
        }).reindex(features_df.index, method='ffill')
        
        # Put/Call ratio
        calls = options_data[options_data['option_type'] == 'C'].groupby('date')['volume'].sum()
        puts = options_data[options_data['option_type'] == 'P'].groupby('date')['volume'].sum()
        
        pc_ratio = puts / calls
        features_df['put_call_ratio'] = pc_ratio.reindex(features_df.index, method='ffill')
        features_df['put_call_ma'] = features_df['put_call_ratio'].rolling(10).mean()
        
        # Options volume indicators
        features_df['options_volume'] = daily_options['volume']
        features_df['options_oi'] = daily_options['open_interest']
        features_df['vol_oi_ratio'] = features_df['options_volume'] / features_df['options_oi']
        
        # Average IV
        features_df['avg_iv'] = daily_options['iv']
        features_df['iv_rank'] = features_df['avg_iv'].rolling(252).apply(
            lambda x: (x.iloc[-1] - x.min()) / (x.max() - x.min()) if x.max() != x.min() else 0.5
        )
        
        return features_df
    
    def _add_strategy_features(self, features_df: pd.DataFrame,
                             market_data: pd.DataFrame) -> pd.DataFrame:
        """Add strategy-specific features"""
        
        if self.strategy_type == 'straddle':
            # Features for straddle strategies (volatility plays)
            features_df = self._add_straddle_features(features_df, market_data)
            
        elif self.strategy_type == 'covered_call':
            # Features for covered call strategies
            features_df = self._add_covered_call_features(features_df, market_data)
            
        elif self.strategy_type == 'iron_condor':
            # Features for iron condor strategies
            features_df = self._add_iron_condor_features(features_df, market_data)
        
        return features_df
    
    def _add_straddle_features(self, features_df: pd.DataFrame,
                             market_data: pd.DataFrame) -> pd.DataFrame:
        """Add features specific to straddle strategies"""
        
        # Earnings announcements proximity (simplified)
        # In real implementation, would use earnings calendar
        features_df['days_to_earnings'] = 30  # Placeholder
        
        # Volatility expansion indicators
        vol_short = market_data['Close'].pct_change().rolling(5).std()
        vol_long = market_data['Close'].pct_change().rolling(20).std()
        features_df['vol_expansion'] = vol_short / vol_long
        
        # Range compression/expansion
        atr_5 = ta.volatility.AverageTrueRange(market_data['High'], market_data['Low'], market_data['Close'], window=5).average_true_range()
        atr_20 = ta.volatility.AverageTrueRange(market_data['High'], market_data['Low'], market_data['Close'], window=20).average_true_range()
        features_df['atr_ratio'] = atr_5 / atr_20
        
        # Breakout probability
        bb_width = features_df.get('bb_width', 0)
        features_df['breakout_probability'] = 1 / (1 + np.exp(-5 * (bb_width - 0.1)))
        
        return features_df
    
    def _add_covered_call_features(self, features_df: pd.DataFrame,
                                 market_data: pd.DataFrame) -> pd.DataFrame:
        """Add features specific to covered call strategies"""
        
        # Trend strength (covered calls work best in sideways/slightly bullish markets)
        features_df['trend_strength'] = features_df.get('adx', 25)
        
        # Support/resistance levels
        rolling_max = market_data['High'].rolling(20).max()
        rolling_min = market_data['Low'].rolling(20).min()
        features_df['resistance_distance'] = (rolling_max - market_data['Close']) / market_data['Close']
        features_df['support_distance'] = (market_data['Close'] - rolling_min) / market_data['Close']
        
        # Dividend yield proximity (for dividend-paying stocks)
        # Placeholder - would need actual dividend data
        features_df['dividend_yield'] = 0.02
        
        return features_df
    
    def _add_iron_condor_features(self, features_df: pd.DataFrame,
                                market_data: pd.DataFrame) -> pd.DataFrame:
        """Add features specific to iron condor strategies"""
        
        # Range-bound market indicators
        features_df['range_bound_score'] = 1 - features_df.get('trend_strength', 25) / 100
        
        # Volatility rank (iron condors profit from high IV that decreases)
        if 'iv_rank' in features_df.columns:
            features_df['iv_crush_probability'] = features_df['iv_rank']
        
        # Market stability
        close_changes = market_data['Close'].pct_change()
        features_df['stability_score'] = 1 / (1 + close_changes.rolling(10).std())
        
        return features_df
    
    def _add_regime_features(self, features_df: pd.DataFrame,
                           market_data: pd.DataFrame) -> pd.DataFrame:
        """Add market regime features"""
        
        # Bull/bear market indicator
        ma_50 = market_data['Close'].rolling(50).mean()
        ma_200 = market_data['Close'].rolling(200).mean()
        features_df['bull_market'] = (ma_50 > ma_200).astype(int)
        
        # Market stress indicator
        returns = market_data['Close'].pct_change()
        features_df['stress_indicator'] = (returns < -0.02).rolling(5).sum()
        
        # Volatility regime
        vol_20 = returns.rolling(20).std() * np.sqrt(252)
        vol_percentile = vol_20.rolling(252).apply(
            lambda x: (x.iloc[-1] - x.min()) / (x.max() - x.min()) if x.max() != x.min() else 0.5
        )
        features_df['high_vol_regime'] = (vol_percentile > 0.8).astype(int)
        
        return features_df
    
    def _clean_features(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Clean and prepare features for modeling"""
        
        # Forward fill missing values
        features_df = features_df.fillna(method='ffill')
        
        # Fill remaining NaN with median values
        for col in features_df.columns:
            if features_df[col].isna().any():
                features_df[col] = features_df[col].fillna(features_df[col].median())
        
        # Remove infinite values
        features_df = features_df.replace([np.inf, -np.inf], np.nan).fillna(0)
        
        # Remove highly correlated features
        correlation_matrix = features_df.corr().abs()
        upper_triangle = correlation_matrix.where(
            np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool)
        )
        
        to_drop = [column for column in upper_triangle.columns if any(upper_triangle[column] > 0.95)]
        features_df = features_df.drop(columns=to_drop)
        
        return features_df
    
    def create_labels(self, market_data: pd.DataFrame, 
                     options_data: Optional[pd.DataFrame] = None,
                     forward_days: int = 5) -> pd.Series:
        """
        Create trading labels based on future profitability
        """
        
        if self.strategy_type == 'straddle':
            return self._create_straddle_labels(market_data, forward_days)
        elif self.strategy_type == 'covered_call':
            return self._create_covered_call_labels(market_data, forward_days)
        elif self.strategy_type == 'iron_condor':
            return self._create_iron_condor_labels(market_data, forward_days)
        else:
            # Default: simple momentum-based labels
            future_return = market_data['Close'].pct_change(forward_days).shift(-forward_days)
            return (future_return > 0.02).astype(int)
    
    def _create_straddle_labels(self, market_data: pd.DataFrame, 
                              forward_days: int = 5) -> pd.Series:
        """Create labels for straddle strategy"""
        
        # Straddle profits from large moves in either direction
        future_return = market_data['Close'].pct_change(forward_days).shift(-forward_days).abs()
        
        # Profitable if absolute return > threshold
        threshold = 0.03  # 3% move required for profitability
        return (future_return > threshold).astype(int)
    
    def _create_covered_call_labels(self, market_data: pd.DataFrame,
                                  forward_days: int = 20) -> pd.Series:
        """Create labels for covered call strategy"""
        
        # Covered call profits from sideways to slightly bullish moves
        future_return = market_data['Close'].pct_change(forward_days).shift(-forward_days)
        
        # Profitable if return is between 0% and 5%
        return ((future_return > 0) & (future_return < 0.05)).astype(int)
    
    def _create_iron_condor_labels(self, market_data: pd.DataFrame,
                                 forward_days: int = 15) -> pd.Series:
        """Create labels for iron condor strategy"""
        
        # Iron condor profits from low volatility and range-bound movement
        future_return = market_data['Close'].pct_change(forward_days).shift(-forward_days).abs()
        
        # Profitable if absolute return < threshold
        threshold = 0.02  # 2% or less movement for profitability
        return (future_return < threshold).astype(int)
    
    def train(self, market_data: pd.DataFrame,
              options_data: Optional[pd.DataFrame] = None,
              vix_data: Optional[pd.DataFrame] = None,
              validation_split: float = 0.2) -> Dict[str, float]:
        """
        Train the strategy signal model
        """
        
        # Prepare features and labels
        features = self.prepare_features(market_data, options_data, vix_data)
        labels = self.create_labels(market_data, options_data)
        
        # Align features and labels
        common_index = features.index.intersection(labels.index)
        X = features.loc[common_index]
        y = labels.loc[common_index]
        
        # Remove rows with missing labels
        valid_mask = ~y.isna()
        X = X[valid_mask]
        y = y[valid_mask]
        
        # Time series split to avoid look-ahead bias
        tscv = TimeSeriesSplit(n_splits=3)
        cv_scores = []
        
        for train_idx, val_idx in tscv.split(X):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            # Scale features for certain models
            if self.model_type in ['logistic', 'svm']:
                X_train_scaled = self.scaler.fit_transform(X_train)
                X_val_scaled = self.scaler.transform(X_val)
            else:
                X_train_scaled = X_train
                X_val_scaled = X_val
            
            # Train model
            temp_model = self._initialize_model()
            temp_model.fit(X_train_scaled, y_train)
            
            # Predict and score
            if hasattr(temp_model, 'predict_proba'):
                y_pred_proba = temp_model.predict_proba(X_val_scaled)[:, 1]
                score = roc_auc_score(y_val, y_pred_proba)
            else:
                y_pred = temp_model.predict(X_val_scaled)
                score = (y_pred == y_val).mean()
            
            cv_scores.append(score)
        
        # Train final model on all data
        if self.model_type in ['logistic', 'svm']:
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = X
        
        self.model.fit(X_scaled, y)
        
        # Calculate final metrics
        if hasattr(self.model, 'predict_proba'):
            y_pred_proba = self.model.predict_proba(X_scaled)[:, 1]
            auc_score = roc_auc_score(y, y_pred_proba)
        else:
            y_pred = self.model.predict(X_scaled)
            auc_score = (y_pred == y).mean()
        
        metrics = {
            'cv_score_mean': np.mean(cv_scores),
            'cv_score_std': np.std(cv_scores),
            'final_auc': auc_score,
            'label_distribution': y.value_counts().to_dict()
        }
        
        self.logger.info(f"Model training completed - CV Score: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")
        
        return metrics
    
    def generate_signal(self, market_data: pd.DataFrame,
                       options_data: Optional[pd.DataFrame] = None,
                       vix_data: Optional[pd.DataFrame] = None,
                       timestamp: str = None) -> StrategySignal:
        """
        Generate trading signal for current market conditions
        """
        
        if self.model is None:
            raise ValueError("Model must be trained before generating signals")
        
        # Prepare features for the latest data point
        features = self.prepare_features(market_data, options_data, vix_data)
        latest_features = features.iloc[-1:].values
        
        # Scale if needed
        if self.model_type in ['logistic', 'svm']:
            latest_features = self.scaler.transform(latest_features)
        
        # Get prediction and confidence
        if hasattr(self.model, 'predict_proba'):
            probabilities = self.model.predict_proba(latest_features)[0]
            signal = 1 if probabilities[1] > 0.5 else 0
            confidence = max(probabilities)
        else:
            signal = self.model.predict(latest_features)[0]
            confidence = 0.7  # Default confidence for non-probabilistic models
        
        # Convert binary to trade signal
        trade_signal = 1 if signal == 1 else 0  # 1: Enter, 0: Stay Out
        
        # Feature importance for interpretation
        feature_dict = dict(zip(self.feature_names, features.iloc[-1].values))
        
        return StrategySignal(
            signal=trade_signal,
            confidence=confidence,
            strategy_type=self.strategy_type,
            timestamp=timestamp or datetime.now().isoformat(),
            features_used=feature_dict
        )
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance from the trained model"""
        
        if self.model is None:
            return {}
        
        if hasattr(self.model, 'feature_importances_'):
            importance_dict = dict(zip(self.feature_names, self.model.feature_importances_))
            return dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
        elif hasattr(self.model, 'coef_'):
            # For linear models, use absolute coefficients
            importance_dict = dict(zip(self.feature_names, np.abs(self.model.coef_[0])))
            return dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
        else:
            return {}
    
    def save_model(self, filepath: str):
        """Save the trained model"""
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'model_type': self.model_type,
            'strategy_type': self.strategy_type,
            'strategy_configs': self.strategy_configs
        }
        
        joblib.dump(model_data, filepath)
    
    def load_model(self, filepath: str):
        """Load a trained model"""
        
        model_data = joblib.load(filepath)
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.feature_names = model_data['feature_names']
        self.model_type = model_data['model_type']
        self.strategy_type = model_data['strategy_type']
        self.strategy_configs = model_data.get('strategy_configs', {})

# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Create sample market data
    dates = pd.date_range('2020-01-01', '2024-01-01', freq='D')
    np.random.seed(42)
    
    # Generate synthetic market data
    returns = np.random.normal(0.0005, 0.02, len(dates))
    prices = 100 * np.exp(np.cumsum(returns))
    
    market_data = pd.DataFrame({
        'Close': prices,
        'High': prices * (1 + np.abs(np.random.normal(0, 0.01, len(dates)))),
        'Low': prices * (1 - np.abs(np.random.normal(0, 0.01, len(dates)))),
        'Volume': np.random.randint(1000000, 10000000, len(dates))
    }, index=dates)
    
    # Test signal generator
    signal_gen = StrategySignalGenerator(strategy_type='straddle', model_type='xgboost')
    metrics = signal_gen.train(market_data)
    
    print("Training metrics:", metrics)
    print("Feature importance:", list(signal_gen.get_feature_importance().items())[:10])
    
    # Generate a signal
    signal = signal_gen.generate_signal(market_data)
    print(f"Signal: {signal.signal}, Confidence: {signal.confidence:.3f}")