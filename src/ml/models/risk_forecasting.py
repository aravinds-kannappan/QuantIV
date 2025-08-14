"""
ML-Based Risk Forecasting
Uses time-series models to forecast volatility, drawdown risk, and tail events
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from arch import arch_model
from typing import Dict, List, Tuple, Optional, Any, Union
import warnings
warnings.filterwarnings('ignore')
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
import joblib

@dataclass
class RiskForecast:
    """Risk forecast output"""
    forecast_date: str
    horizon_days: int
    volatility_forecast: float
    volatility_confidence_interval: Tuple[float, float]
    var_forecast: float  # Value at Risk
    es_forecast: float   # Expected Shortfall
    max_drawdown_prob: float
    tail_risk_indicator: float
    regime_probability: Dict[str, float]

class LSTMVolatilityModel(nn.Module):
    """PyTorch LSTM model for volatility forecasting"""
    
    def __init__(self, input_size: int, hidden_size: int = 64, num_layers: int = 2, dropout: float = 0.2):
        super(LSTMVolatilityModel, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True
        )
        
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        # Initialize hidden state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # LSTM forward pass
        lstm_out, _ = self.lstm(x, (h0, c0))
        
        # Take the last output
        last_output = lstm_out[:, -1, :]
        
        # Apply dropout and linear layer
        output = self.dropout(last_output)
        output = self.linear(output)
        
        return output

class RiskForecaster:
    """
    ML-based risk forecasting system using multiple time-series models
    """
    
    def __init__(self, model_type: str = 'lstm', forecast_horizon: int = 5):
        self.model_type = model_type
        self.forecast_horizon = forecast_horizon
        self.model = None
        self.scaler = MinMaxScaler()
        self.feature_scaler = StandardScaler()
        self.sequence_length = 20  # Look-back window
        self.logger = logging.getLogger(__name__)
        
        # Model performance tracking
        self.training_history = {}
        self.forecast_errors = []
        
    def prepare_sequences(self, data: pd.DataFrame, target_col: str = 'realized_vol',
                         feature_cols: List[str] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare time series sequences for training
        """
        
        if feature_cols is None:
            feature_cols = ['returns', 'volume', 'vix', 'high_low_ratio']
        
        # Ensure we have the required columns
        available_cols = [col for col in feature_cols if col in data.columns]
        if not available_cols:
            # Fallback to basic features
            available_cols = [target_col]
        
        # Prepare target variable
        target_data = data[target_col].values
        target_scaled = self.scaler.fit_transform(target_data.reshape(-1, 1)).flatten()
        
        # Prepare features
        if len(available_cols) > 1:
            feature_data = data[available_cols].values
            feature_scaled = self.feature_scaler.fit_transform(feature_data)
        else:
            feature_scaled = target_scaled.reshape(-1, 1)
        
        # Create sequences
        X, y = [], []
        
        for i in range(self.sequence_length, len(target_scaled) - self.forecast_horizon + 1):
            # Input sequence
            X.append(feature_scaled[i-self.sequence_length:i])
            
            # Target (future volatility)
            if self.forecast_horizon == 1:
                y.append(target_scaled[i])
            else:
                # Multi-step ahead prediction
                y.append(target_scaled[i:i+self.forecast_horizon])
        
        return np.array(X), np.array(y)
    
    def create_tensorflow_model(self, input_shape: Tuple[int, int]) -> keras.Model:
        """
        Create TensorFlow LSTM model for volatility forecasting
        """
        
        model = keras.Sequential([
            layers.LSTM(64, return_sequences=True, input_shape=input_shape),
            layers.Dropout(0.2),
            layers.LSTM(32, return_sequences=False),
            layers.Dropout(0.2),
            layers.Dense(16, activation='relu'),
            layers.Dense(self.forecast_horizon if self.forecast_horizon > 1 else 1)
        ])
        
        model.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def create_temporal_cnn_model(self, input_shape: Tuple[int, int]) -> keras.Model:
        """
        Create Temporal CNN model for volatility forecasting
        """
        
        model = keras.Sequential([
            layers.Conv1D(64, 3, activation='relu', input_shape=input_shape),
            layers.Conv1D(32, 3, activation='relu'),
            layers.MaxPooling1D(2),
            layers.Conv1D(16, 3, activation='relu'),
            layers.GlobalMaxPooling1D(),
            layers.Dense(50, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(self.forecast_horizon if self.forecast_horizon > 1 else 1)
        ])
        
        model.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def train_garch_model(self, returns: pd.Series) -> Any:
        """
        Train GARCH model for volatility forecasting
        """
        
        try:
            # Remove any infinite or NaN values
            clean_returns = returns.dropna().replace([np.inf, -np.inf], np.nan).dropna()
            clean_returns = clean_returns * 100  # Convert to percentage
            
            # Fit GARCH(1,1) model
            model = arch_model(clean_returns, vol='Garch', p=1, q=1, dist='normal')
            fitted_model = model.fit(disp='off')
            
            return fitted_model
            
        except Exception as e:
            self.logger.warning(f"GARCH model training failed: {e}")
            return None
    
    def train(self, data: pd.DataFrame, validation_split: float = 0.2) -> Dict[str, float]:
        """
        Train the risk forecasting model
        """
        
        # Calculate realized volatility if not present
        if 'realized_vol' not in data.columns:
            returns = data['Close'].pct_change().dropna()
            data['realized_vol'] = returns.rolling(window=20).std() * np.sqrt(252)
        
        # Add additional features
        data = self._add_risk_features(data)
        
        # Prepare training data
        feature_cols = ['returns', 'volume_ratio', 'vix_level', 'high_low_ratio', 'momentum']
        available_features = [col for col in feature_cols if col in data.columns]
        
        X, y = self.prepare_sequences(data, 'realized_vol', available_features)
        
        if len(X) == 0:
            raise ValueError("No valid sequences could be created from the data")
        
        # Train-validation split
        split_idx = int(len(X) * (1 - validation_split))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        metrics = {}
        
        if self.model_type == 'lstm_tf':
            # TensorFlow LSTM
            self.model = self.create_tensorflow_model((X.shape[1], X.shape[2]))
            
            callbacks = [
                keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
                keras.callbacks.ReduceLROnPlateau(patience=5, factor=0.5)
            ]
            
            history = self.model.fit(
                X_train, y_train,
                epochs=100,
                batch_size=32,
                validation_data=(X_val, y_val),
                callbacks=callbacks,
                verbose=0
            )
            
            self.training_history = history.history
            
            # Calculate metrics
            y_pred = self.model.predict(X_val, verbose=0)
            mse = mean_squared_error(y_val, y_pred)
            mae = mean_absolute_error(y_val, y_pred)
            
            metrics = {'mse': mse, 'mae': mae, 'final_val_loss': history.history['val_loss'][-1]}
            
        elif self.model_type == 'temporal_cnn':
            # Temporal CNN
            self.model = self.create_temporal_cnn_model((X.shape[1], X.shape[2]))
            
            history = self.model.fit(
                X_train, y_train,
                epochs=50,
                batch_size=32,
                validation_data=(X_val, y_val),
                verbose=0
            )
            
            y_pred = self.model.predict(X_val, verbose=0)
            mse = mean_squared_error(y_val, y_pred)
            mae = mean_absolute_error(y_val, y_pred)
            
            metrics = {'mse': mse, 'mae': mae}
            
        elif self.model_type == 'garch':
            # GARCH model
            returns = data['Close'].pct_change().dropna()
            self.model = self.train_garch_model(returns)
            
            if self.model is not None:
                # Evaluate on validation set
                forecast = self.model.forecast(horizon=self.forecast_horizon)
                metrics = {'log_likelihood': self.model.loglikelihood}
            else:
                metrics = {'error': 'GARCH training failed'}
                
        elif self.model_type == 'ensemble':
            # Ensemble of multiple models
            self.models = {}
            
            # Train LSTM
            lstm_model = self.create_tensorflow_model((X.shape[1], X.shape[2]))
            lstm_model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0)
            self.models['lstm'] = lstm_model
            
            # Train Random Forest
            X_flat = X_train.reshape(X_train.shape[0], -1)
            X_val_flat = X_val.reshape(X_val.shape[0], -1)
            
            rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
            rf_model.fit(X_flat, y_train.flatten())
            self.models['rf'] = rf_model
            
            # Ensemble predictions
            lstm_pred = lstm_model.predict(X_val, verbose=0).flatten()
            rf_pred = rf_model.predict(X_val_flat)
            
            ensemble_pred = 0.7 * lstm_pred + 0.3 * rf_pred
            mse = mean_squared_error(y_val.flatten(), ensemble_pred)
            mae = mean_absolute_error(y_val.flatten(), ensemble_pred)
            
            metrics = {'mse': mse, 'mae': mae, 'ensemble_weights': [0.7, 0.3]}
        
        self.logger.info(f"Risk forecasting model training completed: {metrics}")
        return metrics
    
    def _add_risk_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Add risk-related features to the dataset
        """
        
        # Returns
        data['returns'] = data['Close'].pct_change()
        
        # Volume ratio
        data['volume_ratio'] = data['Volume'] / data['Volume'].rolling(20).mean()
        
        # High-low ratio
        data['high_low_ratio'] = (data['High'] - data['Low']) / data['Close']
        
        # Momentum
        data['momentum'] = data['Close'] / data['Close'].shift(10) - 1
        
        # VIX level (if available)
        if 'VIX' in data.columns:
            data['vix_level'] = data['VIX']
        else:
            # Estimate VIX from realized volatility
            data['vix_level'] = data['returns'].rolling(20).std() * np.sqrt(252) * 100
        
        # Regime indicators
        data['high_vol_regime'] = (data['vix_level'] > data['vix_level'].rolling(252).quantile(0.75)).astype(int)
        
        # Jump indicators
        data['jump_indicator'] = (np.abs(data['returns']) > 2 * data['returns'].rolling(20).std()).astype(int)
        
        return data.fillna(method='ffill').fillna(0)
    
    def forecast_risk(self, data: pd.DataFrame, horizon_days: int = None) -> RiskForecast:
        """
        Generate comprehensive risk forecast
        """
        
        if self.model is None:
            raise ValueError("Model must be trained before forecasting")
        
        horizon = horizon_days or self.forecast_horizon
        
        # Prepare features for latest data
        data_with_features = self._add_risk_features(data)
        
        # Get volatility forecast
        vol_forecast, vol_ci = self._forecast_volatility(data_with_features, horizon)
        
        # Calculate VaR and Expected Shortfall
        returns = data['Close'].pct_change().dropna()
        var_forecast = self._calculate_var(returns, vol_forecast)
        es_forecast = self._calculate_expected_shortfall(returns, vol_forecast)
        
        # Estimate drawdown probability
        drawdown_prob = self._estimate_drawdown_probability(data_with_features, horizon)
        
        # Tail risk indicator
        tail_risk = self._calculate_tail_risk_indicator(data_with_features)
        
        # Market regime probabilities
        regime_probs = self._estimate_regime_probabilities(data_with_features)
        
        forecast_date = data.index[-1].strftime('%Y-%m-%d') if hasattr(data.index[-1], 'strftime') else str(data.index[-1])
        
        return RiskForecast(
            forecast_date=forecast_date,
            horizon_days=horizon,
            volatility_forecast=vol_forecast,
            volatility_confidence_interval=vol_ci,
            var_forecast=var_forecast,
            es_forecast=es_forecast,
            max_drawdown_prob=drawdown_prob,
            tail_risk_indicator=tail_risk,
            regime_probability=regime_probs
        )
    
    def _forecast_volatility(self, data: pd.DataFrame, horizon: int) -> Tuple[float, Tuple[float, float]]:
        """
        Forecast volatility with confidence intervals
        """
        
        if self.model_type == 'garch' and self.model is not None:
            # GARCH forecast
            forecast = self.model.forecast(horizon=horizon)
            vol_forecast = np.sqrt(forecast.variance.iloc[-1, 0]) * np.sqrt(252) / 100
            
            # Simple confidence interval estimation
            vol_std = vol_forecast * 0.2  # Assume 20% uncertainty
            vol_ci = (vol_forecast - 1.96 * vol_std, vol_forecast + 1.96 * vol_std)
            
            return vol_forecast, vol_ci
            
        else:
            # Neural network forecast
            feature_cols = ['returns', 'volume_ratio', 'vix_level', 'high_low_ratio', 'momentum']
            available_features = [col for col in feature_cols if col in data.columns]
            
            # Get the latest sequence
            if len(available_features) > 1:
                latest_features = data[available_features].iloc[-self.sequence_length:].values
                latest_features = self.feature_scaler.transform(latest_features)
            else:
                latest_vol = data['realized_vol'].iloc[-self.sequence_length:].values
                latest_features = self.scaler.transform(latest_vol.reshape(-1, 1))
            
            # Reshape for model input
            X_latest = latest_features.reshape(1, self.sequence_length, -1)
            
            # Make prediction
            if self.model_type == 'ensemble':
                # Ensemble prediction
                lstm_pred = self.models['lstm'].predict(X_latest, verbose=0)[0]
                
                X_flat = X_latest.reshape(1, -1)
                rf_pred = self.models['rf'].predict(X_flat)[0]
                
                vol_pred_scaled = 0.7 * lstm_pred + 0.3 * rf_pred
            else:
                vol_pred_scaled = self.model.predict(X_latest, verbose=0)[0]
            
            # Inverse transform
            if self.forecast_horizon == 1:
                vol_forecast = self.scaler.inverse_transform([[vol_pred_scaled]])[0, 0]
            else:
                vol_forecast = np.mean(self.scaler.inverse_transform(vol_pred_scaled.reshape(-1, 1)))
            
            # Confidence interval estimation
            vol_std = vol_forecast * 0.15  # Assume 15% uncertainty for ML models
            vol_ci = (vol_forecast - 1.96 * vol_std, vol_forecast + 1.96 * vol_std)
            
            return max(vol_forecast, 0.01), vol_ci
    
    def _calculate_var(self, returns: pd.Series, vol_forecast: float, confidence: float = 0.05) -> float:
        """
        Calculate Value at Risk using parametric method
        """
        
        # Assume normal distribution for simplicity
        from scipy.stats import norm
        
        z_score = norm.ppf(confidence)
        var = z_score * vol_forecast / np.sqrt(252)  # Daily VaR
        
        return var
    
    def _calculate_expected_shortfall(self, returns: pd.Series, vol_forecast: float, confidence: float = 0.05) -> float:
        """
        Calculate Expected Shortfall (Conditional VaR)
        """
        
        from scipy.stats import norm
        
        z_score = norm.ppf(confidence)
        phi_z = norm.pdf(z_score)
        
        es = -(phi_z / confidence) * vol_forecast / np.sqrt(252)
        
        return es
    
    def _estimate_drawdown_probability(self, data: pd.DataFrame, horizon: int) -> float:
        """
        Estimate probability of significant drawdown
        """
        
        # Historical drawdown analysis
        returns = data['Close'].pct_change().dropna()
        cumulative_returns = (1 + returns).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdowns = (cumulative_returns - running_max) / running_max
        
        # Probability of >10% drawdown in the forecast horizon
        significant_drawdowns = (drawdowns < -0.1).rolling(horizon).max()
        drawdown_prob = significant_drawdowns.mean()
        
        return min(max(drawdown_prob, 0.01), 0.99)  # Bound between 1% and 99%
    
    def _calculate_tail_risk_indicator(self, data: pd.DataFrame) -> float:
        """
        Calculate composite tail risk indicator
        """
        
        # Combine multiple tail risk measures
        returns = data['returns'].dropna()
        
        # Skewness (negative skewness = left tail risk)
        skewness = returns.rolling(60).skew().iloc[-1] if len(returns) > 60 else 0
        tail_risk_skew = max(-skewness, 0) / 2  # Normalize
        
        # Kurtosis (high kurtosis = fat tails)
        kurtosis = returns.rolling(60).kurt().iloc[-1] if len(returns) > 60 else 3
        tail_risk_kurt = max(kurtosis - 3, 0) / 10  # Excess kurtosis normalized
        
        # VIX level percentile
        vix_percentile = 0.5  # Default
        if 'vix_level' in data.columns:
            vix_current = data['vix_level'].iloc[-1]
            vix_percentile = (data['vix_level'].rank(pct=True)).iloc[-1]
        
        # Composite indicator
        tail_risk = 0.4 * tail_risk_skew + 0.3 * tail_risk_kurt + 0.3 * vix_percentile
        
        return min(max(tail_risk, 0), 1)  # Bound between 0 and 1
    
    def _estimate_regime_probabilities(self, data: pd.DataFrame) -> Dict[str, float]:
        """
        Estimate market regime probabilities
        """
        
        # Simple regime classification based on recent performance
        returns = data['returns'].dropna()
        vol = data.get('realized_vol', data['returns'].rolling(20).std() * np.sqrt(252))
        
        # Recent statistics
        recent_return = returns.tail(20).mean() * 252  # Annualized
        recent_vol = vol.iloc[-1] if len(vol) > 0 else 0.2
        
        # Regime probabilities
        bull_prob = max(min((recent_return + 0.1) / 0.3, 1), 0)  # Bull if return > -10%
        bear_prob = max(min((-recent_return + 0.1) / 0.3, 1), 0)  # Bear if return < -10%
        high_vol_prob = max(min((recent_vol - 0.15) / 0.2, 1), 0)  # High vol if > 15%
        
        # Normalize
        total = bull_prob + bear_prob + high_vol_prob
        if total > 0:
            bull_prob /= total
            bear_prob /= total
            high_vol_prob /= total
        
        sideways_prob = max(1 - bull_prob - bear_prob - high_vol_prob, 0)
        
        return {
            'bull_market': bull_prob,
            'bear_market': bear_prob,
            'high_volatility': high_vol_prob,
            'sideways': sideways_prob
        }
    
    def save_model(self, filepath: str):
        """Save the trained model"""
        
        if self.model_type == 'ensemble':
            # Save ensemble models separately
            for name, model in self.models.items():
                if name == 'lstm':
                    model.save(f"{filepath}_{name}.h5")
                else:
                    joblib.dump(model, f"{filepath}_{name}.pkl")
        elif self.model_type in ['lstm_tf', 'temporal_cnn']:
            self.model.save(f"{filepath}.h5")
        else:
            joblib.dump(self.model, f"{filepath}.pkl")
        
        # Save scalers and metadata
        metadata = {
            'scaler': self.scaler,
            'feature_scaler': self.feature_scaler,
            'model_type': self.model_type,
            'forecast_horizon': self.forecast_horizon,
            'sequence_length': self.sequence_length
        }
        joblib.dump(metadata, f"{filepath}_metadata.pkl")
    
    def load_model(self, filepath: str):
        """Load a trained model"""
        
        # Load metadata
        metadata = joblib.load(f"{filepath}_metadata.pkl")
        self.scaler = metadata['scaler']
        self.feature_scaler = metadata['feature_scaler']
        self.model_type = metadata['model_type']
        self.forecast_horizon = metadata['forecast_horizon']
        self.sequence_length = metadata['sequence_length']
        
        # Load model
        if self.model_type == 'ensemble':
            self.models = {}
            self.models['lstm'] = keras.models.load_model(f"{filepath}_lstm.h5")
            self.models['rf'] = joblib.load(f"{filepath}_rf.pkl")
        elif self.model_type in ['lstm_tf', 'temporal_cnn']:
            self.model = keras.models.load_model(f"{filepath}.h5")
        else:
            self.model = joblib.load(f"{filepath}.pkl")

# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Create sample data
    dates = pd.date_range('2020-01-01', '2024-01-01', freq='D')
    np.random.seed(42)
    
    # Generate synthetic price data with volatility clustering
    returns = np.random.normal(0.0005, 0.02, len(dates))
    vol_process = np.random.normal(0.02, 0.005, len(dates))
    
    prices = 100 * np.exp(np.cumsum(returns))
    volumes = np.random.randint(1000000, 10000000, len(dates))
    
    market_data = pd.DataFrame({
        'Close': prices,
        'High': prices * 1.02,
        'Low': prices * 0.98,
        'Volume': volumes
    }, index=dates)
    
    # Test risk forecaster
    forecaster = RiskForecaster(model_type='lstm_tf', forecast_horizon=5)
    metrics = forecaster.train(market_data)
    
    print("Training metrics:", metrics)
    
    # Generate risk forecast
    risk_forecast = forecaster.forecast_risk(market_data)
    print(f"Volatility forecast: {risk_forecast.volatility_forecast:.4f}")
    print(f"VaR forecast: {risk_forecast.var_forecast:.4f}")
    print(f"Max drawdown probability: {risk_forecast.max_drawdown_prob:.4f}")
    print(f"Regime probabilities: {risk_forecast.regime_probability}")