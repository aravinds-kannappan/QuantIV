"""
ML-Enhanced Volatility Surface Modeling
Uses machine learning to model and predict implied volatility surfaces
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
import xgboost as xgb
import lightgbm as lgb
import tensorflow as tf
from tensorflow import keras
from typing import Dict, List, Tuple, Optional, Any
import joblib
import logging
from dataclasses import dataclass
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

@dataclass
class VolSurfaceFeatures:
    """Features for volatility surface modeling"""
    moneyness: float  # S/K
    time_to_expiry: float  # Years
    log_moneyness: float  # log(S/K)
    delta: Optional[float] = None
    historical_vol: Optional[float] = None
    vix_level: Optional[float] = None
    term_structure_slope: Optional[float] = None
    skew: Optional[float] = None
    underlying_return: Optional[float] = None
    volume_oi_ratio: Optional[float] = None

class VolatilitySurfaceML:
    """
    Machine Learning models for volatility surface prediction and smoothing
    """
    
    def __init__(self, model_type: str = 'xgboost'):
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = []
        self.logger = logging.getLogger(__name__)
        
        # Initialize model based on type
        self._initialize_model()
        
    def _initialize_model(self):
        """Initialize the ML model based on type"""
        
        if self.model_type == 'xgboost':
            self.model = xgb.XGBRegressor(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42
            )
            
        elif self.model_type == 'lightgbm':
            self.model = lgb.LGBMRegressor(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                verbose=-1
            )
            
        elif self.model_type == 'random_forest':
            self.model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42
            )
            
        elif self.model_type == 'gradient_boost':
            self.model = GradientBoostingRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                random_state=42
            )
            
        elif self.model_type == 'neural_network':
            self.model = self._create_neural_network()
            
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
    
    def _create_neural_network(self) -> keras.Model:
        """Create a neural network for volatility surface modeling"""
        
        model = keras.Sequential([
            keras.layers.Dense(128, activation='relu', input_shape=(10,)),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(32, activation='relu'),
            keras.layers.Dense(16, activation='relu'),
            keras.layers.Dense(1, activation='sigmoid')  # IV is between 0 and 1
        ])
        
        model.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def prepare_features(self, options_data: pd.DataFrame, 
                        market_data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Prepare features for volatility surface modeling
        """
        
        features_df = pd.DataFrame()
        
        # Basic features
        features_df['moneyness'] = options_data['underlying_price'] / options_data['strike']
        features_df['log_moneyness'] = np.log(features_df['moneyness'])
        features_df['time_to_expiry'] = self._calculate_time_to_expiry(
            options_data['date'], options_data['expiry']
        )
        
        # Polynomial features for moneyness
        features_df['moneyness_squared'] = features_df['moneyness'] ** 2
        features_df['moneyness_cubed'] = features_df['moneyness'] ** 3
        
        # Time-related features
        features_df['sqrt_time'] = np.sqrt(features_df['time_to_expiry'])
        features_df['time_squared'] = features_df['time_to_expiry'] ** 2
        
        # Interaction features
        features_df['moneyness_time'] = features_df['moneyness'] * features_df['time_to_expiry']
        features_df['log_moneyness_time'] = features_df['log_moneyness'] * features_df['time_to_expiry']
        
        # Option-specific features
        if 'volume' in options_data.columns and 'open_interest' in options_data.columns:
            features_df['volume_oi_ratio'] = np.where(
                options_data['open_interest'] > 0,
                options_data['volume'] / options_data['open_interest'],
                0
            )
        
        # Market data features (if available)
        if market_data is not None:
            features_df = self._add_market_features(features_df, options_data, market_data)
        
        # Technical indicators
        features_df = self._add_technical_features(features_df, options_data)
        
        # Store feature names
        self.feature_names = features_df.columns.tolist()
        
        return features_df
    
    def _calculate_time_to_expiry(self, dates: pd.Series, expiries: pd.Series) -> pd.Series:
        """Calculate time to expiry in years"""
        
        time_to_expiry = []
        
        for date, expiry in zip(dates, expiries):
            try:
                date_obj = pd.to_datetime(date)
                expiry_obj = pd.to_datetime(expiry)
                days_to_expiry = (expiry_obj - date_obj).days
                years_to_expiry = days_to_expiry / 365.25
                time_to_expiry.append(max(years_to_expiry, 1/365.25))  # Minimum 1 day
            except:
                time_to_expiry.append(0.25)  # Default to 3 months
        
        return pd.Series(time_to_expiry)
    
    def _add_market_features(self, features_df: pd.DataFrame, 
                           options_data: pd.DataFrame,
                           market_data: pd.DataFrame) -> pd.DataFrame:
        """Add market-based features"""
        
        # VIX level and changes
        if 'vix_level' in market_data.columns:
            features_df['vix_level'] = market_data['vix_level'].fillna(20)
            features_df['vix_percentile'] = market_data.get('vix_percentile', 0.5)
        
        # Historical volatility
        if 'realized_vol' in market_data.columns:
            features_df['historical_vol'] = market_data['realized_vol'].fillna(0.2)
            features_df['vol_of_vol'] = market_data.get('vol_of_vol', 0.1)
        
        # Term structure
        if 'term_structure_slope' in market_data.columns:
            features_df['term_structure_slope'] = market_data['term_structure_slope'].fillna(0)
        
        # Underlying returns
        if 'underlying_return' in market_data.columns:
            features_df['underlying_return_1d'] = market_data['underlying_return'].fillna(0)
            features_df['underlying_return_5d'] = market_data.get('underlying_return_5d', 0)
        
        return features_df
    
    def _add_technical_features(self, features_df: pd.DataFrame, 
                              options_data: pd.DataFrame) -> pd.DataFrame:
        """Add technical analysis features"""
        
        # Option type encoding
        features_df['is_call'] = (options_data['option_type'] == 'C').astype(int)
        
        # Strike clustering (distance to round numbers)
        features_df['strike_mod_5'] = options_data['strike'] % 5
        features_df['strike_mod_10'] = options_data['strike'] % 10
        
        # Bid-ask spread
        if 'bid' in options_data.columns and 'ask' in options_data.columns:
            features_df['bid_ask_spread'] = options_data['ask'] - options_data['bid']
            features_df['bid_ask_midpoint'] = (options_data['ask'] + options_data['bid']) / 2
            
            # Relative spread
            features_df['relative_spread'] = np.where(
                features_df['bid_ask_midpoint'] > 0,
                features_df['bid_ask_spread'] / features_df['bid_ask_midpoint'],
                0
            )
        
        return features_df
    
    def train(self, options_data: pd.DataFrame, market_data: Optional[pd.DataFrame] = None,
              validation_split: float = 0.2) -> Dict[str, float]:
        """
        Train the volatility surface model
        """
        
        # Prepare features
        X = self.prepare_features(options_data, market_data)
        y = options_data['iv'].fillna(0.2)  # Target implied volatility
        
        # Remove invalid data
        valid_mask = (y > 0.01) & (y < 5.0) & (~np.isnan(y))
        X = X[valid_mask]
        y = y[valid_mask]
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=validation_split, random_state=42
        )
        
        # Scale features for neural networks
        if self.model_type == 'neural_network':
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
        else:
            X_train_scaled = X_train
            X_test_scaled = X_test
        
        # Train model
        if self.model_type == 'neural_network':
            history = self.model.fit(
                X_train_scaled, y_train,
                epochs=100,
                batch_size=32,
                validation_split=0.1,
                verbose=0,
                callbacks=[
                    keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
                ]
            )
        else:
            self.model.fit(X_train_scaled, y_train)
        
        # Make predictions
        if self.model_type == 'neural_network':
            y_pred = self.model.predict(X_test_scaled).flatten()
        else:
            y_pred = self.model.predict(X_test_scaled)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mse)
        
        # Mean absolute percentage error
        mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
        
        metrics = {
            'mse': mse,
            'rmse': rmse,
            'r2': r2,
            'mape': mape
        }
        
        self.logger.info(f"Model training completed - RMSE: {rmse:.4f}, R²: {r2:.4f}")
        
        return metrics
    
    def predict_iv(self, options_data: pd.DataFrame, 
                   market_data: Optional[pd.DataFrame] = None) -> np.ndarray:
        """
        Predict implied volatility for given options
        """
        
        if self.model is None:
            raise ValueError("Model must be trained before making predictions")
        
        # Prepare features
        X = self.prepare_features(options_data, market_data)
        
        # Scale if needed
        if self.model_type == 'neural_network':
            X_scaled = self.scaler.transform(X)
            predictions = self.model.predict(X_scaled).flatten()
        else:
            predictions = self.model.predict(X)
        
        # Ensure predictions are within reasonable bounds
        predictions = np.clip(predictions, 0.01, 5.0)
        
        return predictions
    
    def smooth_surface(self, iv_surface: np.ndarray, strikes: np.ndarray, 
                      maturities: np.ndarray) -> np.ndarray:
        """
        Smooth noisy implied volatility surface using ML
        """
        
        # Create grid of all strike-maturity combinations
        strike_grid, maturity_grid = np.meshgrid(strikes, maturities)
        
        # Flatten for model input
        flat_strikes = strike_grid.flatten()
        flat_maturities = maturity_grid.flatten()
        flat_iv = iv_surface.flatten()
        
        # Create synthetic options data for smoothing
        smooth_data = pd.DataFrame({
            'strike': flat_strikes,
            'time_to_expiry': flat_maturities,
            'iv': flat_iv,
            'underlying_price': np.full_like(flat_strikes, strikes[len(strikes)//2]),
            'option_type': 'C'
        })
        
        # Prepare features (simplified for smoothing)
        features = pd.DataFrame()
        features['moneyness'] = smooth_data['underlying_price'] / smooth_data['strike']
        features['log_moneyness'] = np.log(features['moneyness'])
        features['time_to_expiry'] = smooth_data['time_to_expiry']
        features['moneyness_time'] = features['moneyness'] * features['time_to_expiry']
        
        # Use a simple model for smoothing
        smoother = RandomForestRegressor(n_estimators=50, random_state=42)
        
        # Remove invalid data
        valid_mask = ~np.isnan(flat_iv) & (flat_iv > 0)
        if valid_mask.sum() > 10:  # Need minimum data points
            smoother.fit(features[valid_mask], flat_iv[valid_mask])
            smoothed_iv = smoother.predict(features)
            smoothed_surface = smoothed_iv.reshape(iv_surface.shape)
        else:
            smoothed_surface = iv_surface
        
        return smoothed_surface
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance from tree-based models
        """
        
        if self.model is None:
            return {}
        
        if hasattr(self.model, 'feature_importances_'):
            importance_dict = dict(zip(self.feature_names, self.model.feature_importances_))
            # Sort by importance
            return dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
        else:
            return {}
    
    def calibrate_model(self, options_data: pd.DataFrame, 
                       market_data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        Perform hyperparameter optimization
        """
        
        X = self.prepare_features(options_data, market_data)
        y = options_data['iv'].fillna(0.2)
        
        # Remove invalid data
        valid_mask = (y > 0.01) & (y < 5.0) & (~np.isnan(y))
        X = X[valid_mask]
        y = y[valid_mask]
        
        # Define parameter grids
        if self.model_type == 'xgboost':
            param_grid = {
                'n_estimators': [100, 200],
                'max_depth': [4, 6, 8],
                'learning_rate': [0.05, 0.1, 0.2],
                'subsample': [0.8, 0.9]
            }
        elif self.model_type == 'random_forest':
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, 15],
                'min_samples_split': [2, 5, 10]
            }
        else:
            # Use default parameters for other models
            return self.train(options_data, market_data)
        
        # Grid search
        grid_search = GridSearchCV(
            self.model, param_grid,
            cv=3, scoring='neg_mean_squared_error',
            n_jobs=-1, verbose=1
        )
        
        grid_search.fit(X, y)
        
        # Update model with best parameters
        self.model = grid_search.best_estimator_
        
        return {
            'best_params': grid_search.best_params_,
            'best_score': -grid_search.best_score_,
            'cv_results': grid_search.cv_results_
        }
    
    def save_model(self, filepath: str):
        """Save the trained model"""
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'model_type': self.model_type
        }
        
        if self.model_type == 'neural_network':
            self.model.save(filepath + '_nn.h5')
            # Save other components separately
            joblib.dump({k: v for k, v in model_data.items() if k != 'model'}, 
                       filepath + '_metadata.pkl')
        else:
            joblib.dump(model_data, filepath + '.pkl')
    
    def load_model(self, filepath: str):
        """Load a trained model"""
        
        if self.model_type == 'neural_network':
            self.model = keras.models.load_model(filepath + '_nn.h5')
            metadata = joblib.load(filepath + '_metadata.pkl')
            self.scaler = metadata['scaler']
            self.feature_names = metadata['feature_names']
        else:
            model_data = joblib.load(filepath + '.pkl')
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.feature_names = model_data['feature_names']
    
    def plot_surface_comparison(self, original_surface: np.ndarray, 
                               predicted_surface: np.ndarray,
                               strikes: np.ndarray, maturities: np.ndarray):
        """
        Plot comparison between original and predicted volatility surfaces
        """
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # Original surface
        im1 = axes[0].contourf(maturities, strikes, original_surface, levels=20, cmap='viridis')
        axes[0].set_title('Original IV Surface')
        axes[0].set_xlabel('Time to Expiry')
        axes[0].set_ylabel('Strike')
        plt.colorbar(im1, ax=axes[0])
        
        # Predicted surface
        im2 = axes[1].contourf(maturities, strikes, predicted_surface, levels=20, cmap='viridis')
        axes[1].set_title('ML Predicted IV Surface')
        axes[1].set_xlabel('Time to Expiry')
        axes[1].set_ylabel('Strike')
        plt.colorbar(im2, ax=axes[1])
        
        # Difference
        diff = predicted_surface - original_surface
        im3 = axes[2].contourf(maturities, strikes, diff, levels=20, cmap='RdBu_r')
        axes[2].set_title('Difference (Predicted - Original)')
        axes[2].set_xlabel('Time to Expiry')
        axes[2].set_ylabel('Strike')
        plt.colorbar(im3, ax=axes[2])
        
        plt.tight_layout()
        plt.show()

# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Create sample data for testing
    np.random.seed(42)
    n_options = 1000
    
    sample_data = pd.DataFrame({
        'strike': np.random.uniform(90, 110, n_options),
        'underlying_price': np.full(n_options, 100),
        'time_to_expiry': np.random.uniform(0.01, 1.0, n_options),
        'option_type': np.random.choice(['C', 'P'], n_options),
        'volume': np.random.randint(1, 1000, n_options),
        'open_interest': np.random.randint(10, 10000, n_options),
        'date': '2024-01-01',
        'expiry': '2024-06-01'
    })
    
    # Add synthetic IV with some noise
    moneyness = sample_data['underlying_price'] / sample_data['strike']
    base_iv = 0.2 + 0.1 * (1 - moneyness) + 0.05 * np.sqrt(sample_data['time_to_expiry'])
    sample_data['iv'] = base_iv + np.random.normal(0, 0.02, n_options)
    
    # Test the model
    vol_model = VolatilitySurfaceML(model_type='xgboost')
    metrics = vol_model.train(sample_data)
    
    print("Training metrics:", metrics)
    print("Feature importance:", vol_model.get_feature_importance())