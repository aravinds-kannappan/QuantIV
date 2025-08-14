#!/usr/bin/env python3
"""
ML Analysis Pipeline
Runs comprehensive machine learning analysis on real options data
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import warnings
warnings.filterwarnings('ignore')

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    from ml.models.volatility_surface_ml import VolatilitySurfaceML
except ImportError:
    VolatilitySurfaceML = None

try:
    from ml.models.strategy_signals import StrategySignalGenerator  
except ImportError:
    StrategySignalGenerator = None

try:
    from ml.models.risk_forecasting import RiskForecaster
except ImportError:
    RiskForecaster = None

def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('ml_analysis.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def load_real_data(symbols):
    """Load real data for analysis"""
    
    all_data = {}
    
    for symbol in symbols:
        try:
            # Load options data
            options_file = f'data/real/{symbol}_options.csv'
            historical_file = f'data/real/{symbol}_historical.csv'
            
            if os.path.exists(options_file) and os.path.exists(historical_file):
                options_df = pd.read_csv(options_file)
                hist_df = pd.read_csv(historical_file, index_col=0, parse_dates=True)
                
                # Clean and prepare data
                options_df = clean_options_data(options_df)
                hist_df = prepare_historical_data(hist_df)
                
                all_data[symbol] = {
                    'options': options_df,
                    'historical': hist_df
                }
                
                print(f"✓ Loaded data for {symbol}: {len(options_df)} options, {len(hist_df)} historical")
            else:
                print(f"✗ Data files missing for {symbol}")
                
        except Exception as e:
            print(f"✗ Error loading data for {symbol}: {e}")
            continue
    
    return all_data

def clean_options_data(options_df):
    """Clean and validate options data"""
    
    # Remove rows with missing critical data
    options_df = options_df.dropna(subset=['Strike', 'UnderlyingPrice', 'IV'])
    
    # Filter out unrealistic IVs
    options_df = options_df[(options_df['IV'] > 0.01) & (options_df['IV'] < 10.0)]
    
    # Calculate additional features
    options_df['Moneyness'] = options_df['UnderlyingPrice'] / options_df['Strike']
    options_df['LogMoneyness'] = np.log(options_df['Moneyness'])
    options_df['MidPrice'] = (options_df['Bid'] + options_df['Ask']) / 2
    options_df['BidAskSpread'] = options_df['Ask'] - options_df['Bid']
    
    # Calculate days to expiry
    options_df['Date'] = pd.to_datetime(options_df['Date'])
    options_df['Expiry'] = pd.to_datetime(options_df['Expiry'])
    options_df['DaysToExpiry'] = (options_df['Expiry'] - options_df['Date']).dt.days
    options_df['TimeToExpiry'] = options_df['DaysToExpiry'] / 365.25
    
    return options_df

def prepare_historical_data(hist_df):
    """Prepare historical data with technical indicators"""
    
    # Ensure timezone-naive datetime index
    if hist_df.index.tz is not None:
        hist_df.index = hist_df.index.tz_localize(None)
    
    # Calculate basic features
    hist_df['Returns'] = hist_df['Close'].pct_change()
    hist_df['LogReturns'] = np.log(hist_df['Close'] / hist_df['Close'].shift(1))
    
    # Volatility measures
    for window in [5, 10, 20, 60]:
        hist_df[f'RealizedVol_{window}'] = hist_df['Returns'].rolling(window).std() * np.sqrt(252)
        hist_df[f'SMA_{window}'] = hist_df['Close'].rolling(window).mean()
    
    # Technical indicators
    hist_df['RSI'] = calculate_rsi(hist_df['Close'])
    hist_df['MACD'], hist_df['MACD_Signal'] = calculate_macd(hist_df['Close'])
    hist_df['BollingerUpper'], hist_df['BollingerLower'] = calculate_bollinger_bands(hist_df['Close'])
    
    # Volume indicators
    hist_df['VolumeMA'] = hist_df['Volume'].rolling(20).mean()
    hist_df['VolumeRatio'] = hist_df['Volume'] / hist_df['VolumeMA']
    
    # Price position indicators
    hist_df['HighLowRatio'] = (hist_df['High'] - hist_df['Low']) / hist_df['Close']
    hist_df['ClosePosition'] = (hist_df['Close'] - hist_df['Low']) / (hist_df['High'] - hist_df['Low'])
    
    # Clean data
    hist_df = hist_df.fillna(method='ffill').fillna(method='bfill')
    
    return hist_df

def calculate_rsi(prices, window=14):
    """Calculate RSI indicator"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(prices, fast=12, slow=26, signal=9):
    """Calculate MACD indicator"""
    ema_fast = prices.ewm(span=fast).mean()
    ema_slow = prices.ewm(span=slow).mean()
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=signal).mean()
    return macd, macd_signal

def calculate_bollinger_bands(prices, window=20, num_std=2):
    """Calculate Bollinger Bands"""
    sma = prices.rolling(window=window).mean()
    std = prices.rolling(window=window).std()
    upper = sma + (std * num_std)
    lower = sma - (std * num_std)
    return upper, lower

def train_volatility_models(all_data, logger):
    """Train volatility surface ML models"""
    
    logger.info("Training volatility surface models...")
    
    vol_results = {}
    
    for symbol, data in all_data.items():
        try:
            logger.info(f"  Training volatility model for {symbol}")
            
            options_df = data['options']
            hist_df = data['historical']
            
            # Prepare market data features
            market_features = prepare_market_features(hist_df, options_df)
            
            # Train model
            vol_model = VolatilitySurfaceML(model_type='xgboost')
            metrics = vol_model.train(options_df, market_features)
            
            # Get feature importance
            importance = vol_model.get_feature_importance()
            
            # Generate predictions for analysis
            predictions = vol_model.predict_iv(options_df, market_features)
            
            vol_results[symbol] = {
                'metrics': metrics,
                'feature_importance': importance,
                'model_type': 'xgboost',
                'predictions': predictions.tolist(),
                'actual_iv': options_df['IV'].tolist(),
                'prediction_error': (predictions - options_df['IV']).tolist()
            }
            
            # Save model
            vol_model.save_model(f'models/{symbol}_volatility_model')
            
            logger.info(f"    ✓ {symbol} volatility model - RMSE: {metrics['rmse']:.4f}, R²: {metrics['r2']:.4f}")
            
        except Exception as e:
            logger.error(f"    ✗ Error training volatility model for {symbol}: {e}")
            continue
    
    return vol_results

def prepare_market_features(hist_df, options_df):
    """Prepare market features aligned with options data"""
    
    # Get the latest date from historical data
    latest_date = hist_df.index[-1].strftime('%Y-%m-%d')
    
    # Get latest market features
    latest_features = hist_df.iloc[-1:].copy()
    
    # Create market data DataFrame
    market_data = pd.DataFrame({
        'realized_vol': [latest_features['RealizedVol_20'].iloc[0]],
        'vix_level': [latest_features.get('VIX', 20).iloc[0] if 'VIX' in latest_features.columns else 20],
        'underlying_return': [latest_features['Returns'].iloc[0]],
        'volume_ratio': [latest_features['VolumeRatio'].iloc[0]],
        'rsi': [latest_features['RSI'].iloc[0]],
        'macd': [latest_features['MACD'].iloc[0]]
    })
    
    return market_data

def train_strategy_signals(all_data, logger):
    """Train strategy signal models"""
    
    logger.info("Training strategy signal models...")
    
    signal_results = {}
    strategies = ['straddle', 'covered_call', 'iron_condor']
    
    for symbol, data in all_data.items():
        try:
            logger.info(f"  Training signal models for {symbol}")
            
            hist_df = data['historical']
            options_df = data['options']
            
            symbol_results = {}
            
            for strategy in strategies:
                try:
                    # Train signal generator
                    signal_gen = StrategySignalGenerator(strategy_type=strategy, model_type='xgboost')
                    metrics = signal_gen.train(hist_df, options_df)
                    
                    # Generate current signal
                    current_signal = signal_gen.generate_signal(hist_df, options_df)
                    
                    # Get feature importance
                    importance = signal_gen.get_feature_importance()
                    
                    symbol_results[strategy] = {
                        'metrics': metrics,
                        'current_signal': {
                            'signal': current_signal.signal,
                            'confidence': current_signal.confidence,
                            'timestamp': current_signal.timestamp
                        },
                        'feature_importance': importance
                    }
                    
                    # Save model
                    signal_gen.save_model(f'models/{symbol}_{strategy}_signal_model.pkl')
                    
                    logger.info(f"    ✓ {symbol} {strategy} - Score: {metrics['cv_score_mean']:.4f}")
                    
                except Exception as e:
                    logger.error(f"    ✗ Error training {strategy} signal for {symbol}: {e}")
                    continue
            
            signal_results[symbol] = symbol_results
            
        except Exception as e:
            logger.error(f"  ✗ Error training signals for {symbol}: {e}")
            continue
    
    return signal_results

def train_risk_forecasting(all_data, logger):
    """Train risk forecasting models"""
    
    logger.info("Training risk forecasting models...")
    
    risk_results = {}
    
    for symbol, data in all_data.items():
        try:
            logger.info(f"  Training risk forecasting for {symbol}")
            
            hist_df = data['historical']
            
            # Train risk forecaster
            risk_forecaster = RiskForecaster(model_type='lstm_tf', forecast_horizon=5)
            metrics = risk_forecaster.train(hist_df)
            
            # Generate risk forecast
            risk_forecast = risk_forecaster.forecast_risk(hist_df)
            
            risk_results[symbol] = {
                'metrics': metrics,
                'forecast': {
                    'volatility_forecast': risk_forecast.volatility_forecast,
                    'volatility_ci': risk_forecast.volatility_confidence_interval,
                    'var_forecast': risk_forecast.var_forecast,
                    'es_forecast': risk_forecast.es_forecast,
                    'max_drawdown_prob': risk_forecast.max_drawdown_prob,
                    'tail_risk_indicator': risk_forecast.tail_risk_indicator,
                    'regime_probability': risk_forecast.regime_probability,
                    'forecast_date': risk_forecast.forecast_date,
                    'horizon_days': risk_forecast.horizon_days
                }
            }
            
            # Save model
            risk_forecaster.save_model(f'models/{symbol}_risk_model')
            
            logger.info(f"    ✓ {symbol} risk model - Vol forecast: {risk_forecast.volatility_forecast:.4f}")
            
        except Exception as e:
            logger.error(f"    ✗ Error training risk model for {symbol}: {e}")
            continue
    
    return risk_results

def generate_comprehensive_results(vol_results, signal_results, risk_results, all_data):
    """Generate comprehensive analysis results"""
    
    results = {
        'analysis_timestamp': datetime.now().isoformat(),
        'model_performance': {
            'volatility_models': vol_results,
            'strategy_signals': signal_results,
            'risk_forecasting': risk_results
        },
        'market_summary': {},
        'model_comparison': {},
        'trading_recommendations': {}
    }
    
    # Market summary
    for symbol, data in all_data.items():
        hist_df = data['historical']
        options_df = data['options']
        
        latest_price = hist_df['Close'].iloc[-1]
        latest_vol = hist_df['RealizedVol_20'].iloc[-1]
        avg_iv = options_df['IV'].mean()
        
        results['market_summary'][symbol] = {
            'current_price': float(latest_price),
            'realized_volatility_20d': float(latest_vol),
            'average_implied_volatility': float(avg_iv),
            'iv_realized_vol_ratio': float(avg_iv / latest_vol) if latest_vol > 0 else None,
            'options_count': len(options_df),
            'strike_range': {
                'min': float(options_df['Strike'].min()),
                'max': float(options_df['Strike'].max())
            }
        }
    
    # Model comparison
    if vol_results:
        vol_rmse_scores = {symbol: res['metrics']['rmse'] for symbol, res in vol_results.items()}
        best_vol_model = min(vol_rmse_scores, key=vol_rmse_scores.get)
        
        results['model_comparison']['best_volatility_model'] = {
            'symbol': best_vol_model,
            'rmse': vol_rmse_scores[best_vol_model]
        }
    
    # Trading recommendations
    for symbol in all_data.keys():
        recommendations = []
        
        # Volatility-based recommendations
        if symbol in results['market_summary']:
            iv_rv_ratio = results['market_summary'][symbol]['iv_realized_vol_ratio']
            if iv_rv_ratio and iv_rv_ratio > 1.2:
                recommendations.append({
                    'strategy': 'Short volatility strategies (Iron Condor, Covered Call)',
                    'reason': f'IV/RV ratio of {iv_rv_ratio:.2f} suggests overpriced options',
                    'confidence': 'medium'
                })
            elif iv_rv_ratio and iv_rv_ratio < 0.8:
                recommendations.append({
                    'strategy': 'Long volatility strategies (Straddle, Strangle)',
                    'reason': f'IV/RV ratio of {iv_rv_ratio:.2f} suggests underpriced options',
                    'confidence': 'medium'
                })
        
        # Signal-based recommendations
        if symbol in signal_results:
            for strategy, signal_data in signal_results[symbol].items():
                if signal_data['current_signal']['signal'] == 1:
                    recommendations.append({
                        'strategy': strategy.replace('_', ' ').title(),
                        'reason': f'ML signal indicates favorable conditions',
                        'confidence': f"{signal_data['current_signal']['confidence']:.2f}"
                    })
        
        results['trading_recommendations'][symbol] = recommendations
    
    return results

def create_visualization_data(results, all_data):
    """Create data for visualizations"""
    
    viz_data = {
        'volatility_surfaces': {},
        'signal_evolution': {},
        'risk_metrics': {},
        'performance_comparison': {}
    }
    
    # Volatility surface data
    for symbol, data in all_data.items():
        options_df = data['options']
        
        # Create volatility surface grid
        strikes = sorted(options_df['Strike'].unique())
        
        # Group by time to expiry (simplified to single expiry for now)
        expiry_groups = options_df.groupby('DaysToExpiry')
        
        surface_data = []
        for days, group in expiry_groups:
            for _, row in group.iterrows():
                surface_data.append({
                    'strike': row['Strike'],
                    'days_to_expiry': row['DaysToExpiry'],
                    'implied_vol': row['IV'],
                    'moneyness': row['Moneyness']
                })
        
        viz_data['volatility_surfaces'][symbol] = surface_data
    
    # Risk metrics evolution (using historical volatility as proxy)
    for symbol, data in all_data.items():
        hist_df = data['historical']
        
        risk_evolution = []
        for i in range(max(0, len(hist_df) - 100), len(hist_df)):  # Last 100 days
            row = hist_df.iloc[i]
            risk_evolution.append({
                'date': row.name.strftime('%Y-%m-%d'),
                'realized_vol': row['RealizedVol_20'],
                'price': row['Close'],
                'volume_ratio': row['VolumeRatio']
            })
        
        viz_data['risk_metrics'][symbol] = risk_evolution
    
    return viz_data

def save_results(results, viz_data):
    """Save all results to files"""
    
    # Create results directory
    os.makedirs('docs/data/results', exist_ok=True)
    
    # Save comprehensive results
    with open('docs/data/results/ml_analysis_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    # Save visualization data
    with open('docs/data/results/visualization_data.json', 'w') as f:
        json.dump(viz_data, f, indent=2, default=str)
    
    # Save individual components
    with open('docs/data/results/volatility_models.json', 'w') as f:
        json.dump(results['model_performance']['volatility_models'], f, indent=2, default=str)
    
    with open('docs/data/results/strategy_signals.json', 'w') as f:
        json.dump(results['model_performance']['strategy_signals'], f, indent=2, default=str)
    
    with open('docs/data/results/risk_forecasting.json', 'w') as f:
        json.dump(results['model_performance']['risk_forecasting'], f, indent=2, default=str)
    
    print("✓ Results saved to docs/data/results/")

def main():
    """Main execution function"""
    
    logger = setup_logging()
    logger.info("Starting comprehensive ML analysis...")
    
    # Create models directory
    os.makedirs('models', exist_ok=True)
    
    # Load real data
    symbols = ['SPY', 'QQQ', 'AAPL', 'MSFT', 'NVDA']  # Focus on liquid symbols
    logger.info(f"Loading data for symbols: {symbols}")
    
    all_data = load_real_data(symbols)
    
    if not all_data:
        logger.error("No data loaded. Please run fetch_real_data.py first.")
        return False
    
    logger.info(f"Loaded data for {len(all_data)} symbols")
    
    # Train models
    vol_results = train_volatility_models(all_data, logger)
    signal_results = train_strategy_signals(all_data, logger)
    risk_results = train_risk_forecasting(all_data, logger)
    
    # Generate comprehensive results
    logger.info("Generating comprehensive analysis results...")
    results = generate_comprehensive_results(vol_results, signal_results, risk_results, all_data)
    
    # Create visualization data
    logger.info("Creating visualization data...")
    viz_data = create_visualization_data(results, all_data)
    
    # Save results
    logger.info("Saving results...")
    save_results(results, viz_data)
    
    # Print summary
    logger.info("=" * 60)
    logger.info("ML ANALYSIS COMPLETE")
    logger.info("=" * 60)
    
    logger.info("Model Training Summary:")
    logger.info(f"  - Volatility Models: {len(vol_results)} trained")
    logger.info(f"  - Strategy Signals: {sum(len(signals) for signals in signal_results.values())} trained")
    logger.info(f"  - Risk Models: {len(risk_results)} trained")
    
    logger.info("\nMarket Insights:")
    for symbol, summary in results['market_summary'].items():
        iv_rv_ratio = summary.get('iv_rv_ratio')
        if iv_rv_ratio:
            logger.info(f"  {symbol}: IV/RV = {iv_rv_ratio:.2f}, Price = ${summary['current_price']:.2f}")
    
    logger.info(f"\nResults saved to docs/data/results/")
    logger.info("Ready for research paper generation and interactive dashboard!")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)