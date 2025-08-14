#!/usr/bin/env python3
"""
Simplified ML Analysis Pipeline
Runs basic machine learning analysis on real options data
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

# Basic ML imports
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('simple_ml_analysis.log'),
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
    try:
        if hasattr(hist_df.index, 'tz') and hist_df.index.tz is not None:
            hist_df.index = hist_df.index.tz_localize(None)
    except (AttributeError, TypeError):
        # Index is already timezone-naive or not datetime
        pass
    
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
    hist_df = hist_df.ffill().bfill()
    
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
    """Train simple volatility prediction models"""
    
    logger.info("Training volatility models...")
    
    vol_results = {}
    
    for symbol, data in all_data.items():
        try:
            logger.info(f"  Training volatility model for {symbol}")
            
            options_df = data['options']
            hist_df = data['historical']
            
            # Prepare features for volatility prediction
            features = ['Moneyness', 'LogMoneyness', 'TimeToExpiry', 'Volume']
            X = options_df[features].copy()
            y = options_df['IV'].copy()
            
            # Add market features
            latest_vol = hist_df['RealizedVol_20'].iloc[-1] if not hist_df.empty else 0.2
            latest_return = hist_df['Returns'].iloc[-1] if not hist_df.empty else 0.0
            
            X['MarketVol'] = latest_vol
            X['MarketReturn'] = latest_return
            
            # Clean data
            X = X.fillna(0)
            y = y.fillna(0.2)
            
            if len(X) < 10:
                logger.warning(f"    Insufficient data for {symbol}")
                continue
            
            # Train Random Forest model
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            
            # Evaluate model
            y_pred = model.predict(X_test)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)
            
            # Cross validation
            cv_scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')
            cv_rmse = np.sqrt(-cv_scores.mean())
            
            # Feature importance
            importance = dict(zip(X.columns, model.feature_importances_))
            
            vol_results[symbol] = {
                'metrics': {
                    'rmse': float(rmse),
                    'r2': float(r2),
                    'cv_rmse': float(cv_rmse)
                },
                'feature_importance': {k: float(v) for k, v in importance.items()},
                'model_type': 'RandomForest',
                'n_samples': len(X),
                'n_features': len(X.columns)
            }
            
            logger.info(f"    ✓ {symbol} volatility model - RMSE: {rmse:.4f}, R²: {r2:.4f}")
            
        except Exception as e:
            logger.error(f"    ✗ Error training volatility model for {symbol}: {e}")
            continue
    
    return vol_results

def generate_strategy_signals(all_data, logger):
    """Generate simple strategy signals"""
    
    logger.info("Generating strategy signals...")
    
    signal_results = {}
    
    for symbol, data in all_data.items():
        try:
            logger.info(f"  Generating signals for {symbol}")
            
            hist_df = data['historical']
            options_df = data['options']
            
            # Calculate current market conditions
            latest_price = hist_df['Close'].iloc[-1]
            latest_vol = hist_df['RealizedVol_20'].iloc[-1]
            rsi = hist_df['RSI'].iloc[-1]
            
            # Calculate average IV from options
            avg_iv = options_df['IV'].mean() if not options_df.empty else 0.2
            
            # Generate signals based on conditions
            signals = {}
            
            # Straddle signal (high volatility expected)
            if avg_iv < latest_vol * 0.8 and rsi > 70:
                straddle_signal = 1
                straddle_confidence = 0.7
            elif avg_iv < latest_vol * 0.8 and rsi < 30:
                straddle_signal = 1
                straddle_confidence = 0.7
            else:
                straddle_signal = 0
                straddle_confidence = 0.3
            
            signals['straddle'] = {
                'signal': straddle_signal,
                'confidence': straddle_confidence,
                'reason': 'IV vs RV ratio and RSI conditions'
            }
            
            # Covered call signal (moderate volatility, bullish)
            if latest_vol < 0.2 and rsi < 70:
                covered_call_signal = 1
                covered_call_confidence = 0.6
            else:
                covered_call_signal = 0
                covered_call_confidence = 0.3
            
            signals['covered_call'] = {
                'signal': covered_call_signal,
                'confidence': covered_call_confidence,
                'reason': 'Low volatility and moderate RSI'
            }
            
            # Iron condor signal (low volatility expected)
            if avg_iv > latest_vol * 1.2 and 40 < rsi < 60:
                iron_condor_signal = 1
                iron_condor_confidence = 0.65
            else:
                iron_condor_signal = 0
                iron_condor_confidence = 0.35
            
            signals['iron_condor'] = {
                'signal': iron_condor_signal,
                'confidence': iron_condor_confidence,
                'reason': 'High IV vs RV ratio and neutral RSI'
            }
            
            signal_results[symbol] = {
                'signals': signals,
                'market_conditions': {
                    'current_price': float(latest_price),
                    'realized_vol': float(latest_vol),
                    'implied_vol': float(avg_iv),
                    'rsi': float(rsi),
                    'iv_rv_ratio': float(avg_iv / latest_vol) if latest_vol > 0 else 1.0
                }
            }
            
            logger.info(f"    ✓ Generated signals for {symbol}")
            
        except Exception as e:
            logger.error(f"    ✗ Error generating signals for {symbol}: {e}")
            continue
    
    return signal_results

def perform_risk_analysis(all_data, logger):
    """Perform basic risk analysis"""
    
    logger.info("Performing risk analysis...")
    
    risk_results = {}
    
    for symbol, data in all_data.items():
        try:
            logger.info(f"  Analyzing risk for {symbol}")
            
            hist_df = data['historical']
            
            # Calculate risk metrics
            returns = hist_df['Returns'].dropna()
            
            # Basic risk metrics
            vol_20d = returns.rolling(20).std().iloc[-1] * np.sqrt(252)
            vol_60d = returns.rolling(60).std().iloc[-1] * np.sqrt(252)
            
            # VaR calculation (95% confidence)
            var_95 = np.percentile(returns, 5)
            
            # Expected Shortfall (CVaR)
            es_95 = returns[returns <= var_95].mean()
            
            # Maximum drawdown
            cumulative_returns = (1 + returns).cumprod()
            rolling_max = cumulative_returns.expanding().max()
            drawdown = (cumulative_returns - rolling_max) / rolling_max
            max_drawdown = drawdown.min()
            
            # Sharpe ratio (assuming 0 risk-free rate)
            sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252)
            
            # Forecast next 5 days volatility (simple model)
            recent_vol = returns.iloc[-20:].std() * np.sqrt(252)
            vol_forecast = recent_vol * 1.02  # Simple trend adjustment
            
            risk_results[symbol] = {
                'current_metrics': {
                    'volatility_20d': float(vol_20d),
                    'volatility_60d': float(vol_60d),
                    'var_95': float(var_95),
                    'expected_shortfall_95': float(es_95),
                    'max_drawdown': float(max_drawdown),
                    'sharpe_ratio': float(sharpe_ratio)
                },
                'forecast': {
                    'volatility_forecast_5d': float(vol_forecast),
                    'confidence_interval': [float(vol_forecast * 0.8), float(vol_forecast * 1.2)],
                    'forecast_date': datetime.now().isoformat(),
                    'horizon_days': 5
                }
            }
            
            logger.info(f"    ✓ Risk analysis completed for {symbol}")
            
        except Exception as e:
            logger.error(f"    ✗ Error in risk analysis for {symbol}: {e}")
            continue
    
    return risk_results

def generate_comprehensive_results(vol_results, signal_results, risk_results, all_data):
    """Generate comprehensive analysis results"""
    
    results = {
        'analysis_timestamp': datetime.now().isoformat(),
        'model_performance': {
            'volatility_models': vol_results,
            'strategy_signals': signal_results,
            'risk_analysis': risk_results
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
            signals = signal_results[symbol]['signals']
            for strategy, signal_data in signals.items():
                if signal_data['signal'] == 1:
                    recommendations.append({
                        'strategy': strategy.replace('_', ' ').title(),
                        'reason': signal_data['reason'],
                        'confidence': f"{signal_data['confidence']:.2f}"
                    })
        
        results['trading_recommendations'][symbol] = recommendations
    
    return results

def save_results(results):
    """Save all results to files"""
    
    # Create results directory
    os.makedirs('docs/data/results', exist_ok=True)
    
    # Save comprehensive results
    with open('docs/data/results/simple_ml_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print("✓ Results saved to docs/data/results/simple_ml_results.json")

def main():
    """Main execution function"""
    
    logger = setup_logging()
    logger.info("Starting simplified ML analysis...")
    
    # Load real data
    symbols = ['SPY', 'QQQ', 'AAPL', 'MSFT', 'NVDA']  # Focus on liquid symbols
    logger.info(f"Loading data for symbols: {symbols}")
    
    all_data = load_real_data(symbols)
    
    if not all_data:
        logger.error("No data loaded. Please run fetch_real_data.py first.")
        return False
    
    logger.info(f"Loaded data for {len(all_data)} symbols")
    
    # Run analysis
    vol_results = train_volatility_models(all_data, logger)
    signal_results = generate_strategy_signals(all_data, logger)
    risk_results = perform_risk_analysis(all_data, logger)
    
    # Generate comprehensive results
    logger.info("Generating comprehensive analysis results...")
    results = generate_comprehensive_results(vol_results, signal_results, risk_results, all_data)
    
    # Save results
    logger.info("Saving results...")
    save_results(results)
    
    # Print summary
    logger.info("=" * 60)
    logger.info("SIMPLIFIED ML ANALYSIS COMPLETE")
    logger.info("=" * 60)
    
    logger.info("Analysis Summary:")
    logger.info(f"  - Volatility Models: {len(vol_results)} trained")
    logger.info(f"  - Strategy Signals: {len(signal_results)} generated")
    logger.info(f"  - Risk Analysis: {len(risk_results)} completed")
    
    logger.info("\nMarket Insights:")
    for symbol, summary in results['market_summary'].items():
        iv_rv_ratio = summary.get('iv_realized_vol_ratio')
        if iv_rv_ratio:
            logger.info(f"  {symbol}: IV/RV = {iv_rv_ratio:.2f}, Price = ${summary['current_price']:.2f}")
    
    logger.info(f"\nResults saved to docs/data/results/simple_ml_results.json")
    logger.info("Ready for research paper generation and interactive dashboard!")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)