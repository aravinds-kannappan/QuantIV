#!/usr/bin/env python3
"""
Real Data Fetching Script
Fetches actual options and market data from various sources
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import time

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from ml.data_sources.real_data_fetcher import RealDataFetcher

def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('data_fetch.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def create_data_directories():
    """Create necessary data directories"""
    dirs = [
        'data/real',
        'data/processed',
        'data/ml_features',
        'docs/data/results'
    ]
    
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)
        print(f"✓ Created directory: {dir_path}")

def fetch_symbols_data(symbols, fetcher, logger):
    """Fetch data for multiple symbols"""
    
    all_data = {}
    failed_symbols = []
    
    for symbol in symbols:
        try:
            logger.info(f"Fetching data for {symbol}...")
            
            # Fetch options data
            logger.info(f"  - Fetching options data for {symbol}")
            options_data = fetcher.fetch_yahoo_options(symbol)
            
            if len(options_data) == 0:
                logger.warning(f"  - No options data found for {symbol}")
                failed_symbols.append(symbol)
                continue
            
            # Convert to DataFrame
            options_df = pd.DataFrame([{
                'Symbol': opt.symbol,
                'Date': opt.date,
                'Expiry': opt.expiry,
                'Strike': opt.strike,
                'Type': opt.option_type,
                'Bid': opt.bid,
                'Ask': opt.ask,
                'Last': opt.last,
                'Volume': opt.volume,
                'OpenInterest': opt.open_interest,
                'UnderlyingPrice': opt.underlying_price,
                'IV': opt.iv
            } for opt in options_data])
            
            # Fetch historical underlying data
            logger.info(f"  - Fetching historical data for {symbol}")
            end_date = datetime.now().strftime('%Y-%m-%d')
            start_date = (datetime.now() - timedelta(days=365*2)).strftime('%Y-%m-%d')
            
            hist_data = fetcher.fetch_historical_underlying(symbol, start_date, end_date)
            
            if hist_data.empty:
                logger.warning(f"  - No historical data found for {symbol}")
                failed_symbols.append(symbol)
                continue
            
            # Store data
            all_data[symbol] = {
                'options': options_df,
                'historical': hist_data,
                'fetch_date': datetime.now().isoformat()
            }
            
            # Save individual symbol data
            options_df.to_csv(f'data/real/{symbol}_options.csv', index=False)
            hist_data.to_csv(f'data/real/{symbol}_historical.csv')
            
            logger.info(f"  ✓ Successfully fetched data for {symbol}")
            logger.info(f"    - Options records: {len(options_df)}")
            logger.info(f"    - Historical records: {len(hist_data)}")
            
            # Rate limiting
            time.sleep(1)
            
        except Exception as e:
            logger.error(f"  ✗ Error fetching data for {symbol}: {e}")
            failed_symbols.append(symbol)
            continue
    
    return all_data, failed_symbols

def fetch_vix_data(fetcher, logger):
    """Fetch VIX data"""
    
    try:
        logger.info("Fetching VIX data...")
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=365*3)).strftime('%Y-%m-%d')
        
        vix_data = fetcher.fetch_vix_data(start_date, end_date)
        
        if not vix_data.empty:
            vix_data.to_csv('data/real/VIX_data.csv')
            logger.info(f"✓ VIX data saved: {len(vix_data)} records")
            return vix_data
        else:
            logger.warning("No VIX data found")
            return pd.DataFrame()
            
    except Exception as e:
        logger.error(f"Error fetching VIX data: {e}")
        return pd.DataFrame()

def create_feature_dataset(all_data, vix_data, logger):
    """Create comprehensive feature dataset for ML"""
    
    logger.info("Creating feature dataset...")
    
    feature_datasets = {}
    
    for symbol, data in all_data.items():
        try:
            hist_data = data['historical'].copy()
            options_data = data['options'].copy()
            
            # Add technical indicators
            logger.info(f"  - Adding technical features for {symbol}")
            
            # Price-based features
            hist_data['SMA_20'] = hist_data['Close'].rolling(20).mean()
            hist_data['SMA_50'] = hist_data['Close'].rolling(50).mean()
            hist_data['Price_SMA20_Ratio'] = hist_data['Close'] / hist_data['SMA_20']
            
            # Volatility features
            hist_data['Realized_Vol_20'] = hist_data['Returns'].rolling(20).std() * np.sqrt(252)
            hist_data['Realized_Vol_60'] = hist_data['Returns'].rolling(60).std() * np.sqrt(252)
            
            # Volume features
            hist_data['Volume_SMA'] = hist_data['Volume'].rolling(20).mean()
            hist_data['Volume_Ratio'] = hist_data['Volume'] / hist_data['Volume_SMA']
            
            # Add VIX data if available
            if not vix_data.empty:
                # Merge VIX data with historical data
                vix_resampled = vix_data[['Close']].rename(columns={'Close': 'VIX'})
                hist_data = hist_data.join(vix_resampled, how='left')
                hist_data['VIX'].fillna(method='ffill', inplace=True)
                hist_data['VIX_Change'] = hist_data['VIX'].pct_change()
            
            # Add options-derived features
            if not options_data.empty:
                # Calculate put/call ratios
                daily_options = options_data.groupby(['Date', 'Type']).agg({
                    'Volume': 'sum',
                    'OpenInterest': 'sum',
                    'IV': 'mean'
                }).reset_index()
                
                # Pivot to get calls and puts
                pc_pivot = daily_options.pivot_table(
                    index='Date', 
                    columns='Type', 
                    values='Volume', 
                    fill_value=0
                )
                
                if 'C' in pc_pivot.columns and 'P' in pc_pivot.columns:
                    pc_pivot['Put_Call_Ratio'] = pc_pivot['P'] / (pc_pivot['C'] + 1e-6)
                    
                    # Convert Date index to datetime
                    pc_pivot.index = pd.to_datetime(pc_pivot.index)
                    hist_data = hist_data.join(pc_pivot[['Put_Call_Ratio']], how='left')
                    hist_data['Put_Call_Ratio'].fillna(method='ffill', inplace=True)
                
                # Add average IV
                avg_iv = options_data.groupby('Date')['IV'].mean()
                avg_iv.index = pd.to_datetime(avg_iv.index)
                hist_data = hist_data.join(avg_iv.rename('Avg_IV'), how='left')
                hist_data['Avg_IV'].fillna(method='ffill', inplace=True)
            
            # Clean data
            hist_data = hist_data.fillna(method='ffill').fillna(method='bfill')
            
            # Store feature dataset
            feature_datasets[symbol] = hist_data
            
            # Save to file
            hist_data.to_csv(f'data/ml_features/{symbol}_features.csv')
            
            logger.info(f"  ✓ Created feature dataset for {symbol}: {len(hist_data)} records, {len(hist_data.columns)} features")
            
        except Exception as e:
            logger.error(f"  ✗ Error creating features for {symbol}: {e}")
            continue
    
    return feature_datasets

def create_summary_report(all_data, failed_symbols, feature_datasets, logger):
    """Create data summary report"""
    
    logger.info("Creating summary report...")
    
    summary = {
        'fetch_timestamp': datetime.now().isoformat(),
        'successful_symbols': list(all_data.keys()),
        'failed_symbols': failed_symbols,
        'total_symbols_attempted': len(all_data) + len(failed_symbols),
        'success_rate': len(all_data) / (len(all_data) + len(failed_symbols)) if (len(all_data) + len(failed_symbols)) > 0 else 0,
        'data_summary': {}
    }
    
    for symbol, data in all_data.items():
        summary['data_summary'][symbol] = {
            'options_records': len(data['options']),
            'historical_records': len(data['historical']),
            'options_expiries': data['options']['Expiry'].nunique() if not data['options'].empty else 0,
            'options_strikes': data['options']['Strike'].nunique() if not data['options'].empty else 0,
            'date_range': {
                'start': str(data['historical'].index.min()) if not data['historical'].empty else None,
                'end': str(data['historical'].index.max()) if not data['historical'].empty else None
            },
            'avg_iv': float(data['options']['IV'].mean()) if not data['options'].empty and 'IV' in data['options'].columns else None,
            'feature_count': len(feature_datasets[symbol].columns) if symbol in feature_datasets else 0
        }
    
    # Save summary
    with open('data/real/data_summary.json', 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    logger.info("✓ Summary report saved to data/real/data_summary.json")
    
    return summary

def main():
    """Main execution function"""
    
    logger = setup_logging()
    logger.info("Starting real data fetching process...")
    
    # Create directories
    create_data_directories()
    
    # Initialize data fetcher
    fetcher = RealDataFetcher()
    
    # Define symbols to fetch
    symbols = ['SPY', 'QQQ', 'AAPL', 'MSFT', 'NVDA', 'TSLA', 'AMZN', 'GOOGL']
    logger.info(f"Target symbols: {symbols}")
    
    # Fetch VIX data first
    vix_data = fetch_vix_data(fetcher, logger)
    
    # Fetch data for all symbols
    all_data, failed_symbols = fetch_symbols_data(symbols, fetcher, logger)
    
    if not all_data:
        logger.error("No data was successfully fetched for any symbol!")
        return False
    
    # Create feature datasets
    feature_datasets = create_feature_dataset(all_data, vix_data, logger)
    
    # Create summary report
    summary = create_summary_report(all_data, failed_symbols, feature_datasets, logger)
    
    # Print final summary
    logger.info("=" * 50)
    logger.info("DATA FETCHING COMPLETE")
    logger.info("=" * 50)
    logger.info(f"Successfully fetched data for: {len(all_data)} symbols")
    logger.info(f"Failed symbols: {len(failed_symbols)}")
    logger.info(f"Success rate: {summary['success_rate']:.1%}")
    
    if all_data:
        logger.info("\nData Summary:")
        for symbol, info in summary['data_summary'].items():
            logger.info(f"  {symbol}: {info['options_records']} options, {info['historical_records']} historical, {info['feature_count']} features")
    
    if failed_symbols:
        logger.warning(f"\nFailed to fetch data for: {failed_symbols}")
    
    logger.info(f"\nData files saved to:")
    logger.info(f"  - Raw data: data/real/")
    logger.info(f"  - Features: data/ml_features/")
    logger.info(f"  - Summary: data/real/data_summary.json")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)