"""
Real Options Data Fetcher
Fetches historical options data from viable free and paid sources
"""

import pandas as pd
import numpy as np
import yfinance as yf
import requests
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging
from dataclasses import dataclass
try:
    from alpha_vantage.timeseries import TimeSeries
except ImportError:
    TimeSeries = None

try:
    from polygon import RESTClient
except ImportError:
    RESTClient = None
import os
from pathlib import Path

@dataclass
class OptionData:
    symbol: str
    date: str
    expiry: str
    strike: float
    option_type: str  # 'C' or 'P'
    bid: float
    ask: float
    last: float
    volume: int
    open_interest: int
    underlying_price: float
    iv: Optional[float] = None
    delta: Optional[float] = None
    gamma: Optional[float] = None
    theta: Optional[float] = None
    vega: Optional[float] = None

class RealDataFetcher:
    """
    Fetches real historical options data from multiple sources
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # API keys from environment or config
        self.alpha_vantage_key = os.getenv('ALPHA_VANTAGE_API_KEY', 
                                          self.config.get('alpha_vantage_key'))
        self.polygon_key = os.getenv('POLYGON_API_KEY', 
                                   self.config.get('polygon_key'))
        
        # Initialize API clients
        if self.alpha_vantage_key and TimeSeries:
            self.av_client = TimeSeries(key=self.alpha_vantage_key, output_format='pandas')
        else:
            self.av_client = None
        
        if self.polygon_key and RESTClient:
            self.polygon_client = RESTClient(api_key=self.polygon_key)
        else:
            self.polygon_client = None
        
        # Rate limiting
        self.last_request_time = {}
        self.min_request_interval = 1.0  # seconds between requests
        
    def fetch_yahoo_options(self, symbol: str, expiry_date: Optional[str] = None) -> List[OptionData]:
        """
        Fetch options data from Yahoo Finance
        """
        try:
            ticker = yf.Ticker(symbol)
            
            # Get available expiry dates
            expiry_dates = ticker.options
            
            if not expiry_dates:
                self.logger.warning(f"No options data available for {symbol}")
                return []
            
            # Use specified expiry or the nearest one
            if expiry_date and expiry_date in expiry_dates:
                target_expiry = expiry_date
            else:
                target_expiry = expiry_dates[0]  # Nearest expiry
            
            # Get options chain
            options_chain = ticker.option_chain(target_expiry)
            
            # Get current stock price
            info = ticker.info
            current_price = info.get('regularMarketPrice', info.get('previousClose', 0))
            
            options_data = []
            current_date = datetime.now().strftime('%Y-%m-%d')
            
            # Process calls
            for _, row in options_chain.calls.iterrows():
                option = OptionData(
                    symbol=symbol,
                    date=current_date,
                    expiry=target_expiry,
                    strike=row['strike'],
                    option_type='C',
                    bid=row.get('bid', 0),
                    ask=row.get('ask', 0),
                    last=row.get('lastPrice', 0),
                    volume=row.get('volume', 0),
                    open_interest=row.get('openInterest', 0),
                    underlying_price=current_price,
                    iv=row.get('impliedVolatility')
                )
                options_data.append(option)
            
            # Process puts
            for _, row in options_chain.puts.iterrows():
                option = OptionData(
                    symbol=symbol,
                    date=current_date,
                    expiry=target_expiry,
                    strike=row['strike'],
                    option_type='P',
                    bid=row.get('bid', 0),
                    ask=row.get('ask', 0),
                    last=row.get('lastPrice', 0),
                    volume=row.get('volume', 0),
                    open_interest=row.get('openInterest', 0),
                    underlying_price=current_price,
                    iv=row.get('impliedVolatility')
                )
                options_data.append(option)
            
            self.logger.info(f"Fetched {len(options_data)} options for {symbol} expiry {target_expiry}")
            return options_data
            
        except Exception as e:
            self.logger.error(f"Error fetching Yahoo options for {symbol}: {e}")
            return []
    
    def fetch_polygon_options(self, symbol: str, date: str) -> List[OptionData]:
        """
        Fetch options data from Polygon.io
        """
        if not self.polygon_key:
            self.logger.warning("Polygon API key not available")
            return []
        
        try:
            self._rate_limit('polygon')
            
            # Get options contracts
            contracts = self.polygon_client.list_options_contracts(
                underlying_ticker=symbol,
                limit=1000
            )
            
            options_data = []
            
            for contract in contracts:
                # Get option details for the specific date
                try:
                    bars = self.polygon_client.get_aggs(
                        ticker=contract.ticker,
                        multiplier=1,
                        timespan="day",
                        from_=date,
                        to=date
                    )
                    
                    if bars:
                        bar = bars[0]
                        option = OptionData(
                            symbol=symbol,
                            date=date,
                            expiry=contract.expiration_date,
                            strike=contract.strike_price,
                            option_type=contract.contract_type,
                            bid=0,  # Not available in bars
                            ask=0,  # Not available in bars
                            last=bar.close,
                            volume=bar.volume,
                            open_interest=0,  # Not available
                            underlying_price=0  # Need to fetch separately
                        )
                        options_data.append(option)
                        
                except Exception as e:
                    self.logger.debug(f"Error fetching polygon data for {contract.ticker}: {e}")
                    continue
            
            return options_data
            
        except Exception as e:
            self.logger.error(f"Error fetching Polygon options for {symbol}: {e}")
            return []
    
    def fetch_dxfeed_sample(self) -> List[OptionData]:
        """
        Fetch sample data from dxFeed
        """
        try:
            # dxFeed sample URL for AAPL option
            sample_url = "https://dxfeed.com/samples/equity-options/AAPL190927C210-TnS.csv"
            
            self._rate_limit('dxfeed')
            
            response = requests.get(sample_url, timeout=30)
            response.raise_for_status()
            
            # Parse the CSV data
            from io import StringIO
            df = pd.read_csv(StringIO(response.text))
            
            options_data = []
            
            for _, row in df.iterrows():
                option = OptionData(
                    symbol="AAPL",
                    date=row.get('Date', '2019-09-27'),
                    expiry="2019-09-27",
                    strike=210.0,
                    option_type='C',
                    bid=row.get('Bid', 0),
                    ask=row.get('Ask', 0),
                    last=row.get('Last', 0),
                    volume=row.get('Volume', 0),
                    open_interest=row.get('OpenInterest', 0),
                    underlying_price=row.get('UnderlyingPrice', 0)
                )
                options_data.append(option)
            
            self.logger.info(f"Fetched {len(options_data)} sample options from dxFeed")
            return options_data
            
        except Exception as e:
            self.logger.error(f"Error fetching dxFeed sample data: {e}")
            return []
    
    def fetch_cboe_volume_data(self, symbol: str, year: str) -> pd.DataFrame:
        """
        Fetch CBOE volume and put/call ratio data
        """
        try:
            # CBOE historical data URLs
            base_url = "https://cdn.cboe.com/data/us/options/market_statistics/archive/"
            filename = f"{year}_volume_and_price.csv"
            url = base_url + filename
            
            self._rate_limit('cboe')
            
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            # Parse CSV
            from io import StringIO
            df = pd.read_csv(StringIO(response.text))
            
            # Filter for the specific symbol if column exists
            if 'Symbol' in df.columns:
                df = df[df['Symbol'] == symbol]
            
            self.logger.info(f"Fetched CBOE volume data for {symbol} year {year}")
            return df
            
        except Exception as e:
            self.logger.error(f"Error fetching CBOE data for {symbol}: {e}")
            return pd.DataFrame()
    
    def fetch_multi_source_data(self, symbol: str, start_date: str, end_date: str) -> List[OptionData]:
        """
        Fetch options data from multiple sources and combine
        """
        all_options = []
        
        self.logger.info(f"Fetching options data for {symbol} from {start_date} to {end_date}")
        
        # Try Yahoo Finance first (most reliable)
        yahoo_options = self.fetch_yahoo_options(symbol)
        all_options.extend(yahoo_options)
        
        # Try Polygon if API key available
        if self.polygon_key:
            polygon_options = self.fetch_polygon_options(symbol, start_date)
            all_options.extend(polygon_options)
        
        # Get sample data if symbol is AAPL
        if symbol.upper() == 'AAPL':
            sample_options = self.fetch_dxfeed_sample()
            all_options.extend(sample_options)
        
        # Remove duplicates based on key fields
        unique_options = self._deduplicate_options(all_options)
        
        self.logger.info(f"Total unique options collected: {len(unique_options)}")
        return unique_options
    
    def fetch_historical_underlying(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Fetch historical underlying stock prices
        """
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(start=start_date, end=end_date)
            
            # Add additional columns for ML features
            hist['Returns'] = hist['Close'].pct_change()
            hist['HL_Ratio'] = (hist['High'] - hist['Low']) / hist['Close']
            hist['Volume_MA'] = hist['Volume'].rolling(window=20).mean()
            hist['Price_MA_20'] = hist['Close'].rolling(window=20).mean()
            hist['Price_MA_50'] = hist['Close'].rolling(window=50).mean()
            hist['RSI'] = self._calculate_rsi(hist['Close'])
            hist['Volatility_20'] = hist['Returns'].rolling(window=20).std() * np.sqrt(252)
            
            return hist
            
        except Exception as e:
            self.logger.error(f"Error fetching historical data for {symbol}: {e}")
            return pd.DataFrame()
    
    def fetch_vix_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Fetch VIX data for volatility analysis
        """
        try:
            vix = yf.Ticker("^VIX")
            hist = vix.history(start=start_date, end=end_date)
            
            # Add VIX-specific features
            hist['VIX_Change'] = hist['Close'].pct_change()
            hist['VIX_MA_10'] = hist['Close'].rolling(window=10).mean()
            hist['VIX_Percentile'] = hist['Close'].rolling(window=252).apply(
                lambda x: (x.iloc[-1] - x.min()) / (x.max() - x.min()) if x.max() != x.min() else 0.5
            )
            
            return hist
            
        except Exception as e:
            self.logger.error(f"Error fetching VIX data: {e}")
            return pd.DataFrame()
    
    def _rate_limit(self, source: str):
        """
        Implement rate limiting for API calls
        """
        current_time = time.time()
        last_time = self.last_request_time.get(source, 0)
        
        if current_time - last_time < self.min_request_interval:
            sleep_time = self.min_request_interval - (current_time - last_time)
            time.sleep(sleep_time)
        
        self.last_request_time[source] = time.time()
    
    def _deduplicate_options(self, options: List[OptionData]) -> List[OptionData]:
        """
        Remove duplicate options based on key fields
        """
        seen = set()
        unique_options = []
        
        for option in options:
            key = (option.symbol, option.date, option.expiry, 
                  option.strike, option.option_type)
            
            if key not in seen:
                seen.add(key)
                unique_options.append(option)
        
        return unique_options
    
    def _calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """
        Calculate Relative Strength Index
        """
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def save_data_to_csv(self, options: List[OptionData], filepath: str):
        """
        Save options data to CSV format
        """
        try:
            # Convert to DataFrame
            data = []
            for option in options:
                data.append({
                    'Symbol': option.symbol,
                    'Date': option.date,
                    'Expiry': option.expiry,
                    'Strike': option.strike,
                    'Type': option.option_type,
                    'Bid': option.bid,
                    'Ask': option.ask,
                    'Last': option.last,
                    'Volume': option.volume,
                    'OpenInterest': option.open_interest,
                    'UnderlyingPrice': option.underlying_price,
                    'IV': option.iv,
                    'Delta': option.delta,
                    'Gamma': option.gamma,
                    'Theta': option.theta,
                    'Vega': option.vega
                })
            
            df = pd.DataFrame(data)
            
            # Ensure directory exists
            Path(filepath).parent.mkdir(parents=True, exist_ok=True)
            
            df.to_csv(filepath, index=False)
            self.logger.info(f"Saved {len(data)} options to {filepath}")
            
        except Exception as e:
            self.logger.error(f"Error saving data to {filepath}: {e}")

# Example usage and testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    fetcher = RealDataFetcher()
    
    # Test Yahoo Finance data
    options = fetcher.fetch_yahoo_options("SPY")
    print(f"Fetched {len(options)} options for SPY")
    
    # Test historical underlying data
    hist = fetcher.fetch_historical_underlying("SPY", "2024-01-01", "2024-12-31")
    print(f"Fetched {len(hist)} days of historical data for SPY")
    
    # Test VIX data
    vix = fetcher.fetch_vix_data("2024-01-01", "2024-12-31")
    print(f"Fetched {len(vix)} days of VIX data")