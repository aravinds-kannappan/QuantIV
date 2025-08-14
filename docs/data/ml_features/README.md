# Machine Learning Features

This directory contains feature-engineered datasets specifically prepared for ML model training and inference.

## Feature Categories

### Primary Option Features
- **Moneyness**: `ln(S/K)` - Log ratio of underlying price to strike price
- **Log Moneyness**: Natural logarithm transformation for symmetry
- **Time to Expiry**: Days until expiration / 365.25 (annualized)
- **Volume**: Normalized trading volume relative to historical averages

### Market Microstructure Features  
- **Bid-Ask Spread**: `(Ask - Bid) / Mid_Price`
- **Mid Price**: `(Bid + Ask) / 2`
- **Volume Ratio**: Current volume / 20-day moving average volume
- **Open Interest**: Outstanding contracts for liquidity assessment

### Technical Analysis Features
- **RSI (14-day)**: Relative Strength Index momentum oscillator
- **MACD**: Moving Average Convergence Divergence indicator
- **Bollinger Bands**: Price volatility bands (±2 standard deviations)
- **Moving Averages**: SMA 20, 50 day periods for trend identification

### Volatility Features
- **Realized Volatility**: Historical volatility over multiple windows (5, 10, 20, 60 days)
- **Implied Volatility**: Market-derived volatility expectations
- **IV-RV Ratio**: Implied volatility premium relative to realized volatility
- **VIX Integration**: Market-wide volatility regime indicator

## Feature Importance Analysis

Based on our Random Forest model results:

| Feature | Average Importance | Description |
|---------|-------------------|-------------|
| **Moneyness** | 44.5% | Strike-to-spot relationship (primary pricing factor) |
| **Log Moneyness** | 47.7% | Symmetric transformation of moneyness |
| **Volume** | 7.8% | Trading activity indicator |
| **Time to Expiry** | 0.0% | Minimal impact in short-dated options |
| **Market Vol** | 0.0% | Market volatility regime |
| **Market Return** | 0.0% | Underlying return momentum |

**Key Finding**: Moneyness variables dominate predictive power (92.2% combined importance), confirming theoretical expectations from option pricing models.

## Data Processing Pipeline

### 1. Raw Data Ingestion
```
Historical Data (501 days × 8 securities) + Options Data (1,799 contracts)
↓
```

### 2. Technical Indicator Calculation
```python
# RSI Calculation (14-day)
delta = prices.diff()
gain = (delta.where(delta > 0, 0)).rolling(14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
rsi = 100 - (100 / (1 + gain/loss))
```

### 3. Volatility Computation  
```python
# Realized Volatility (20-day)
returns = prices.pct_change()
realized_vol = returns.rolling(20).std() * sqrt(252)
```

### 4. Feature Standardization
- **Z-score normalization** with 20-day rolling windows
- **Missing value imputation** via forward-fill methodology
- **Outlier detection** using 3-sigma bounds

## Model Training Results

Features engineered through this pipeline enabled:

- **Volatility Models**: R² scores from 0.42 (AAPL) to 0.97 (SPY)
- **Signal Generation**: 65% confidence iron condor recommendations
- **Risk Forecasting**: 68% accuracy over 5-day horizons
- **Cross-Validation**: Stable performance across temporal splits

The feature engineering pipeline successfully transforms raw market data into predictive variables suitable for institutional-grade quantitative analysis.