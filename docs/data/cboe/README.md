# CBOE Historical Data

This directory would contain Chicago Board Options Exchange (CBOE) historical market statistics and options data. 

## Data Sources (Future Integration)

The following CBOE data sources could be integrated:

### Market Statistics
- **Daily Options Volume**: Total contract volumes across all options
- **Put/Call Ratios**: Market sentiment indicators
- **Volatility Index (VIX)**: Market fear gauge and volatility expectations
- **SKEW Index**: Tail risk measurements
- **Options Market Maker Activities**: Liquidity provision metrics

### Historical Data Files (Examples)
- `CBOE_daily_volume_2023.csv` - Daily trading volumes by underlying
- `CBOE_put_call_ratios.csv` - Historical put/call ratio data
- `CBOE_vix_historical.csv` - VIX term structure data
- `CBOE_market_statistics.csv` - Comprehensive market metrics

## Data Processing Notes

CBOE data typically requires:
- **Date normalization** to consistent UTC timestamps
- **Volume aggregation** across different option series
- **Volatility calculations** for term structure analysis
- **Cross-validation** with other data sources

## Current Status

This directory is prepared for future CBOE data integration. The current analysis uses Yahoo Finance data which provides sufficient coverage for our machine learning models and research findings.

For our research paper results, VIX data was successfully integrated from Yahoo Finance (`^VIX` symbol) with 752 historical records spanning 3 years of volatility regime data.