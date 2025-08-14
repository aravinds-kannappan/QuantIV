#!/bin/bash

# Volatility Alchemist Data Download Script

set -e

echo "📥 Downloading Options Data..."
echo "=============================="

# Default symbols to download
SYMBOLS=("SPY" "AAPL" "QQQ" "TSLA" "NVDA")
YEAR="2025"
MONTH="05"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -s|--symbols)
            IFS=',' read -ra SYMBOLS <<< "$2"
            shift 2
            ;;
        -y|--year)
            YEAR="$2"
            shift 2
            ;;
        -m|--month)
            MONTH="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  -s, --symbols SYMBOLS    Comma-separated list of symbols (default: SPY,AAPL,QQQ,TSLA,NVDA)"
            echo "  -y, --year YEAR          Year to download (default: 2025)"
            echo "  -m, --month MONTH        Month to download (default: 05)"
            echo "  -h, --help               Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0                       # Download default symbols"
            echo "  $0 -s SPY,AAPL          # Download specific symbols"
            echo "  $0 -y 2024 -m 12        # Download December 2024 data"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use -h or --help for usage information"
            exit 1
            ;;
    esac
done

# Ensure data directories exist
mkdir -p data/cboe
mkdir -p data/dxfeed

# Function to download CBOE data
download_cboe_data() {
    local symbol=$1
    local url="https://www.cboe.com/publish/scheduledtask/mktdata/datahouse/market_statistics/OPRA/OPTION_VOLUME_SYMBOL_${symbol}_${YEAR}-${MONTH}.csv"
    local filename="data/cboe/${symbol}_${YEAR}-${MONTH}.csv"
    
    echo "📊 Downloading CBOE data for $symbol..."
    
    if curl -L -o "$filename" "$url" --fail --silent --show-error; then
        if [[ -s "$filename" ]]; then
            echo "✅ Successfully downloaded: $filename"
            
            # Check if file contains data (more than just headers)
            local line_count=$(wc -l < "$filename")
            if [[ $line_count -gt 1 ]]; then
                echo "   📈 Data file contains $((line_count - 1)) records"
            else
                echo "   ⚠️  Warning: Data file appears to contain only headers"
            fi
        else
            echo "❌ Downloaded file is empty: $filename"
            rm -f "$filename"
            return 1
        fi
    else
        echo "❌ Failed to download: $url"
        rm -f "$filename"
        return 1
    fi
}

# Function to download sample dxFeed data
download_sample_data() {
    local symbol=$1
    
    # Try to download sample data from dxFeed (these are historical samples)
    case $symbol in
        "AAPL")
            local sample_url="https://dxfeed.com/downloads/samples/equity-options/AAPL190927C210.csv"
            local filename="data/dxfeed/${symbol}_sample.csv"
            ;;
        *)
            echo "   ℹ️  No sample data available for $symbol"
            return 0
            ;;
    esac
    
    echo "📦 Downloading sample data for $symbol..."
    
    if curl -L -o "$filename" "$sample_url" --fail --silent --show-error; then
        if [[ -s "$filename" ]]; then
            echo "✅ Successfully downloaded sample: $filename"
        else
            echo "❌ Downloaded sample file is empty: $filename"
            rm -f "$filename"
        fi
    else
        echo "❌ Failed to download sample: $sample_url"
        rm -f "$filename"
    fi
}

# Main download loop
echo "🎯 Target symbols: ${SYMBOLS[*]}"
echo "📅 Period: $YEAR-$MONTH"
echo ""

successful_downloads=0
total_attempts=0

for symbol in "${SYMBOLS[@]}"; do
    echo "Processing $symbol..."
    
    # Download CBOE data
    ((total_attempts++))
    if download_cboe_data "$symbol"; then
        ((successful_downloads++))
    fi
    
    # Download sample data if available
    download_sample_data "$symbol"
    
    echo ""
done

echo "📊 Download Summary:"
echo "   Total attempts: $total_attempts"
echo "   Successful downloads: $successful_downloads"
echo "   Success rate: $(( (successful_downloads * 100) / total_attempts ))%"

if [[ $successful_downloads -eq 0 ]]; then
    echo ""
    echo "❌ No data was successfully downloaded!"
    echo "   This might be due to:"
    echo "   • Network connectivity issues"
    echo "   • CBOE website being temporarily unavailable"
    echo "   • Data not yet available for the specified period"
    echo "   • Invalid symbol names"
    echo ""
    echo "💡 Try running the analysis with sample data generation:"
    echo "   ./build/volatility_alchemist analyze SPY"
    exit 1
else
    echo ""
    echo "✅ Data download completed!"
    echo "📁 Files saved to:"
    echo "   • data/cboe/ (CBOE market data)"
    echo "   • data/dxfeed/ (Sample data)"
    echo ""
    echo "🚀 Next steps:"
    echo "   1. Build the project: ./scripts/build.sh"
    echo "   2. Run analysis: ./build/volatility_alchemist analyze SPY"
    echo "   3. View dashboard: open docs/index.html"
fi