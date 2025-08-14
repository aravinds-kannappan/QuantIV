#!/bin/bash

# Volatility Alchemist Build Script

set -e  # Exit on any error

echo "🔧 Building Volatility Alchemist..."
echo "===================================="

# Check if we're in the right directory
if [[ ! -f "CMakeLists.txt" ]]; then
    echo "❌ Error: CMakeLists.txt not found. Please run this script from the project root."
    exit 1
fi

# Create build directory
BUILD_DIR="build"
if [[ ! -d "$BUILD_DIR" ]]; then
    echo "📁 Creating build directory..."
    mkdir -p "$BUILD_DIR"
fi

cd "$BUILD_DIR"

# Configure CMake
echo "⚙️  Configuring CMake..."
cmake .. -DCMAKE_BUILD_TYPE=Release

# Check if CMake configuration was successful
if [[ $? -ne 0 ]]; then
    echo "❌ CMake configuration failed!"
    exit 1
fi

# Build the project
echo "🔨 Building project..."
make -j$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)

# Check if build was successful
if [[ $? -ne 0 ]]; then
    echo "❌ Build failed!"
    exit 1
fi

cd ..

echo "✅ Build completed successfully!"
echo "📍 Executable: build/volatility_alchemist"
echo ""
echo "🚀 Usage examples:"
echo "  ./build/volatility_alchemist analyze SPY"
echo "  ./build/volatility_alchemist backtest covered_call"
echo "  ./build/volatility_alchemist surface AAPL"
echo ""
echo "📊 To view results, open docs/index.html in a web browser"