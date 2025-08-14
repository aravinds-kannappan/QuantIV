#include "core/data_loader.hpp"
#include "core/volatility_model.hpp"
#include "core/strategy_engine.hpp"
#include "core/metrics.hpp"
#include "core/exporter.hpp"
#include <iostream>
#include <vector>
#include <memory>

using namespace volatility;

void printUsage() {
    std::cout << "Volatility Alchemist - Options Strategy Analytics Engine\n";
    std::cout << "Usage: volatility_alchemist <command> [options]\n\n";
    std::cout << "Commands:\n";
    std::cout << "  download <symbol>     Download options data for symbol\n";
    std::cout << "  backtest <strategy>   Run strategy backtest\n";
    std::cout << "  surface <symbol>      Generate volatility surface\n";
    std::cout << "  analyze <symbol>      Full analysis pipeline\n\n";
    std::cout << "Strategies:\n";
    std::cout << "  covered_call         Covered call strategy\n";
    std::cout << "  vertical_spread      Bull call spread strategy\n";
    std::cout << "  straddle             Long straddle strategy\n\n";
    std::cout << "Examples:\n";
    std::cout << "  volatility_alchemist analyze SPY\n";
    std::cout << "  volatility_alchemist backtest covered_call\n";
    std::cout << "  volatility_alchemist surface AAPL\n";
}

bool downloadData(const std::string& symbol) {
    std::cout << "Downloading options data for " << symbol << "...\n";
    
    DataLoader loader;
    
    // Download CBOE data
    std::string cboe_url = "https://www.cboe.com/publish/scheduledtask/mktdata/datahouse/market_statistics/OPRA/OPTION_VOLUME_SYMBOL_" + symbol + "_2025-05.csv";
    std::string cboe_file = "data/cboe/" + symbol + "_2025-05.csv";
    
    if (loader.downloadData(cboe_url, cboe_file)) {
        std::cout << "✓ Downloaded CBOE data for " << symbol << "\n";
        return true;
    } else {
        std::cout << "✗ Failed to download CBOE data for " << symbol << "\n";
        return false;
    }
}

bool generateVolatilitySurface(const std::string& symbol) {
    std::cout << "Generating volatility surface for " << symbol << "...\n";
    
    DataLoader loader;
    VolatilityModel vol_model;
    Exporter exporter;
    
    // Load data
    std::string data_file = "data/cboe/" + symbol + "_2025-05.csv";
    auto options_data = loader.parseCBOE(data_file);
    
    if (options_data.empty()) {
        std::cout << "✗ No options data found for " << symbol << "\n";
        return false;
    }
    
    std::cout << "✓ Loaded " << options_data.size() << " option records\n";
    
    // Build volatility surface
    auto surface = vol_model.buildIVSurface(options_data, "2025-05-20");
    
    if (surface.strikes.empty()) {
        std::cout << "✗ Failed to build volatility surface\n";
        return false;
    }
    
    std::cout << "✓ Built volatility surface: " << surface.strikes.size() << " strikes, " 
              << surface.maturities.size() << " maturities\n";
    
    // Export surface
    std::string filename = symbol + "_iv_surface_2025-05-20.json";
    if (exporter.exportIVSurface(surface, filename)) {
        std::cout << "✓ Exported volatility surface to " << filename << "\n";
        return true;
    } else {
        std::cout << "✗ Failed to export volatility surface\n";
        return false;
    }
}

bool runStrategyBacktest(const std::string& strategy_name) {
    std::cout << "Running " << strategy_name << " backtest...\n";
    
    DataLoader loader;
    StrategyEngine engine;
    MetricsCalculator metrics_calc;
    Exporter exporter;
    
    // For demo purposes, use SPY data
    std::string symbol = "SPY";
    std::string data_file = "data/cboe/" + symbol + "_2025-05.csv";
    auto options_data = loader.parseCBOE(data_file);
    
    if (options_data.empty()) {
        std::cout << "✗ No options data found. Please download data first.\n";
        return false;
    }
    
    std::cout << "✓ Loaded " << options_data.size() << " option records\n";
    
    // Set up strategy parameters
    StrategyParams params;
    params.initial_capital = 100000;
    params.commission_per_contract = 1.0;
    params.max_positions = 5;
    params.profit_target = 0.5;
    params.loss_limit = 0.1;
    
    // Determine strategy type
    StrategyType strategy_type;
    if (strategy_name == "covered_call") {
        strategy_type = StrategyType::COVERED_CALL;
    } else if (strategy_name == "vertical_spread") {
        strategy_type = StrategyType::VERTICAL_SPREAD;
    } else if (strategy_name == "straddle") {
        strategy_type = StrategyType::STRADDLE;
    } else {
        std::cout << "✗ Unknown strategy: " << strategy_name << "\n";
        return false;
    }
    
    // Run backtest
    auto results = engine.backtestStrategy(strategy_type, options_data, params);
    
    if (results.dates.empty()) {
        std::cout << "✗ Backtest failed - no results generated\n";
        return false;
    }
    
    std::cout << "✓ Backtest completed: " << results.dates.size() << " trading days\n";
    
    // Calculate metrics
    auto performance_metrics = metrics_calc.calculatePerformanceMetrics(results);
    auto greeks_profile = metrics_calc.calculateGreeksProfile(results);
    
    std::cout << "✓ Performance Analysis:\n";
    std::cout << "  Total Return: " << (performance_metrics.total_return * 100) << "%\n";
    std::cout << "  Sharpe Ratio: " << performance_metrics.sharpe_ratio << "\n";
    std::cout << "  Max Drawdown: " << (performance_metrics.max_drawdown * 100) << "%\n";
    std::cout << "  Win Rate: " << (performance_metrics.win_rate * 100) << "%\n";
    std::cout << "  Total Trades: " << results.total_trades << "\n";
    
    // Export results
    std::string json_filename = strategy_name + "_" + symbol + "_backtest.json";
    std::string csv_filename = strategy_name + "_" + symbol + "_timeseries.csv";
    std::string positions_filename = strategy_name + "_" + symbol + "_positions.csv";
    
    bool export_success = true;
    export_success &= exporter.exportStrategyResults(results, performance_metrics, json_filename);
    export_success &= exporter.exportTimeseriesCSV(results, csv_filename);
    export_success &= exporter.exportPositions(results.all_positions, positions_filename);
    
    if (export_success) {
        std::cout << "✓ Exported backtest results\n";
        return true;
    } else {
        std::cout << "✗ Failed to export some results\n";
        return false;
    }
}

bool runFullAnalysis(const std::string& symbol) {
    std::cout << "Running full analysis for " << symbol << "...\n\n";
    
    bool success = true;
    
    // Step 1: Download data
    success &= downloadData(symbol);
    if (!success) return false;
    
    std::cout << "\n";
    
    // Step 2: Generate volatility surface
    success &= generateVolatilitySurface(symbol);
    if (!success) return false;
    
    std::cout << "\n";
    
    // Step 3: Run strategy backtests
    std::vector<std::string> strategies = {"covered_call"};
    
    for (const auto& strategy : strategies) {
        success &= runStrategyBacktest(strategy);
        if (!success) return false;
        std::cout << "\n";
    }
    
    if (success) {
        std::cout << "✓ Full analysis completed successfully!\n";
        std::cout << "✓ Results exported to docs/data/ directory\n";
        std::cout << "✓ Open docs/index.html in a web browser to view dashboard\n";
    }
    
    return success;
}

int main(int argc, char* argv[]) {
    std::cout << "Volatility Alchemist v1.0\n";
    std::cout << "Options Strategy Analytics Engine\n";
    std::cout << "================================\n\n";
    
    if (argc < 2) {
        printUsage();
        return 1;
    }
    
    std::string command = argv[1];
    
    try {
        if (command == "download" && argc >= 3) {
            std::string symbol = argv[2];
            return downloadData(symbol) ? 0 : 1;
            
        } else if (command == "surface" && argc >= 3) {
            std::string symbol = argv[2];
            return generateVolatilitySurface(symbol) ? 0 : 1;
            
        } else if (command == "backtest" && argc >= 3) {
            std::string strategy = argv[2];
            return runStrategyBacktest(strategy) ? 0 : 1;
            
        } else if (command == "analyze" && argc >= 3) {
            std::string symbol = argv[2];
            return runFullAnalysis(symbol) ? 0 : 1;
            
        } else {
            std::cout << "✗ Invalid command or missing arguments\n\n";
            printUsage();
            return 1;
        }
        
    } catch (const std::exception& e) {
        std::cerr << "✗ Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}