#pragma once

#include "strategy_engine.hpp"
#include "volatility_model.hpp"
#include "metrics.hpp"
#include <string>
#include <memory>

namespace volatility {

class Exporter {
public:
    Exporter(const std::string& output_dir = "docs/data/");
    ~Exporter();
    
    // Export volatility surface data
    bool exportIVSurface(const IVSurface& surface, const std::string& filename);
    
    // Export strategy backtest results
    bool exportStrategyResults(const StrategyResult& results, 
                              const PerformanceMetrics& metrics,
                              const std::string& filename);
    
    // Export strategy performance summary
    bool exportPerformanceSummary(const std::vector<StrategyResult>& all_results,
                                 const std::vector<PerformanceMetrics>& all_metrics,
                                 const std::string& filename);
    
    // Export CSV timeseries data
    bool exportTimeseriesCSV(const StrategyResult& results, const std::string& filename);
    
    // Export volatility term structure
    bool exportVolTermStructure(const std::vector<IVSurface>& surfaces, 
                               const std::string& filename);
    
    // Export positions data
    bool exportPositions(const std::vector<Position>& positions, const std::string& filename);
    
    // Export Greeks timeseries
    bool exportGreeksTimeseries(const GreeksProfile& greeks_profile,
                               const std::vector<std::string>& dates,
                               const std::string& filename);

private:
    std::string output_directory_;
    
    // JSON export helpers
    std::string generateIVSurfaceJSON(const IVSurface& surface);
    std::string generateStrategyResultJSON(const StrategyResult& results, 
                                          const PerformanceMetrics& metrics);
    std::string generatePerformanceSummaryJSON(const std::vector<StrategyResult>& all_results,
                                              const std::vector<PerformanceMetrics>& all_metrics);
    
    // CSV export helpers
    std::string generateTimeseriesCSV(const StrategyResult& results);
    std::string generateVolTermStructureCSV(const std::vector<IVSurface>& surfaces);
    std::string generatePositionsCSV(const std::vector<Position>& positions);
    std::string generateGreeksCSV(const GreeksProfile& greeks_profile,
                                 const std::vector<std::string>& dates);
    
    // Utility functions
    bool writeToFile(const std::string& filepath, const std::string& content);
    std::string escapeJSON(const std::string& str);
    std::string formatDouble(double value, int precision = 6);
    std::string vectorToJSONArray(const std::vector<double>& vec);
    std::string vectorToJSONArray(const std::vector<std::string>& vec);
    std::string matrixToJSONArray(const std::vector<std::vector<double>>& matrix);
    
    // Path utilities
    std::string getFullPath(const std::string& filename);
    bool createDirectoryIfNotExists(const std::string& dir_path);
};

} // namespace volatility