#include "exporter.hpp"
#include <fstream>
#include <sstream>
#include <iomanip>
#include <iostream>
#include <sys/stat.h>

namespace volatility {

Exporter::Exporter(const std::string& output_dir) : output_directory_(output_dir) {
    createDirectoryIfNotExists(output_directory_);
    createDirectoryIfNotExists(output_directory_ + "iv_surfaces/");
    createDirectoryIfNotExists(output_directory_ + "strategies/");
    createDirectoryIfNotExists(output_directory_ + "metrics/");
    createDirectoryIfNotExists(output_directory_ + "raw/");
}

Exporter::~Exporter() {}

bool Exporter::exportIVSurface(const IVSurface& surface, const std::string& filename) {
    std::string json_content = generateIVSurfaceJSON(surface);
    std::string filepath = getFullPath("iv_surfaces/" + filename);
    return writeToFile(filepath, json_content);
}

bool Exporter::exportStrategyResults(const StrategyResult& results, 
                                    const PerformanceMetrics& metrics,
                                    const std::string& filename) {
    std::string json_content = generateStrategyResultJSON(results, metrics);
    std::string filepath = getFullPath("strategies/" + filename);
    return writeToFile(filepath, json_content);
}

bool Exporter::exportPerformanceSummary(const std::vector<StrategyResult>& all_results,
                                       const std::vector<PerformanceMetrics>& all_metrics,
                                       const std::string& filename) {
    std::string json_content = generatePerformanceSummaryJSON(all_results, all_metrics);
    std::string filepath = getFullPath("metrics/" + filename);
    return writeToFile(filepath, json_content);
}

bool Exporter::exportTimeseriesCSV(const StrategyResult& results, const std::string& filename) {
    std::string csv_content = generateTimeseriesCSV(results);
    std::string filepath = getFullPath("strategies/" + filename);
    return writeToFile(filepath, csv_content);
}

bool Exporter::exportVolTermStructure(const std::vector<IVSurface>& surfaces, 
                                     const std::string& filename) {
    std::string csv_content = generateVolTermStructureCSV(surfaces);
    std::string filepath = getFullPath("raw/" + filename);
    return writeToFile(filepath, csv_content);
}

bool Exporter::exportPositions(const std::vector<Position>& positions, const std::string& filename) {
    std::string csv_content = generatePositionsCSV(positions);
    std::string filepath = getFullPath("strategies/" + filename);
    return writeToFile(filepath, csv_content);
}

bool Exporter::exportGreeksTimeseries(const GreeksProfile& greeks_profile,
                                     const std::vector<std::string>& dates,
                                     const std::string& filename) {
    std::string csv_content = generateGreeksCSV(greeks_profile, dates);
    std::string filepath = getFullPath("strategies/" + filename);
    return writeToFile(filepath, csv_content);
}

std::string Exporter::generateIVSurfaceJSON(const IVSurface& surface) {
    std::ostringstream json;
    
    json << "{\n";
    json << "  \"metadata\": {\n";
    json << "    \"symbol\": \"" << escapeJSON(surface.symbol) << "\",\n";
    json << "    \"date\": \"" << escapeJSON(surface.date) << "\",\n";
    json << "    \"underlying_price\": " << formatDouble(surface.underlying_price) << "\n";
    json << "  },\n";
    json << "  \"surface\": {\n";
    json << "    \"strikes\": " << vectorToJSONArray(surface.strikes) << ",\n";
    json << "    \"maturities\": " << vectorToJSONArray(surface.maturities) << ",\n";
    json << "    \"iv_matrix\": " << matrixToJSONArray(surface.iv_matrix) << "\n";
    json << "  }\n";
    json << "}";
    
    return json.str();
}

std::string Exporter::generateStrategyResultJSON(const StrategyResult& results, 
                                                const PerformanceMetrics& metrics) {
    std::ostringstream json;
    
    json << "{\n";
    json << "  \"strategy\": \"" << escapeJSON(results.strategy_name) << "\",\n";
    json << "  \"symbol\": \"" << escapeJSON(results.symbol) << "\",\n";
    json << "  \"period\": {\n";
    json << "    \"start\": \"" << escapeJSON(results.start_date) << "\",\n";
    json << "    \"end\": \"" << escapeJSON(results.end_date) << "\"\n";
    json << "  },\n";
    json << "  \"metrics\": {\n";
    json << "    \"total_return\": " << formatDouble(metrics.total_return) << ",\n";
    json << "    \"annualized_return\": " << formatDouble(metrics.annualized_return) << ",\n";
    json << "    \"sharpe_ratio\": " << formatDouble(metrics.sharpe_ratio) << ",\n";
    json << "    \"sortino_ratio\": " << formatDouble(metrics.sortino_ratio) << ",\n";
    json << "    \"max_drawdown\": " << formatDouble(metrics.max_drawdown) << ",\n";
    json << "    \"win_rate\": " << formatDouble(metrics.win_rate) << ",\n";
    json << "    \"profit_factor\": " << formatDouble(metrics.profit_factor) << ",\n";
    json << "    \"var_95\": " << formatDouble(metrics.var_95) << ",\n";
    json << "    \"cvar_95\": " << formatDouble(metrics.cvar_95) << "\n";
    json << "  },\n";
    json << "  \"timeseries\": [\n";
    
    for (size_t i = 0; i < results.dates.size(); ++i) {
        json << "    {\n";
        json << "      \"date\": \"" << escapeJSON(results.dates[i]) << "\",\n";
        json << "      \"pnl\": " << formatDouble(i < results.pnl_curve.size() ? results.pnl_curve[i] : 0) << ",\n";
        json << "      \"equity\": " << formatDouble(i < results.equity_curve.size() ? results.equity_curve[i] : 0) << ",\n";
        json << "      \"drawdown\": " << formatDouble(i < results.drawdown_curve.size() ? results.drawdown_curve[i] : 0) << ",\n";
        json << "      \"delta\": " << formatDouble(i < results.delta_exposure.size() ? results.delta_exposure[i] : 0) << ",\n";
        json << "      \"gamma\": " << formatDouble(i < results.gamma_exposure.size() ? results.gamma_exposure[i] : 0) << ",\n";
        json << "      \"vega\": " << formatDouble(i < results.vega_exposure.size() ? results.vega_exposure[i] : 0) << ",\n";
        json << "      \"theta\": " << formatDouble(i < results.theta_exposure.size() ? results.theta_exposure[i] : 0) << "\n";
        json << "    }";
        
        if (i < results.dates.size() - 1) {
            json << ",";
        }
        json << "\n";
    }
    
    json << "  ],\n";
    json << "  \"total_trades\": " << results.total_trades << ",\n";
    json << "  \"winning_trades\": " << results.winning_trades << "\n";
    json << "}";
    
    return json.str();
}

std::string Exporter::generatePerformanceSummaryJSON(const std::vector<StrategyResult>& all_results,
                                                    const std::vector<PerformanceMetrics>& all_metrics) {
    std::ostringstream json;
    
    json << "{\n";
    json << "  \"summary\": {\n";
    json << "    \"total_strategies\": " << all_results.size() << ",\n";
    json << "    \"generation_date\": \"" << "2025-01-01" << "\"\n"; // Should use actual date
    json << "  },\n";
    json << "  \"strategies\": [\n";
    
    for (size_t i = 0; i < all_results.size() && i < all_metrics.size(); ++i) {
        const auto& result = all_results[i];
        const auto& metrics = all_metrics[i];
        
        json << "    {\n";
        json << "      \"name\": \"" << escapeJSON(result.strategy_name) << "\",\n";
        json << "      \"symbol\": \"" << escapeJSON(result.symbol) << "\",\n";
        json << "      \"total_return\": " << formatDouble(metrics.total_return) << ",\n";
        json << "      \"sharpe_ratio\": " << formatDouble(metrics.sharpe_ratio) << ",\n";
        json << "      \"max_drawdown\": " << formatDouble(metrics.max_drawdown) << ",\n";
        json << "      \"win_rate\": " << formatDouble(metrics.win_rate) << ",\n";
        json << "      \"total_trades\": " << result.total_trades << "\n";
        json << "    }";
        
        if (i < all_results.size() - 1) {
            json << ",";
        }
        json << "\n";
    }
    
    json << "  ]\n";
    json << "}";
    
    return json.str();
}

std::string Exporter::generateTimeseriesCSV(const StrategyResult& results) {
    std::ostringstream csv;
    
    csv << "date,pnl,equity,drawdown,delta,gamma,vega,theta\n";
    
    for (size_t i = 0; i < results.dates.size(); ++i) {
        csv << results.dates[i] << ",";
        csv << (i < results.pnl_curve.size() ? formatDouble(results.pnl_curve[i]) : "0") << ",";
        csv << (i < results.equity_curve.size() ? formatDouble(results.equity_curve[i]) : "0") << ",";
        csv << (i < results.drawdown_curve.size() ? formatDouble(results.drawdown_curve[i]) : "0") << ",";
        csv << (i < results.delta_exposure.size() ? formatDouble(results.delta_exposure[i]) : "0") << ",";
        csv << (i < results.gamma_exposure.size() ? formatDouble(results.gamma_exposure[i]) : "0") << ",";
        csv << (i < results.vega_exposure.size() ? formatDouble(results.vega_exposure[i]) : "0") << ",";
        csv << (i < results.theta_exposure.size() ? formatDouble(results.theta_exposure[i]) : "0") << "\n";
    }
    
    return csv.str();
}

std::string Exporter::generateVolTermStructureCSV(const std::vector<IVSurface>& surfaces) {
    std::ostringstream csv;
    
    csv << "date,symbol,maturity_days,atm_strike,atm_iv,skew_25d,convexity\n";
    
    for (const auto& surface : surfaces) {
        for (size_t i = 0; i < surface.maturities.size(); ++i) {
            double maturity_days = surface.maturities[i] * 365;
            
            // Find ATM strike index (closest to underlying price)
            size_t atm_idx = 0;
            double min_diff = std::abs(surface.strikes[0] - surface.underlying_price);
            for (size_t j = 1; j < surface.strikes.size(); ++j) {
                double diff = std::abs(surface.strikes[j] - surface.underlying_price);
                if (diff < min_diff) {
                    min_diff = diff;
                    atm_idx = j;
                }
            }
            
            double atm_iv = (atm_idx < surface.iv_matrix.size() && i < surface.iv_matrix[atm_idx].size()) 
                           ? surface.iv_matrix[atm_idx][i] : 0.0;
            
            csv << surface.date << ",";
            csv << surface.symbol << ",";
            csv << formatDouble(maturity_days, 0) << ",";
            csv << formatDouble(surface.strikes[atm_idx]) << ",";
            csv << formatDouble(atm_iv) << ",";
            csv << "0.02,"; // Placeholder for skew
            csv << "0.01\n"; // Placeholder for convexity
        }
    }
    
    return csv.str();
}

std::string Exporter::generatePositionsCSV(const std::vector<Position>& positions) {
    std::ostringstream csv;
    
    csv << "symbol,strike,option_type,quantity,entry_price,entry_date,expiry_date,is_open\n";
    
    for (const auto& pos : positions) {
        csv << pos.symbol << ",";
        csv << formatDouble(pos.strike) << ",";
        csv << pos.option_type << ",";
        csv << pos.quantity << ",";
        csv << formatDouble(pos.entry_price) << ",";
        csv << pos.entry_date << ",";
        csv << pos.expiry_date << ",";
        csv << (pos.is_open ? "true" : "false") << "\n";
    }
    
    return csv.str();
}

std::string Exporter::generateGreeksCSV(const GreeksProfile& greeks_profile,
                                       const std::vector<std::string>& dates) {
    std::ostringstream csv;
    
    csv << "date,delta,gamma,vega,theta\n";
    
    size_t min_size = std::min({dates.size(), 
                               greeks_profile.delta_timeseries.size(),
                               greeks_profile.gamma_timeseries.size(),
                               greeks_profile.vega_timeseries.size(),
                               greeks_profile.theta_timeseries.size()});
    
    for (size_t i = 0; i < min_size; ++i) {
        csv << dates[i] << ",";
        csv << formatDouble(greeks_profile.delta_timeseries[i]) << ",";
        csv << formatDouble(greeks_profile.gamma_timeseries[i]) << ",";
        csv << formatDouble(greeks_profile.vega_timeseries[i]) << ",";
        csv << formatDouble(greeks_profile.theta_timeseries[i]) << "\n";
    }
    
    return csv.str();
}

bool Exporter::writeToFile(const std::string& filepath, const std::string& content) {
    std::ofstream file(filepath);
    
    if (!file.is_open()) {
        std::cerr << "Error: Cannot create file " << filepath << std::endl;
        return false;
    }
    
    file << content;
    file.close();
    
    std::cout << "Exported: " << filepath << std::endl;
    return true;
}

std::string Exporter::escapeJSON(const std::string& str) {
    std::string escaped;
    for (char c : str) {
        switch (c) {
            case '\"': escaped += "\\\""; break;
            case '\\': escaped += "\\\\"; break;
            case '\n': escaped += "\\n"; break;
            case '\r': escaped += "\\r"; break;
            case '\t': escaped += "\\t"; break;
            default: escaped += c; break;
        }
    }
    return escaped;
}

std::string Exporter::formatDouble(double value, int precision) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(precision) << value;
    return oss.str();
}

std::string Exporter::vectorToJSONArray(const std::vector<double>& vec) {
    std::ostringstream json;
    json << "[";
    
    for (size_t i = 0; i < vec.size(); ++i) {
        json << formatDouble(vec[i]);
        if (i < vec.size() - 1) {
            json << ", ";
        }
    }
    
    json << "]";
    return json.str();
}

std::string Exporter::vectorToJSONArray(const std::vector<std::string>& vec) {
    std::ostringstream json;
    json << "[";
    
    for (size_t i = 0; i < vec.size(); ++i) {
        json << "\"" << escapeJSON(vec[i]) << "\"";
        if (i < vec.size() - 1) {
            json << ", ";
        }
    }
    
    json << "]";
    return json.str();
}

std::string Exporter::matrixToJSONArray(const std::vector<std::vector<double>>& matrix) {
    std::ostringstream json;
    json << "[";
    
    for (size_t i = 0; i < matrix.size(); ++i) {
        json << vectorToJSONArray(matrix[i]);
        if (i < matrix.size() - 1) {
            json << ",\n      ";
        }
    }
    
    json << "]";
    return json.str();
}

std::string Exporter::getFullPath(const std::string& filename) {
    return output_directory_ + filename;
}

bool Exporter::createDirectoryIfNotExists(const std::string& dir_path) {
    struct stat info;
    
    if (stat(dir_path.c_str(), &info) != 0) {
        // Directory doesn't exist, create it
        return mkdir(dir_path.c_str(), 0755) == 0;
    } else if (info.st_mode & S_IFDIR) {
        // Directory already exists
        return true;
    } else {
        // Path exists but is not a directory
        return false;
    }
}

} // namespace volatility