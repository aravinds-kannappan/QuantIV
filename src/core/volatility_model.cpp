#include "volatility_model.hpp"
#include "../utils/date_utils.hpp"
#include <algorithm>
#include <set>
#include <cmath>
#include <iostream>

namespace volatility {

VolatilityModel::VolatilityModel() {}

VolatilityModel::~VolatilityModel() {}

IVSurface VolatilityModel::buildIVSurface(const std::vector<OptionData>& options_data, 
                                         const std::string& date,
                                         double risk_free_rate) {
    IVSurface surface;
    surface.symbol = options_data.empty() ? "" : options_data[0].symbol;
    surface.date = date;
    surface.underlying_price = options_data.empty() ? 0.0 : options_data[0].underlying_price;
    
    // Calculate all IV points
    auto iv_points = calculateImpliedVolatilities(options_data, risk_free_rate);
    iv_points = cleanIVData(iv_points);
    
    if (iv_points.empty()) {
        return surface;
    }
    
    // Extract unique strikes and maturities
    std::set<double> unique_strikes;
    std::set<double> unique_maturities;
    
    for (const auto& point : iv_points) {
        unique_strikes.insert(point.strike);
        unique_maturities.insert(point.maturity);
    }
    
    surface.strikes.assign(unique_strikes.begin(), unique_strikes.end());
    surface.maturities.assign(unique_maturities.begin(), unique_maturities.end());
    
    // Initialize matrix
    surface.iv_matrix.resize(surface.strikes.size());
    for (auto& row : surface.iv_matrix) {
        row.resize(surface.maturities.size(), 0.0);
    }
    
    // Fill matrix with IV values
    for (const auto& point : iv_points) {
        auto strike_it = std::find(surface.strikes.begin(), surface.strikes.end(), point.strike);
        auto maturity_it = std::find(surface.maturities.begin(), surface.maturities.end(), point.maturity);
        
        if (strike_it != surface.strikes.end() && maturity_it != surface.maturities.end()) {
            int strike_idx = std::distance(surface.strikes.begin(), strike_it);
            int maturity_idx = std::distance(surface.maturities.begin(), maturity_it);
            surface.iv_matrix[strike_idx][maturity_idx] = point.implied_vol;
        }
    }
    
    return surface;
}

std::vector<IVPoint> VolatilityModel::calculateImpliedVolatilities(const std::vector<OptionData>& options_data,
                                                                  double risk_free_rate) {
    std::vector<IVPoint> iv_points;
    
    for (const auto& option : options_data) {
        if (option.mid_price <= 0 || option.underlying_price <= 0) continue;
        
        double T = timeToExpiration(option.date, option.expiry);
        if (T <= 0) continue;
        
        double iv = BlackScholes::impliedVolatility(
            option.underlying_price,
            option.strike,
            T,
            risk_free_rate,
            option.mid_price,
            option.option_type
        );
        
        if (iv > 0 && isValidIV(iv, option.strike, option.underlying_price, T)) {
            double moneyness = calculateMoneyness(option.strike, option.underlying_price);
            iv_points.emplace_back(option.strike, T, iv, moneyness);
        }
    }
    
    return iv_points;
}

double VolatilityModel::interpolateIV(const IVSurface& surface, double strike, double maturity) {
    if (surface.strikes.empty() || surface.maturities.empty()) {
        return 0.0;
    }
    
    return bilinearInterpolation(surface.iv_matrix, surface.strikes, surface.maturities, strike, maturity);
}

VolatilityMetrics VolatilityModel::extractMetrics(const IVSurface& surface) {
    VolatilityMetrics metrics;
    
    if (surface.strikes.empty() || surface.maturities.empty()) {
        return metrics;
    }
    
    // Find ATM volatility (closest to current price)
    double min_diff = std::numeric_limits<double>::max();
    int atm_strike_idx = 0;
    
    for (size_t i = 0; i < surface.strikes.size(); ++i) {
        double diff = std::abs(surface.strikes[i] - surface.underlying_price);
        if (diff < min_diff) {
            min_diff = diff;
            atm_strike_idx = i;
        }
    }
    
    // ATM vol (30-day if available)
    for (size_t j = 0; j < surface.maturities.size(); ++j) {
        if (surface.maturities[j] >= 0.08 && surface.maturities[j] <= 0.12) { // ~30 days
            metrics.atm_vol = surface.iv_matrix[atm_strike_idx][j];
            break;
        }
    }
    
    // Term structure slope (short vs long term)
    if (surface.maturities.size() >= 2) {
        double short_term_vol = surface.iv_matrix[atm_strike_idx][0];
        double long_term_vol = surface.iv_matrix[atm_strike_idx][surface.maturities.size() - 1];
        metrics.term_structure_slope = (long_term_vol - short_term_vol) / 
                                     (surface.maturities.back() - surface.maturities.front());
    }
    
    return metrics;
}

std::vector<double> VolatilityModel::fitVolatilitySmile(const std::vector<IVPoint>& iv_points, 
                                                       double target_maturity,
                                                       double tolerance) {
    std::vector<double> smile_points;
    
    // Filter points near target maturity
    std::vector<IVPoint> filtered_points;
    for (const auto& point : iv_points) {
        if (std::abs(point.maturity - target_maturity) <= tolerance) {
            filtered_points.push_back(point);
        }
    }
    
    // Sort by strike
    std::sort(filtered_points.begin(), filtered_points.end(),
              [](const IVPoint& a, const IVPoint& b) {
                  return a.strike < b.strike;
              });
    
    // Extract IV values
    for (const auto& point : filtered_points) {
        smile_points.push_back(point.implied_vol);
    }
    
    return smile_points;
}

std::vector<double> VolatilityModel::extractTermStructure(const IVSurface& surface, double strike_ratio) {
    std::vector<double> term_structure;
    
    if (surface.strikes.empty() || surface.maturities.empty()) {
        return term_structure;
    }
    
    // Find strike closest to target ratio
    double target_strike = surface.underlying_price * strike_ratio;
    int strike_idx = 0;
    double min_diff = std::numeric_limits<double>::max();
    
    for (size_t i = 0; i < surface.strikes.size(); ++i) {
        double diff = std::abs(surface.strikes[i] - target_strike);
        if (diff < min_diff) {
            min_diff = diff;
            strike_idx = i;
        }
    }
    
    // Extract volatilities across maturities
    for (size_t j = 0; j < surface.maturities.size(); ++j) {
        term_structure.push_back(surface.iv_matrix[strike_idx][j]);
    }
    
    return term_structure;
}

double VolatilityModel::calculateSkew(const std::vector<IVPoint>& iv_points, 
                                     double maturity, 
                                     double delta_level) {
    // Simplified skew calculation as difference between OTM put and call IVs
    // In practice, would use delta-based selection
    
    std::vector<IVPoint> filtered_points;
    for (const auto& point : iv_points) {
        if (std::abs(point.maturity - maturity) <= 0.01) {
            filtered_points.push_back(point);
        }
    }
    
    if (filtered_points.size() < 2) return 0.0;
    
    // Sort by moneyness
    std::sort(filtered_points.begin(), filtered_points.end(),
              [](const IVPoint& a, const IVPoint& b) {
                  return a.moneyness < b.moneyness;
              });
    
    // Calculate skew as IV difference between extremes
    double low_strike_iv = filtered_points.front().implied_vol;
    double high_strike_iv = filtered_points.back().implied_vol;
    
    return low_strike_iv - high_strike_iv;
}

double VolatilityModel::timeToExpiration(const std::string& current_date, 
                                       const std::string& expiry_date) {
    // Simple implementation - in practice would use proper date library
    // Assuming YYYY-MM-DD format
    return 0.25; // Default to 3 months for now
}

double VolatilityModel::calculateMoneyness(double strike, double spot) {
    return spot > 0 ? strike / spot : 1.0;
}

double VolatilityModel::bilinearInterpolation(const std::vector<std::vector<double>>& matrix,
                                            const std::vector<double>& x_axis,
                                            const std::vector<double>& y_axis,
                                            double x, double y) {
    if (matrix.empty() || x_axis.empty() || y_axis.empty()) return 0.0;
    
    // Find bounding indices
    auto x_it = std::lower_bound(x_axis.begin(), x_axis.end(), x);
    auto y_it = std::lower_bound(y_axis.begin(), y_axis.end(), y);
    
    if (x_it == x_axis.begin()) x_it++;
    if (y_it == y_axis.begin()) y_it++;
    if (x_it == x_axis.end()) x_it--;
    if (y_it == y_axis.end()) y_it--;
    
    int x1_idx = std::distance(x_axis.begin(), x_it) - 1;
    int x2_idx = std::distance(x_axis.begin(), x_it);
    int y1_idx = std::distance(y_axis.begin(), y_it) - 1;
    int y2_idx = std::distance(y_axis.begin(), y_it);
    
    double x1 = x_axis[x1_idx], x2 = x_axis[x2_idx];
    double y1 = y_axis[y1_idx], y2 = y_axis[y2_idx];
    
    double q11 = matrix[x1_idx][y1_idx];
    double q12 = matrix[x1_idx][y2_idx];
    double q21 = matrix[x2_idx][y1_idx];
    double q22 = matrix[x2_idx][y2_idx];
    
    double r1 = ((x2 - x) / (x2 - x1)) * q11 + ((x - x1) / (x2 - x1)) * q21;
    double r2 = ((x2 - x) / (x2 - x1)) * q12 + ((x - x1) / (x2 - x1)) * q22;
    
    return ((y2 - y) / (y2 - y1)) * r1 + ((y - y1) / (y2 - y1)) * r2;
}

double VolatilityModel::linearInterpolation(const std::vector<double>& x, 
                                          const std::vector<double>& y, 
                                          double target_x) {
    if (x.size() != y.size() || x.empty()) return 0.0;
    
    auto it = std::lower_bound(x.begin(), x.end(), target_x);
    
    if (it == x.begin()) return y[0];
    if (it == x.end()) return y.back();
    
    int idx = std::distance(x.begin(), it);
    double x1 = x[idx - 1], x2 = x[idx];
    double y1 = y[idx - 1], y2 = y[idx];
    
    return y1 + (y2 - y1) * (target_x - x1) / (x2 - x1);
}

std::vector<IVPoint> VolatilityModel::cleanIVData(const std::vector<IVPoint>& raw_iv_data) {
    std::vector<IVPoint> clean_data;
    
    for (const auto& point : raw_iv_data) {
        if (isValidIV(point.implied_vol, point.strike, 
                     point.strike / point.moneyness, point.maturity)) {
            clean_data.push_back(point);
        }
    }
    
    return clean_data;
}

bool VolatilityModel::isValidIV(double iv, double strike, double spot, double maturity) {
    return iv > 0.01 && iv < 5.0 &&  // Reasonable IV range
           strike > 0 && spot > 0 && maturity > 0 &&
           strike / spot > 0.5 && strike / spot < 2.0;  // Reasonable moneyness range
}

} // namespace volatility