#pragma once

#include "data_loader.hpp"
#include "../models/black_scholes.hpp"
#include <vector>
#include <map>
#include <memory>

namespace volatility {

struct IVPoint {
    double strike;
    double maturity; // Time to expiration in years
    double implied_vol;
    double moneyness; // Strike / Spot
    
    IVPoint(double k, double t, double iv, double m) 
        : strike(k), maturity(t), implied_vol(iv), moneyness(m) {}
};

struct IVSurface {
    std::string symbol;
    std::string date;
    double underlying_price;
    std::vector<double> strikes;
    std::vector<double> maturities;
    std::vector<std::vector<double>> iv_matrix; // [strike_idx][maturity_idx]
    
    IVSurface() : underlying_price(0.0) {}
};

struct VolatilityMetrics {
    double atm_vol;
    double vol_25d_skew;
    double vol_10d_skew;
    double term_structure_slope;
    double convexity;
    
    VolatilityMetrics() : atm_vol(0), vol_25d_skew(0), vol_10d_skew(0), 
                         term_structure_slope(0), convexity(0) {}
};

class VolatilityModel {
public:
    VolatilityModel();
    ~VolatilityModel();
    
    // Build volatility surface from options data
    IVSurface buildIVSurface(const std::vector<OptionData>& options_data, 
                            const std::string& date,
                            double risk_free_rate = 0.05);
    
    // Calculate implied volatilities for all options
    std::vector<IVPoint> calculateImpliedVolatilities(const std::vector<OptionData>& options_data,
                                                      double risk_free_rate = 0.05);
    
    // Interpolate volatility surface
    double interpolateIV(const IVSurface& surface, double strike, double maturity);
    
    // Extract volatility metrics
    VolatilityMetrics extractMetrics(const IVSurface& surface);
    
    // Fit volatility smile for a given maturity
    std::vector<double> fitVolatilitySmile(const std::vector<IVPoint>& iv_points, 
                                          double target_maturity,
                                          double tolerance = 0.01);
    
    // Term structure analysis
    std::vector<double> extractTermStructure(const IVSurface& surface, 
                                           double strike_ratio = 1.0); // ATM by default
    
    // Calculate volatility skew
    double calculateSkew(const std::vector<IVPoint>& iv_points, 
                        double maturity, 
                        double delta_level = 0.25);

private:
    // Helper functions
    double timeToExpiration(const std::string& current_date, 
                          const std::string& expiry_date);
    
    double calculateMoneyness(double strike, double spot);
    
    // Interpolation methods
    double bilinearInterpolation(const std::vector<std::vector<double>>& matrix,
                               const std::vector<double>& x_axis,
                               const std::vector<double>& y_axis,
                               double x, double y);
    
    double linearInterpolation(const std::vector<double>& x, 
                             const std::vector<double>& y, 
                             double target_x);
    
    // Outlier detection and cleaning
    std::vector<IVPoint> cleanIVData(const std::vector<IVPoint>& raw_iv_data);
    
    bool isValidIV(double iv, double strike, double spot, double maturity);
};

} // namespace volatility