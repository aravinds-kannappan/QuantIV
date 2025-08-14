#pragma once

#include "strategy_engine.hpp"
#include <vector>
#include <string>

namespace volatility {

struct PerformanceMetrics {
    double total_return;
    double annualized_return;
    double sharpe_ratio;
    double sortino_ratio;
    double max_drawdown;
    double win_rate;
    double profit_factor;
    double average_win;
    double average_loss;
    double largest_win;
    double largest_loss;
    double calmar_ratio;
    double var_95; // Value at Risk at 95% confidence
    double cvar_95; // Conditional VaR at 95%
    std::vector<double> monthly_returns;
    
    PerformanceMetrics() : total_return(0), annualized_return(0), sharpe_ratio(0), 
                          sortino_ratio(0), max_drawdown(0), win_rate(0), profit_factor(0),
                          average_win(0), average_loss(0), largest_win(0), largest_loss(0),
                          calmar_ratio(0), var_95(0), cvar_95(0) {}
};

struct GreeksProfile {
    std::vector<double> delta_timeseries;
    std::vector<double> gamma_timeseries;
    std::vector<double> vega_timeseries;
    std::vector<double> theta_timeseries;
    
    double avg_delta;
    double max_delta;
    double min_delta;
    double avg_gamma;
    double max_gamma;
    double avg_vega;
    double max_vega;
    double avg_theta;
    double min_theta; // Most negative theta
    
    GreeksProfile() : avg_delta(0), max_delta(0), min_delta(0), avg_gamma(0), 
                     max_gamma(0), avg_vega(0), max_vega(0), avg_theta(0), min_theta(0) {}
};

struct RiskMetrics {
    double volatility_of_returns;
    double skewness;
    double kurtosis;
    double beta; // Relative to underlying
    double information_ratio;
    double tracking_error;
    double max_consecutive_losses;
    double recovery_factor;
    
    RiskMetrics() : volatility_of_returns(0), skewness(0), kurtosis(0), beta(0),
                   information_ratio(0), tracking_error(0), max_consecutive_losses(0),
                   recovery_factor(0) {}
};

class MetricsCalculator {
public:
    MetricsCalculator();
    ~MetricsCalculator();
    
    // Calculate comprehensive performance metrics
    PerformanceMetrics calculatePerformanceMetrics(const StrategyResult& strategy_result);
    
    // Calculate Greeks-based metrics
    GreeksProfile calculateGreeksProfile(const StrategyResult& strategy_result);
    
    // Calculate risk metrics
    RiskMetrics calculateRiskMetrics(const StrategyResult& strategy_result,
                                    const std::vector<double>& benchmark_returns = {});
    
    // Individual metric calculations
    double calculateSharpeRatio(const std::vector<double>& returns, 
                               double risk_free_rate = 0.02);
    
    double calculateSortinoRatio(const std::vector<double>& returns, 
                                double target_return = 0.0);
    
    double calculateMaxDrawdown(const std::vector<double>& equity_curve);
    
    double calculateWinRate(const std::vector<Position>& positions);
    
    double calculateProfitFactor(const std::vector<Position>& positions);
    
    double calculateVaR(const std::vector<double>& returns, double confidence = 0.95);
    
    double calculateCVaR(const std::vector<double>& returns, double confidence = 0.95);
    
    // Utility functions
    std::vector<double> calculateReturns(const std::vector<double>& equity_curve);
    
    std::vector<double> calculateDrawdowns(const std::vector<double>& equity_curve);
    
    std::vector<double> calculateMonthlyReturns(const std::vector<std::string>& dates,
                                               const std::vector<double>& equity_curve);
    
    // Statistical functions
    double calculateMean(const std::vector<double>& data);
    double calculateStdDev(const std::vector<double>& data);
    double calculateSkewness(const std::vector<double>& data);
    double calculateKurtosis(const std::vector<double>& data);
    
    // Benchmark comparison
    double calculateBeta(const std::vector<double>& strategy_returns,
                        const std::vector<double>& benchmark_returns);
    
    double calculateInformationRatio(const std::vector<double>& strategy_returns,
                                    const std::vector<double>& benchmark_returns);

private:
    // Helper functions
    std::vector<double> getPositionPnLs(const std::vector<Position>& positions);
    
    double calculatePositionPnL(const Position& position);
    
    std::vector<double> getNegativeReturns(const std::vector<double>& returns);
    
    std::pair<int, int> getWinLossCount(const std::vector<Position>& positions);
    
    std::vector<double> getConsecutiveLosses(const std::vector<double>& returns);
    
    // Statistical utilities
    double percentile(std::vector<double> data, double p);
    
    std::vector<double> rollingWindow(const std::vector<double>& data, size_t window_size);
};

} // namespace volatility