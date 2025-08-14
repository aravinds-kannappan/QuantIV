#include "metrics.hpp"
#include <algorithm>
#include <numeric>
#include <cmath>
#include <iostream>

namespace volatility {

MetricsCalculator::MetricsCalculator() {}

MetricsCalculator::~MetricsCalculator() {}

PerformanceMetrics MetricsCalculator::calculatePerformanceMetrics(const StrategyResult& strategy_result) {
    PerformanceMetrics metrics;
    
    if (strategy_result.equity_curve.empty()) {
        return metrics;
    }
    
    // Calculate returns
    auto returns = calculateReturns(strategy_result.equity_curve);
    
    // Total return
    metrics.total_return = (strategy_result.equity_curve.back() - strategy_result.equity_curve.front()) 
                          / strategy_result.equity_curve.front();
    
    // Annualized return (assuming daily data)
    double years = strategy_result.dates.size() / 252.0;
    if (years > 0) {
        metrics.annualized_return = std::pow(1 + metrics.total_return, 1.0 / years) - 1.0;
    }
    
    // Sharpe ratio
    metrics.sharpe_ratio = calculateSharpeRatio(returns);
    
    // Sortino ratio
    metrics.sortino_ratio = calculateSortinoRatio(returns);
    
    // Max drawdown
    metrics.max_drawdown = calculateMaxDrawdown(strategy_result.equity_curve);
    
    // Win rate
    metrics.win_rate = calculateWinRate(strategy_result.all_positions);
    
    // Profit factor
    metrics.profit_factor = calculateProfitFactor(strategy_result.all_positions);
    
    // VaR and CVaR
    metrics.var_95 = calculateVaR(returns, 0.95);
    metrics.cvar_95 = calculateCVaR(returns, 0.95);
    
    // Calmar ratio
    if (metrics.max_drawdown != 0) {
        metrics.calmar_ratio = metrics.annualized_return / std::abs(metrics.max_drawdown);
    }
    
    // Win/Loss statistics
    auto position_pnls = getPositionPnLs(strategy_result.all_positions);
    std::vector<double> wins, losses;
    
    for (double pnl : position_pnls) {
        if (pnl > 0) {
            wins.push_back(pnl);
        } else if (pnl < 0) {
            losses.push_back(pnl);
        }
    }
    
    if (!wins.empty()) {
        metrics.average_win = calculateMean(wins);
        metrics.largest_win = *std::max_element(wins.begin(), wins.end());
    }
    
    if (!losses.empty()) {
        metrics.average_loss = calculateMean(losses);
        metrics.largest_loss = *std::min_element(losses.begin(), losses.end());
    }
    
    // Monthly returns
    metrics.monthly_returns = calculateMonthlyReturns(strategy_result.dates, strategy_result.equity_curve);
    
    return metrics;
}

GreeksProfile MetricsCalculator::calculateGreeksProfile(const StrategyResult& strategy_result) {
    GreeksProfile profile;
    
    profile.delta_timeseries = strategy_result.delta_exposure;
    profile.gamma_timeseries = strategy_result.gamma_exposure;
    profile.vega_timeseries = strategy_result.vega_exposure;
    profile.theta_timeseries = strategy_result.theta_exposure;
    
    if (!profile.delta_timeseries.empty()) {
        profile.avg_delta = calculateMean(profile.delta_timeseries);
        profile.max_delta = *std::max_element(profile.delta_timeseries.begin(), profile.delta_timeseries.end());
        profile.min_delta = *std::min_element(profile.delta_timeseries.begin(), profile.delta_timeseries.end());
    }
    
    if (!profile.gamma_timeseries.empty()) {
        profile.avg_gamma = calculateMean(profile.gamma_timeseries);
        profile.max_gamma = *std::max_element(profile.gamma_timeseries.begin(), profile.gamma_timeseries.end());
    }
    
    if (!profile.vega_timeseries.empty()) {
        profile.avg_vega = calculateMean(profile.vega_timeseries);
        profile.max_vega = *std::max_element(profile.vega_timeseries.begin(), profile.vega_timeseries.end());
    }
    
    if (!profile.theta_timeseries.empty()) {
        profile.avg_theta = calculateMean(profile.theta_timeseries);
        profile.min_theta = *std::min_element(profile.theta_timeseries.begin(), profile.theta_timeseries.end());
    }
    
    return profile;
}

RiskMetrics MetricsCalculator::calculateRiskMetrics(const StrategyResult& strategy_result,
                                                   const std::vector<double>& benchmark_returns) {
    RiskMetrics risk_metrics;
    
    auto returns = calculateReturns(strategy_result.equity_curve);
    
    if (returns.empty()) {
        return risk_metrics;
    }
    
    // Volatility of returns
    risk_metrics.volatility_of_returns = calculateStdDev(returns) * std::sqrt(252); // Annualized
    
    // Skewness and kurtosis
    risk_metrics.skewness = calculateSkewness(returns);
    risk_metrics.kurtosis = calculateKurtosis(returns);
    
    // Beta and information ratio (if benchmark provided)
    if (!benchmark_returns.empty() && benchmark_returns.size() == returns.size()) {
        risk_metrics.beta = calculateBeta(returns, benchmark_returns);
        risk_metrics.information_ratio = calculateInformationRatio(returns, benchmark_returns);
        
        // Tracking error
        std::vector<double> excess_returns;
        for (size_t i = 0; i < returns.size(); ++i) {
            excess_returns.push_back(returns[i] - benchmark_returns[i]);
        }
        risk_metrics.tracking_error = calculateStdDev(excess_returns) * std::sqrt(252);
    }
    
    // Max consecutive losses
    auto consecutive_losses = getConsecutiveLosses(returns);
    if (!consecutive_losses.empty()) {
        risk_metrics.max_consecutive_losses = *std::max_element(consecutive_losses.begin(), consecutive_losses.end());
    }
    
    // Recovery factor
    if (risk_metrics.volatility_of_returns != 0) {
        risk_metrics.recovery_factor = std::abs(calculateMaxDrawdown(strategy_result.equity_curve)) / 
                                      risk_metrics.volatility_of_returns;
    }
    
    return risk_metrics;
}

double MetricsCalculator::calculateSharpeRatio(const std::vector<double>& returns, double risk_free_rate) {
    if (returns.empty()) return 0.0;
    
    double mean_return = calculateMean(returns);
    double std_dev = calculateStdDev(returns);
    
    if (std_dev == 0) return 0.0;
    
    double daily_rf_rate = risk_free_rate / 252.0;
    return (mean_return - daily_rf_rate) * std::sqrt(252) / (std_dev * std::sqrt(252));
}

double MetricsCalculator::calculateSortinoRatio(const std::vector<double>& returns, double target_return) {
    if (returns.empty()) return 0.0;
    
    double mean_return = calculateMean(returns);
    auto negative_returns = getNegativeReturns(returns);
    
    if (negative_returns.empty()) return 0.0;
    
    double downside_deviation = calculateStdDev(negative_returns);
    
    if (downside_deviation == 0) return 0.0;
    
    return (mean_return - target_return / 252.0) * std::sqrt(252) / (downside_deviation * std::sqrt(252));
}

double MetricsCalculator::calculateMaxDrawdown(const std::vector<double>& equity_curve) {
    if (equity_curve.empty()) return 0.0;
    
    double peak = equity_curve[0];
    double max_dd = 0.0;
    
    for (double value : equity_curve) {
        if (value > peak) {
            peak = value;
        }
        double drawdown = (peak - value) / peak;
        max_dd = std::max(max_dd, drawdown);
    }
    
    return max_dd;
}

double MetricsCalculator::calculateWinRate(const std::vector<Position>& positions) {
    auto win_loss = getWinLossCount(positions);
    int total_trades = win_loss.first + win_loss.second;
    
    return total_trades > 0 ? static_cast<double>(win_loss.first) / total_trades : 0.0;
}

double MetricsCalculator::calculateProfitFactor(const std::vector<Position>& positions) {
    auto position_pnls = getPositionPnLs(positions);
    
    double total_profit = 0.0;
    double total_loss = 0.0;
    
    for (double pnl : position_pnls) {
        if (pnl > 0) {
            total_profit += pnl;
        } else if (pnl < 0) {
            total_loss += std::abs(pnl);
        }
    }
    
    return total_loss > 0 ? total_profit / total_loss : 0.0;
}

double MetricsCalculator::calculateVaR(const std::vector<double>& returns, double confidence) {
    if (returns.empty()) return 0.0;
    
    std::vector<double> sorted_returns = returns;
    std::sort(sorted_returns.begin(), sorted_returns.end());
    
    return percentile(sorted_returns, 1.0 - confidence);
}

double MetricsCalculator::calculateCVaR(const std::vector<double>& returns, double confidence) {
    if (returns.empty()) return 0.0;
    
    std::vector<double> sorted_returns = returns;
    std::sort(sorted_returns.begin(), sorted_returns.end());
    
    double var = percentile(sorted_returns, 1.0 - confidence);
    
    // Calculate mean of returns below VaR
    std::vector<double> tail_returns;
    for (double ret : sorted_returns) {
        if (ret <= var) {
            tail_returns.push_back(ret);
        }
    }
    
    return tail_returns.empty() ? 0.0 : calculateMean(tail_returns);
}

std::vector<double> MetricsCalculator::calculateReturns(const std::vector<double>& equity_curve) {
    std::vector<double> returns;
    
    for (size_t i = 1; i < equity_curve.size(); ++i) {
        if (equity_curve[i-1] != 0) {
            double ret = (equity_curve[i] - equity_curve[i-1]) / equity_curve[i-1];
            returns.push_back(ret);
        }
    }
    
    return returns;
}

std::vector<double> MetricsCalculator::calculateDrawdowns(const std::vector<double>& equity_curve) {
    std::vector<double> drawdowns;
    
    if (equity_curve.empty()) return drawdowns;
    
    double peak = equity_curve[0];
    
    for (double value : equity_curve) {
        if (value > peak) {
            peak = value;
        }
        double drawdown = (peak - value) / peak;
        drawdowns.push_back(drawdown);
    }
    
    return drawdowns;
}

std::vector<double> MetricsCalculator::calculateMonthlyReturns(const std::vector<std::string>& dates,
                                                              const std::vector<double>& equity_curve) {
    std::vector<double> monthly_returns;
    
    // Simplified implementation - should group by actual months
    if (dates.size() >= 21 && equity_curve.size() >= 21) { // Assume ~21 trading days per month
        for (size_t i = 21; i < equity_curve.size(); i += 21) {
            double monthly_ret = (equity_curve[i] - equity_curve[i-21]) / equity_curve[i-21];
            monthly_returns.push_back(monthly_ret);
        }
    }
    
    return monthly_returns;
}

double MetricsCalculator::calculateMean(const std::vector<double>& data) {
    if (data.empty()) return 0.0;
    return std::accumulate(data.begin(), data.end(), 0.0) / data.size();
}

double MetricsCalculator::calculateStdDev(const std::vector<double>& data) {
    if (data.size() <= 1) return 0.0;
    
    double mean = calculateMean(data);
    double variance = 0.0;
    
    for (double value : data) {
        variance += (value - mean) * (value - mean);
    }
    
    variance /= (data.size() - 1);
    return std::sqrt(variance);
}

double MetricsCalculator::calculateSkewness(const std::vector<double>& data) {
    if (data.size() < 3) return 0.0;
    
    double mean = calculateMean(data);
    double std_dev = calculateStdDev(data);
    
    if (std_dev == 0) return 0.0;
    
    double skewness = 0.0;
    for (double value : data) {
        skewness += std::pow((value - mean) / std_dev, 3);
    }
    
    return skewness / data.size();
}

double MetricsCalculator::calculateKurtosis(const std::vector<double>& data) {
    if (data.size() < 4) return 0.0;
    
    double mean = calculateMean(data);
    double std_dev = calculateStdDev(data);
    
    if (std_dev == 0) return 0.0;
    
    double kurtosis = 0.0;
    for (double value : data) {
        kurtosis += std::pow((value - mean) / std_dev, 4);
    }
    
    return (kurtosis / data.size()) - 3.0; // Excess kurtosis
}

double MetricsCalculator::calculateBeta(const std::vector<double>& strategy_returns,
                                       const std::vector<double>& benchmark_returns) {
    if (strategy_returns.size() != benchmark_returns.size() || strategy_returns.empty()) {
        return 0.0;
    }
    
    double strategy_mean = calculateMean(strategy_returns);
    double benchmark_mean = calculateMean(benchmark_returns);
    
    double covariance = 0.0;
    double benchmark_variance = 0.0;
    
    for (size_t i = 0; i < strategy_returns.size(); ++i) {
        covariance += (strategy_returns[i] - strategy_mean) * (benchmark_returns[i] - benchmark_mean);
        benchmark_variance += (benchmark_returns[i] - benchmark_mean) * (benchmark_returns[i] - benchmark_mean);
    }
    
    if (benchmark_variance == 0) return 0.0;
    
    return covariance / benchmark_variance;
}

double MetricsCalculator::calculateInformationRatio(const std::vector<double>& strategy_returns,
                                                   const std::vector<double>& benchmark_returns) {
    if (strategy_returns.size() != benchmark_returns.size() || strategy_returns.empty()) {
        return 0.0;
    }
    
    std::vector<double> excess_returns;
    for (size_t i = 0; i < strategy_returns.size(); ++i) {
        excess_returns.push_back(strategy_returns[i] - benchmark_returns[i]);
    }
    
    double mean_excess = calculateMean(excess_returns);
    double tracking_error = calculateStdDev(excess_returns);
    
    return tracking_error > 0 ? mean_excess / tracking_error : 0.0;
}

std::vector<double> MetricsCalculator::getPositionPnLs(const std::vector<Position>& positions) {
    std::vector<double> pnls;
    
    for (const auto& pos : positions) {
        if (!pos.is_open) { // Only closed positions
            double pnl = calculatePositionPnL(pos);
            pnls.push_back(pnl);
        }
    }
    
    return pnls;
}

double MetricsCalculator::calculatePositionPnL(const Position& position) {
    // Simplified - in real implementation would need exit price
    return 0.0; // Placeholder
}

std::vector<double> MetricsCalculator::getNegativeReturns(const std::vector<double>& returns) {
    std::vector<double> negative_returns;
    
    for (double ret : returns) {
        if (ret < 0) {
            negative_returns.push_back(ret);
        }
    }
    
    return negative_returns;
}

std::pair<int, int> MetricsCalculator::getWinLossCount(const std::vector<Position>& positions) {
    int wins = 0, losses = 0;
    
    auto pnls = getPositionPnLs(positions);
    for (double pnl : pnls) {
        if (pnl > 0) wins++;
        else if (pnl < 0) losses++;
    }
    
    return {wins, losses};
}

std::vector<double> MetricsCalculator::getConsecutiveLosses(const std::vector<double>& returns) {
    std::vector<double> consecutive_losses;
    int current_streak = 0;
    
    for (double ret : returns) {
        if (ret < 0) {
            current_streak++;
        } else {
            if (current_streak > 0) {
                consecutive_losses.push_back(current_streak);
                current_streak = 0;
            }
        }
    }
    
    if (current_streak > 0) {
        consecutive_losses.push_back(current_streak);
    }
    
    return consecutive_losses;
}

double MetricsCalculator::percentile(std::vector<double> data, double p) {
    if (data.empty()) return 0.0;
    
    std::sort(data.begin(), data.end());
    
    if (p <= 0) return data.front();
    if (p >= 1) return data.back();
    
    double index = p * (data.size() - 1);
    int lower_index = static_cast<int>(std::floor(index));
    int upper_index = static_cast<int>(std::ceil(index));
    
    if (lower_index == upper_index) {
        return data[lower_index];
    }
    
    double weight = index - lower_index;
    return data[lower_index] * (1 - weight) + data[upper_index] * weight;
}

} // namespace volatility