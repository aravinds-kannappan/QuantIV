#include "strategy_engine.hpp"
#include "../utils/date_utils.hpp"
#include <algorithm>
#include <numeric>
#include <cmath>
#include <iostream>

namespace volatility {

StrategyEngine::StrategyEngine() {}

StrategyEngine::~StrategyEngine() {}

StrategyResult StrategyEngine::backtestStrategy(StrategyType type,
                                               const std::vector<OptionData>& historical_data,
                                               const StrategyParams& params) {
    switch (type) {
        case StrategyType::COVERED_CALL:
            return backtestCoveredCall(historical_data, params);
        case StrategyType::VERTICAL_SPREAD:
            return backtestVerticalSpread(historical_data, params);
        case StrategyType::STRADDLE:
            return backtestStraddle(historical_data, params);
        default:
            return StrategyResult();
    }
}

StrategyResult StrategyEngine::backtestCoveredCall(const std::vector<OptionData>& data, 
                                                  const StrategyParams& params) {
    StrategyResult result;
    result.strategy_name = "Covered Call";
    
    if (data.empty()) return result;
    
    result.symbol = data[0].symbol;
    auto unique_dates = getUniqueDates(data);
    std::sort(unique_dates.begin(), unique_dates.end());
    
    if (unique_dates.empty()) return result;
    
    result.start_date = unique_dates.front();
    result.end_date = unique_dates.back();
    
    std::vector<Position> all_positions;
    double current_equity = params.initial_capital;
    double peak_equity = current_equity;
    
    for (const auto& date : unique_dates) {
        auto daily_data = filterByDate(data, date);
        if (daily_data.empty()) continue;
        
        double underlying_price = daily_data[0].underlying_price;
        auto open_positions = getOpenPositions(all_positions);
        
        // Check exit conditions
        auto positions_to_close = checkExitConditions(open_positions, daily_data, params);
        for (auto& pos : positions_to_close) {
            pos.is_open = false;
            // Update position in all_positions
            for (auto& all_pos : all_positions) {
                if (all_pos.symbol == pos.symbol && 
                    all_pos.strike == pos.strike && 
                    all_pos.option_type == pos.option_type &&
                    all_pos.entry_date == pos.entry_date) {
                    all_pos.is_open = false;
                    break;
                }
            }
        }
        
        // Check entry conditions
        if (shouldEnterPosition(StrategyType::COVERED_CALL, daily_data, 
                               getOpenPositions(all_positions), params)) {
            auto new_positions = generateEntryPositions(StrategyType::COVERED_CALL, 
                                                       daily_data, params);
            all_positions.insert(all_positions.end(), new_positions.begin(), new_positions.end());
        }
        
        // Calculate portfolio value and Greeks
        open_positions = getOpenPositions(all_positions);
        double portfolio_value = calculatePortfolioValue(open_positions, daily_data, underlying_price);
        Greeks portfolio_greeks = calculatePortfolioGreeks(open_positions, daily_data, underlying_price);
        
        current_equity = params.initial_capital + portfolio_value;
        peak_equity = std::max(peak_equity, current_equity);
        double drawdown = (peak_equity - current_equity) / peak_equity;
        
        // Store daily results
        result.dates.push_back(date);
        result.pnl_curve.push_back(portfolio_value);
        result.equity_curve.push_back(current_equity);
        result.drawdown_curve.push_back(drawdown);
        result.delta_exposure.push_back(portfolio_greeks.delta);
        result.gamma_exposure.push_back(portfolio_greeks.gamma);
        result.vega_exposure.push_back(portfolio_greeks.vega);
        result.theta_exposure.push_back(portfolio_greeks.theta);
    }
    
    // Calculate final metrics
    result.all_positions = all_positions;
    result.total_trades = all_positions.size();
    
    if (!result.equity_curve.empty()) {
        result.total_return = (result.equity_curve.back() - params.initial_capital) / params.initial_capital;
        result.max_drawdown = *std::max_element(result.drawdown_curve.begin(), result.drawdown_curve.end());
        
        // Calculate Sharpe ratio (simplified)
        if (result.equity_curve.size() > 1) {
            std::vector<double> returns;
            for (size_t i = 1; i < result.equity_curve.size(); ++i) {
                double ret = (result.equity_curve[i] - result.equity_curve[i-1]) / result.equity_curve[i-1];
                returns.push_back(ret);
            }
            
            double mean_return = std::accumulate(returns.begin(), returns.end(), 0.0) / returns.size();
            double variance = 0.0;
            for (double ret : returns) {
                variance += (ret - mean_return) * (ret - mean_return);
            }
            variance /= returns.size();
            double std_dev = std::sqrt(variance);
            
            result.sharpe_ratio = std_dev > 0 ? (mean_return * 252) / (std_dev * std::sqrt(252)) : 0.0;
        }
    }
    
    return result;
}

StrategyResult StrategyEngine::backtestVerticalSpread(const std::vector<OptionData>& data, 
                                                     const StrategyParams& params) {
    StrategyResult result;
    result.strategy_name = "Vertical Spread";
    // Implementation similar to covered call but with different position generation
    return result;
}

StrategyResult StrategyEngine::backtestStraddle(const std::vector<OptionData>& data, 
                                               const StrategyParams& params) {
    StrategyResult result;
    result.strategy_name = "Long Straddle";
    // Implementation similar to covered call but with different position generation
    return result;
}

std::vector<Position> StrategyEngine::getOpenPositions(const std::vector<Position>& all_positions) {
    std::vector<Position> open_positions;
    for (const auto& pos : all_positions) {
        if (pos.is_open) {
            open_positions.push_back(pos);
        }
    }
    return open_positions;
}

double StrategyEngine::calculatePortfolioValue(const std::vector<Position>& positions,
                                              const std::vector<OptionData>& current_data,
                                              double underlying_price) {
    double total_value = 0.0;
    
    for (const auto& pos : positions) {
        double current_price = getOptionPrice(current_data, pos.strike, 
                                            pos.option_type, pos.expiry_date);
        if (current_price > 0) {
            double position_value = (current_price - pos.entry_price) * pos.quantity * 100; // Options are per 100 shares
            total_value += position_value;
        }
    }
    
    return total_value;
}

Greeks StrategyEngine::calculatePortfolioGreeks(const std::vector<Position>& positions,
                                               const std::vector<OptionData>& current_data,
                                               double underlying_price,
                                               double risk_free_rate) {
    Greeks total_greeks;
    
    for (const auto& pos : positions) {
        // Find matching option data
        for (const auto& option : current_data) {
            if (std::abs(option.strike - pos.strike) < 0.01 && 
                option.option_type == pos.option_type &&
                option.expiry == pos.expiry_date) {
                
                double T = 0.25; // Simplified - should use actual time to expiry
                Greeks pos_greeks = BlackScholes::calculateGreeks(
                    underlying_price, pos.strike, T, risk_free_rate, 
                    0.2, pos.option_type); // Simplified - should use actual IV
                
                total_greeks.delta += pos_greeks.delta * pos.quantity;
                total_greeks.gamma += pos_greeks.gamma * pos.quantity;
                total_greeks.theta += pos_greeks.theta * pos.quantity;
                total_greeks.vega += pos_greeks.vega * pos.quantity;
                total_greeks.rho += pos_greeks.rho * pos.quantity;
                break;
            }
        }
    }
    
    return total_greeks;
}

bool StrategyEngine::shouldEnterPosition(StrategyType type,
                                        const std::vector<OptionData>& current_data,
                                        const std::vector<Position>& open_positions,
                                        const StrategyParams& params) {
    // Simple logic: enter if we have less than max positions
    return open_positions.size() < static_cast<size_t>(params.max_positions);
}

std::vector<Position> StrategyEngine::generateEntryPositions(StrategyType type,
                                                           const std::vector<OptionData>& current_data,
                                                           const StrategyParams& params) {
    std::vector<Position> new_positions;
    
    if (current_data.empty()) return new_positions;
    
    if (type == StrategyType::COVERED_CALL) {
        // Find ATM call option
        double underlying_price = current_data[0].underlying_price;
        
        for (const auto& option : current_data) {
            if (option.option_type == 'C' && 
                std::abs(option.strike - underlying_price) < underlying_price * 0.05) { // Within 5% of ATM
                
                Position pos;
                pos.symbol = option.symbol;
                pos.strike = option.strike;
                pos.option_type = 'C';
                pos.quantity = -1; // Short call
                pos.entry_price = option.mid_price;
                pos.entry_date = option.date;
                pos.expiry_date = option.expiry;
                pos.is_open = true;
                
                new_positions.push_back(pos);
                break; // Only one position for now
            }
        }
    }
    
    return new_positions;
}

std::vector<Position> StrategyEngine::checkExitConditions(const std::vector<Position>& open_positions,
                                                        const std::vector<OptionData>& current_data,
                                                        const StrategyParams& params) {
    std::vector<Position> positions_to_close;
    
    for (const auto& pos : open_positions) {
        // Check if close to expiry
        if (isCloseToExpiry(current_data.empty() ? "" : current_data[0].date, 
                           pos.expiry_date, params.days_to_expiry_close)) {
            positions_to_close.push_back(pos);
            continue;
        }
        
        // Check profit target
        double current_price = getOptionPrice(current_data, pos.strike, 
                                            pos.option_type, pos.expiry_date);
        if (current_price > 0) {
            double pnl_pct = (pos.entry_price - current_price) / pos.entry_price; // For short positions
            if (pnl_pct >= params.profit_target) {
                positions_to_close.push_back(pos);
            }
        }
    }
    
    return positions_to_close;
}

double StrategyEngine::getOptionPrice(const std::vector<OptionData>& data, 
                                     double strike, char option_type, 
                                     const std::string& expiry) {
    for (const auto& option : data) {
        if (std::abs(option.strike - strike) < 0.01 && 
            option.option_type == option_type &&
            option.expiry == expiry) {
            return option.mid_price;
        }
    }
    return 0.0;
}

std::vector<OptionData> StrategyEngine::filterByDate(const std::vector<OptionData>& data, 
                                                    const std::string& date) {
    std::vector<OptionData> filtered;
    for (const auto& option : data) {
        if (option.date == date) {
            filtered.push_back(option);
        }
    }
    return filtered;
}

std::vector<OptionData> StrategyEngine::filterByExpiry(const std::vector<OptionData>& data,
                                                      const std::string& min_expiry,
                                                      const std::string& max_expiry) {
    std::vector<OptionData> filtered;
    for (const auto& option : data) {
        if (option.expiry >= min_expiry && option.expiry <= max_expiry) {
            filtered.push_back(option);
        }
    }
    return filtered;
}

double StrategyEngine::calculateCommissions(const std::vector<Position>& positions, 
                                           double commission_per_contract) {
    double total_commissions = 0.0;
    for (const auto& pos : positions) {
        total_commissions += std::abs(pos.quantity) * commission_per_contract;
    }
    return total_commissions;
}

std::vector<std::string> StrategyEngine::getUniqueDates(const std::vector<OptionData>& data) {
    std::set<std::string> unique_dates_set;
    for (const auto& option : data) {
        unique_dates_set.insert(option.date);
    }
    return std::vector<std::string>(unique_dates_set.begin(), unique_dates_set.end());
}

bool StrategyEngine::isCloseToExpiry(const std::string& current_date, 
                                    const std::string& expiry_date, 
                                    int days_threshold) {
    int days_to_expiry = DateUtils::daysBetween(current_date, expiry_date);
    return days_to_expiry <= days_threshold;
}

} // namespace volatility