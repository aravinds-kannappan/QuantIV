#pragma once

#include "data_loader.hpp"
#include "volatility_model.hpp"
#include "../models/black_scholes.hpp"
#include <vector>
#include <memory>
#include <map>

namespace volatility {

enum class StrategyType {
    COVERED_CALL,
    VERTICAL_SPREAD,
    STRADDLE,
    IRON_CONDOR
};

struct Position {
    std::string symbol;
    double strike;
    char option_type; // 'C' or 'P'
    int quantity; // Positive for long, negative for short
    double entry_price;
    std::string entry_date;
    std::string expiry_date;
    bool is_open;
    
    Position() : strike(0), option_type('C'), quantity(0), entry_price(0), is_open(true) {}
};

struct StrategyParams {
    double initial_capital;
    double commission_per_contract;
    int max_positions;
    double profit_target; // % of max profit to close at
    double loss_limit; // % of capital to stop at
    int days_to_expiry_close; // Close positions X days before expiry
    
    StrategyParams() : initial_capital(100000), commission_per_contract(1.0), 
                      max_positions(10), profit_target(0.5), loss_limit(0.1),
                      days_to_expiry_close(7) {}
};

struct StrategyResult {
    std::string strategy_name;
    std::string symbol;
    std::string start_date;
    std::string end_date;
    
    std::vector<std::string> dates;
    std::vector<double> pnl_curve;
    std::vector<double> equity_curve;
    std::vector<double> drawdown_curve;
    std::vector<double> delta_exposure;
    std::vector<double> gamma_exposure;
    std::vector<double> vega_exposure;
    std::vector<double> theta_exposure;
    
    std::vector<Position> all_positions;
    int total_trades;
    int winning_trades;
    double total_return;
    double max_drawdown;
    double sharpe_ratio;
    double profit_factor;
    
    StrategyResult() : total_trades(0), winning_trades(0), total_return(0), 
                      max_drawdown(0), sharpe_ratio(0), profit_factor(0) {}
};

class StrategyEngine {
public:
    StrategyEngine();
    ~StrategyEngine();
    
    // Main backtesting function
    StrategyResult backtestStrategy(StrategyType type,
                                   const std::vector<OptionData>& historical_data,
                                   const StrategyParams& params);
    
    // Strategy-specific implementations
    StrategyResult backtestCoveredCall(const std::vector<OptionData>& data, 
                                      const StrategyParams& params);
    
    StrategyResult backtestVerticalSpread(const std::vector<OptionData>& data, 
                                         const StrategyParams& params);
    
    StrategyResult backtestStraddle(const std::vector<OptionData>& data, 
                                   const StrategyParams& params);
    
    // Portfolio management
    std::vector<Position> getOpenPositions(const std::vector<Position>& all_positions);
    
    double calculatePortfolioValue(const std::vector<Position>& positions,
                                  const std::vector<OptionData>& current_data,
                                  double underlying_price);
    
    Greeks calculatePortfolioGreeks(const std::vector<Position>& positions,
                                   const std::vector<OptionData>& current_data,
                                   double underlying_price,
                                   double risk_free_rate = 0.05);
    
    // Entry/exit logic
    bool shouldEnterPosition(StrategyType type,
                           const std::vector<OptionData>& current_data,
                           const std::vector<Position>& open_positions,
                           const StrategyParams& params);
    
    std::vector<Position> generateEntryPositions(StrategyType type,
                                               const std::vector<OptionData>& current_data,
                                               const StrategyParams& params);
    
    std::vector<Position> checkExitConditions(const std::vector<Position>& open_positions,
                                            const std::vector<OptionData>& current_data,
                                            const StrategyParams& params);

private:
    // Helper functions
    double getOptionPrice(const std::vector<OptionData>& data, 
                         double strike, char option_type, 
                         const std::string& expiry);
    
    std::vector<OptionData> filterByDate(const std::vector<OptionData>& data, 
                                        const std::string& date);
    
    std::vector<OptionData> filterByExpiry(const std::vector<OptionData>& data,
                                          const std::string& min_expiry,
                                          const std::string& max_expiry);
    
    double calculateCommissions(const std::vector<Position>& positions, 
                               double commission_per_contract);
    
    double calculatePositionPnL(const Position& position,
                               const std::vector<OptionData>& current_data,
                               double current_underlying_price);
    
    // Risk management
    bool isWithinRiskLimits(const std::vector<Position>& positions,
                           const std::vector<OptionData>& current_data,
                           const StrategyParams& params);
    
    // Date utilities
    std::vector<std::string> getUniqueDates(const std::vector<OptionData>& data);
    bool isCloseToExpiry(const std::string& current_date, 
                        const std::string& expiry_date, 
                        int days_threshold);
};

} // namespace volatility