#pragma once

#include <string>
#include <vector>
#include <memory>

namespace volatility {

struct OptionData {
    std::string symbol;
    std::string date;
    std::string expiry;
    double strike;
    char option_type; // 'C' for call, 'P' for put
    double bid;
    double ask;
    double mid_price;
    long volume;
    long open_interest;
    double underlying_price;
    
    OptionData() : strike(0.0), option_type('C'), bid(0.0), ask(0.0), 
                   mid_price(0.0), volume(0), open_interest(0), underlying_price(0.0) {}
};

class DataLoader {
public:
    DataLoader();
    ~DataLoader();
    
    // Download data from URL
    bool downloadData(const std::string& url, const std::string& filepath);
    
    // Parse CBOE CSV format
    std::vector<OptionData> parseCBOE(const std::string& filepath);
    
    // Parse dxFeed CSV format
    std::vector<OptionData> parseDxFeed(const std::string& filepath);
    
    // Load historical data for a symbol
    std::vector<OptionData> loadHistoricalData(const std::string& symbol, 
                                               const std::string& start_date,
                                               const std::string& end_date);
    
    // Validate and clean data
    std::vector<OptionData> cleanData(const std::vector<OptionData>& raw_data);
    
private:
    std::string trim(const std::string& str);
    std::vector<std::string> split(const std::string& str, char delimiter);
    bool isValidOptionData(const OptionData& data);
    double parseDouble(const std::string& str);
    long parseLong(const std::string& str);
};

} // namespace volatility