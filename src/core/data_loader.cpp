#include "data_loader.hpp"
#include "../utils/http_client.hpp"
#include <fstream>
#include <sstream>
#include <algorithm>
#include <iostream>

namespace volatility {

DataLoader::DataLoader() {}

DataLoader::~DataLoader() {}

bool DataLoader::downloadData(const std::string& url, const std::string& filepath) {
    HttpClient client;
    return client.downloadFile(url, filepath);
}

std::vector<OptionData> DataLoader::parseCBOE(const std::string& filepath) {
    std::vector<OptionData> options;
    std::ifstream file(filepath);
    
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open file " << filepath << std::endl;
        return options;
    }
    
    std::string line;
    bool isHeader = true;
    
    while (std::getline(file, line)) {
        if (isHeader) {
            isHeader = false;
            continue; // Skip header
        }
        
        auto fields = split(line, ',');
        if (fields.size() < 10) continue;
        
        OptionData option;
        
        try {
            option.symbol = trim(fields[0]);
            option.date = trim(fields[1]);
            option.expiry = trim(fields[2]);
            option.strike = parseDouble(fields[3]);
            option.option_type = trim(fields[4])[0];
            option.bid = parseDouble(fields[5]);
            option.ask = parseDouble(fields[6]);
            option.mid_price = (option.bid + option.ask) / 2.0;
            option.volume = parseLong(fields[7]);
            option.open_interest = parseLong(fields[8]);
            option.underlying_price = parseDouble(fields[9]);
            
            if (isValidOptionData(option)) {
                options.push_back(option);
            }
        } catch (const std::exception& e) {
            std::cerr << "Error parsing line: " << line << std::endl;
            continue;
        }
    }
    
    file.close();
    return options;
}

std::vector<OptionData> DataLoader::parseDxFeed(const std::string& filepath) {
    std::vector<OptionData> options;
    std::ifstream file(filepath);
    
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open file " << filepath << std::endl;
        return options;
    }
    
    std::string line;
    bool isHeader = true;
    
    while (std::getline(file, line)) {
        if (isHeader) {
            isHeader = false;
            continue; // Skip header
        }
        
        auto fields = split(line, ',');
        if (fields.size() < 8) continue;
        
        OptionData option;
        
        try {
            // dxFeed format: Symbol,Date,Strike,Type,Bid,Ask,Volume,OpenInterest
            option.symbol = trim(fields[0]);
            option.date = trim(fields[1]);
            option.strike = parseDouble(fields[2]);
            option.option_type = trim(fields[3])[0];
            option.bid = parseDouble(fields[4]);
            option.ask = parseDouble(fields[5]);
            option.mid_price = (option.bid + option.ask) / 2.0;
            option.volume = parseLong(fields[6]);
            option.open_interest = parseLong(fields[7]);
            
            // Extract expiry from symbol (e.g., AAPL190927C210)
            std::string symbol_str = option.symbol;
            size_t pos = symbol_str.find_first_of("CP");
            if (pos != std::string::npos && pos >= 6) {
                option.expiry = "20" + symbol_str.substr(pos - 6, 6);
            }
            
            if (isValidOptionData(option)) {
                options.push_back(option);
            }
        } catch (const std::exception& e) {
            std::cerr << "Error parsing line: " << line << std::endl;
            continue;
        }
    }
    
    file.close();
    return options;
}

std::vector<OptionData> DataLoader::loadHistoricalData(const std::string& symbol, 
                                                       const std::string& start_date,
                                                       const std::string& end_date) {
    std::vector<OptionData> all_data;
    
    // Download CBOE data for the specified period
    std::string base_url = "https://www.cboe.com/publish/scheduledtask/mktdata/datahouse/market_statistics/OPRA/OPTION_VOLUME_SYMBOL_";
    
    // For now, download current month data
    std::string url = base_url + symbol + "_2025-05.csv";
    std::string filepath = "data/cboe/" + symbol + "_2025-05.csv";
    
    if (downloadData(url, filepath)) {
        auto cboe_data = parseCBOE(filepath);
        all_data.insert(all_data.end(), cboe_data.begin(), cboe_data.end());
    }
    
    return cleanData(all_data);
}

std::vector<OptionData> DataLoader::cleanData(const std::vector<OptionData>& raw_data) {
    std::vector<OptionData> clean_data;
    
    for (const auto& option : raw_data) {
        if (isValidOptionData(option)) {
            clean_data.push_back(option);
        }
    }
    
    return clean_data;
}

std::string DataLoader::trim(const std::string& str) {
    size_t start = str.find_first_not_of(" \t\r\n\"");
    if (start == std::string::npos) return "";
    
    size_t end = str.find_last_not_of(" \t\r\n\"");
    return str.substr(start, end - start + 1);
}

std::vector<std::string> DataLoader::split(const std::string& str, char delimiter) {
    std::vector<std::string> tokens;
    std::stringstream ss(str);
    std::string token;
    
    while (std::getline(ss, token, delimiter)) {
        tokens.push_back(token);
    }
    
    return tokens;
}

bool DataLoader::isValidOptionData(const OptionData& data) {
    return !data.symbol.empty() &&
           !data.date.empty() &&
           data.strike > 0 &&
           (data.option_type == 'C' || data.option_type == 'P') &&
           data.bid >= 0 &&
           data.ask >= data.bid &&
           data.volume >= 0 &&
           data.open_interest >= 0;
}

double DataLoader::parseDouble(const std::string& str) {
    try {
        return std::stod(trim(str));
    } catch (...) {
        return 0.0;
    }
}

long DataLoader::parseLong(const std::string& str) {
    try {
        return std::stol(trim(str));
    } catch (...) {
        return 0;
    }
}

} // namespace volatility