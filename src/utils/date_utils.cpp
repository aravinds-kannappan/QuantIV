#include "date_utils.hpp"
#include <sstream>
#include <iomanip>
#include <ctime>

namespace volatility {

std::chrono::system_clock::time_point DateUtils::parseDate(const std::string& date_str) {
    auto parts = splitDate(date_str);
    if (parts.size() != 3) {
        return std::chrono::system_clock::now();
    }
    
    std::tm tm = {};
    tm.tm_year = std::stoi(parts[0]) - 1900;
    tm.tm_mon = std::stoi(parts[1]) - 1;
    tm.tm_mday = std::stoi(parts[2]);
    
    std::time_t tt = std::mktime(&tm);
    return std::chrono::system_clock::from_time_t(tt);
}

int DateUtils::daysBetween(const std::string& start_date, const std::string& end_date) {
    auto start_tp = parseDate(start_date);
    auto end_tp = parseDate(end_date);
    
    auto duration = end_tp - start_tp;
    return std::chrono::duration_cast<std::chrono::days>(duration).count();
}

double DateUtils::yearsBetween(const std::string& start_date, const std::string& end_date) {
    int days = daysBetween(start_date, end_date);
    return static_cast<double>(days) / 365.25;
}

std::string DateUtils::getCurrentDate() {
    auto now = std::chrono::system_clock::now();
    return formatDate(now);
}

std::string DateUtils::formatDate(const std::chrono::system_clock::time_point& tp) {
    std::time_t tt = std::chrono::system_clock::to_time_t(tp);
    std::tm* tm = std::localtime(&tt);
    
    std::ostringstream oss;
    oss << std::put_time(tm, "%Y-%m-%d");
    return oss.str();
}

std::string DateUtils::addDays(const std::string& date_str, int days) {
    auto tp = parseDate(date_str);
    tp += std::chrono::hours(24 * days);
    return formatDate(tp);
}

bool DateUtils::isValidDate(const std::string& date_str) {
    auto parts = splitDate(date_str);
    if (parts.size() != 3) return false;
    
    try {
        int year = std::stoi(parts[0]);
        int month = std::stoi(parts[1]);
        int day = std::stoi(parts[2]);
        
        return year >= 1900 && year <= 2100 &&
               month >= 1 && month <= 12 &&
               day >= 1 && day <= 31;
    } catch (...) {
        return false;
    }
}

std::string DateUtils::parseOptionExpiry(const std::string& expiry_str) {
    // Handle different expiry formats
    if (expiry_str.length() == 6) {
        // Format: YYMMDD
        std::string year = "20" + expiry_str.substr(0, 2);
        std::string month = expiry_str.substr(2, 2);
        std::string day = expiry_str.substr(4, 2);
        return year + "-" + month + "-" + day;
    } else if (expiry_str.length() == 8) {
        // Format: YYYYMMDD
        std::string year = expiry_str.substr(0, 4);
        std::string month = expiry_str.substr(4, 2);
        std::string day = expiry_str.substr(6, 2);
        return year + "-" + month + "-" + day;
    }
    
    return expiry_str; // Return as-is if already in correct format
}

std::vector<std::string> DateUtils::splitDate(const std::string& date_str, char delimiter) {
    std::vector<std::string> parts;
    std::stringstream ss(date_str);
    std::string part;
    
    while (std::getline(ss, part, delimiter)) {
        parts.push_back(part);
    }
    
    return parts;
}

} // namespace volatility