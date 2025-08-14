#pragma once

#include <string>
#include <chrono>

namespace volatility {

class DateUtils {
public:
    // Convert date string to time_point
    static std::chrono::system_clock::time_point parseDate(const std::string& date_str);
    
    // Calculate days between two dates
    static int daysBetween(const std::string& start_date, const std::string& end_date);
    
    // Calculate years between two dates
    static double yearsBetween(const std::string& start_date, const std::string& end_date);
    
    // Get current date as string
    static std::string getCurrentDate();
    
    // Format date as YYYY-MM-DD
    static std::string formatDate(const std::chrono::system_clock::time_point& tp);
    
    // Add days to a date
    static std::string addDays(const std::string& date_str, int days);
    
    // Check if date string is valid
    static bool isValidDate(const std::string& date_str);
    
    // Convert option expiry format to standard date
    static std::string parseOptionExpiry(const std::string& expiry_str);

private:
    static std::vector<std::string> splitDate(const std::string& date_str, char delimiter = '-');
};

} // namespace volatility