#pragma once

#include <string>

namespace volatility {

class HttpClient {
public:
    HttpClient();
    ~HttpClient();
    
    // Download file from URL and save to filepath
    bool downloadFile(const std::string& url, const std::string& filepath);
    
    // Get content from URL as string
    std::string getContent(const std::string& url);
    
private:
    static size_t writeCallback(void* contents, size_t size, size_t nmemb, void* userp);
    static size_t writeFileCallback(void* contents, size_t size, size_t nmemb, void* userp);
};

} // namespace volatility