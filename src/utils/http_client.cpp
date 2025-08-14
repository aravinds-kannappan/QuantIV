#include "http_client.hpp"
#include <curl/curl.h>
#include <fstream>
#include <iostream>

namespace volatility {

HttpClient::HttpClient() {
    curl_global_init(CURL_GLOBAL_DEFAULT);
}

HttpClient::~HttpClient() {
    curl_global_cleanup();
}

bool HttpClient::downloadFile(const std::string& url, const std::string& filepath) {
    CURL* curl = curl_easy_init();
    if (!curl) {
        std::cerr << "Error: Failed to initialize CURL" << std::endl;
        return false;
    }
    
    std::ofstream file(filepath, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot create file " << filepath << std::endl;
        curl_easy_cleanup(curl);
        return false;
    }
    
    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, writeFileCallback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &file);
    curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);
    curl_easy_setopt(curl, CURLOPT_TIMEOUT, 30L);
    curl_easy_setopt(curl, CURLOPT_USERAGENT, "VolatilityAlchemist/1.0");
    
    CURLcode res = curl_easy_perform(curl);
    
    long response_code;
    curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &response_code);
    
    curl_easy_cleanup(curl);
    file.close();
    
    if (res != CURLE_OK || response_code != 200) {
        std::cerr << "Error: Failed to download " << url 
                  << " (HTTP " << response_code << ")" << std::endl;
        return false;
    }
    
    std::cout << "Downloaded: " << url << " -> " << filepath << std::endl;
    return true;
}

std::string HttpClient::getContent(const std::string& url) {
    CURL* curl = curl_easy_init();
    if (!curl) {
        return "";
    }
    
    std::string content;
    
    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, writeCallback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &content);
    curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);
    curl_easy_setopt(curl, CURLOPT_TIMEOUT, 30L);
    curl_easy_setopt(curl, CURLOPT_USERAGENT, "VolatilityAlchemist/1.0");
    
    CURLcode res = curl_easy_perform(curl);
    curl_easy_cleanup(curl);
    
    if (res != CURLE_OK) {
        return "";
    }
    
    return content;
}

size_t HttpClient::writeCallback(void* contents, size_t size, size_t nmemb, void* userp) {
    size_t realsize = size * nmemb;
    std::string* str = static_cast<std::string*>(userp);
    str->append(static_cast<char*>(contents), realsize);
    return realsize;
}

size_t HttpClient::writeFileCallback(void* contents, size_t size, size_t nmemb, void* userp) {
    size_t realsize = size * nmemb;
    std::ofstream* file = static_cast<std::ofstream*>(userp);
    file->write(static_cast<char*>(contents), realsize);
    return realsize;
}

} // namespace volatility