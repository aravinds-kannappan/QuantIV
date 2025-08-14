#include "black_scholes.hpp"
#include <cmath>
#include <algorithm>

namespace volatility {

double BlackScholes::callPrice(double S, double K, double T, double r, double sigma) {
    if (T <= 0 || sigma <= 0 || S <= 0 || K <= 0) return 0.0;
    
    double d1_val = d1(S, K, T, r, sigma);
    double d2_val = d2(S, K, T, r, sigma);
    
    return S * N(d1_val) - K * std::exp(-r * T) * N(d2_val);
}

double BlackScholes::putPrice(double S, double K, double T, double r, double sigma) {
    if (T <= 0 || sigma <= 0 || S <= 0 || K <= 0) return 0.0;
    
    double d1_val = d1(S, K, T, r, sigma);
    double d2_val = d2(S, K, T, r, sigma);
    
    return K * std::exp(-r * T) * N(-d2_val) - S * N(-d1_val);
}

double BlackScholes::impliedVolatility(double S, double K, double T, double r, 
                                      double market_price, char option_type, 
                                      double tolerance, int max_iterations) {
    if (T <= 0 || S <= 0 || K <= 0 || market_price <= 0) return 0.0;
    
    double sigma_low = 0.001;
    double sigma_high = 5.0;
    double sigma_mid = 0.5;
    
    for (int i = 0; i < max_iterations; ++i) {
        double price;
        if (option_type == 'C' || option_type == 'c') {
            price = callPrice(S, K, T, r, sigma_mid);
        } else {
            price = putPrice(S, K, T, r, sigma_mid);
        }
        
        double diff = price - market_price;
        
        if (std::abs(diff) < tolerance) {
            return sigma_mid;
        }
        
        if (diff > 0) {
            sigma_high = sigma_mid;
        } else {
            sigma_low = sigma_mid;
        }
        
        sigma_mid = (sigma_low + sigma_high) / 2.0;
    }
    
    return sigma_mid;
}

Greeks BlackScholes::calculateGreeks(double S, double K, double T, double r, double sigma, char option_type) {
    Greeks greeks;
    
    greeks.delta = delta(S, K, T, r, sigma, option_type);
    greeks.gamma = gamma(S, K, T, r, sigma);
    greeks.theta = theta(S, K, T, r, sigma, option_type);
    greeks.vega = vega(S, K, T, r, sigma);
    greeks.rho = rho(S, K, T, r, sigma, option_type);
    
    return greeks;
}

double BlackScholes::delta(double S, double K, double T, double r, double sigma, char option_type) {
    if (T <= 0 || sigma <= 0 || S <= 0 || K <= 0) return 0.0;
    
    double d1_val = d1(S, K, T, r, sigma);
    
    if (option_type == 'C' || option_type == 'c') {
        return N(d1_val);
    } else {
        return N(d1_val) - 1.0;
    }
}

double BlackScholes::gamma(double S, double K, double T, double r, double sigma) {
    if (T <= 0 || sigma <= 0 || S <= 0 || K <= 0) return 0.0;
    
    double d1_val = d1(S, K, T, r, sigma);
    return n(d1_val) / (S * sigma * std::sqrt(T));
}

double BlackScholes::theta(double S, double K, double T, double r, double sigma, char option_type) {
    if (T <= 0 || sigma <= 0 || S <= 0 || K <= 0) return 0.0;
    
    double d1_val = d1(S, K, T, r, sigma);
    double d2_val = d2(S, K, T, r, sigma);
    
    double theta_common = -(S * n(d1_val) * sigma) / (2.0 * std::sqrt(T));
    
    if (option_type == 'C' || option_type == 'c') {
        return (theta_common - r * K * std::exp(-r * T) * N(d2_val)) / 365.0;
    } else {
        return (theta_common + r * K * std::exp(-r * T) * N(-d2_val)) / 365.0;
    }
}

double BlackScholes::vega(double S, double K, double T, double r, double sigma) {
    if (T <= 0 || sigma <= 0 || S <= 0 || K <= 0) return 0.0;
    
    double d1_val = d1(S, K, T, r, sigma);
    return S * n(d1_val) * std::sqrt(T) / 100.0; // Per 1% change in volatility
}

double BlackScholes::rho(double S, double K, double T, double r, double sigma, char option_type) {
    if (T <= 0 || sigma <= 0 || S <= 0 || K <= 0) return 0.0;
    
    double d2_val = d2(S, K, T, r, sigma);
    
    if (option_type == 'C' || option_type == 'c') {
        return K * T * std::exp(-r * T) * N(d2_val) / 100.0;
    } else {
        return -K * T * std::exp(-r * T) * N(-d2_val) / 100.0;
    }
}

double BlackScholes::N(double x) {
    // Approximation of cumulative standard normal distribution
    double a1 =  0.31938153;
    double a2 = -0.356563782;
    double a3 =  1.781477937;
    double a4 = -1.821255978;
    double a5 =  1.330274429;
    
    double k = 1.0 / (1.0 + 0.2316419 * std::abs(x));
    double cnd = 1.0 - (1.0 / std::sqrt(2.0 * M_PI)) * std::exp(-0.5 * x * x) *
                 (a1 * k + a2 * k * k + a3 * k * k * k + a4 * k * k * k * k + a5 * k * k * k * k * k);
    
    if (x < 0) {
        return 1.0 - cnd;
    }
    
    return cnd;
}

double BlackScholes::n(double x) {
    // Standard normal probability density function
    return (1.0 / std::sqrt(2.0 * M_PI)) * std::exp(-0.5 * x * x);
}

double BlackScholes::d1(double S, double K, double T, double r, double sigma) {
    return (std::log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * std::sqrt(T));
}

double BlackScholes::d2(double S, double K, double T, double r, double sigma) {
    return d1(S, K, T, r, sigma) - sigma * std::sqrt(T);
}

} // namespace volatility