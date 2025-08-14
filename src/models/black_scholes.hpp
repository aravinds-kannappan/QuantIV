#pragma once

#include <cmath>

namespace volatility {

struct Greeks {
    double delta;
    double gamma;
    double theta;
    double vega;
    double rho;
    
    Greeks() : delta(0), gamma(0), theta(0), vega(0), rho(0) {}
};

class BlackScholes {
public:
    // Option pricing
    static double callPrice(double S, double K, double T, double r, double sigma);
    static double putPrice(double S, double K, double T, double r, double sigma);
    
    // Implied volatility calculation
    static double impliedVolatility(double S, double K, double T, double r, 
                                   double market_price, char option_type, 
                                   double tolerance = 1e-6, int max_iterations = 100);
    
    // Greeks calculation
    static Greeks calculateGreeks(double S, double K, double T, double r, double sigma, char option_type);
    
    // Individual Greeks
    static double delta(double S, double K, double T, double r, double sigma, char option_type);
    static double gamma(double S, double K, double T, double r, double sigma);
    static double theta(double S, double K, double T, double r, double sigma, char option_type);
    static double vega(double S, double K, double T, double r, double sigma);
    static double rho(double S, double K, double T, double r, double sigma, char option_type);

private:
    // Cumulative standard normal distribution
    static double N(double x);
    
    // Standard normal probability density function
    static double n(double x);
    
    // Calculate d1 and d2 parameters
    static double d1(double S, double K, double T, double r, double sigma);
    static double d2(double S, double K, double T, double r, double sigma);
};

} // namespace volatility