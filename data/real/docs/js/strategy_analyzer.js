// Strategy analysis and comparison utilities
class StrategyAnalyzer {
    constructor() {
        this.strategies = new Map();
        this.benchmarks = new Map();
    }
    
    addStrategy(name, data) {
        this.strategies.set(name, data);
    }
    
    addBenchmark(name, data) {
        this.benchmarks.set(name, data);
    }
    
    compareStrategies(strategyNames) {
        const comparison = {
            strategies: [],
            metrics: {},
            rankings: {}
        };
        
        const metricsToCompare = [
            'total_return',
            'sharpe_ratio',
            'max_drawdown',
            'win_rate',
            'profit_factor',
            'var_95',
            'sortino_ratio'
        ];
        
        // Collect data for each strategy
        strategyNames.forEach(name => {
            const strategy = this.strategies.get(name);
            if (strategy && strategy.metrics) {
                comparison.strategies.push({
                    name: name,
                    data: strategy,
                    metrics: strategy.metrics
                });
            }
        });
        
        // Calculate rankings for each metric
        metricsToCompare.forEach(metric => {
            const values = comparison.strategies.map(s => ({
                name: s.name,
                value: s.metrics[metric] || 0
            }));
            
            // Sort by metric value (higher is better except for max_drawdown and var_95)
            const isLowerBetter = ['max_drawdown', 'var_95'].includes(metric);
            values.sort((a, b) => isLowerBetter ? a.value - b.value : b.value - a.value);
            
            comparison.rankings[metric] = values.map((item, index) => ({
                ...item,
                rank: index + 1
            }));
        });
        
        return comparison;
    }
    
    calculateRiskAdjustedReturns(strategy) {
        if (!strategy.timeseries || !strategy.metrics) {
            return null;
        }
        
        const returns = this.calculateDailyReturns(strategy.timeseries);
        const metrics = strategy.metrics;
        
        return {
            sharpe_ratio: metrics.sharpe_ratio || 0,
            sortino_ratio: metrics.sortino_ratio || 0,
            calmar_ratio: metrics.annualized_return / Math.abs(metrics.max_drawdown) || 0,
            information_ratio: this.calculateInformationRatio(returns),
            treynor_ratio: this.calculateTreynorRatio(returns, metrics),
            jensen_alpha: this.calculateJensenAlpha(returns)
        };
    }
    
    calculateDailyReturns(timeseries) {
        if (!timeseries || timeseries.length < 2) {
            return [];
        }
        
        const returns = [];
        for (let i = 1; i < timeseries.length; i++) {
            const prevEquity = timeseries[i - 1].equity;
            const currentEquity = timeseries[i].equity;
            
            if (prevEquity > 0) {
                returns.push((currentEquity - prevEquity) / prevEquity);
            }
        }
        
        return returns;
    }
    
    calculateInformationRatio(returns, benchmarkReturns = null) {
        if (!benchmarkReturns) {
            // Use a simple market return approximation if no benchmark provided
            benchmarkReturns = returns.map(() => 0.0003); // ~8% annual return
        }
        
        if (returns.length !== benchmarkReturns.length) {
            return 0;
        }
        
        const excessReturns = returns.map((r, i) => r - benchmarkReturns[i]);
        const meanExcess = this.calculateMean(excessReturns);
        const trackingError = this.calculateStandardDeviation(excessReturns);
        
        return trackingError > 0 ? meanExcess / trackingError : 0;
    }
    
    calculateTreynorRatio(returns, metrics) {
        // Simplified calculation - would need market data for proper beta
        const beta = metrics.beta || 1.0;
        const riskFreeRate = 0.02 / 252; // 2% annual risk-free rate
        const meanReturn = this.calculateMean(returns);
        
        return beta !== 0 ? (meanReturn - riskFreeRate) / beta : 0;
    }
    
    calculateJensenAlpha(returns) {
        // Simplified Jensen's Alpha calculation
        const marketReturn = 0.0003; // ~8% annual return
        const riskFreeRate = 0.02 / 252;
        const beta = 1.0; // Simplified
        const portfolioReturn = this.calculateMean(returns);
        
        return portfolioReturn - (riskFreeRate + beta * (marketReturn - riskFreeRate));
    }
    
    analyzeDrawdownPeriods(timeseries) {
        if (!timeseries || timeseries.length === 0) {
            return { periods: [], statistics: {} };
        }
        
        const drawdowns = timeseries.map(t => t.drawdown || 0);
        const periods = [];
        let currentPeriod = null;
        
        drawdowns.forEach((dd, index) => {
            if (dd > 0) {
                if (!currentPeriod) {
                    currentPeriod = {
                        start: index,
                        startDate: timeseries[index].date,
                        peak: dd,
                        duration: 1
                    };
                } else {
                    currentPeriod.duration++;
                    currentPeriod.peak = Math.max(currentPeriod.peak, dd);
                }
            } else {
                if (currentPeriod) {
                    currentPeriod.end = index - 1;
                    currentPeriod.endDate = timeseries[index - 1].date;
                    periods.push(currentPeriod);
                    currentPeriod = null;
                }
            }
        });
        
        // Close any open period
        if (currentPeriod) {
            currentPeriod.end = drawdowns.length - 1;
            currentPeriod.endDate = timeseries[drawdowns.length - 1].date;
            periods.push(currentPeriod);
        }
        
        const statistics = {
            total_periods: periods.length,
            avg_duration: periods.length > 0 ? this.calculateMean(periods.map(p => p.duration)) : 0,
            max_duration: periods.length > 0 ? Math.max(...periods.map(p => p.duration)) : 0,
            avg_depth: periods.length > 0 ? this.calculateMean(periods.map(p => p.peak)) : 0,
            max_depth: periods.length > 0 ? Math.max(...periods.map(p => p.peak)) : 0
        };
        
        return { periods, statistics };
    }
    
    calculateCorrelationMatrix(strategies) {
        const names = Array.from(strategies.keys());
        const matrix = {};
        
        names.forEach(name1 => {
            matrix[name1] = {};
            names.forEach(name2 => {
                const corr = this.calculateCorrelation(
                    strategies.get(name1),
                    strategies.get(name2)
                );
                matrix[name1][name2] = corr;
            });
        });
        
        return matrix;
    }
    
    calculateCorrelation(strategy1, strategy2) {
        if (!strategy1.timeseries || !strategy2.timeseries) {
            return 0;
        }
        
        const returns1 = this.calculateDailyReturns(strategy1.timeseries);
        const returns2 = this.calculateDailyReturns(strategy2.timeseries);
        
        return this.pearsonCorrelation(returns1, returns2);
    }
    
    pearsonCorrelation(x, y) {
        if (x.length !== y.length || x.length === 0) {
            return 0;
        }
        
        const n = x.length;
        const meanX = this.calculateMean(x);
        const meanY = this.calculateMean(y);
        
        let numerator = 0;
        let sumXSquared = 0;
        let sumYSquared = 0;
        
        for (let i = 0; i < n; i++) {
            const xDiff = x[i] - meanX;
            const yDiff = y[i] - meanY;
            
            numerator += xDiff * yDiff;
            sumXSquared += xDiff * xDiff;
            sumYSquared += yDiff * yDiff;
        }
        
        const denominator = Math.sqrt(sumXSquared * sumYSquared);
        return denominator > 0 ? numerator / denominator : 0;
    }
    
    identifyRegimes(timeseries, windowSize = 20) {
        if (!timeseries || timeseries.length < windowSize * 2) {
            return [];
        }
        
        const returns = this.calculateDailyReturns(timeseries);
        const regimes = [];
        
        for (let i = windowSize; i < returns.length - windowSize; i++) {
            const window = returns.slice(i - windowSize, i + windowSize);
            const volatility = this.calculateStandardDeviation(window);
            const meanReturn = this.calculateMean(window);
            
            let regime = 'normal';
            if (volatility > 0.03) { // High volatility threshold
                regime = meanReturn > 0 ? 'bull_volatile' : 'bear_volatile';
            } else if (volatility < 0.01) { // Low volatility threshold
                regime = 'low_volatility';
            } else if (meanReturn > 0.002) { // Strong positive trend
                regime = 'bull_market';
            } else if (meanReturn < -0.002) { // Strong negative trend
                regime = 'bear_market';
            }
            
            regimes.push({
                date: timeseries[i].date,
                regime: regime,
                volatility: volatility,
                return: meanReturn
            });
        }
        
        return regimes;
    }
    
    calculateMaximumAdverseExcursion(positions) {
        // Simplified MAE calculation
        if (!positions || positions.length === 0) {
            return { avg_mae: 0, max_mae: 0, mae_ratio: 0 };
        }
        
        // This would require intraday position data in a real implementation
        const estimatedMAE = positions.map(pos => {
            // Rough estimation based on position size and typical market movements
            return Math.abs(pos.quantity) * pos.entry_price * 0.02; // 2% adverse move
        });
        
        return {
            avg_mae: this.calculateMean(estimatedMAE),
            max_mae: Math.max(...estimatedMAE),
            mae_ratio: this.calculateMean(estimatedMAE) / (positions[0].entry_price || 1)
        };
    }
    
    generatePerformanceAttribution(strategy) {
        if (!strategy.timeseries || !strategy.metrics) {
            return null;
        }
        
        const returns = this.calculateDailyReturns(strategy.timeseries);
        const totalReturn = strategy.metrics.total_return;
        
        // Simplified attribution analysis
        return {
            market_return: totalReturn * 0.7, // Assume 70% from market
            alpha_return: totalReturn * 0.2,  // 20% from skill
            other_factors: totalReturn * 0.1,  // 10% from other factors
            volatility_contribution: this.calculateStandardDeviation(returns),
            concentration_risk: this.calculateConcentrationRisk(strategy)
        };
    }
    
    calculateConcentrationRisk(strategy) {
        // Simplified concentration risk based on position count
        const totalTrades = strategy.total_trades || 1;
        return totalTrades < 10 ? 'High' : totalTrades < 30 ? 'Medium' : 'Low';
    }
    
    // Utility statistical functions
    calculateMean(array) {
        if (!array || array.length === 0) return 0;
        return array.reduce((sum, val) => sum + val, 0) / array.length;
    }
    
    calculateStandardDeviation(array) {
        if (!array || array.length === 0) return 0;
        const mean = this.calculateMean(array);
        const variance = array.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / array.length;
        return Math.sqrt(variance);
    }
    
    calculatePercentile(array, percentile) {
        if (!array || array.length === 0) return 0;
        const sorted = [...array].sort((a, b) => a - b);
        const index = (percentile / 100) * (sorted.length - 1);
        const lower = Math.floor(index);
        const upper = Math.ceil(index);
        const weight = index - lower;
        
        return sorted[lower] * (1 - weight) + sorted[upper] * weight;
    }
    
    // Export analysis results
    exportAnalysis(strategies, format = 'json') {
        const analysis = {
            timestamp: new Date().toISOString(),
            strategies: {},
            comparison: {},
            risk_metrics: {}
        };
        
        // Analyze each strategy
        strategies.forEach((strategy, name) => {
            analysis.strategies[name] = {
                risk_adjusted_returns: this.calculateRiskAdjustedReturns(strategy),
                drawdown_analysis: this.analyzeDrawdownPeriods(strategy.timeseries),
                performance_attribution: this.generatePerformanceAttribution(strategy)
            };
        });
        
        // Compare strategies
        if (strategies.size > 1) {
            analysis.comparison = this.compareStrategies(Array.from(strategies.keys()));
            analysis.risk_metrics.correlation_matrix = this.calculateCorrelationMatrix(strategies);
        }
        
        if (format === 'json') {
            return JSON.stringify(analysis, null, 2);
        } else if (format === 'csv') {
            return this.convertAnalysisToCSV(analysis);
        }
        
        return analysis;
    }
    
    convertAnalysisToCSV(analysis) {
        // Simplified CSV export of key metrics
        let csv = 'Strategy,Total Return,Sharpe Ratio,Max Drawdown,Win Rate,Total Trades\n';
        
        Object.keys(analysis.strategies).forEach(strategyName => {
            const strategy = this.strategies.get(strategyName);
            if (strategy && strategy.metrics) {
                csv += `${strategyName},${strategy.metrics.total_return},${strategy.metrics.sharpe_ratio},${strategy.metrics.max_drawdown},${strategy.metrics.win_rate},${strategy.total_trades}\n`;
            }
        });
        
        return csv;
    }
}