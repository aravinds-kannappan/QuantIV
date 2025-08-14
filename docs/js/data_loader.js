// Data loading and management class
class DataLoader {
    constructor() {
        this.baseUrl = 'data/';
        this.cache = new Map();
        this.cacheTimeout = 5 * 60 * 1000; // 5 minutes
    }
    
    async loadStrategyData(symbol) {
        try {
            const strategies = [];
            const strategyTypes = ['covered_call', 'vertical_spread', 'straddle'];
            
            for (const strategyType of strategyTypes) {
                try {
                    const data = await this.loadJSON(`strategies/${strategyType}_${symbol}_backtest.json`);
                    if (data) {
                        strategies.push(data);
                    }
                } catch (error) {
                    console.warn(`Failed to load ${strategyType} data for ${symbol}:`, error);
                    // Continue loading other strategies
                }
            }
            
            return strategies;
        } catch (error) {
            console.error('Error loading strategy data:', error);
            throw new Error(`Failed to load strategy data for ${symbol}`);
        }
    }
    
    async loadVolatilitySurface(symbol) {
        try {
            // Try to find the most recent volatility surface file
            const possibleFiles = [
                `iv_surfaces/${symbol}_iv_surface_2025-05-20.json`,
                `iv_surfaces/${symbol}_iv_surface.json`,
                `iv_surfaces/${symbol.toLowerCase()}_iv_surface.json`
            ];
            
            for (const filename of possibleFiles) {
                try {
                    const data = await this.loadJSON(filename);
                    if (data) {
                        return data;
                    }
                } catch (error) {
                    // Continue to next file
                    console.warn(`Failed to load ${filename}:`, error);
                }
            }
            
            throw new Error(`No volatility surface data found for ${symbol}`);
        } catch (error) {
            console.error('Error loading volatility surface:', error);
            throw error;
        }
    }
    
    async loadPerformanceSummary() {
        try {
            return await this.loadJSON('metrics/performance_summary.json');
        } catch (error) {
            console.error('Error loading performance summary:', error);
            return null;
        }
    }
    
    async loadTimeseriesCSV(symbol, strategy) {
        try {
            const filename = `strategies/${strategy}_${symbol}_timeseries.csv`;
            return await this.loadCSV(filename);
        } catch (error) {
            console.error('Error loading timeseries CSV:', error);
            return null;
        }
    }
    
    async loadPositionsData(symbol, strategy) {
        try {
            const filename = `strategies/${strategy}_${symbol}_positions.csv`;
            return await this.loadCSV(filename);
        } catch (error) {
            console.error('Error loading positions data:', error);
            return null;
        }
    }
    
    async loadJSON(filename) {
        const cacheKey = `json_${filename}`;
        
        // Check cache first
        if (this.cache.has(cacheKey)) {
            const cached = this.cache.get(cacheKey);
            if (Date.now() - cached.timestamp < this.cacheTimeout) {
                return cached.data;
            }
        }
        
        try {
            const response = await fetch(this.baseUrl + filename);
            
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }
            
            const data = await response.json();
            
            // Cache the result
            this.cache.set(cacheKey, {
                data: data,
                timestamp: Date.now()
            });
            
            return data;
        } catch (error) {
            console.error(`Failed to load JSON file ${filename}:`, error);
            throw error;
        }
    }
    
    async loadCSV(filename) {
        const cacheKey = `csv_${filename}`;
        
        // Check cache first
        if (this.cache.has(cacheKey)) {
            const cached = this.cache.get(cacheKey);
            if (Date.now() - cached.timestamp < this.cacheTimeout) {
                return cached.data;
            }
        }
        
        try {
            const response = await fetch(this.baseUrl + filename);
            
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }
            
            const text = await response.text();
            const data = this.parseCSV(text);
            
            // Cache the result
            this.cache.set(cacheKey, {
                data: data,
                timestamp: Date.now()
            });
            
            return data;
        } catch (error) {
            console.error(`Failed to load CSV file ${filename}:`, error);
            throw error;
        }
    }
    
    parseCSV(csvText) {
        const lines = csvText.trim().split('\n');
        if (lines.length === 0) {
            return [];
        }
        
        const headers = lines[0].split(',').map(h => h.trim());
        const data = [];
        
        for (let i = 1; i < lines.length; i++) {
            const values = lines[i].split(',').map(v => v.trim());
            
            if (values.length === headers.length) {
                const row = {};
                headers.forEach((header, index) => {
                    const value = values[index];
                    // Try to parse as number, otherwise keep as string
                    if (!isNaN(value) && value !== '') {
                        row[header] = parseFloat(value);
                    } else {
                        row[header] = value;
                    }
                });
                data.push(row);
            }
        }
        
        return data;
    }
    
    // Generate sample data if files don't exist
    generateSampleStrategyData(symbol, strategy) {
        const dates = this.generateDateRange('2024-01-01', '2024-12-31', 21); // Monthly data
        const timeseries = dates.map((date, index) => {
            const t = index / dates.length;
            const volatility = 0.02 + 0.01 * Math.sin(t * 2 * Math.PI * 3); // Seasonal volatility
            const drift = 0.08 / 252; // 8% annual return
            const randomWalk = (Math.random() - 0.5) * volatility * Math.sqrt(1/252);
            
            const baseReturn = drift + randomWalk;
            const pnl = index === 0 ? 0 : (1000 * baseReturn * (index + 1));
            const equity = 100000 + pnl;
            
            return {
                date: date,
                pnl: pnl,
                equity: equity,
                drawdown: Math.max(0, (Math.max(...dates.slice(0, index + 1).map((_, i) => 100000 + 1000 * baseReturn * (i + 1))) - equity) / 100000),
                delta: 0.5 + 0.3 * Math.sin(t * 2 * Math.PI) * (Math.random() - 0.5),
                gamma: 0.02 + 0.01 * Math.random(),
                vega: 50 + 25 * Math.random(),
                theta: -25 - 10 * Math.random()
            };
        });
        
        return {
            strategy: this.formatStrategyName(strategy),
            symbol: symbol,
            period: {
                start: dates[0],
                end: dates[dates.length - 1]
            },
            metrics: {
                total_return: (timeseries[timeseries.length - 1].equity - 100000) / 100000,
                annualized_return: 0.08 + (Math.random() - 0.5) * 0.04,
                sharpe_ratio: 1.2 + (Math.random() - 0.5) * 0.8,
                sortino_ratio: 1.5 + (Math.random() - 0.5) * 0.8,
                max_drawdown: Math.max(...timeseries.map(t => t.drawdown)),
                win_rate: 0.55 + (Math.random() - 0.5) * 0.2,
                profit_factor: 1.3 + (Math.random() - 0.5) * 0.6,
                var_95: -0.02 - Math.random() * 0.01,
                cvar_95: -0.025 - Math.random() * 0.015
            },
            timeseries: timeseries,
            total_trades: 15 + Math.floor(Math.random() * 20),
            winning_trades: Math.floor((15 + Math.random() * 20) * (0.55 + (Math.random() - 0.5) * 0.2))
        };
    }
    
    generateSampleVolatilitySurface(symbol) {
        const strikes = [];
        const basePrice = symbol === 'SPY' ? 520 : symbol === 'AAPL' ? 180 : 350;
        
        for (let i = -10; i <= 10; i++) {
            strikes.push(basePrice + i * (basePrice * 0.05));
        }
        
        const maturities = [7/365, 14/365, 30/365, 60/365, 90/365, 180/365]; // Days to years
        
        const iv_matrix = strikes.map((strike, i) => {
            return maturities.map((maturity, j) => {
                const moneyness = strike / basePrice;
                const atmVol = 0.20;
                const skew = 0.1 * (1 - moneyness); // Volatility skew
                const termStructure = 0.02 * Math.sqrt(maturity); // Term structure effect
                
                return Math.max(0.05, atmVol + skew + termStructure + (Math.random() - 0.5) * 0.02);
            });
        });
        
        return {
            metadata: {
                symbol: symbol,
                date: '2025-05-20',
                underlying_price: basePrice
            },
            surface: {
                strikes: strikes,
                maturities: maturities,
                iv_matrix: iv_matrix
            }
        };
    }
    
    generateDateRange(startDate, endDate, step = 1) {
        const dates = [];
        const start = new Date(startDate);
        const end = new Date(endDate);
        
        let current = new Date(start);
        while (current <= end) {
            dates.push(current.toISOString().split('T')[0]);
            current.setDate(current.getDate() + step);
        }
        
        return dates;
    }
    
    formatStrategyName(strategy) {
        return strategy.split('_').map(word => 
            word.charAt(0).toUpperCase() + word.slice(1)
        ).join(' ');
    }
    
    clearCache() {
        this.cache.clear();
    }
    
    // Check if data files exist
    async checkDataAvailability(symbol) {
        const availability = {
            strategies: {},
            volatility: false,
            summary: false
        };
        
        const strategyTypes = ['covered_call', 'vertical_spread', 'straddle'];
        
        for (const strategy of strategyTypes) {
            try {
                await this.loadJSON(`strategies/${strategy}_${symbol}_backtest.json`);
                availability.strategies[strategy] = true;
            } catch {
                availability.strategies[strategy] = false;
            }
        }
        
        try {
            await this.loadVolatilitySurface(symbol);
            availability.volatility = true;
        } catch {
            availability.volatility = false;
        }
        
        try {
            await this.loadPerformanceSummary();
            availability.summary = true;
        } catch {
            availability.summary = false;
        }
        
        return availability;
    }
}