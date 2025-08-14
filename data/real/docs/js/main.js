// Main application controller
class VolatilityAlchemist {
    constructor() {
        this.dataLoader = new DataLoader();
        this.chartManager = new ChartManager();
        this.strategyAnalyzer = new StrategyAnalyzer();
        
        this.currentStrategy = 'all';
        this.currentSymbol = 'SPY';
        this.currentData = null;
        
        this.init();
    }
    
    async init() {
        this.setupEventListeners();
        this.showLoading();
        
        try {
            await this.loadAndDisplayData();
        } catch (error) {
            console.error('Initialization error:', error);
            this.showError('Failed to load initial data. Please check if data files exist.');
        } finally {
            this.hideLoading();
        }
    }
    
    setupEventListeners() {
        // Strategy selector
        const strategySelect = document.getElementById('strategySelect');
        strategySelect.addEventListener('change', (e) => {
            this.currentStrategy = e.target.value;
            this.updateDashboard();
        });
        
        // Symbol selector
        const symbolSelect = document.getElementById('symbolSelect');
        symbolSelect.addEventListener('change', (e) => {
            this.currentSymbol = e.target.value;
            this.loadAndDisplayData();
        });
        
        // Refresh button
        const refreshBtn = document.getElementById('refreshBtn');
        refreshBtn.addEventListener('click', () => {
            this.loadAndDisplayData();
        });
        
        // Handle window resize for chart responsiveness
        window.addEventListener('resize', () => {
            this.chartManager.resizeAllCharts();
        });
    }
    
    async loadAndDisplayData() {
        this.showLoading();
        
        try {
            // Load strategy data
            const strategyData = await this.dataLoader.loadStrategyData(this.currentSymbol);
            
            // Load volatility surface data
            const volatilityData = await this.dataLoader.loadVolatilitySurface(this.currentSymbol);
            
            // Store current data
            this.currentData = {
                strategies: strategyData,
                volatility: volatilityData
            };
            
            // Update dashboard
            this.updateDashboard();
            
            this.showSuccess('Data loaded successfully');
            
        } catch (error) {
            console.error('Data loading error:', error);
            this.showError(`Failed to load data for ${this.currentSymbol}. Please ensure data files exist.`);
        } finally {
            this.hideLoading();
        }
    }
    
    updateDashboard() {
        if (!this.currentData) {
            return;
        }
        
        try {
            // Filter data based on current strategy selection
            const filteredData = this.filterStrategyData();
            
            // Update charts
            this.updatePerformanceChart(filteredData);
            this.updateVolatilitySurface();
            this.updateGreeksChart(filteredData);
            this.updateDrawdownChart(filteredData);
            this.updateComparisonChart();
            
            // Update metrics table and summary
            this.updateMetricsTable(filteredData);
            this.updateSummaryCards(filteredData);
            
        } catch (error) {
            console.error('Dashboard update error:', error);
            this.showError('Failed to update dashboard visualizations');
        }
    }
    
    filterStrategyData() {
        if (!this.currentData.strategies) {
            return [];
        }
        
        if (this.currentStrategy === 'all') {
            return this.currentData.strategies;
        }
        
        return this.currentData.strategies.filter(strategy => 
            strategy.strategy.toLowerCase().includes(this.currentStrategy.toLowerCase())
        );
    }
    
    updatePerformanceChart(data) {
        if (!data || data.length === 0) {
            this.chartManager.showEmptyChart('performanceChart', 'No performance data available');
            return;
        }
        
        const primaryStrategy = data[0];
        if (!primaryStrategy.timeseries) {
            this.chartManager.showEmptyChart('performanceChart', 'No timeseries data available');
            return;
        }
        
        const dates = primaryStrategy.timeseries.map(t => t.date);
        const pnl = primaryStrategy.timeseries.map(t => t.pnl);
        const equity = primaryStrategy.timeseries.map(t => t.equity);
        
        this.chartManager.createPerformanceChart('performanceChart', {
            dates: dates,
            pnl: pnl,
            equity: equity,
            title: `${primaryStrategy.strategy} - ${primaryStrategy.symbol}`
        });
    }
    
    updateVolatilitySurface() {
        if (!this.currentData.volatility) {
            this.chartManager.showEmptyChart('volatilityChart', 'No volatility surface data available');
            return;
        }
        
        const surface = this.currentData.volatility.surface;
        this.chartManager.createVolatilitySurface('volatilityChart', {
            strikes: surface.strikes,
            maturities: surface.maturities,
            ivMatrix: surface.iv_matrix,
            symbol: this.currentData.volatility.metadata.symbol
        });
    }
    
    updateGreeksChart(data) {
        if (!data || data.length === 0) {
            this.chartManager.showEmptyChart('greeksChart', 'No Greeks data available');
            return;
        }
        
        const primaryStrategy = data[0];
        if (!primaryStrategy.timeseries) {
            this.chartManager.showEmptyChart('greeksChart', 'No Greeks timeseries available');
            return;
        }
        
        const dates = primaryStrategy.timeseries.map(t => t.date);
        const delta = primaryStrategy.timeseries.map(t => t.delta);
        const gamma = primaryStrategy.timeseries.map(t => t.gamma);
        const vega = primaryStrategy.timeseries.map(t => t.vega);
        const theta = primaryStrategy.timeseries.map(t => t.theta);
        
        this.chartManager.createGreeksChart('greeksChart', {
            dates: dates,
            delta: delta,
            gamma: gamma,
            vega: vega,
            theta: theta
        });
    }
    
    updateDrawdownChart(data) {
        if (!data || data.length === 0) {
            this.chartManager.showEmptyChart('drawdownChart', 'No drawdown data available');
            return;
        }
        
        const primaryStrategy = data[0];
        if (!primaryStrategy.timeseries) {
            this.chartManager.showEmptyChart('drawdownChart', 'No drawdown timeseries available');
            return;
        }
        
        const dates = primaryStrategy.timeseries.map(t => t.date);
        const drawdowns = primaryStrategy.timeseries.map(t => -t.drawdown * 100); // Convert to percentage
        
        this.chartManager.createDrawdownChart('drawdownChart', {
            dates: dates,
            drawdowns: drawdowns,
            title: `Drawdown Analysis - ${primaryStrategy.strategy}`
        });
    }
    
    updateComparisonChart() {
        if (!this.currentData.strategies || this.currentData.strategies.length === 0) {
            this.chartManager.showEmptyChart('comparisonChart', 'No strategy data for comparison');
            return;
        }
        
        const strategies = this.currentData.strategies.map(strategy => ({
            name: strategy.strategy,
            totalReturn: strategy.metrics.total_return * 100,
            sharpeRatio: strategy.metrics.sharpe_ratio,
            maxDrawdown: strategy.metrics.max_drawdown * 100
        }));
        
        this.chartManager.createComparisonChart('comparisonChart', strategies);
    }
    
    updateMetricsTable(data) {
        const container = document.getElementById('metricsTable');
        
        if (!data || data.length === 0) {
            container.innerHTML = '<p class="text-center">No metrics data available</p>';
            return;
        }
        
        const primaryStrategy = data[0];
        const metrics = primaryStrategy.metrics;
        
        const metricsData = [
            { label: 'Total Return', value: this.formatPercentage(metrics.total_return) },
            { label: 'Annualized Return', value: this.formatPercentage(metrics.annualized_return) },
            { label: 'Sharpe Ratio', value: this.formatNumber(metrics.sharpe_ratio) },
            { label: 'Sortino Ratio', value: this.formatNumber(metrics.sortino_ratio) },
            { label: 'Maximum Drawdown', value: this.formatPercentage(metrics.max_drawdown) },
            { label: 'Win Rate', value: this.formatPercentage(metrics.win_rate) },
            { label: 'Profit Factor', value: this.formatNumber(metrics.profit_factor) },
            { label: 'VaR (95%)', value: this.formatPercentage(metrics.var_95) },
            { label: 'CVaR (95%)', value: this.formatPercentage(metrics.cvar_95) }
        ];
        
        let tableHTML = '<table class="metrics-table"><thead><tr><th>Metric</th><th>Value</th></tr></thead><tbody>';
        
        metricsData.forEach(metric => {
            tableHTML += `<tr><td>${metric.label}</td><td>${metric.value}</td></tr>`;
        });
        
        tableHTML += '</tbody></table>';
        container.innerHTML = tableHTML;
    }
    
    updateSummaryCards(data) {
        if (!data || data.length === 0) {
            this.clearSummaryCards();
            return;
        }
        
        const primaryStrategy = data[0];
        const metrics = primaryStrategy.metrics;
        
        // Update summary cards
        this.updateSummaryCard('totalReturn', this.formatPercentage(metrics.total_return), metrics.total_return >= 0);
        this.updateSummaryCard('sharpeRatio', this.formatNumber(metrics.sharpe_ratio), metrics.sharpe_ratio >= 1);
        this.updateSummaryCard('maxDrawdown', this.formatPercentage(metrics.max_drawdown), false);
        this.updateSummaryCard('winRate', this.formatPercentage(metrics.win_rate), metrics.win_rate >= 0.5);
        this.updateSummaryCard('totalTrades', primaryStrategy.total_trades.toString(), true);
    }
    
    updateSummaryCard(elementId, value, isPositive) {
        const element = document.getElementById(elementId);
        if (element) {
            element.textContent = value;
            element.className = 'metric-value';
            
            if (elementId !== 'totalTrades') {
                element.classList.add(isPositive ? 'positive' : 'negative');
            }
        }
    }
    
    clearSummaryCards() {
        ['totalReturn', 'sharpeRatio', 'maxDrawdown', 'winRate', 'totalTrades'].forEach(id => {
            const element = document.getElementById(id);
            if (element) {
                element.textContent = '-';
                element.className = 'metric-value';
            }
        });
    }
    
    // Utility methods
    formatPercentage(value) {
        if (typeof value !== 'number' || isNaN(value)) return '-';
        return (value * 100).toFixed(2) + '%';
    }
    
    formatNumber(value) {
        if (typeof value !== 'number' || isNaN(value)) return '-';
        return value.toFixed(3);
    }
    
    showLoading() {
        const overlay = document.getElementById('loadingOverlay');
        if (overlay) {
            overlay.classList.remove('hidden');
        }
    }
    
    hideLoading() {
        const overlay = document.getElementById('loadingOverlay');
        if (overlay) {
            overlay.classList.add('hidden');
        }
    }
    
    showError(message) {
        this.showMessage(message, 'error-message');
    }
    
    showSuccess(message) {
        this.showMessage(message, 'success-message');
    }
    
    showMessage(message, className) {
        // Remove existing messages
        document.querySelectorAll('.error-message, .success-message').forEach(el => el.remove());
        
        const messageDiv = document.createElement('div');
        messageDiv.className = className;
        messageDiv.textContent = message;
        
        const dashboard = document.querySelector('.dashboard');
        dashboard.insertBefore(messageDiv, dashboard.firstChild);
        
        // Auto-remove success messages
        if (className === 'success-message') {
            setTimeout(() => {
                messageDiv.remove();
            }, 3000);
        }
    }
}

// Initialize application when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.app = new VolatilityAlchemist();
});