// Chart management and visualization class
class ChartManager {
    constructor() {
        this.defaultLayout = {
            autosize: true,
            margin: { l: 50, r: 50, t: 50, b: 50 },
            font: { family: 'Segoe UI, Tahoma, Geneva, Verdana, sans-serif', size: 12 },
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(0,0,0,0)',
            showlegend: true,
            legend: { x: 0, y: 1, bgcolor: 'rgba(255,255,255,0.8)' }
        };
        
        this.colorScheme = {
            primary: '#2a5298',
            secondary: '#1e3c72',
            success: '#10b981',
            danger: '#ef4444',
            warning: '#f59e0b',
            info: '#3b82f6',
            light: '#f8f9fa',
            dark: '#343a40'
        };
    }
    
    createPerformanceChart(containerId, data) {
        const traces = [
            {
                x: data.dates,
                y: data.equity,
                type: 'scatter',
                mode: 'lines',
                name: 'Equity Curve',
                line: { color: this.colorScheme.primary, width: 2 },
                hovertemplate: 'Date: %{x}<br>Equity: $%{y:,.2f}<extra></extra>'
            },
            {
                x: data.dates,
                y: data.pnl,
                type: 'scatter',
                mode: 'lines',
                name: 'P&L',
                yaxis: 'y2',
                line: { color: this.colorScheme.success, width: 2 },
                hovertemplate: 'Date: %{x}<br>P&L: $%{y:,.2f}<extra></extra>'
            }
        ];
        
        const layout = {
            ...this.defaultLayout,
            title: data.title || 'Strategy Performance',
            xaxis: { title: 'Date', type: 'date' },
            yaxis: { title: 'Equity ($)', side: 'left', tickformat: '$,.0f' },
            yaxis2: {
                title: 'P&L ($)',
                side: 'right',
                overlaying: 'y',
                tickformat: '$,.0f',
                zeroline: true,
                zerolinecolor: '#ccc'
            }
        };
        
        Plotly.newPlot(containerId, traces, layout, { responsive: true });
    }
    
    createVolatilitySurface(containerId, data) {
        if (!data.strikes || !data.maturities || !data.ivMatrix) {
            this.showEmptyChart(containerId, 'Invalid volatility surface data');
            return;
        }
        
        // Convert maturities from years to days for better readability
        const maturityDays = data.maturities.map(m => Math.round(m * 365));
        
        const trace = {
            x: maturityDays,
            y: data.strikes,
            z: data.ivMatrix,
            type: 'surface',
            colorscale: [
                [0, '#1e3c72'],
                [0.25, '#2a5298'],
                [0.5, '#3b82f6'],
                [0.75, '#10b981'],
                [1, '#f59e0b']
            ],
            hovertemplate: 'Maturity: %{x} days<br>Strike: $%{y}<br>IV: %{z:.2%}<extra></extra>',
            colorbar: {
                title: 'Implied Volatility',
                tickformat: '.1%'
            }
        };
        
        const layout = {
            ...this.defaultLayout,
            title: `Implied Volatility Surface - ${data.symbol}`,
            scene: {
                xaxis: { title: 'Time to Expiry (Days)' },
                yaxis: { title: 'Strike Price ($)' },
                zaxis: { title: 'Implied Volatility', tickformat: '.1%' },
                camera: {
                    eye: { x: 1.5, y: 1.5, z: 1.5 }
                }
            },
            showlegend: false
        };
        
        Plotly.newPlot(containerId, [trace], layout, { responsive: true });
    }
    
    createGreeksChart(containerId, data) {
        const traces = [
            {
                x: data.dates,
                y: data.delta,
                type: 'scatter',
                mode: 'lines',
                name: 'Delta',
                line: { color: this.colorScheme.primary, width: 2 },
                hovertemplate: 'Date: %{x}<br>Delta: %{y:.3f}<extra></extra>'
            },
            {
                x: data.dates,
                y: data.gamma,
                type: 'scatter',
                mode: 'lines',
                name: 'Gamma',
                yaxis: 'y2',
                line: { color: this.colorScheme.success, width: 2 },
                hovertemplate: 'Date: %{x}<br>Gamma: %{y:.4f}<extra></extra>'
            },
            {
                x: data.dates,
                y: data.vega,
                type: 'scatter',
                mode: 'lines',
                name: 'Vega',
                yaxis: 'y3',
                line: { color: this.colorScheme.warning, width: 2 },
                hovertemplate: 'Date: %{x}<br>Vega: %{y:.2f}<extra></extra>'
            },
            {
                x: data.dates,
                y: data.theta,
                type: 'scatter',
                mode: 'lines',
                name: 'Theta',
                yaxis: 'y4',
                line: { color: this.colorScheme.danger, width: 2 },
                hovertemplate: 'Date: %{x}<br>Theta: %{y:.2f}<extra></extra>'
            }
        ];
        
        const layout = {
            ...this.defaultLayout,
            title: 'Greeks Exposure Over Time',
            xaxis: { title: 'Date', type: 'date', domain: [0, 0.95] },
            yaxis: { title: 'Delta', side: 'left', position: 0 },
            yaxis2: { title: 'Gamma', side: 'right', overlaying: 'y', position: 0.95 },
            yaxis3: { title: 'Vega', side: 'left', overlaying: 'y', position: 0.05 },
            yaxis4: { title: 'Theta', side: 'right', overlaying: 'y', position: 0.9 },
            legend: { x: 0, y: 1.1, orientation: 'h' }
        };
        
        Plotly.newPlot(containerId, traces, layout, { responsive: true });
    }
    
    createDrawdownChart(containerId, data) {
        const trace = {
            x: data.dates,
            y: data.drawdowns,
            type: 'scatter',
            mode: 'lines',
            fill: 'tonexty',
            fillcolor: 'rgba(239, 68, 68, 0.3)',
            line: { color: this.colorScheme.danger, width: 2 },
            name: 'Drawdown',
            hovertemplate: 'Date: %{x}<br>Drawdown: %{y:.2f}%<extra></extra>'
        };
        
        // Add zero line
        const zeroTrace = {
            x: data.dates,
            y: new Array(data.dates.length).fill(0),
            type: 'scatter',
            mode: 'lines',
            line: { color: '#666', width: 1, dash: 'dash' },
            name: 'Zero Line',
            showlegend: false
        };
        
        const layout = {
            ...this.defaultLayout,
            title: data.title || 'Drawdown Analysis',
            xaxis: { title: 'Date', type: 'date' },
            yaxis: { 
                title: 'Drawdown (%)', 
                tickformat: '.2f',
                range: [Math.min(...data.drawdowns) * 1.1, 1]
            }
        };
        
        Plotly.newPlot(containerId, [zeroTrace, trace], layout, { responsive: true });
    }
    
    createComparisonChart(containerId, strategies) {
        const traces = [
            {
                x: strategies.map(s => s.name),
                y: strategies.map(s => s.totalReturn),
                type: 'bar',
                name: 'Total Return (%)',
                marker: { color: this.colorScheme.primary },
                hovertemplate: 'Strategy: %{x}<br>Total Return: %{y:.2f}%<extra></extra>'
            },
            {
                x: strategies.map(s => s.name),
                y: strategies.map(s => s.sharpeRatio),
                type: 'scatter',
                mode: 'markers+lines',
                name: 'Sharpe Ratio',
                yaxis: 'y2',
                marker: { color: this.colorScheme.success, size: 10 },
                line: { color: this.colorScheme.success, width: 2 },
                hovertemplate: 'Strategy: %{x}<br>Sharpe Ratio: %{y:.2f}<extra></extra>'
            }
        ];
        
        const layout = {
            ...this.defaultLayout,
            title: 'Strategy Performance Comparison',
            xaxis: { title: 'Strategy' },
            yaxis: { title: 'Total Return (%)', side: 'left' },
            yaxis2: {
                title: 'Sharpe Ratio',
                side: 'right',
                overlaying: 'y'
            },
            barmode: 'group'
        };
        
        Plotly.newPlot(containerId, traces, layout, { responsive: true });
    }
    
    createVolatilityTermStructure(containerId, data) {
        const traces = data.dates.map((date, index) => ({
            x: data.maturities,
            y: data.termStructures[index],
            type: 'scatter',
            mode: 'lines+markers',
            name: date,
            hovertemplate: 'Maturity: %{x:.0f} days<br>IV: %{y:.2%}<extra></extra>'
        }));
        
        const layout = {
            ...this.defaultLayout,
            title: 'Volatility Term Structure Evolution',
            xaxis: { title: 'Time to Expiry (Days)' },
            yaxis: { title: 'Implied Volatility', tickformat: '.1%' }
        };
        
        Plotly.newPlot(containerId, traces, layout, { responsive: true });
    }
    
    createRiskMetricsRadar(containerId, data) {
        const trace = {
            type: 'scatterpolar',
            r: data.values,
            theta: data.metrics,
            fill: 'toself',
            fillcolor: 'rgba(42, 82, 152, 0.3)',
            line: { color: this.colorScheme.primary, width: 2 },
            marker: { color: this.colorScheme.primary, size: 8 },
            hovertemplate: '%{theta}: %{r:.2f}<extra></extra>'
        };
        
        const layout = {
            ...this.defaultLayout,
            title: 'Risk Metrics Profile',
            polar: {
                radialaxis: {
                    visible: true,
                    range: [0, Math.max(...data.values) * 1.1]
                }
            },
            showlegend: false
        };
        
        Plotly.newPlot(containerId, [trace], layout, { responsive: true });
    }
    
    showEmptyChart(containerId, message) {
        const container = document.getElementById(containerId);
        if (container) {
            container.innerHTML = `
                <div style="display: flex; align-items: center; justify-content: center; height: 300px; color: #666; font-style: italic;">
                    <div style="text-align: center;">
                        <div style="font-size: 2rem; margin-bottom: 1rem;">📊</div>
                        <div>${message}</div>
                    </div>
                </div>
            `;
        }
    }
    
    resizeAllCharts() {
        // Resize all Plotly charts
        const chartContainers = document.querySelectorAll('.chart-container > div');
        chartContainers.forEach(container => {
            if (container.id && window.Plotly) {
                Plotly.Plots.resize(container.id);
            }
        });
    }
    
    // Utility method to create custom color scales
    createColorScale(colors) {
        const step = 1 / (colors.length - 1);
        return colors.map((color, index) => [index * step, color]);
    }
    
    // Method to update chart data without full redraw
    updateChartData(containerId, newData, traceIndex = 0) {
        if (window.Plotly && document.getElementById(containerId)) {
            Plotly.restyle(containerId, newData, traceIndex);
        }
    }
    
    // Method to update chart layout
    updateChartLayout(containerId, newLayout) {
        if (window.Plotly && document.getElementById(containerId)) {
            Plotly.relayout(containerId, newLayout);
        }
    }
    
    // Method to add annotation to charts
    addAnnotation(containerId, annotation) {
        const update = {
            'annotations[0]': annotation
        };
        this.updateChartLayout(containerId, update);
    }
    
    // Export chart as image
    async exportChart(containerId, filename, format = 'png') {
        if (window.Plotly && document.getElementById(containerId)) {
            try {
                const img = await Plotly.toImage(containerId, {
                    format: format,
                    width: 1200,
                    height: 800
                });
                
                const link = document.createElement('a');
                link.download = filename;
                link.href = img;
                link.click();
            } catch (error) {
                console.error('Error exporting chart:', error);
            }
        }
    }
}