/**
 * CUSTOMER SEGMENTATION — CHARTS VISUALIZATION ENGINE
 * Built with Chart.js
 */

class DashboardCharts {
    constructor() {
        this.pieChart = null;
        this.barChart = null;
        this.scatterChart = null;
    }

    renderRevenuePieChart(canvasId, clusters) {
        const ctx = document.getElementById(canvasId).getContext('2d');
        
        if (this.pieChart) {
            this.pieChart.destroy();
        }

        const labels = clusters.map(c => c.persona_title);
        const data = clusters.map(c => c.metrics.total_revenue);
        const colors = clusters.map(c => c.color);

        this.pieChart = new Chart(ctx, {
            type: 'doughnut',
            data: {
                labels: labels,
                datasets: [{
                    data: data,
                    backgroundColor: colors,
                    borderWidth: 2,
                    borderColor: '#0B0F17'
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: 'right',
                        labels: {
                            color: '#94A3B8',
                            font: { family: 'Plus Jakarta Sans', size: 12 }
                        }
                    },
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                const val = context.raw || 0;
                                return ` Revenue: $${val.toLocaleString()}`;
                            }
                        }
                    }
                },
                cutout: '65%'
            }
        });
    }

    renderRecencyMonetaryChart(canvasId, clusters) {
        const ctx = document.getElementById(canvasId).getContext('2d');

        if (this.barChart) {
            this.barChart.destroy();
        }

        const labels = clusters.map(c => c.persona_title.split(' ')[1] || c.persona_title);
        const monetaryData = clusters.map(c => c.metrics.avg_monetary_spend);
        const recencyData = clusters.map(c => c.metrics.avg_recency_days);

        this.barChart = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: labels,
                datasets: [
                    {
                        label: 'Avg Monetary Spend ($)',
                        data: monetaryData,
                        backgroundColor: '#3B82F6',
                        borderRadius: 6,
                        yAxisID: 'y'
                    },
                    {
                        label: 'Avg Recency Gap (Days)',
                        data: recencyData,
                        backgroundColor: '#EF4444',
                        borderRadius: 6,
                        yAxisID: 'y1'
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    x: {
                        ticks: { color: '#94A3B8' },
                        grid: { color: 'rgba(255, 255, 255, 0.05)' }
                    },
                    y: {
                        type: 'linear',
                        display: true,
                        position: 'left',
                        ticks: { color: '#3B82F6' },
                        grid: { color: 'rgba(255, 255, 255, 0.05)' },
                        title: { display: true, text: 'Spend ($)', color: '#3B82F6' }
                    },
                    y1: {
                        type: 'linear',
                        display: true,
                        position: 'right',
                        ticks: { color: '#EF4444' },
                        grid: { drawOnChartArea: false },
                        title: { display: true, text: 'Recency (Days)', color: '#EF4444' }
                    }
                },
                plugins: {
                    legend: {
                        labels: { color: '#94A3B8' }
                    }
                }
            }
        });
    }

    renderPcaScatterChart(canvasId, points) {
        const ctx = document.getElementById(canvasId).getContext('2d');

        if (this.scatterChart) {
            this.scatterChart.destroy();
        }

        // Group points by cluster persona for separate legend items
        const clustersMap = {};
        points.forEach(p => {
            if (!clustersMap[p.persona]) {
                clustersMap[p.persona] = {
                    label: p.persona,
                    color: p.color,
                    data: []
                };
            }
            clustersMap[p.persona].data.push({
                x: p.x,
                y: p.y,
                id: p.id,
                spend: p.spend,
                freq: p.freq,
                recency: p.recency
            });
        });

        const datasets = Object.values(clustersMap).map(c => ({
            label: c.label,
            data: c.data,
            backgroundColor: c.color,
            pointRadius: 4,
            pointHoverRadius: 7
        }));

        this.scatterChart = new Chart(ctx, {
            type: 'scatter',
            data: { datasets: datasets },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    x: {
                        title: { display: true, text: 'PCA Dimension 1', color: '#94A3B8' },
                        ticks: { color: '#64748B' },
                        grid: { color: 'rgba(255, 255, 255, 0.05)' }
                    },
                    y: {
                        title: { display: true, text: 'PCA Dimension 2', color: '#94A3B8' },
                        ticks: { color: '#64748B' },
                        grid: { color: 'rgba(255, 255, 255, 0.05)' }
                    }
                },
                plugins: {
                    legend: {
                        position: 'top',
                        labels: { color: '#94A3B8' }
                    },
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                const pt = context.raw;
                                return `${pt.id}: Spend $${pt.spend.toLocaleString()} | Orders: ${pt.freq} | Recency: ${pt.recency}d`;
                            }
                        }
                    }
                }
            }
        });
    }
}

window.dashboardCharts = new DashboardCharts();
