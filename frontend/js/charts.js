/**
 * CUSTOMER SEGMENTATION — CHARTS VISUALIZATION ENGINE
 * Built with Chart.js
 */

class DashboardCharts {
    constructor() {
        this.pieChart = null;
        this.barChart = null;
        this.scatterChart = null;
        this.elbowChart = null;
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

    renderElbowSilhouetteChart(canvasId, gridData) {
        const ctx = document.getElementById(canvasId).getContext('2d');

        if (this.elbowChart) {
            this.elbowChart.destroy();
        }

        const ks = gridData.map(g => `K=${g.k}`);
        const inertias = gridData.map(g => g.inertia);
        const silhouettes = gridData.map(g => g.silhouette_score);

        this.elbowChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: ks,
                datasets: [
                    {
                        label: 'Inertia (Elbow Drop)',
                        data: inertias,
                        borderColor: '#3B82F6',
                        backgroundColor: 'rgba(59, 130, 246, 0.1)',
                        tension: 0.3,
                        pointRadius: 6,
                        yAxisID: 'y'
                    },
                    {
                        label: 'Silhouette Score (Peak Optimal)',
                        data: silhouettes,
                        borderColor: '#10B981',
                        backgroundColor: 'rgba(16, 185, 129, 0.1)',
                        tension: 0.3,
                        pointRadius: 6,
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
                        position: 'left',
                        ticks: { color: '#3B82F6' },
                        grid: { color: 'rgba(255, 255, 255, 0.05)' },
                        title: { display: true, text: 'Inertia', color: '#3B82F6' }
                    },
                    y1: {
                        type: 'linear',
                        position: 'right',
                        ticks: { color: '#10B981' },
                        grid: { drawOnChartArea: false },
                        title: { display: true, text: 'Silhouette Score', color: '#10B981' }
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

    renderPlotly3dPcaPlot(containerId, points) {
        const container = document.getElementById(containerId);
        if (!container || !window.Plotly) return;

        const clustersMap = {};
        points.forEach(p => {
            if (!clustersMap[p.persona]) {
                clustersMap[p.persona] = {
                    name: p.persona,
                    color: p.color,
                    x: [],
                    y: [],
                    z: [],
                    text: []
                };
            }
            clustersMap[p.persona].x.push(p.x);
            clustersMap[p.persona].y.push(p.y);
            clustersMap[p.persona].z.push(p.z);
            clustersMap[p.persona].text.push(
                `<b>${p.id}</b> (${p.persona})<br>` +
                `Monetary Spend: $${p.spend.toLocaleString()}<br>` +
                `Frequency: ${p.freq} orders<br>` +
                `Recency: ${p.recency} days<br>` +
                `Churn Risk: ${(p.churn * 100).toFixed(1)}%`
            );
        });

        const data = Object.values(clustersMap).map(c => ({
            type: 'scatter3d',
            mode: 'markers',
            name: c.name,
            x: c.x,
            y: c.y,
            z: c.z,
            hoverinfo: 'text',
            hovertext: c.text,
            marker: {
                size: 4,
                color: c.color,
                opacity: 0.85,
                line: { width: 0.5, color: '#ffffff' }
            }
        }));

        const layout = {
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(0,0,0,0)',
            margin: { l: 0, r: 0, b: 0, t: 0 },
            legend: {
                font: { family: 'Plus Jakarta Sans', color: '#94A3B8', size: 12 },
                bgcolor: 'rgba(15, 23, 42, 0.6)',
                bordercolor: 'rgba(255, 255, 255, 0.1)',
                borderwidth: 1
            },
            scene: {
                xaxis: { title: 'PCA 1', color: '#94A3B8', gridcolor: 'rgba(255,255,255,0.08)', zerolinecolor: 'rgba(255,255,255,0.15)' },
                yaxis: { title: 'PCA 2', color: '#94A3B8', gridcolor: 'rgba(255,255,255,0.08)', zerolinecolor: 'rgba(255,255,255,0.15)' },
                zaxis: { title: 'PCA 3', color: '#94A3B8', gridcolor: 'rgba(255,255,255,0.08)', zerolinecolor: 'rgba(255,255,255,0.15)' },
                bgcolor: 'rgba(0, 0, 0, 0)'
            }
        };

        Plotly.newPlot(containerId, data, layout, { responsive: true, displayModeBar: true, displaylogo: false });
    }

    renderCohortRetentionChart(canvasId, cohortData) {
        const canvas = document.getElementById(canvasId);
        if (!canvas) return;
        const ctx = canvas.getContext('2d');

        if (this.cohortChart) {
            this.cohortChart.destroy();
        }

        const months = cohortData.months;
        const datasets = cohortData.cohorts.map(c => ({
            label: `${c.persona_title} (M12: ${c.m12_retention}%)`,
            data: c.retention_curve,
            borderColor: c.color,
            backgroundColor: `${c.color}15`,
            tension: 0.35,
            pointRadius: 4,
            pointHoverRadius: 7,
            borderWidth: 2.5
        }));

        this.cohortChart = new Chart(ctx, {
            type: 'line',
            data: { labels: months, datasets: datasets },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    x: {
                        ticks: { color: '#94A3B8' },
                        grid: { color: 'rgba(255, 255, 255, 0.05)' },
                        title: { display: true, text: 'Months Since Onboarding', color: '#94A3B8' }
                    },
                    y: {
                        ticks: { color: '#94A3B8' },
                        grid: { color: 'rgba(255, 255, 255, 0.05)' },
                        title: { display: true, text: 'Active Retention Rate (%)', color: '#94A3B8' },
                        min: 0,
                        max: 100
                    }
                },
                plugins: {
                    legend: {
                        position: 'top',
                        labels: { color: '#94A3B8', font: { family: 'Plus Jakarta Sans', size: 12 } }
                    },
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                return ` ${context.dataset.label.split(' (')[0]}: ${context.raw}% active retention`;
                            }
                        }
                    }
                }
            }
        });
    }
}

window.dashboardCharts = new DashboardCharts();
