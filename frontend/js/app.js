/**
 * CUSTOMER SEGMENTATION — MAIN FRONTEND CONTROLLER
 */

document.addEventListener('DOMContentLoaded', () => {
    initNavigation();
    loadDashboardData();
    initPredictorForm();

    document.getElementById('btn-refresh').addEventListener('click', () => {
        loadDashboardData();
    });

    document.getElementById('btn-export').addEventListener('click', () => {
        window.open('/api/overview', '_blank');
        alert("Downloading Customer Segmentation Report Data!");
    });
});

/* Navigation Tab Switcher */
function initNavigation() {
    const navItems = document.querySelectorAll('.nav-item');
    const tabContents = document.querySelectorAll('.tab-content');
    const pageTitle = document.getElementById('page-title');

    const titlesMap = {
        'tab-overview': 'Executive Overview',
        'tab-personas': 'Customer Personas & Strategies',
        'tab-clusters': 'PCA 3D Cluster Visualizer',
        'tab-benchmark': 'Clustering Algorithm Benchmarks',
        'tab-predictor': 'Real-Time Customer Predictor'
    };

    navItems.forEach(item => {
        item.addEventListener('click', (e) => {
            e.preventDefault();
            const targetTab = item.getAttribute('data-tab');

            navItems.forEach(n => n.classList.remove('active'));
            tabContents.forEach(tc => tc.classList.remove('active'));

            item.classList.add('active');
            document.getElementById(targetTab).classList.add('active');
            pageTitle.innerText = titlesMap[targetTab] || 'Customer Segmentation Engine';
        });
    });
}

/* Load Data from FastAPI Endpoints */
async function loadDashboardData() {
    try {
        // 1. Fetch Overview KPIs
        const overviewRes = await fetch('/api/overview');
        if (overviewRes.ok) {
            const data = await overviewRes.json();
            document.getElementById('stat-total-customers').innerText = data.total_customers.toLocaleString();
            document.getElementById('stat-total-revenue').innerText = `$${data.total_revenue.toLocaleString()}`;
            document.getElementById('stat-avg-spend').innerText = `$${data.avg_spend_per_customer.toLocaleString()}`;
            document.getElementById('stat-cluster-count').innerText = `${data.optimal_clusters_count} Clusters`;
        }

        // 2. Fetch Personas & Render Charts
        const personasRes = await fetch('/api/personas');
        if (personasRes.ok) {
            const personasData = await personasRes.json();
            renderPersonasGrid(personasData.clusters);
            window.dashboardCharts.renderRevenuePieChart('revenuePieChart', personasData.clusters);
            window.dashboardCharts.renderRecencyMonetaryChart('recencyMonetaryChart', personasData.clusters);
        }

        // 3. Fetch PCA Scatterplot Points
        const pcaRes = await fetch('/api/visualization/pca3d');
        if (pcaRes.ok) {
            const pcaData = await pcaRes.json();
            window.dashboardCharts.renderPcaScatterChart('pcaScatterChart', pcaData.points);
        }

        // 4. Fetch Benchmark Table
        const benchRes = await fetch('/api/benchmark');
        if (benchRes.ok) {
            const benchData = await benchRes.json();
            renderBenchmarkTable(benchData);
        }

    } catch (err) {
        console.error("Error loading dashboard data:", err);
    }
}

/* Render Persona Cards */
function renderPersonasGrid(clusters) {
    const container = document.getElementById('personas-container');
    container.innerHTML = '';

    clusters.forEach(c => {
        const card = document.createElement('div');
        card.className = 'persona-card';
        card.innerHTML = `
            <div>
                <div class="persona-header">
                    <div class="persona-badge-icon" style="background:${c.color};">
                        <i class="fa-solid fa-${c.icon}"></i>
                    </div>
                    <div class="persona-title-box">
                        <h3>${c.persona_title}</h3>
                        <p class="persona-tagline">${c.tagline}</p>
                    </div>
                </div>
                <div class="persona-body">
                    <p>${c.description}</p>
                    <div class="persona-metrics-list">
                        <div class="p-metric">
                            <span class="p-label">Customer Count</span>
                            <span class="p-val">${c.metrics.customer_count.toLocaleString()} (${c.metrics.customer_percentage}%)</span>
                        </div>
                        <div class="p-metric">
                            <span class="p-label">Total Revenue</span>
                            <span class="p-val">$${c.metrics.total_revenue.toLocaleString()}</span>
                        </div>
                        <div class="p-metric">
                            <span class="p-label">Avg Order Value</span>
                            <span class="p-val">$${c.metrics.avg_order_value}</span>
                        </div>
                        <div class="p-metric">
                            <span class="p-label">Avg Recency Gap</span>
                            <span class="p-val">${c.metrics.avg_recency_days} days</span>
                        </div>
                        <div class="p-metric">
                            <span class="p-label">Avg Orders / Year</span>
                            <span class="p-val">${c.metrics.avg_frequency_orders}</span>
                        </div>
                        <div class="p-metric">
                            <span class="p-label">Avg Churn Index</span>
                            <span class="p-val">${(c.metrics.avg_churn_risk * 100).toFixed(1)}%</span>
                        </div>
                    </div>
                </div>
            </div>
            <div class="persona-strategy" style="border-left-color:${c.color};">
                <strong>Target Action Playbook:</strong> ${c.strategy}
            </div>
        `;
        container.appendChild(card);
    });
}

/* Render Benchmark Table */
function renderBenchmarkTable(benchmarkList) {
    const tbody = document.getElementById('benchmark-tbody');
    tbody.innerHTML = '';

    benchmarkList.forEach(row => {
        const tr = document.createElement('tr');
        
        let ratingBadge = `<span class="badge-rating badge-good">Good</span>`;
        if (row.Silhouette_Score >= 0.50) {
            ratingBadge = `<span class="badge-rating badge-optimal">Optimal</span>`;
        } else if (row.Silhouette_Score < 0.25) {
            ratingBadge = `<span class="badge-rating badge-fair">Fair / Dense</span>`;
        }

        tr.innerHTML = `
            <td><strong>${row.Algorithm}</strong></td>
            <td>${row.Clusters}</td>
            <td>${row['Outliers/Noise']}</td>
            <td><strong style="color:#10B981;">${row.Silhouette_Score}</strong></td>
            <td>${row.Davies_Bouldin_Index}</td>
            <td>${row.Calinski_Harabasz_Index.toLocaleString()}</td>
            <td>${ratingBadge}</td>
        `;
        tbody.appendChild(tr);
    });
}

/* Handle Predictor Form Submission */
function initPredictorForm() {
    const form = document.getElementById('predictor-form');
    const resultCard = document.getElementById('result-card');

    form.addEventListener('submit', async (e) => {
        e.preventDefault();

        const payload = {
            Recency_Days: parseInt(document.getElementById('pred-recency').value),
            Frequency_Orders: parseInt(document.getElementById('pred-frequency').value),
            Monetary_Spend: parseFloat(document.getElementById('pred-monetary').value),
            Category_Diversity: parseInt(document.getElementById('pred-category').value),
            Engagement_Score: parseFloat(document.getElementById('pred-engagement').value),
            Support_Tickets: parseInt(document.getElementById('pred-tickets').value),
            Discount_Ratio: parseFloat(document.getElementById('pred-discount').value),
            Return_Rate: parseFloat(document.getElementById('pred-return').value),
            Age: 35,
            Preferred_Channel: "Mobile App",
            Gender: "Female"
        };

        try {
            const res = await fetch('/api/segment/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });

            if (res.ok) {
                const pred = await res.json();
                renderPredictionResult(pred, resultCard);
            } else {
                alert("Failed to classify customer. Make sure backend engine is running.");
            }
        } catch (err) {
            console.error("Prediction error:", err);
        }
    });
}

/* Render Prediction Result Card */
function renderPredictionResult(pred, resultCard) {
    const churnPct = (pred.churn_risk_index * 100).toFixed(1);
    let churnColor = "#10B981";
    if (pred.churn_risk_index > 0.60) churnColor = "#EF4444";
    else if (pred.churn_risk_index > 0.35) churnColor = "#F59E0B";

    resultCard.innerHTML = `
        <div class="prediction-output">
            <div class="pred-badge-box" style="background:${pred.color}20; border: 1px solid ${pred.color};">
                <div class="pred-icon-circle" style="background:${pred.color};">
                    <i class="fa-solid fa-${pred.icon}"></i>
                </div>
                <div class="pred-title-wrap">
                    <h2 style="color:${pred.color};">${pred.persona_title}</h2>
                    <p>${pred.tagline}</p>
                </div>
            </div>

            <div class="strategy-box" style="border-left-color:${pred.color};">
                <h4><i class="fa-solid fa-bullseye"></i> Recommended Strategic Playbook</h4>
                <p>${pred.recommended_strategy}</p>
            </div>

            <div class="persona-metrics-list">
                <div class="p-metric">
                    <span class="p-label">Predicted Cluster ID</span>
                    <span class="p-val">Cluster ${pred.predicted_cluster}</span>
                </div>
                <div class="p-metric">
                    <span class="p-label">Predicted Churn Risk</span>
                    <span class="p-val" style="color:${churnColor};">${churnPct}% Risk</span>
                </div>
                <div class="p-metric">
                    <span class="p-label">PCA 3D Coordinates</span>
                    <span class="p-val">[${pred.pca_coordinates.join(', ')}]</span>
                </div>
                <div class="p-metric">
                    <span class="p-label">Segment Avg Spend</span>
                    <span class="p-val">$${pred.metrics_summary.avg_monetary_spend}</span>
                </div>
            </div>
        </div>
    `;
}
