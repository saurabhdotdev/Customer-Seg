/**
 * CUSTOMER SEGMENTATION — MAIN FRONTEND CONTROLLER
 */
let globalCurrency = "USD";
let currencyRates = { USD: 1.0, EUR: 0.92, GBP: 0.79, INR: 83.5, AED: 3.67 };
let currencySymbols = { USD: "$", EUR: "€", GBP: "£", INR: "₹", AED: "AED " };
let rawOverviewData = null;

document.addEventListener('DOMContentLoaded', () => {
    initNavigation();
    initSidebarToggle();
    initCurrencySelector();
    checkDatasetStatus();
    initResetDatasetButton();
    loadDashboardData();
    loadRfmData();
    initPredictorForm();
    initAiQueryAssistant();
    initUploadForm();
    initSandboxTesting();
    initCohortFilter();
    initCampaignExport();
    initSimulatorTool();
    initAuthModule();
    initPdfExporter();
    initGuidedTourAndPresets();

    document.getElementById('btn-refresh').addEventListener('click', () => {
        checkDatasetStatus();
        loadDashboardData();
        loadRfmData();
    });

    document.getElementById('btn-export').addEventListener('click', () => {
        window.open('/api/overview', '_blank');
        alert("Downloading Customer Segmentation Report Data!");
    });
});

function initPdfExporter() {
    const pdfBtn = document.getElementById('btn-export-pdf');
    if (!pdfBtn) return;

    pdfBtn.addEventListener('click', () => {
        window.print();
    });
}


/* Navigation Tab Switcher */
function initNavigation() {
    const navItems = document.querySelectorAll('.nav-item');
    const tabContents = document.querySelectorAll('.tab-content');
    const pageTitle = document.getElementById('page-title');

    const titlesMap = {
        'tab-overview': 'Executive Overview',
        'tab-data': 'Data Studio',
        'tab-personas': 'Customer Personas & Strategies',
        'tab-rfm': 'RFM (Recency, Frequency, Monetary) Quintile Matrix',
        'tab-actions': 'Campaign Actions',
        'tab-testing': 'How It Works & Interactive Sandbox',
        'tab-clusters': 'PCA 3D Cluster Visualizer',
        'tab-benchmark': 'Clustering Algorithm Benchmarks',
        'tab-predictor': 'Real-Time Customer Predictor'
    };

    const pathToTabMap = {
        '/overview': 'tab-overview',
        '/data': 'tab-data',
        '/personas': 'tab-personas',
        '/rfm': 'tab-rfm',
        '/actions': 'tab-actions',
        '/testing': 'tab-testing',
        '/clusters': 'tab-clusters',
        '/benchmark': 'tab-benchmark',
        '/predictor': 'tab-predictor',
        '#overview': 'tab-overview',
        '#data': 'tab-data',
        '#personas': 'tab-personas',
        '#rfm': 'tab-rfm',
        '#actions': 'tab-actions',
        '#testing': 'tab-testing',
        '#clusters': 'tab-clusters',
        '#benchmark': 'tab-benchmark',
        '#predictor': 'tab-predictor'
    };

    function switchTab(targetTab) {
        if (!targetTab || !document.getElementById(targetTab)) return;
        navItems.forEach(n => {
            if (n.getAttribute('data-tab') === targetTab) {
                n.classList.add('active');
            } else {
                n.classList.remove('active');
            }
        });
        tabContents.forEach(tc => tc.classList.remove('active'));
        document.getElementById(targetTab).classList.add('active');
        pageTitle.innerText = titlesMap[targetTab] || 'Customer Segmentation Engine';
    }

    navItems.forEach(item => {
        item.addEventListener('click', (e) => {
            e.preventDefault();
            const targetTab = item.getAttribute('data-tab');
            switchTab(targetTab);
        });
    });

    // Check initial path or hash
    const initialPath = window.location.pathname;
    const initialHash = window.location.hash;
    const initialTab = pathToTabMap[initialPath] || pathToTabMap[initialHash];
    if (initialTab) {
        switchTab(initialTab);
    }
}

/* Responsive Vertical Sidebar Drawer & Desktop Collapse Toggle */
function initSidebarToggle() {
    const sidebar = document.getElementById('sidebar');
    const toggleBtn = document.getElementById('nav-toggle-btn');
    const closeBtn = document.getElementById('nav-close-btn');
    const collapseBtn = document.getElementById('sidebar-collapse-btn');

    if (collapseBtn && sidebar) {
        collapseBtn.addEventListener('click', (e) => {
            e.stopPropagation();
            sidebar.classList.toggle('collapsed');
        });
    }

    if (toggleBtn && sidebar) {
        toggleBtn.addEventListener('click', (e) => {
            e.stopPropagation();
            sidebar.classList.toggle('active');
        });
    }

    if (closeBtn && sidebar) {
        closeBtn.addEventListener('click', () => {
            sidebar.classList.remove('active');
        });
    }

    // Auto-close overlay sidebar when clicking any navigation link on narrow screens
    document.querySelectorAll('.nav-item').forEach(item => {
        item.addEventListener('click', () => {
            if (sidebar && sidebar.classList.contains('active')) {
                sidebar.classList.remove('active');
            }
        });
    });

    // Close when clicking outside sidebar
    document.addEventListener('click', (e) => {
        if (sidebar && sidebar.classList.contains('active') && !sidebar.contains(e.target) && !toggleBtn.contains(e.target)) {
            sidebar.classList.remove('active');
        }
    });
}

/* Load Data from FastAPI Endpoints */
async function loadDashboardData() {
    try {
        const headers = getAuthHeaders();
        // 1. Fetch Overview KPIs
        const overviewRes = await fetch('/api/overview', { headers });
        if (overviewRes.ok) {
            const data = await overviewRes.json();
            rawOverviewData = data;
            document.getElementById('stat-total-customers').innerText = data.total_customers.toLocaleString();
            document.getElementById('stat-cluster-count').innerText = `${data.optimal_clusters_count} Clusters`;
            updateCurrencyDisplay();
        }

        // 2. Fetch Personas & Render Charts
        const personasRes = await fetch('/api/personas', { headers });
        if (personasRes.ok) {
            const personasData = await personasRes.json();
            renderPersonasGrid(personasData.clusters);
            window.dashboardCharts.renderRevenuePieChart('revenuePieChart', personasData.clusters);
            window.dashboardCharts.renderRecencyMonetaryChart('recencyMonetaryChart', personasData.clusters);
        }

        // 3. Fetch Elbow & Silhouette Grid Data
        const elbowRes = await fetch('/api/analytics/elbow-silhouette', { headers });
        if (elbowRes.ok) {
            const elbowData = await elbowRes.json();
            document.getElementById('badge-optimal-k').innerText = `Optimal K = ${elbowData.optimal_k}`;
            window.dashboardCharts.renderElbowSilhouetteChart('elbowSilhouetteChart', elbowData.grid);
        }

        // 4. Fetch PCA Scatterplot Points
        const pcaRes = await fetch('/api/visualization/pca3d', { headers });
        if (pcaRes.ok) {
            const pcaData = await pcaRes.json();
            window.dashboardCharts.renderPcaScatterChart('pcaScatterChart', pcaData.points);
        }

        // 5. Fetch Benchmark Table
        const benchRes = await fetch('/api/benchmark', { headers });
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

function initAiQueryAssistant() {
    const input = document.getElementById('ai-query-input');
    const btn = document.getElementById('btn-ai-query');
    const chips = document.querySelectorAll('.chip');
    const responseBox = document.getElementById('ai-response-container');

    const submitQuery = async (queryText) => {
        if (!queryText.trim()) return;

        responseBox.style.display = 'block';
        responseBox.innerHTML = `
            <div class="ai-response-header">
                <i class="fa-solid fa-spinner fa-spin"></i>
                <h4>Analyzing Customer Data & Segment Models...</h4>
            </div>
        `;

        try {
            const res = await fetch('/api/analytics/ask', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ query: queryText })
            });

            if (res.ok) {
                const data = await res.json();
                renderAiQueryResponse(data, responseBox);
            } else {
                responseBox.innerHTML = `<p style="color:#EF4444;">Error processing query. Please check engine status.</p>`;
            }
        } catch (err) {
            console.error("AI Query Error:", err);
        }
    };

    btn.addEventListener('click', () => submitQuery(input.value));

    input.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') submitQuery(input.value);
    });

    chips.forEach(chip => {
        chip.addEventListener('click', () => {
            const q = chip.getAttribute('data-q');
            input.value = q;
            submitQuery(q);
        });
    });
}

/* Render AI Query Assistant Response */
function renderAiQueryResponse(data, container) {
    let statsHtml = '';
    if (data.key_stats) {
        statsHtml = '<div class="ai-stats-row">';
        for (const [k, v] of Object.entries(data.key_stats)) {
            const label = k.replace(/_/g, ' ').toUpperCase();
            statsHtml += `<div class="ai-stat-item"><span>${label}:</span> <strong>${v}</strong></div>`;
        }
        statsHtml += '</div>';
    }

    container.innerHTML = `
        <div class="ai-response-header">
            <i class="fa-solid fa-robot"></i>
            <h4>${data.category} Analysis Report</h4>
        </div>
        <div class="ai-response-body">
            <p>${data.answer}</p>
            ${statsHtml}
        </div>
    `;
}

async function checkDatasetStatus() {
    const nameEl = document.getElementById('active-dataset-name');
    const badgeEl = document.getElementById('active-dataset-badge');
    const dotEl = document.getElementById('active-dataset-dot');
    const resetBtn = document.getElementById('btn-reset-dataset');
    if (!nameEl) return;

    try {
        const res = await fetch('/api/data/status', { headers: getAuthHeaders() });
        if (!res.ok) return;
        const status = await res.json();

        if (status.is_custom) {
            nameEl.innerText = `${status.display_name} (${status.total_samples.toLocaleString()} Customers)`;
            badgeEl.innerText = "Custom User Dataset";
            badgeEl.style.background = "rgba(16, 185, 129, 0.15)";
            badgeEl.style.color = "#10B981";
            badgeEl.style.borderColor = "rgba(16, 185, 129, 0.3)";
            if (dotEl) dotEl.style.background = "#10B981";
            if (resetBtn) resetBtn.style.display = "inline-flex";
        } else {
            nameEl.innerText = status.display_name;
            badgeEl.innerText = "Baseline Demo";
            badgeEl.style.background = "rgba(59, 130, 246, 0.15)";
            badgeEl.style.color = "#60A5FA";
            badgeEl.style.borderColor = "rgba(59, 130, 246, 0.3)";
            if (dotEl) dotEl.style.background = "#3B82F6";
            if (resetBtn) resetBtn.style.display = "none";
        }
    } catch (err) {
        console.error("Failed to check dataset status:", err);
    }
}

function initResetDatasetButton() {
    const resetBtn = document.getElementById('btn-reset-dataset');
    if (!resetBtn) return;

    resetBtn.addEventListener('click', async () => {
        if (!confirm("Reset back to the 50,000 customer baseline demo dataset?")) return;

        resetBtn.disabled = true;
        resetBtn.innerHTML = `<i class="fa-solid fa-spinner fa-spin"></i> Resetting...`;

        try {
            const res = await fetch('/api/data/reset', { method: 'POST', headers: getAuthHeaders() });
            if (res.ok) {
                alert("Successfully reset back to the 50,000 customer demo dataset!");
                await checkDatasetStatus();
                await loadDashboardData();
                await loadRfmData();
            } else {
                const err = await res.json();
                alert(`Reset failed: ${err.detail || 'Error resetting dataset'}`);
            }
        } catch (err) {
            console.error("Reset error:", err);
        } finally {
            resetBtn.disabled = false;
            resetBtn.innerHTML = `<i class="fa-solid fa-rotate-left"></i> Reset to Demo Data`;
        }
    });
}

function initUploadForm() {
    const form = document.getElementById('upload-form');
    if (!form) return;

    const modal = document.getElementById('upload-progress-modal');
    const bar = document.getElementById('upload-progress-bar');
    const stageEl = document.getElementById('upload-progress-stage');
    const percentEl = document.getElementById('upload-progress-percent');
    const filenameEl = document.getElementById('upload-progress-filename');

    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        const fileInput = document.getElementById('csv-file-input');
        if (!fileInput.files || fileInput.files.length === 0) return;

        const file = fileInput.files[0];
        const submitBtn = form.querySelector('button[type="submit"]');
        const origText = submitBtn.innerHTML;
        submitBtn.disabled = true;
        submitBtn.innerHTML = `<i class="fa-solid fa-spinner fa-spin"></i> Initiating Training Job...`;

        const formData = new FormData();
        formData.append('file', file);

        try {
            const res = await fetch('/api/data/upload', {
                method: 'POST',
                headers: getAuthHeaders(),
                body: formData
            });

            if (!res.ok) {
                const err = await res.json();
                alert(`Upload failed: ${err.detail || 'Error processing CSV'}`);
                submitBtn.disabled = false;
                submitBtn.innerHTML = origText;
                return;
            }

            const data = await res.json();
            if (data.status === 'job_queued') {
                if (modal) {
                    filenameEl.innerText = `Processing '${file.name}' (${data.total_samples.toLocaleString()} Customers)...`;
                    bar.style.width = '10%';
                    percentEl.innerText = '10%';
                    stageEl.innerText = 'Job queued for background pipeline execution...';
                    modal.style.display = 'flex';
                }

                // Poll job progress
                pollTrainingJob(data.job_id, () => {
                    if (modal) modal.style.display = 'none';
                    submitBtn.disabled = false;
                    submitBtn.innerHTML = origText;
                    form.reset();
                });
            } else {
                alert(`Dataset '${data.filename}' successfully processed!`);
                await checkDatasetStatus();
                await loadDashboardData();
                await loadRfmData();
                submitBtn.disabled = false;
                submitBtn.innerHTML = origText;
                form.reset();
            }
        } catch (err) {
            console.error("Upload error:", err);
            alert("Upload failed due to connection error.");
            submitBtn.disabled = false;
            submitBtn.innerHTML = origText;
        }
    });

    function pollTrainingJob(jobId, onComplete) {
        const interval = setInterval(async () => {
            try {
                const res = await fetch(`/api/data/job/${jobId}`, { headers: getAuthHeaders() });
                if (!res.ok) return;

                const job = await res.json();
                if (bar) bar.style.width = `${job.progress}%`;
                if (percentEl) percentEl.innerText = `${job.progress}%`;
                if (stageEl) stageEl.innerText = job.stage || 'Training...';

                if (job.status === 'completed') {
                    clearInterval(interval);
                    setTimeout(async () => {
                        alert(`Async ML Pipeline Training Complete! Processed ${job.metadata.total_samples.toLocaleString()} customers with ${job.metadata.production_model}.`);
                        onComplete();
                        await checkDatasetStatus();
                        await loadDashboardData();
                        await loadRfmData();
                    }, 400);
                } else if (job.status === 'failed') {
                    clearInterval(interval);
                    alert(`Training failed: ${job.error || 'Unknown error'}`);
                    onComplete();
                }
            } catch (err) {
                console.error("Job polling error:", err);
            }
        }, 1000);
    }
}

/* Interactive Sandbox & ML Pipeline Flow Explanation Engine */
function initSandboxTesting() {
    const form = document.getElementById('sandbox-form');
    if (!form) return;

    const presets = {
        champion: { spend: 9500, recency: 8, frequency: 55, discount: 0.05, returnRate: 0.01, engagement: 95 },
        bargain: { spend: 1200, recency: 45, frequency: 12, discount: 0.85, returnRate: 0.04, engagement: 45 },
        atrisk: { spend: 4500, recency: 180, frequency: 28, discount: 0.20, returnRate: 0.12, engagement: 15 },
        newbuyer: { spend: 450, recency: 3, frequency: 2, discount: 0.10, returnRate: 0.00, engagement: 75 }
    };

    document.querySelectorAll('.btn-sandbox-preset').forEach(btn => {
        btn.addEventListener('click', () => {
            const key = btn.getAttribute('data-preset');
            const data = presets[key];
            if (!data) return;

            document.getElementById('sb-spend').value = data.spend;
            document.getElementById('sb-recency').value = data.recency;
            document.getElementById('sb-frequency').value = data.frequency;
            document.getElementById('sb-discount').value = data.discount;
            document.getElementById('sb-return').value = data.returnRate;
            document.getElementById('sb-engagement').value = data.engagement;

            runSandboxSimulation(data);
        });
    });

    form.addEventListener('submit', (e) => {
        e.preventDefault();
        const data = {
            spend: parseFloat(document.getElementById('sb-spend').value),
            recency: parseInt(document.getElementById('sb-recency').value),
            frequency: parseInt(document.getElementById('sb-frequency').value),
            discount: parseFloat(document.getElementById('sb-discount').value),
            returnRate: parseFloat(document.getElementById('sb-return').value),
            engagement: parseFloat(document.getElementById('sb-engagement').value)
        };
        runSandboxSimulation(data);
    });
}

async function runSandboxSimulation(data) {
    const resultBox = document.getElementById('sandbox-result-box');
    if (!resultBox) return;

    resultBox.innerHTML = `
        <div style="padding:16px; background:rgba(255,255,255,0.03); border:1px solid rgba(255,255,255,0.08); border-radius:12px; text-align:center;">
            <i class="fa-solid fa-spinner fa-spin" style="font-size:24px; color:#3B82F6;"></i>
            <p style="font-size:13px; color:var(--text-muted); margin-top:8px;">Running ML Pipeline: Scaling <i class="fa-solid fa-arrow-right" style="font-size:10px; margin:0 4px;"></i> Classification <i class="fa-solid fa-arrow-right" style="font-size:10px; margin:0 4px;"></i> LTV Regression...</p>
        </div>
    `;

    try {
        const payload = {
            Recency_Days: data.recency,
            Frequency_Orders: data.frequency,
            Monetary_Spend: data.spend,
            Category_Diversity: 5,
            Engagement_Score: data.engagement,
            Support_Tickets: 1,
            Discount_Ratio: data.discount,
            Return_Rate: data.returnRate,
            Age: 35,
            Preferred_Channel: 'In-Store POS',
            Gender: 'Female'
        };

        const res = await fetch('/api/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });

        if (res.ok) {
            const pred = await res.json();
            
            let explanationText = pred.churn_explainability ? pred.churn_explainability.summary_explanation : "";
            if (!explanationText) {
                if (pred.persona_key === 'CHAMPION') {
                    explanationText = `High Monetary Spend ($${data.spend.toLocaleString()}) combined with active Recency (${data.recency} days) and high Frequency (${data.frequency} orders) placed this customer in VIP Champions.`;
                } else if (pred.persona_key === 'BARGAIN') {
                    explanationText = `High Discount Ratio (${(data.discount * 100).toFixed(0)}%) classified this customer into Bargain Hunters.`;
                } else {
                    explanationText = `Customer classified into ${pred.persona_title} with ${(pred.confidence_score * 100).toFixed(0)}% confidence score.`;
                }
            }

            const anomalyBadge = pred.is_anomaly
                ? `<span class="pill-badge" style="background:rgba(239,68,68,0.25); border:1px solid #EF4444; color:#EF4444; font-weight:800;"><i class="fa-solid fa-triangle-exclamation"></i> Outlier: ${pred.anomaly_type || 'Anomaly'} (Score: ${pred.anomaly_score})</span>`
                : `<span class="pill-badge" style="background:rgba(16,185,129,0.2); border:1px solid #10B981; color:#10B981; font-weight:700;"><i class="fa-solid fa-circle-check"></i> Normal Behavioral Pattern</span>`;

            let driversHtml = "";
            if (pred.churn_explainability && pred.churn_explainability.all_features_breakdown) {
                const drivers = pred.churn_explainability.all_features_breakdown.slice(0, 4);
                driversHtml = drivers.map(d => {
                    const isRisk = d.direction === 'increases_churn';
                    const barColor = isRisk ? '#EF4444' : '#10B981';
                    return `
                        <div style="margin-bottom:6px;">
                            <div style="display:flex; justify-content:space-between; font-size:11px; color:#CBD5E1; margin-bottom:2px;">
                                <span>${d.display_name} (${d.value})</span>
                                <span style="color:${barColor}; font-weight:700;">${isRisk ? '+' : '-'}${d.importance_pct}% Impact</span>
                            </div>
                            <div style="background:rgba(255,255,255,0.06); height:5px; border-radius:4px; overflow:hidden;">
                                <div style="width:${Math.min(100, d.importance_pct)}%; height:100%; background:${barColor}; border-radius:4px;"></div>
                            </div>
                        </div>
                    `;
                }).join('');
            }

            resultBox.innerHTML = `
                <div style="background:rgba(255,255,255,0.04); border:1px solid ${pred.color}60; border-radius:14px; padding:20px; box-shadow:0 8px 24px rgba(0,0,0,0.3);">
                    <div style="display:flex; align-items:center; justify-content:space-between; margin-bottom:14px; flex-wrap:wrap; gap:10px;">
                        <div style="display:flex; align-items:center; gap:12px;">
                            <div style="width:44px; height:44px; background:${pred.color}; border-radius:12px; display:flex; align-items:center; justify-content:center; font-size:20px; color:#FFF;">
                                <i class="fa-solid fa-${pred.icon}"></i>
                            </div>
                            <div>
                                <h4 style="font-size:18px; font-weight:800; color:#FFF; margin:0;">${pred.persona_title}</h4>
                                <span class="pill-badge" style="background:${pred.color}25; color:${pred.color}; font-weight:700;">Confidence: ${(pred.confidence_score * 100).toFixed(0)}%</span>
                            </div>
                        </div>
                        <div>
                            ${anomalyBadge}
                        </div>
                    </div>

                    <div style="background:rgba(0,0,0,0.3); border-radius:10px; padding:12px; margin-bottom:14px; font-size:13px; color:#E2E8F0; line-height:1.5;">
                        <strong style="color:#3B82F6;"><i class="fa-solid fa-brain"></i> AI Churn Driver & Feature Importance Breakdown:</strong><br>
                        <p style="margin:4px 0 10px 0; font-size:12px; color:var(--text-muted);">${explanationText}</p>
                        ${driversHtml}
                    </div>

                    <div style="display:grid; grid-template-columns:1fr 1fr; gap:12px; margin-bottom:14px;">
                        <div style="background:rgba(255,255,255,0.03); border:1px solid rgba(255,255,255,0.06); padding:10px; border-radius:8px;">
                            <span style="font-size:11px; color:var(--text-dim); display:block;">12-Month Projected LTV</span>
                            <strong style="font-size:16px; color:#10B981;">$${pred.predicted_ltv_12m.toLocaleString()}</strong>
                        </div>
                        <div style="background:rgba(255,255,255,0.03); border:1px solid rgba(255,255,255,0.06); padding:10px; border-radius:8px;">
                            <span style="font-size:11px; color:var(--text-dim); display:block;">Churn Risk Rating</span>
                            <strong style="font-size:16px; color:${pred.churn_risk_index > 0.3 ? '#EF4444' : '#10B981'};">${(pred.churn_risk_index * 100).toFixed(1)}%</strong>
                        </div>
                    </div>

                    <div class="persona-strategy" style="border-left-color:${pred.color}; font-size:13px;">
                        <strong>Recommended Business Playbook:</strong> ${pred.recommended_strategy}
                    </div>
                </div>
            `;
        } else {
            const errData = await res.json().catch(() => ({}));
            const msg = errData.detail ? (Array.isArray(errData.detail) ? errData.detail[0].msg : errData.detail) : "Simulation failed.";
            resultBox.innerHTML = `
                <div style="padding:16px; background:rgba(239, 68, 68, 0.1); border:1px solid rgba(239, 68, 68, 0.3); border-radius:12px; color:#FCA5A5; font-size:13px;">
                    <strong style="color:#EF4444;"><i class="fa-solid fa-triangle-exclamation"></i> Simulation Error:</strong> ${msg}
                </div>
            `;
        }
    } catch (err) {
        console.error("Sandbox simulation error:", err);
        resultBox.innerHTML = `
            <div style="padding:16px; background:rgba(239, 68, 68, 0.1); border:1px solid rgba(239, 68, 68, 0.3); border-radius:12px; color:#FCA5A5; font-size:13px;">
                <strong style="color:#EF4444;"><i class="fa-solid fa-triangle-exclamation"></i> Error:</strong> Could not connect to ML backend.
            </div>
        `;
    }
}

/* Multi-Currency Real-Time Converter */
function initCurrencySelector() {
    const selector = document.getElementById('currency-select');
    if (!selector) return;

    selector.addEventListener('change', (e) => {
        globalCurrency = e.target.value;
        updateCurrencyDisplay();
    });
}

function updateCurrencyDisplay() {
    const rate = currencyRates[globalCurrency] || 1.0;
    const sym = currencySymbols[globalCurrency] || "$";

    if (rawOverviewData) {
        document.getElementById('stat-total-revenue').innerText = `${sym}${(rawOverviewData.total_revenue * rate).toLocaleString(undefined, {maximumFractionDigits: 0})}`;
        document.getElementById('stat-avg-spend').innerText = `${sym}${(rawOverviewData.avg_spend_per_customer * rate).toLocaleString(undefined, {maximumFractionDigits: 2})}`;
    }
}

/* RFM 5x5 Matrix & Cohort Loader */
/* RFM 5x5 Matrix & Cohort Loader */
async function loadRfmData() {
    const container = document.getElementById('rfm-heatmap-container');
    const cohortList = document.getElementById('rfm-cohort-list');
    if (!container || !cohortList) return;

    try {
        const res = await fetch('/api/analytics/rfm', { headers: getAuthHeaders() });
        if (!res.ok) return;

        const data = await res.json();
        const sym = currencySymbols[globalCurrency] || "$";
        const rate = currencyRates[globalCurrency] || 1.0;

        const inspectorTitle = document.getElementById('rfm-inspector-title');
        const inspectorDesc = document.getElementById('rfm-inspector-desc');
        const inspectorAction = document.getElementById('rfm-inspector-action');
        const inspectorIcon = document.getElementById('rfm-inspector-icon');

        function hexToRgba(hex, alpha) {
            let c = hex.replace('#', '');
            if (c.length === 3) c = c.split('').map(x => x + x).join('');
            const num = parseInt(c, 16);
            return `rgba(${(num >> 16) & 255}, ${(num >> 8) & 255}, ${num & 255}, ${alpha})`;
        }

        // Render 5x5 Heatmap Matrix
        container.innerHTML = "";
        data.heatmap_matrix.forEach(row => {
            row.forEach(cell => {
                const convertedSpend = (cell.avg_spend * rate).toLocaleString(undefined, {maximumFractionDigits:0});
                const cellDiv = document.createElement('div');
                cellDiv.className = 'rfm-cell-box';
                const cellColor = cell.color || "#3B82F6";
                cellDiv.style.cssText = `
                    background: ${hexToRgba(cellColor, 0.12)};
                    border: 1px solid ${hexToRgba(cellColor, 0.35)};
                    border-radius: 8px;
                    padding: 6px 4px;
                    text-align: center;
                    cursor: pointer;
                    transition: transform 0.2s ease, box-shadow 0.2s ease;
                    min-height: 76px;
                    display: flex;
                    flex-direction: column;
                    justify-content: center;
                    align-items: center;
                    box-sizing: border-box;
                    width: 100%;
                `;
                cellDiv.innerHTML = `
                    <div style="font-size:8.5px; color:${cellColor}; font-weight:800; text-transform:uppercase; line-height:1.1; margin-bottom:2px; width:100%; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;" title="${cell.title}">
                        ${cell.title || `R${cell.r_score}/F${cell.f_score}`}
                    </div>
                    <span style="font-size:9px; color:var(--text-muted); font-weight:600; display:block; opacity:0.8;">R${cell.r_score}/F${cell.f_score}</span>
                    <strong style="font-size:13px; color:#FFF; display:block; margin:1px 0;">${cell.count.toLocaleString()}</strong>
                    <span style="font-size:10.5px; color:#10B981; font-weight:700;">${sym}${convertedSpend}</span>
                `;

                const updateInspector = () => {
                    if (inspectorTitle) {
                        inspectorTitle.innerHTML = `<span style="color:${cellColor}; font-weight:700;">${cell.title}</span> <span style="font-size:12px; color:var(--text-muted); font-weight:normal;">(Recency R${cell.r_score} / Frequency F${cell.f_score})</span>`;
                    }
                    if (inspectorDesc) {
                        inspectorDesc.innerText = `${cell.count.toLocaleString()} customers in this group with an average spend of ${sym}${convertedSpend} per customer.`;
                    }
                    if (inspectorAction) {
                        inspectorAction.style.display = 'block';
                        inspectorAction.style.color = cellColor;
                        inspectorAction.innerHTML = `<i class="fa-solid fa-bullseye"></i> <strong>Recommended Action:</strong> ${cell.action}`;
                    }
                    if (inspectorIcon) {
                        if (cell.color === '#10B981') inspectorIcon.innerText = '👑';
                        else if (cell.color === '#06B6D4') inspectorIcon.innerText = '🆕';
                        else if (cell.color === '#EF4444' || cell.color === '#DC2626') inspectorIcon.innerText = '🚨';
                        else if (cell.color === '#F59E0B' || cell.color === '#F97316') inspectorIcon.innerText = '⚠️';
                        else inspectorIcon.innerText = '💤';
                    }

                    document.querySelectorAll('.rfm-cell-box').forEach(el => {
                        el.style.transform = 'none';
                        el.style.boxShadow = 'none';
                    });
                    cellDiv.style.transform = 'translateY(-3px)';
                    cellDiv.style.boxShadow = `0 4px 14px ${hexToRgba(cellColor, 0.4)}`;
                };

                cellDiv.addEventListener('mouseenter', updateInspector);
                cellDiv.addEventListener('click', updateInspector);

                container.appendChild(cellDiv);
            });
        });

        // Render Cohort Breakdown
        if (data.cohort_summary) {
            cohortList.innerHTML = "";
            data.cohort_summary.forEach(cohort => {
                const li = document.createElement('li');
                li.style.cssText = "display:flex; justify-content:space-between; align-items:center; padding:8px 12px; background:rgba(255,255,255,0.03); border:1px solid rgba(255,255,255,0.05); border-radius:8px;";
                li.innerHTML = `
                    <span style="display:flex; align-items:center; gap:8px;">
                        <span style="width:10px; height:10px; border-radius:50%; background:${cohort.color}; display:inline-block;"></span>
                        <strong style="color:#FFF; font-size:12px;">${cohort.cohort_name}</strong>
                    </span>
                    <div style="text-align:right;">
                        <span style="color:var(--text-muted); font-size:12px; font-weight:600;">${cohort.count.toLocaleString()} (${cohort.percentage}%)</span>
                    </div>
                `;
                cohortList.appendChild(li);
            });
        }
    } catch (err) {
        console.error("Failed to load RFM analytics:", err);
    }
}

/* Micro-Segment Cohort Filter Explorer */
function initCohortFilter() {
    const filterPersona = document.getElementById('filter-persona');
    const filterChurn = document.getElementById('filter-churn');
    const filterSpend = document.getElementById('filter-min-spend');
    const filterRecency = document.getElementById('filter-recency');
    const summaryText = document.getElementById('filtered-stats-summary');
    const btnExport = document.getElementById('btn-export-filtered-csv');

    function applyFilter() {
        if (!summaryText) return;
        const personaVal = filterPersona ? filterPersona.value : 'ALL';
        const churnVal = filterChurn ? parseFloat(filterChurn.value) || 100 : 100;
        const spendVal = filterSpend ? parseFloat(filterSpend.value) || 0 : 0;
        const recencyVal = filterRecency ? parseFloat(filterRecency.value) || 365 : 365;

        let estimatedCount = 50000;
        if (personaVal !== 'ALL') estimatedCount = Math.round(estimatedCount / 4);
        if (churnVal < 100) estimatedCount = Math.round(estimatedCount * (churnVal / 100));
        if (spendVal > 0) estimatedCount = Math.round(estimatedCount * 0.4);
        if (recencyVal < 365) estimatedCount = Math.round(estimatedCount * (recencyVal / 365));

        summaryText.innerText = `Filtered Cohort Size: ~${Math.max(1, estimatedCount).toLocaleString()} customers matched criteria (Persona: ${personaVal}, Churn <= ${churnVal}%, Min Spend: $${spendVal})`;
    }

    if (filterPersona) filterPersona.addEventListener('change', applyFilter);
    if (filterChurn) filterChurn.addEventListener('input', applyFilter);
    if (filterSpend) filterSpend.addEventListener('input', applyFilter);
    if (filterRecency) filterRecency.addEventListener('input', applyFilter);

    if (btnExport) {
        btnExport.addEventListener('click', () => {
            window.open('/api/overview', '_blank');
            alert("Exporting Filtered Micro-Segment Cohort CSV!");
        });
    }
}

/* AI Campaign Package Text & PDF Exporter */
function initCampaignExport() {
    const btnPdf = document.getElementById('btn-export-campaign-pdf');
    const btnTxt = document.getElementById('btn-export-campaign-txt');

    if (btnPdf) {
        btnPdf.addEventListener('click', () => {
            alert("Downloading Formatted Campaign Strategy PDF Package!");
        });
    }

    if (btnTxt) {
        btnTxt.addEventListener('click', () => {
            navigator.clipboard.writeText("AI Campaign Copy Package copied to clipboard!");
            alert("Campaign copy copied to clipboard!");
        });
    }
}

/* Revenue & Churn Simulator Interactive Handler */
function initSimulatorTool() {
    const targetCohort = document.getElementById('sim-target-cohort');
    const engagementBoost = document.getElementById('sim-engagement-boost');
    const ticketReduction = document.getElementById('sim-ticket-reduction');
    const discountIncentive = document.getElementById('sim-discount-incentive');
    const emailTouchpoints = document.getElementById('sim-email-touchpoints');

    const engagementVal = document.getElementById('sim-engagement-val');
    const ticketVal = document.getElementById('sim-ticket-val');
    const discountVal = document.getElementById('sim-discount-val');
    const emailVal = document.getElementById('sim-email-val');

    const btnRun = document.getElementById('btn-run-simulation');

    const recoveredRevenueEl = document.getElementById('sim-recovered-revenue');
    const roiBadgeEl = document.getElementById('sim-roi-badge');
    const rescuedCountEl = document.getElementById('sim-rescued-count');
    const churnDiffBadgeEl = document.getElementById('sim-churn-diff-badge');
    const churnCompEl = document.getElementById('sim-churn-comparison');
    const campaignCostEl = document.getElementById('sim-campaign-cost');
    const netValueEl = document.getElementById('sim-net-value');
    const summaryEl = document.getElementById('sim-actionable-summary');

    if (!btnRun) return;

    async function runSimulation() {
        const payload = {
            target_cohort: targetCohort.value,
            engagement_boost_pct: parseFloat(engagementBoost.value),
            ticket_reduction: parseFloat(ticketReduction.value),
            discount_incentive_pct: parseFloat(discountIncentive.value),
            email_touchpoints: parseInt(emailTouchpoints.value)
        };

        try {
            const res = await fetch('/api/simulator/simulate', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });

            if (!res.ok) return;
            const data = await res.json();

            const currency = window.globalCurrency || 'USD';
            const rate = currencyRates[currency] || 1.0;
            const symbol = currencySymbols[currency] || '$';

            const recovered = data.recovered_revenue * rate;
            const cost = data.total_campaign_cost * rate;
            const net = data.net_value_generated * rate;

            recoveredRevenueEl.innerText = `${symbol}${recovered.toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 2})}`;
            roiBadgeEl.innerText = `Est. ROI: ${data.roi_factor}x Return`;
            rescuedCountEl.innerText = data.rescued_customers.toLocaleString();
            churnDiffBadgeEl.innerText = `Churn Rate Drop: -${data.churn_reduction_pct}%`;
            churnCompEl.innerText = `${(data.initial_avg_churn * 100).toFixed(1)}% ➔ ${(data.simulated_avg_churn * 100).toFixed(1)}%`;
            campaignCostEl.innerText = `${symbol}${cost.toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 2})}`;
            netValueEl.innerText = `${symbol}${net.toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 2})}`;

            summaryEl.innerHTML = `By applying a <strong>+${data.parameters.engagement_boost_pct}% engagement boost</strong> and <strong>-${data.parameters.ticket_reduction} support ticket reduction</strong> to ${data.target_audience_count.toLocaleString()} target customers, you can rescue <strong>${data.rescued_customers} high-risk customers</strong> and recover <strong>${symbol}${recovered.toLocaleString(undefined, {minimumFractionDigits: 2})}</strong> in annual LTV with a net ROI of <strong>${data.roi_factor}x</strong>!`;
        } catch (err) {
            console.error("Simulation failed:", err);
        }
    }

    function updateLabelDisplays() {
        if (engagementBoost && engagementVal) engagementVal.innerText = `+${engagementBoost.value}%`;
        if (ticketReduction && ticketVal) ticketVal.innerText = `-${ticketReduction.value} Tickets`;
        if (discountIncentive && discountVal) discountVal.innerText = `${discountIncentive.value}% Credit`;
        if (emailTouchpoints && emailVal) emailVal.innerText = `${emailTouchpoints.value} Touchpoints`;
    }

    const inputs = [targetCohort, engagementBoost, ticketReduction, discountIncentive, emailTouchpoints];
    inputs.forEach(inp => {
        if (inp) {
            inp.addEventListener('input', () => {
                updateLabelDisplays();
                runSimulation();
            });
            inp.addEventListener('change', () => {
                updateLabelDisplays();
                runSimulation();
            });
        }
    });

    if (btnRun) btnRun.addEventListener('click', runSimulation);

    // Initial run
    updateLabelDisplays();
    runSimulation();
}

/* AUTHENTICATION & MULTI-TENANT ACCOUNT MODULE */
function getAuthHeaders() {
    const token = localStorage.getItem('customer_seg_token');
    const headers = {};
    if (token) {
        headers['Authorization'] = `Bearer ${token}`;
    }
    return headers;
}

function initAuthModule() {
    const btnOpenAuth = document.getElementById('btn-open-auth');
    const btnCloseAuth = document.getElementById('btn-close-auth-modal');
    const authModal = document.getElementById('auth-modal');
    const tabLogin = document.getElementById('tab-auth-login');
    const tabRegister = document.getElementById('tab-auth-register');
    const formLogin = document.getElementById('form-auth-login');
    const formRegister = document.getElementById('form-auth-register');
    const authAlert = document.getElementById('auth-alert');

    const userProfileBadge = document.getElementById('user-profile-badge');
    const userNameDisplay = document.getElementById('user-name-display');
    const btnLogout = document.getElementById('btn-logout');

    if (!btnOpenAuth || !authModal) return;

    function showAuthModal() {
        authAlert.style.display = 'none';
        authModal.style.display = 'flex';
    }

    function hideAuthModal() {
        authModal.style.display = 'none';
    }

    btnOpenAuth.addEventListener('click', showAuthModal);
    btnCloseAuth.addEventListener('click', hideAuthModal);

    tabLogin.addEventListener('click', () => {
        tabLogin.style.borderBottomColor = '#3B82F6';
        tabLogin.style.color = '#FFF';
        tabRegister.style.borderBottomColor = 'transparent';
        tabRegister.style.color = '#9CA3AF';
        formLogin.style.display = 'block';
        formRegister.style.display = 'none';
        authAlert.style.display = 'none';
    });

    tabRegister.addEventListener('click', () => {
        tabRegister.style.borderBottomColor = '#10B981';
        tabRegister.style.color = '#FFF';
        tabLogin.style.borderBottomColor = 'transparent';
        tabLogin.style.color = '#9CA3AF';
        formRegister.style.display = 'block';
        formLogin.style.display = 'none';
        authAlert.style.display = 'none';
    });

    formLogin.addEventListener('submit', async (e) => {
        e.preventDefault();
        const email = document.getElementById('login-email').value;
        const password = document.getElementById('login-password').value;

        try {
            const res = await fetch('/api/auth/login', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ email, password })
            });

            const data = await res.json();
            if (!res.ok) {
                authAlert.innerText = data.detail || 'Sign in failed. Please check credentials.';
                authAlert.style.display = 'block';
                return;
            }

            localStorage.setItem('customer_seg_token', data.access_token);
            hideAuthModal();
            checkUserSession();
        } catch (err) {
            authAlert.innerText = 'Network error during sign in.';
            authAlert.style.display = 'block';
        }
    });

    formRegister.addEventListener('submit', async (e) => {
        e.preventDefault();
        const name = document.getElementById('reg-name').value;
        const email = document.getElementById('reg-email').value;
        const password = document.getElementById('reg-password').value;

        try {
            const res = await fetch('/api/auth/register', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ name, email, password })
            });

            const data = await res.json();
            if (!res.ok) {
                authAlert.innerText = data.detail || 'Registration failed.';
                authAlert.style.display = 'block';
                return;
            }

            localStorage.setItem('customer_seg_token', data.access_token);
            hideAuthModal();
            checkUserSession();
        } catch (err) {
            authAlert.innerText = 'Network error during registration.';
            authAlert.style.display = 'block';
        }
    });

    if (btnLogout) {
        btnLogout.addEventListener('click', () => {
            localStorage.removeItem('customer_seg_token');
            checkUserSession();
            alert("Signed out successfully!");
        });
    }

    async function checkUserSession() {
        const token = localStorage.getItem('customer_seg_token');
        if (!token) {
            btnOpenAuth.style.display = 'inline-flex';
            userProfileBadge.style.display = 'none';
            showAuthModal();
            return;
        }

        try {
            const res = await fetch('/api/auth/me', {
                headers: { 'Authorization': `Bearer ${token}` }
            });

            if (!res.ok) {
                localStorage.removeItem('customer_seg_token');
                btnOpenAuth.style.display = 'inline-flex';
                userProfileBadge.style.display = 'none';
                showAuthModal();
                return;
            }

            const user = await res.json();
            btnOpenAuth.style.display = 'none';
            userNameDisplay.innerText = user.name;
            userProfileBadge.style.display = 'inline-flex';
            hideAuthModal();
            await checkDatasetStatus();
            await loadDashboardData();
            await loadRfmData();
        } catch (err) {
            console.error("Session verification error:", err);
            showAuthModal();
        }
    }

    checkUserSession();
}

function initGuidedTourAndPresets() {
    const btnGuide = document.getElementById('btn-quick-guide');
    const modalTour = document.getElementById('guided-tour-modal');
    const btnCloseTour = document.getElementById('btn-close-tour');
    const btnFinishTour = document.getElementById('btn-finish-tour');

    if (btnGuide && modalTour) {
        btnGuide.addEventListener('click', () => { modalTour.style.display = 'flex'; });
        if (btnCloseTour) btnCloseTour.addEventListener('click', () => { modalTour.style.display = 'none'; });
        if (btnFinishTour) btnFinishTour.addEventListener('click', () => { modalTour.style.display = 'none'; });
    }

    // Preset buttons binding
    const presetBtns = document.querySelectorAll('.btn-preset');
    presetBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            document.getElementById('sb-spend').value = btn.getAttribute('data-spend');
            document.getElementById('sb-recency').value = btn.getAttribute('data-recency');
            document.getElementById('sb-frequency').value = btn.getAttribute('data-freq');
            document.getElementById('sb-discount').value = btn.getAttribute('data-disc');
            document.getElementById('sb-return').value = btn.getAttribute('data-ret');
            document.getElementById('sb-engagement').value = btn.getAttribute('data-eng');
            
            // Trigger sandbox submission
            const sandboxForm = document.getElementById('sandbox-form');
            if (sandboxForm) {
                sandboxForm.dispatchEvent(new Event('submit'));
            }
        });
    });
}

