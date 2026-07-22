# 🚀 Enterprise Customer Intelligence & Behavioral ML Platform

[![Live App on Vercel](https://img.shields.io/badge/Live_App-Customer_Seg-10B981?style=for-the-badge&logo=vercel)](https://customer-seg.vercel.app)
[![Python 3.12](https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109-009688?style=for-the-badge&logo=fastapi)](https://fastapi.tiangolo.com)
[![PyTest Status](https://img.shields.io/badge/Tests-30%2F30%20PASSED-34D399?style=for-the-badge&logo=pytest)](https://github.com/saurabhdotdev/Customer-Seg)

An enterprise-grade Customer Behavioral Intelligence & AI Platform combining multi-algorithm clustering (**K-Means, GMM, DBSCAN, Hierarchical**), supervised ML classification (**RandomForest**), 12-month LTV forecasting (**Gradient Boosting**), **IsolationForest Anomaly Detection**, **SHAP-style Churn Explainability**, **Auto-ML Model Selector**, **Google Gemini 2.5 Flash Campaign Copy Generation**, and **JWT Multi-Tenant Data Isolation**.

---

## 🌟 Key Platform Features

- **🔐 Mandatory JWT Multi-Tenant Security & Isolation**: Fully isolated model training and CSV dataset storage per user account (`/tmp/users/{user_id}/`). Explicit actionable authentication validation with 1-click tab auto-switching.
- **🌲 IsolationForest Fraud & Anomaly Detector**: Real-time identification of VIP Whale spenders, fraud suspect discount abusers, abnormal return behavior, and dormant high-value customers.
- **📊 SHAP-Style Churn Explainability Engine**: Calculates exact percentage contribution drivers (e.g. `Support Tickets (+38.2%)`, `Recency (+29.4%)`, `Engagement (-22.1%)`) for every customer prediction.
- **⚡ Auto-ML Production Model Selector**: Benchmarks K-Means vs GMM vs DBSCAN vs Hierarchical on **Silhouette Score**, **Calinski-Harabasz Index**, and **Davies-Bouldin Index** with 1-click active production engine swapping.
- **📅 12-Month Cohort Retention Heatmap Matrix**: Calculates signup cohort decay curves month-over-month (M0 to M12) across 12 monthly cohorts.
- **🔔 Automated Webhook Alert Engine**: Real-time event notification system dispatching webhooks with SQLite audit trails when high churn risk (>70%) or anomalies are detected.
- **🤖 Gemini 2.5 Flash Generative AI Campaign Copy Studio**: Auto-generates personalized subject lines, email body copy, SMS text, and promotional push notifications tailored per persona.
- **📄 1-Click Executive PDF Exporter**: Instant browser print styling generating executive PDF reports formatted for board presentations.
- **💡 Usability & Guided Tour**: Interactive 3-step platform guided tour and 1-click preset simulator buttons (`💎 VIP Whale`, `⚠️ High Churn`, `🛒 Bargain Hunter`).

---

## 🏛️ System Architecture

```text
Customer-Seg/
├── backend/
│   ├── main.py                  # FastAPI Application Entry & Routing
│   ├── config.py                # System Directories & Multi-Tenant Paths
│   └── api/
│       ├── auth.py              # JWT Token Generation & Passlib Verification
│       ├── db.py                # SQLite Persistence & Audit Logger
│       ├── routes.py            # REST API Endpoints & Multi-Tenant Middleware
│       ├── webhooks.py          # Real-time Webhook Alert Engine
│       └── schemas.py           # Pydantic Schemas
├── src/
│   ├── data/                    # Data Preprocessor & Synthetic Generator
│   ├── models/
│   │   ├── kmeans_model.py      # K-Means Clustering Engine
│   │   ├── gmm_model.py         # Gaussian Mixture Model Engine
│   │   ├── dbscan_model.py      # DBSCAN Density Engine
│   │   ├── hierarchical_model.py# Hierarchical Agglomerative Engine
│   │   ├── classifier.py        # Supervised RandomForest Classifier
│   │   ├── ltv_regressor.py     # GradientBoosting 12-Month LTV Regressor
│   │   ├── anomaly_detector.py  # IsolationForest Outlier & Fraud Detector
│   │   ├── explainability.py    # SHAP Feature Importance Explainability Engine
│   │   └── automl.py            # Auto-ML Model Selector & Leaderboard
│   ├── insights/
│   │   ├── cohort_engine.py     # 12-Month Cohort Retention Decay Matrix
│   │   ├── rfm_engine.py        # 5x5 RFM Quintile Matrix
│   │   ├── persona_generator.py # Persona Profiling & Business Strategies
│   │   └── query_engine.py      # Natural Language Analytics Assistant
│   └── visualization/
│       └── dimensionality.py    # 2D/3D PCA Dimensionality Reduction
├── frontend/
│   ├── index.html               # Modern Dark-Glassmorphism UI Shell
│   ├── css/styles.css           # Vanilla CSS Styling & Responsive Grid
│   └── js/
│       ├── app.js               # Main Dashboard Logic & API Connector
│       └── charts.js            # Chart.js & Plotly 3D WebGL Scatterplots
├── tests/
│   ├── test_api.py              # 16 API & Auth Unit Tests
│   ├── test_models.py           # 10 Model Unit Tests
│   ├── test_cohort_engine.py    # Cohort Unit Tests
│   ├── test_generator.py        # Synthetic Data Unit Tests
│   └── test_preprocessor.py     # Preprocessing Unit Tests
└── run_demo.py                  # CLI Pipeline & Server Launcher
```

---

## 📡 REST API Endpoints

### 🔐 Authentication & Profile
- `POST /api/auth/register`: User registration with password hashing.
- `POST /api/auth/login`: Authentication returning 7-day JWT Bearer tokens.
- `GET /api/auth/me`: Decodes JWT token and returns user profile.

### 📊 Intelligence & Analytics
- `GET /api/overview`: Executive KPIs, revenue contribution, and persona summaries.
- `GET /api/personas`: Segment personas, metric breakdowns, and actionable playbooks.
- `GET /api/analytics/rfm`: 5x5 RFM Quintile matrix and cohort distribution.
- `GET /api/analytics/cohort-retention-matrix`: 12-month signup cohort retention decay matrix.
- `POST /api/analytics/ask`: Gemini 2.5 Flash natural language analytics query engine.

### 🤖 Auto-ML & Production Model Selection
- `GET /api/automl/benchmark`: Runs Auto-ML evaluation across Silhouette, Calinski-Harabasz, and Davies-Bouldin metrics.
- `POST /api/automl/select-model`: 1-click active production engine swap.

### 🎯 Predictions & Sandbox Simulation
- `POST /api/predict`: Single-customer segment prediction with IsolationForest anomaly score and SHAP feature driver breakdown.
- `POST /api/simulator`: Interactive marketing intervention simulator predicting recovered revenue, rescued customers, and campaign ROI.

### 🔔 Webhooks & System Audit
- `POST /api/webhooks/test`: Triggers a test notification webhook.
- `GET /api/webhooks/logs`: Retrieves persistent SQLite audit trail of fired webhook notifications.
- `POST /api/ingest/transaction`: Real-time transaction ingestion API.

---

## 🧪 Testing & Verification Suite

Run the full automated test suite (30 Unit Tests):

```bash
.\venv\Scripts\python.exe -m pytest tests/ -v
```

```text
======================= 30 passed in 40.97s =======================
```

---

## 🚀 Deployment

- **Live Production App:** [https://customer-seg.vercel.app](https://customer-seg.vercel.app)
- **GitHub Repository:** [https://github.com/saurabhdotdev/Customer-Seg](https://github.com/saurabhdotdev/Customer-Seg)

---

## 📄 License

Apache-2.0 © 2026 Saurabh — Built with Python, FastAPI, Scikit-Learn, PyTest, and Google Gemini.
