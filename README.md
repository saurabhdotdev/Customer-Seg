# High-Yield Customer Segmentation & Behavioral Intelligence Platform 🚀

A production-grade, multi-algorithm machine learning platform designed for e-commerce and retail customer segmentation. Features **K-Means**, **DBSCAN**, **Hierarchical Agglomerative Clustering**, and **Gaussian Mixture Models (GMM)**, complete with RFM feature engineering, PCA 2D/3D dimensionality reduction, business persona playbooks, a **FastAPI backend**, an **interactive glassmorphism dark-mode web dashboard**, and a **Jupyter Notebook**.

---

## 🌟 Key Features

- **Multi-Algorithm ML Suite**: Compare K-Means, DBSCAN, Agglomerative Hierarchical Clustering, and GMM side-by-side.
- **RFM Feature Engineering**: Computes Recency, Frequency, and Monetary scores alongside skewness log transformations, standard scaling, and categorical one-hot encoding.
- **Dimensionality Reduction**: 2D and 3D PCA projections for customer cluster separation and visualization.
- **Evaluation Benchmark Engine**: Evaluates Silhouette Score, Davies-Bouldin Index, Calinski-Harabasz Index, and density noise ratio across all algorithms.
- **Automated Business Personas & Strategies**: Maps cluster centroids to actionable marketing playbooks (VIP Champions, Steady Loyalists, At-Risk Hibernating, Bargain Hunters, New Buyers).
- **Executive Dark-Mode Web Dashboard**: Built with HTML5, CSS3, JavaScript, and Chart.js featuring real-time cluster visualizer, algorithm matrix, and live customer prediction tool.
- **FastAPI REST API**: Serving predictions, metrics, persona playbooks, and dataset exports.
- **Data Science Notebook**: Complete Jupyter Notebook (`notebooks/customer_segmentation_analysis.ipynb`) adhering strictly to ML best practices.

---

## 📁 Repository Structure

```
Customer-Seg/
├── data/
│   ├── raw/
│   │   └── customer_transactions.csv
│   └── processed/
│       ├── customer_features.csv
│       └── customer_segments.csv
├── src/
│   ├── data/
│   │   ├── generator.py           # Synthetic customer dataset generator (5,000 samples)
│   │   └── preprocessor.py        # RFM calculator, log transformer, scaler & encoder
│   ├── models/
│   │   ├── kmeans_model.py        # K-Means + Elbow & Silhouette search
│   │   ├── dbscan_model.py        # DBSCAN + Epsilon grid search
│   │   ├── hierarchical_model.py  # Agglomerative clustering
│   │   ├── gmm_model.py           # Gaussian Mixture Model + BIC/AIC search
│   │   └── evaluator.py           # Benchmark comparison evaluator
│   ├── visualization/
│   │   └── dimensionality.py      # 2D/3D PCA & t-SNE projections
│   └── insights/
│       └── persona_generator.py   # Business personas & recommendation engine
├── backend/
│   ├── main.py                    # FastAPI application entrypoint
│   ├── config.py                  # Path and environment configuration
│   └── api/
│       ├── routes.py              # Endpoints: predict, benchmark, personas, pca3d
│       └── schemas.py             # Pydantic data schemas
├── frontend/
│   ├── index.html                 # Single-page web app shell
│   ├── css/
│   │   └── styles.css             # Glassmorphism dark-mode styles
│   └── js/
│       ├── app.js                 # Dashboard controller
│       └── charts.js              # Chart.js visualization engine
├── notebooks/
│   └── customer_segmentation_analysis.ipynb # Complete ML notebook
├── models_saved/                  # Fitted scalers, PCA models & metadata
├── requirements.txt               # Python package dependencies
├── run_demo.py                    # Complete pipeline runner & server launcher
└── README.md
```

---

## 🚀 Quick Start Guide

### 1. Environment Setup
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment (Windows)
.\venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Complete Pipeline & Launch Web Server
Execute the automated orchestrator script:
```bash
python run_demo.py
```
This will automatically:
1. Generate the 5,000 customer transaction dataset.
2. Run feature preprocessing and RFM scoring.
3. Fit and tune K-Means, DBSCAN, HAC, and GMM models.
4. Calculate PCA 3D coordinates.
5. Generate model comparison benchmarks and persona metadata.
6. Launch the FastAPI server at `http://127.0.0.1:8000`.

---

## 📊 Customer Persona Strategy Matrix

| Persona | Description | Key Metric Indicator | Strategic Playbook |
| :--- | :--- | :--- | :--- |
| **🌟 VIP Champions** | Highest spenders & frequent buyers | Spend > $3,500, Recency < 30 days | Exclusive VIP perks, early product access |
| **💎 Steady Loyalists** | Consistent repeat purchasers | Freq > 20 orders, Recency < 60 days | Cross-sell & tier upgrade incentives |
| **⚠️ At-Risk / Hibernating** | Formerly high spenders now inactive | Recency > 120 days, high support tickets | Win-back email drip & re-engagement discounts |
| **🏷️ Bargain Hunters** | Sale & promo driven buyers | Discount usage ratio > 60% | Flash clearance & volume bundle offers |
| **🌱 New Buyers** | Recent first-time purchasers | Orders 1-3, Recency < 25 days | Product onboarding series & welcome offers |

---

## 📡 API Endpoints Summary

- `GET /api/overview`: Overview KPIs (Total customers, total revenue, average spend).
- `GET /api/benchmark`: Comparative evaluation table across all 4 algorithms.
- `GET /api/personas`: Persona descriptions, metric breakdowns, and strategic playbooks.
- `GET /api/visualization/pca3d`: 2D/3D PCA point coordinates for scatterplot visualizer.
- `POST /api/segment/predict`: Classifies new customer inputs in real-time.

---

## 📜 License
Apache-2.0 License.
