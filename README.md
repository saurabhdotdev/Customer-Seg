# Customer Intelligence Platform

An end-to-end AIML project for customer segmentation, anomaly detection, and marketing decision support. The system combines unsupervised learning, RFM feature engineering, model benchmarking, explainable segment profiles, campaign recommendations, a FastAPI backend, and an interactive web dashboard.

This project is designed to go beyond a basic clustering notebook. It is structured like a production-style customer analytics system that a retail, e-commerce, fintech, or subscription business could adapt for real customer lifecycle decisions.

## Why This Project Is Differentiable

- **Multi-model unsupervised learning**: K-Means, DBSCAN, Hierarchical Agglomerative Clustering, and Gaussian Mixture Models are trained and benchmarked side by side.
- **Production model strategy**: K-Means is used for stable customer segmentation, while DBSCAN acts as an anomaly and risk-detection layer.
- **Business-ready intelligence**: Each segment is converted into personas, revenue contribution, churn risk, and campaign recommendations.
- **Explainable clustering**: The system identifies the strongest drivers behind every segment using feature-level deviations from the global customer profile.
- **Model stability analysis**: K-Means reproducibility is evaluated with repeated resampling and Adjusted Rand Index.
- **Real API surface**: FastAPI endpoints serve predictions, benchmark results, personas, explainability, high-risk customers, anomaly summaries, and campaign recommendations.
- **Future-ready architecture**: The repo separates data generation, preprocessing, modeling, insights, backend API, frontend dashboard, and saved model artifacts.

## Project Architecture

```text
Customer-Seg/
├── backend/
│   ├── main.py                  # FastAPI application
│   ├── config.py                # Path and artifact configuration
│   └── api/
│       ├── routes.py            # REST API endpoints
│       └── schemas.py           # Pydantic request/response schemas
├── data/
│   ├── raw/                     # Generated or imported transactions
│   └── processed/               # Feature and segment outputs
├── frontend/
│   ├── index.html               # Dashboard shell
│   ├── css/styles.css
│   └── js/
├── models_saved/
│   ├── scaler.joblib            # Preprocessing pipeline
│   ├── kmeans_model.joblib      # Production segmentation model
│   ├── pca.joblib               # PCA projection model
│   └── model_metadata.json      # Benchmarks, personas, explanations
├── notebooks/
│   └── customer_segmentation_analysis.ipynb
├── src/
│   ├── data/                    # Synthetic dataset generation and preprocessing
│   ├── models/                  # Clustering algorithms and evaluation
│   ├── visualization/           # PCA dimensionality reduction
│   └── insights/                # Personas, analytics, explainability, campaigns
├── requirements.txt
└── run_demo.py                  # Full pipeline runner
```

## Machine Learning Workflow

1. **Generate or load customer data**
   - Recency, frequency, monetary value, engagement, channel, returns, discounts, support tickets, and churn-risk indicators.

2. **Feature engineering**
   - RFM scoring
   - Average order value
   - Churn risk index
   - Log transforms for skewed financial and frequency variables
   - Standard scaling and categorical one-hot encoding

3. **Model training**
   - K-Means with optimal K search
   - DBSCAN with epsilon and min-samples grid search
   - Hierarchical Agglomerative Clustering
   - Gaussian Mixture Model

4. **Model evaluation**
   - Silhouette Score
   - Davies-Bouldin Index
   - Calinski-Harabasz Index
   - DBSCAN noise ratio
   - K-Means stability using Adjusted Rand Index

5. **Business intelligence layer**
   - Segment personas
   - Segment explanations
   - Revenue and churn priority
   - High-risk customer ranking
   - Campaign recommendations
   - DBSCAN anomaly summary

## API Endpoints

| Endpoint | Purpose |
| :--- | :--- |
| `GET /api/health` | API health check |
| `GET /api/overview` | Portfolio-level customer KPIs |
| `GET /api/model/info` | Production model, artifacts, PCA, and stability report |
| `GET /api/data/schema` | Required CSV schema for uploads |
| `POST /api/data/upload` | Upload CSV and retrain the full ML pipeline |
| `GET /api/benchmark` | Multi-algorithm clustering comparison |
| `GET /api/personas` | Segment personas and business metrics |
| `GET /api/segments/explainability` | Feature drivers behind each segment |
| `GET /api/recommendations/campaigns` | Campaign recommendations ranked by priority |
| `GET /api/anomalies` | DBSCAN anomaly summary and top anomalous customers |
| `GET /api/customers/high-risk` | Customers with highest churn and revenue risk |
| `GET /api/reports/executive` | Download a plain-text executive report |
| `GET /api/visualization/pca3d` | PCA coordinates for 2D/3D visualization |
| `POST /api/segment/predict` | Predict the segment for a new customer |
| `POST /api/analytics/ask` | Query the analytics layer with natural-language-like prompts |

## Quick Start

```bash
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
python run_demo.py
```

The pipeline will:

1. Generate 5,000 customer records.
2. Build RFM and behavioral features.
3. Train and benchmark four clustering algorithms.
4. Save preprocessing, K-Means, PCA, and metadata artifacts.
5. Generate personas, explainability, anomaly, stability, and campaign reports.
6. Launch the FastAPI server at `http://127.0.0.1:8000`.

For pipeline execution without launching the server:

```bash
python run_demo.py --no-server
```

## Run With Docker

If Docker Desktop is running:

```bash
docker compose up --build
```

Then open:

```text
http://127.0.0.1:8000
```

The Compose setup persists generated datasets and model artifacts through the local `data/` and `models_saved/` folders.

## CSV Upload Workflow

Open the dashboard and go to **Data Studio**:

```text
Upload Data -> Train Models -> View Segments -> Predict Customers -> Download Executive Report
```

Required columns:

```text
Recency_Days
Frequency_Orders
Monetary_Spend
Category_Diversity
Engagement_Score
Support_Tickets
Discount_Ratio
Return_Rate
Preferred_Channel
Gender
```

Optional columns:

```text
Customer_ID
Age
```

If `Customer_ID` is missing, stable IDs are generated. If `Age` is missing, a default value is used.

## Example Prediction Payload

```json
{
  "Recency_Days": 18,
  "Frequency_Orders": 32,
  "Monetary_Spend": 4200.0,
  "Category_Diversity": 6,
  "Engagement_Score": 82.0,
  "Support_Tickets": 1,
  "Discount_Ratio": 0.18,
  "Return_Rate": 0.04,
  "Age": 34,
  "Preferred_Channel": "Mobile App",
  "Gender": "Female"
}
```

## Real-World Use Cases

- Customer lifecycle segmentation
- Personalized marketing automation
- Churn-risk prioritization
- Campaign budget allocation
- VIP customer retention
- Discount sensitivity analysis
- Anomaly and unusual behavior detection
- Executive customer analytics dashboarding

## Future Enhancements

- Add MLflow or DVC for experiment tracking
- Add cloud deployment
- Add scheduled retraining
- Add SHAP-style local explanations for individual predictions
- Add uplift modeling for campaign response optimization

## License

Apache-2.0
