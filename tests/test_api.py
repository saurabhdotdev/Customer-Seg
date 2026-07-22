import pytest
from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)

def test_api_overview_endpoint():
    response = client.get("/api/overview")
    assert response.status_code == 200
    data = response.json()
    assert "total_customers" in data
    assert "total_revenue" in data
    assert "optimal_clusters_count" in data

def test_api_personas_endpoint():
    response = client.get("/api/personas")
    assert response.status_code == 200
    data = response.json()
    assert "clusters" in data
    assert len(data["clusters"]) >= 1

def test_api_elbow_silhouette_endpoint():
    response = client.get("/api/analytics/elbow-silhouette")
    assert response.status_code == 200
    data = response.json()
    assert "optimal_k" in data
    assert "grid" in data

def test_api_visualization_pca3d_endpoint():
    response = client.get("/api/visualization/pca3d")
    assert response.status_code == 200
    data = response.json()
    assert "count" in data
    assert "points" in data
    assert data["count"] > 0

def test_api_predict_endpoint():
    payload = {
        "Recency_Days": 10,
        "Frequency_Orders": 50,
        "Monetary_Spend": 10000.0,
        "Category_Diversity": 7,
        "Engagement_Score": 95.0,
        "Support_Tickets": 0,
        "Discount_Ratio": 0.05,
        "Return_Rate": 0.01,
        "Age": 36,
        "Preferred_Channel": "Mobile App",
        "Gender": "Female"
    }
    response = client.post("/api/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "predicted_cluster" in data
    assert "confidence_score" in data
    assert "persona_title" in data
    assert "churn_risk_index" in data
    assert "is_anomaly" in data

def test_api_generate_campaign_copy_endpoint():
    payload = {
        "persona_key": "CHAMPION",
        "persona_title": "VIP Champions"
    }
    response = client.post("/api/campaigns/generate-copy", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "email_subject" in data
    assert "sms_text" in data
    assert "ad_headline" in data

def test_api_export_csv_endpoint():
    response = client.get("/api/export/csv")
    assert response.status_code == 200
    assert "attachment" in response.headers.get("content-disposition", "")

def test_api_reports_executive_endpoint():
    response = client.get("/api/reports/executive")
    assert response.status_code == 200
    assert "CUSTOMER INTELLIGENCE EXECUTIVE REPORT" in response.text
