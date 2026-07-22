import secrets
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

def test_api_simulator_endpoint():
    payload = {
        "target_cohort": "all",
        "engagement_boost_pct": 15.0,
        "ticket_reduction": 2.0,
        "discount_incentive_pct": 10.0,
        "email_touchpoints": 2
    }
    response = client.post("/api/simulator/simulate", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "recovered_revenue" in data
    assert "rescued_customers" in data
    assert "roi_factor" in data

def test_api_auth_register_login_me():
    email = f"unittest_{secrets.token_hex(4)}@example.com"
    reg = client.post("/api/auth/register", json={
        "name": "Test User",
        "email": email,
        "password": "mysecretpassword123"
    })
    assert reg.status_code == 200
    token = reg.json()["access_token"]
    assert token is not None

    login = client.post("/api/auth/login", json={
        "email": email,
        "password": "mysecretpassword123"
    })
    assert login.status_code == 200
    assert login.json()["access_token"] is not None

    me = client.get("/api/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert me.status_code == 200
    assert me.json()["email"] == email

def test_api_multi_tenant_user_isolation():
    import io
    import pandas as pd
    reg_a = client.post("/api/auth/register", json={
        "name": "User A",
        "email": f"usera_{secrets.token_hex(3)}@example.com",
        "password": "password123"
    })
    token_a = reg_a.json()["access_token"]
    csv_a = pd.DataFrame({"Recency_Days": [10]*10, "Frequency_Orders": [5]*10, "Monetary_Spend": [500.0]*10}).to_csv(index=False)
    client.post("/api/data/upload", headers={"Authorization": f"Bearer {token_a}"}, files={"file": ("usera.csv", io.BytesIO(csv_a.encode()), "text/csv")})
    
    user_a_overview = client.get("/api/overview", headers={"Authorization": f"Bearer {token_a}"})
    assert user_a_overview.status_code == 200
    assert user_a_overview.json()["total_customers"] == 10

    reg_b = client.post("/api/auth/register", json={
        "name": "User B",
        "email": f"userb_{secrets.token_hex(3)}@example.com",
        "password": "password123"
    })
    token_b = reg_b.json()["access_token"]
    csv_b = pd.DataFrame({"Recency_Days": [15]*15, "Frequency_Orders": [8]*15, "Monetary_Spend": [1200.0]*15}).to_csv(index=False)
    client.post("/api/data/upload", headers={"Authorization": f"Bearer {token_b}"}, files={"file": ("userb.csv", io.BytesIO(csv_b.encode()), "text/csv")})
    
    user_b_overview = client.get("/api/overview", headers={"Authorization": f"Bearer {token_b}"})
    assert user_b_overview.status_code == 200
    assert user_b_overview.json()["total_customers"] == 15

def test_api_async_job_status():
    import io
    import pandas as pd
    df_test = pd.DataFrame({"Recency_Days": list(range(1, 10)), "Frequency_Orders": list(range(1, 10)), "Monetary_Spend": [100.0 * i for i in range(1, 10)]})
    csv_bytes = df_test.to_csv(index=False).encode()
    up = client.post("/api/data/upload", files={"file": ("async_test.csv", io.BytesIO(csv_bytes), "text/csv")})
    assert up.status_code == 200
    job_id = up.json()["job_id"]
    job_res = client.get(f"/api/data/job/{job_id}")
    assert job_res.status_code == 200
    assert "status" in job_res.json()

def test_api_analytics_ask_endpoint():
    payload = {"query": "Which segment has the highest churn risk?"}
    response = client.post("/api/analytics/ask", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert "category" in data
