import pytest
import pandas as pd
from src.data.generator import generate_customer_dataset
from src.insights.cohort_engine import CustomerCohortEngine
from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)

def test_cohort_engine_calculation():
    df = generate_customer_dataset(n_samples=300, random_state=42)
    df["KMeans_Cluster"] = 0
    res = CustomerCohortEngine.calculate_cohort_retention(df, cluster_col="KMeans_Cluster")
    
    assert "months" in res
    assert "cohorts" in res
    assert len(res["months"]) == 13
    assert len(res["cohorts"]) >= 1
    
    first_cohort = res["cohorts"][0]
    assert first_cohort["retention_curve"][0] == 100.0
    assert first_cohort["m12_retention"] > 0
    assert first_cohort["m12_churn_rate"] + first_cohort["m12_retention"] == 100.0

def test_api_cohorts_endpoint():
    response = client.get("/api/analytics/cohorts")
    assert response.status_code == 200
    data = response.json()
    assert "months" in data
    assert "cohorts" in data
    assert len(data["cohorts"]) > 0
