"""
Unit and Integration Tests for Next-Best-Action (NBA) Engine and API Routes
"""

import pytest
from fastapi.testclient import TestClient
from src.models.next_best_action import NextBestActionEngine
from backend.main import app

client = TestClient(app)


def test_next_best_action_engine_at_risk_churn():
    engine = NextBestActionEngine()
    customer = {
        "Recency_Days": 120,
        "Frequency_Orders": 3,
        "Monetary_Spend": 450.0,
        "Category_Diversity": 2,
        "Engagement_Score": 30.0,
        "Discount_Ratio": 0.20
    }
    result = engine.evaluate_customer(
        customer_data=customer,
        churn_risk_score=0.85,
        predicted_ltv_12m=900.0,
        is_anomaly=True,
        persona_key="AT_RISK"
    )

    assert result["total_actions_evaluated"] == 6
    assert result["top_action"] is not None
    assert result["top_action"]["action_id"] == "WINBACK_CREDIT"
    assert result["top_action"]["priority_score"] >= 80.0
    assert len(result["recommendations"]) == 6


def test_next_best_action_engine_vip_champion():
    engine = NextBestActionEngine()
    customer = {
        "Recency_Days": 10,
        "Frequency_Orders": 15,
        "Monetary_Spend": 3500.0,
        "Category_Diversity": 5,
        "Engagement_Score": 90.0,
        "Discount_Ratio": 0.10
    }
    result = engine.evaluate_customer(
        customer_data=customer,
        churn_risk_score=0.10,
        predicted_ltv_12m=5000.0,
        is_anomaly=False,
        persona_key="CHAMPION"
    )

    assert result["top_action"]["action_id"] == "VIP_CONCIERGE"
    assert result["top_action"]["badge_color"] == "emerald"
    assert result["top_action"]["net_roi_percent"] > 0


def test_api_recommend_next_best_action_route():
    payload = {
        "customer": {
            "Recency_Days": 45,
            "Frequency_Orders": 4,
            "Monetary_Spend": 600.0,
            "Category_Diversity": 1,
            "Engagement_Score": 60.0,
            "Discount_Ratio": 0.35,
            "Return_Rate": 0.05,
            "Support_Tickets": 1,
            "Age": 32,
            "Preferred_Channel": "Web",
            "Gender": "Female"
        },
        "churn_risk_score": 0.40,
        "predicted_ltv_12m": 1400.0,
        "is_anomaly": False,
        "persona_key": "BARGAIN"
    }

    response = client.post("/api/recommendations/next-best-action", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "total_actions_evaluated" in data
    assert data["total_actions_evaluated"] == 6
    assert "top_action" in data
    assert data["top_action"]["title"] is not None


def test_api_segment_matrix_route():
    response = client.get("/api/recommendations/segment-matrix")
    assert response.status_code == 200
    data = response.json()
    assert "matrix" in data
    assert len(data["matrix"]) == 5
    assert data["matrix"][0]["persona_key"] == "CHAMPION"
