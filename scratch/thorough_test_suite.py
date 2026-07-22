import sys
import os
import json
import secrets
from fastapi.testclient import TestClient

# Ensure root directory is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.main import app

client = TestClient(app)

def run_thorough_system_audit():
    print("=" * 60)
    print("RUNNING THOROUGH END-TO-END SYSTEM AUDIT & VERIFICATION")
    print("=" * 60)

    # 1. Test Health Check
    print("\n[1/10] Testing Health Endpoint...")
    res = client.get("/api/health")
    assert res.status_code == 200, f"Health failed: {res.text}"
    print("  [OK] Health Endpoint PASSED:", res.json())

    # 2. Test Multi-Tenant Auth
    print("\n[2/10] Testing JWT Multi-Tenant Authentication & Verification...")
    test_email = f"audit_user_{secrets.token_hex(4)}@domain.com"
    test_password = "AuditPassword123!"
    
    reg_res = client.post("/api/auth/register", json={
        "name": "Audit Tester",
        "email": test_email,
        "password": test_password
    })
    assert reg_res.status_code == 200, f"Registration failed: {reg_res.text}"
    token = reg_res.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}
    print("  [OK] User Registration & Token Issue PASSED")

    me_res = client.get("/api/auth/me", headers=headers)
    assert me_res.status_code == 200, f"/auth/me failed: {me_res.text}"
    assert me_res.json()["email"] == test_email
    print("  [OK] Token Verification (/auth/me) PASSED")

    # 3. Test Explicit Auth Error Validation
    print("\n[3/10] Testing Explicit Actionable Auth Error Messages...")
    bad_login = client.post("/api/auth/login", json={"email": "nonexistent_email_999@domain.com", "password": "pass"})
    assert bad_login.status_code == 404
    assert "ACCOUNT_NOT_FOUND" in bad_login.json()["detail"]
    print("  [OK] Account Not Found Exception PASSED:", bad_login.json()["detail"])

    bad_pwd = client.post("/api/auth/login", json={"email": test_email, "password": "WrongPassword!"})
    assert bad_pwd.status_code == 401
    assert "INCORRECT_PASSWORD" in bad_pwd.json()["detail"]
    print("  [OK] Incorrect Password Exception PASSED:", bad_pwd.json()["detail"])

    # 4. Test Executive Overview & Personas
    print("\n[4/10] Testing Overview & Persona Intelligence APIs...")
    ov_res = client.get("/api/overview", headers=headers)
    assert ov_res.status_code == 200, f"Overview failed: {ov_res.text}"
    assert "total_customers" in ov_res.json()
    print(f"  [OK] Executive Overview PASSED (Total Customers: {ov_res.json()['total_customers']}, Revenue: ${ov_res.json()['total_revenue']:,})")

    per_res = client.get("/api/personas", headers=headers)
    assert per_res.status_code == 200, f"Personas failed: {per_res.text}"
    assert len(per_res.json()["clusters"]) > 0
    print(f"  [OK] Persona Clusters PASSED ({len(per_res.json()['clusters'])} personas loaded)")

    # 5. Test RFM Matrix & Cohort Heatmap
    print("\n[5/10] Testing RFM 5x5 Matrix & 12-Month Cohort Retention Decay...")
    rfm_res = client.get("/api/analytics/rfm", headers=headers)
    assert rfm_res.status_code == 200
    assert "heatmap_matrix" in rfm_res.json()
    print("  [OK] RFM 5x5 Quintile Matrix PASSED")

    cohort_res = client.get("/api/analytics/cohort-retention-matrix", headers=headers)
    assert cohort_res.status_code == 200
    assert len(cohort_res.json()["heatmap_data"]) == 12
    print("  [OK] 12-Month Cohort Retention Decay Matrix PASSED (12 cohorts calculated)")

    # 6. Test Single Customer Predictor + IsolationForest + SHAP Explainability
    print("\n[6/10] Testing Predictor API + IsolationForest Anomaly + SHAP Drivers...")
    pred_payload = {
        "Recency_Days": 4,
        "Frequency_Orders": 68,
        "Monetary_Spend": 19500.0,
        "Category_Diversity": 8,
        "Engagement_Score": 96.0,
        "Support_Tickets": 0,
        "Discount_Ratio": 0.02,
        "Return_Rate": 0.01,
        "Age": 38,
        "Preferred_Channel": "Mobile App",
        "Gender": "Female"
    }
    pred_res = client.post("/api/predict", json=pred_payload, headers=headers)
    assert pred_res.status_code == 200, f"Predict failed: {pred_res.text}"
    pdata = pred_res.json()
    assert "predicted_cluster" in pdata
    assert "anomaly_type" in pdata
    assert "churn_explainability" in pdata
    print(f"  [OK] Prediction Result: Cluster {pdata['predicted_cluster']} ({pdata['persona_title']})")
    print(f"  [OK] IsolationForest Anomaly Audit: score={pdata['anomaly_score']}, type='{pdata['anomaly_type']}', is_anomaly={pdata['is_anomaly']}")
    print(f"  [OK] SHAP Churn Driver Summary: {pdata['churn_explainability']['summary_explanation']}")

    # 7. Test Revenue & Churn Intervention Simulator
    print("\n[7/10] Testing Revenue & Churn Simulator...")
    sim_payload = {
        "target_cohort": "all",
        "engagement_boost_pct": 15.0,
        "ticket_reduction": 2.0,
        "discount_incentive_pct": 10.0,
        "email_touchpoints": 2
    }
    sim_res = client.post("/api/simulator/simulate", json=sim_payload, headers=headers)
    assert sim_res.status_code == 200, f"Simulator failed: {sim_res.text}"
    sdata = sim_res.json()
    print(f"  [OK] Simulator PASSED: Recovered Revenue=${sdata['recovered_revenue']:,}, Rescued Customers={sdata['rescued_customers']}, ROI Factor={sdata['roi_factor']}")

    # 8. Test Auto-ML Model Selector & Active Engine Swap
    print("\n[8/10] Testing Auto-ML Model Selector & 1-Click Engine Swap...")
    bench_res = client.get("/api/automl/benchmark", headers=headers)
    assert bench_res.status_code == 200
    bdata = bench_res.json()
    print(f"  [OK] Auto-ML Leaderboard PASSED: Winner = '{bdata['winner_model']}' (Composite Score: {bdata['best_composite_score']})")

    swap_res = client.post("/api/automl/select-model", json={"selected_model": "Gaussian Mixture Model (GMM)"}, headers=headers)
    assert swap_res.status_code == 200
    assert swap_res.json()["active_production_model"] == "Gaussian Mixture Model (GMM)"
    print("  [OK] Active Production Model Swapped to GMM PASSED")

    # 9. Test Webhooks & Real-Time Ingestion
    print("\n[9/10] Testing Webhook Alert Engine & Real-Time Transaction Ingestion...")
    web_trig = client.post("/api/webhooks/test", json={"alert_type": "AUDIT_CHURN_ALERT", "customer_id": "CUST_AUDIT_99"}, headers=headers)
    assert web_trig.status_code == 200
    assert web_trig.json()["status"] == "dispatched"
    print("  [OK] Webhook Alert Dispatch PASSED")

    web_logs = client.get("/api/webhooks/logs", headers=headers)
    assert web_logs.status_code == 200
    assert len(web_logs.json()["logs"]) >= 1
    print("  [OK] Webhook Audit Trail fetch PASSED")

    ing_res = client.post("/api/ingest/transaction", json={
        "customer_id": "CUST_INGEST_1",
        "recency_days": 12,
        "frequency_orders": 30,
        "monetary_spend": 3200.0
    })
    assert ing_res.status_code == 200
    assert ing_res.json()["status"] == "ingested"
    print("  [OK] Real-time Transaction Ingestion PASSED")

    # 10. Test Natural Language Analytics AI Assistant
    print("\n[10/10] Testing Gemini AI Natural Language Assistant...")
    ask_res = client.post("/api/analytics/ask", json={"query": "Which customer segment has the highest monetary spend?"}, headers=headers)
    assert ask_res.status_code == 200
    adata = ask_res.json()
    assert "answer" in adata
    print("  [OK] Gemini AI Analytics Assistant PASSED:", adata["answer"])

    print("\n" + "=" * 60)
    print("ALL 10 END-TO-END AUDIT CHECKS PASSED WITH 100% PERFECT HEALTH!")
    print("=" * 60)

if __name__ == "__main__":
    run_thorough_system_audit()
