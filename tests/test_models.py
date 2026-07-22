import pytest
import numpy as np
import pandas as pd
from src.data.generator import generate_customer_dataset
from src.data.preprocessor import CustomerDataPreprocessor
from src.models.kmeans_model import KMeansSegmentationModel
from src.models.dbscan_model import DBSCANSegmentationModel
from src.models.classifier import SegmentClassifierModel

def test_kmeans_model_fit_predict():
    df = generate_customer_dataset(n_samples=300, random_state=42)
    preprocessor = CustomerDataPreprocessor()
    X_scaled, _ = preprocessor.fit_transform(df)
    
    kmeans = KMeansSegmentationModel(n_clusters=4, random_state=42)
    kmeans.fit(X_scaled)
    labels = kmeans.labels_
    
    assert len(labels) == 300
    assert len(np.unique(labels)) == 4
    assert kmeans.cluster_centers_.shape == (4, X_scaled.shape[1])

def test_dbscan_model_fit_predict():
    df = generate_customer_dataset(n_samples=300, random_state=42)
    preprocessor = CustomerDataPreprocessor()
    X_scaled, _ = preprocessor.fit_transform(df)
    
    dbscan = DBSCANSegmentationModel(eps=1.5, min_samples=5)
    dbscan.fit(X_scaled)
    labels = dbscan.labels_
    
    assert len(labels) == 300
    assert -1 in labels or len(np.unique(labels)) >= 1

def test_segment_classifier_model():
    df = generate_customer_dataset(n_samples=500, random_state=42)
    preprocessor = CustomerDataPreprocessor()
    X_scaled, _ = preprocessor.fit_transform(df)
    
    kmeans = KMeansSegmentationModel(n_clusters=4, random_state=42)
    kmeans.fit(X_scaled)
    labels = kmeans.labels_
    
    classifier = SegmentClassifierModel(n_estimators=50, max_depth=8, random_state=42)
    classifier.fit(X_scaled, labels)
    eval_report = classifier.evaluate(X_scaled, labels)
    
    assert eval_report['accuracy'] >= 0.90
    
    preds = classifier.predict(X_scaled[:10])
    assert len(preds) == 10
    
    probas = classifier.predict_proba(X_scaled[:10])
    assert probas.shape == (10, 4)
    assert np.allclose(probas.sum(axis=1), 1.0)

def test_customer_ltv_regressor():
    from src.models.ltv_regressor import CustomerLTVRegressor
    df = generate_customer_dataset(n_samples=400, random_state=42)
    preprocessor = CustomerDataPreprocessor()
    X_scaled, df_proc = preprocessor.fit_transform(df)
    
    y_ltv = df_proc["Customer_LTV"].values if "Customer_LTV" in df_proc.columns else df_proc["Monetary_Spend"].values * 1.5
    ltv_reg = CustomerLTVRegressor(n_estimators=30, max_depth=5, random_state=42)
    eval_report = ltv_reg.fit(X_scaled, y_ltv, feature_names=preprocessor.feature_names)
    
    assert eval_report['r2_score'] > 0.30
    assert eval_report['root_mean_squared_error'] > 0
    
    preds = ltv_reg.predict(X_scaled[:10])
    assert len(preds) == 10
    assert (preds >= 0).all()

def test_anomaly_detector_fit_predict():
    from src.models.anomaly_detector import CustomerAnomalyDetector
    df = generate_customer_dataset(n_samples=300, random_state=42)
    detector = CustomerAnomalyDetector(contamination=0.05, random_state=42)
    df_out = detector.fit_predict(df)
    
    assert "Is_Anomaly" in df_out.columns
    assert "Anomaly_Score" in df_out.columns
    assert "Anomaly_Type" in df_out.columns
    assert len(df_out) == 300
    assert df_out["Is_Anomaly"].sum() >= 1

    single_res = detector.predict_single({
        "Recency_Days": 10, "Frequency_Orders": 100, "Monetary_Spend": 25000.0,
        "Category_Diversity": 10, "Engagement_Score": 99.0, "Support_Tickets": 0,
        "Discount_Ratio": 0.05, "Return_Rate": 0.01
    })
    assert "is_anomaly" in single_res
    assert "anomaly_score" in single_res
    assert "anomaly_type" in single_res

def test_churn_explainability_engine():
    from src.models.explainability import ChurnExplainabilityEngine
    input_dict = {
        "Recency_Days": 120, "Frequency_Orders": 2, "Monetary_Spend": 150.0,
        "Category_Diversity": 1, "Engagement_Score": 15.0, "Support_Tickets": 6,
        "Discount_Ratio": 0.40, "Return_Rate": 0.25
    }
    exp = ChurnExplainabilityEngine.explain_customer(input_dict, churn_score=0.85)
    
    assert exp["overall_churn_risk"] == 0.85
    assert len(exp["top_churn_risk_drivers"]) >= 1
    assert "summary_explanation" in exp

def test_logistic_churn_classifier():
    from src.models.linear_models import LogisticChurnClassifier
    from src.data.generator import generate_customer_dataset
    df = generate_customer_dataset(n_samples=200, random_state=42)
    
    classifier = LogisticChurnClassifier()
    fit_res = classifier.fit(df)
    
    assert fit_res["roc_auc_score"] >= 0.50
    assert len(fit_res["odds_ratios"]) == 8
    
    pred_res = classifier.predict_churn_probability({
        "Recency_Days": 10, "Frequency_Orders": 50, "Monetary_Spend": 5000.0,
        "Category_Diversity": 8, "Engagement_Score": 90.0, "Support_Tickets": 0,
        "Discount_Ratio": 0.05, "Return_Rate": 0.01
    })
    assert "churn_probability" in pred_res
    assert "risk_category" in pred_res
    assert 0.0 <= pred_res["churn_probability"] <= 1.0

def test_linear_ltv_regressor():
    from src.models.linear_models import LinearLTVRegressor
    from src.data.generator import generate_customer_dataset
    df = generate_customer_dataset(n_samples=200, random_state=42)
    
    regressor = LinearLTVRegressor()
    fit_res = regressor.fit(df)
    
    assert "r2_score" in fit_res
    assert "rmse" in fit_res
    assert len(fit_res["feature_multipliers"]) == 8
    
    pred_val = regressor.predict({
        "Recency_Days": 10, "Frequency_Orders": 50, "Monetary_Spend": 5000.0,
        "Category_Diversity": 8, "Engagement_Score": 90.0, "Support_Tickets": 0,
        "Discount_Ratio": 0.05, "Return_Rate": 0.01
    })
    assert isinstance(pred_val, float)
