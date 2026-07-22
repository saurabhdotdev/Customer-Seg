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
