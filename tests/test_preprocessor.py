import os
import pytest
import numpy as np
import pandas as pd
from src.data.generator import generate_customer_dataset
from src.data.preprocessor import CustomerDataPreprocessor

def test_preprocessor_fit_transform():
    df = generate_customer_dataset(n_samples=200, random_state=42)
    preprocessor = CustomerDataPreprocessor()
    X_scaled, df_processed = preprocessor.fit_transform(df)
    
    assert isinstance(X_scaled, np.ndarray)
    assert X_scaled.shape[0] == 200
    assert X_scaled.shape[1] == len(preprocessor.feature_names)
    assert np.abs(X_scaled[:, :9].mean(axis=0)).max() < 0.1

def test_preprocessor_single_customer(tmp_path):
    df = generate_customer_dataset(n_samples=200, random_state=42)
    scaler_path = os.path.join(tmp_path, "scaler.joblib")
    preprocessor = CustomerDataPreprocessor(scaler_save_path=scaler_path)
    X_scaled, _ = preprocessor.fit_transform(df)
    
    assert os.path.exists(scaler_path)
    
    single_customer = {
        "Recency_Days": 15,
        "Frequency_Orders": 45,
        "Monetary_Spend": 8500.0,
        "Category_Diversity": 6,
        "Engagement_Score": 92.0,
        "Support_Tickets": 1,
        "Discount_Ratio": 0.10,
        "Return_Rate": 0.02,
        "Age": 38,
        "Preferred_Channel": "Mobile App",
        "Gender": "Female"
    }
    
    X_single = preprocessor.transform_single_customer(single_customer, pipeline_path=scaler_path)
    assert isinstance(X_single, np.ndarray)
    assert X_single.shape == (1, X_scaled.shape[1])
