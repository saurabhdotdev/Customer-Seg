import pytest
import pandas as pd
from src.data.generator import generate_customer_dataset
from src.pipeline.training import REQUIRED_COLUMNS

def test_generate_customer_dataset_default():
    df = generate_customer_dataset(n_samples=500, random_state=42)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 500
    for col in REQUIRED_COLUMNS:
        assert col in df.columns, f"Missing required column: {col}"
    assert df['Customer_ID'].nunique() == 500
    assert not df.isnull().values.any(), "Dataset contains null values"

def test_generate_customer_dataset_columns_types():
    df = generate_customer_dataset(n_samples=100, random_state=123)
    assert (df['Recency_Days'] >= 1).all() and (df['Recency_Days'] <= 365).all()
    assert (df['Frequency_Orders'] >= 1).all()
    assert (df['Monetary_Spend'] >= 1.0).all()
    assert (df['Discount_Ratio'] >= 0.0).all() and (df['Discount_Ratio'] <= 1.0).all()
    assert 'Customer_LTV' in df.columns
    assert 'Region' in df.columns
