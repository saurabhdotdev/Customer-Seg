import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_RAW_PATH = os.path.join(BASE_DIR, "data", "raw", "customer_transactions.csv")
DATA_PROCESSED_PATH = os.path.join(BASE_DIR, "data", "processed", "customer_features.csv")
DATA_SEGMENTS_PATH = os.path.join(BASE_DIR, "data", "processed", "customer_segments.csv")

MODELS_DIR = os.path.join(BASE_DIR, "models_saved")
SCALER_PATH = os.path.join(MODELS_DIR, "scaler.joblib")
PCA_PATH = os.path.join(MODELS_DIR, "pca.joblib")
METADATA_PATH = os.path.join(MODELS_DIR, "model_metadata.json")

FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")
NOTEBOOKS_DIR = os.path.join(BASE_DIR, "notebooks")
