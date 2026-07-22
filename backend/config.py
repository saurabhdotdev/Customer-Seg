import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IS_VERCEL = os.environ.get("VERCEL") == "1" or os.environ.get("VERCEL_ENV") is not None

# Fallback pre-built artifact paths in repository
BASE_DATA_SEGMENTS = os.path.join(BASE_DIR, "data", "processed", "customer_segments.csv")
BASE_METADATA_PATH = os.path.join(BASE_DIR, "models_saved", "model_metadata.json")

if IS_VERCEL:
    WORK_DIR = "/tmp"
    DATA_RAW_PATH = os.path.join(WORK_DIR, "data", "raw", "customer_transactions.csv")
    DATA_PROCESSED_PATH = os.path.join(WORK_DIR, "data", "processed", "customer_features.csv")
    DATA_SEGMENTS_PATH = os.path.join(WORK_DIR, "data", "processed", "customer_segments.csv")
    MODELS_DIR = os.path.join(WORK_DIR, "models_saved")
else:
    WORK_DIR = BASE_DIR
    DATA_RAW_PATH = os.path.join(BASE_DIR, "data", "raw", "customer_transactions.csv")
    DATA_PROCESSED_PATH = os.path.join(BASE_DIR, "data", "processed", "customer_features.csv")
    DATA_SEGMENTS_PATH = os.path.join(BASE_DIR, "data", "processed", "customer_segments.csv")
    MODELS_DIR = os.path.join(BASE_DIR, "models_saved")

SCALER_PATH = os.path.join(MODELS_DIR, "scaler.joblib")
KMEANS_MODEL_PATH = os.path.join(MODELS_DIR, "kmeans_model.joblib")
CLASSIFIER_PATH = os.path.join(MODELS_DIR, "classifier.joblib")
LTV_REGRESSOR_PATH = os.path.join(MODELS_DIR, "ltv_regressor.joblib")
PCA_PATH = os.path.join(MODELS_DIR, "pca.joblib")
METADATA_PATH = os.path.join(MODELS_DIR, "model_metadata.json")

FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")
NOTEBOOKS_DIR = os.path.join(BASE_DIR, "notebooks")

def get_existing_path(target_path: str) -> str:
    if os.path.exists(target_path):
        return target_path
    if IS_VERCEL and target_path.startswith(WORK_DIR):
        relative = os.path.relpath(target_path, WORK_DIR)
        fallback = os.path.join(BASE_DIR, relative)
        if os.path.exists(fallback):
            return fallback
    return target_path
