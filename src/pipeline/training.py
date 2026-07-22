import json
import os
from datetime import datetime, timezone

import joblib
import pandas as pd

from backend.config import DATA_PROCESSED_PATH, DATA_SEGMENTS_PATH, KMEANS_MODEL_PATH, CLASSIFIER_PATH, LTV_REGRESSOR_PATH, METADATA_PATH, PCA_PATH, SCALER_PATH
from src.data.preprocessor import CustomerDataPreprocessor
from src.insights.persona_generator import CustomerPersonaGenerator
from src.insights.segment_intelligence import SegmentIntelligenceEngine
from src.models.dbscan_model import DBSCANSegmentationModel
from src.models.evaluator import ModelBenchmarkEvaluator
from src.models.gmm_model import GMMSegmentationModel
from src.models.hierarchical_model import HierarchicalSegmentationModel
from src.models.kmeans_model import KMeansSegmentationModel
from src.models.classifier import SegmentClassifierModel
from src.models.ltv_regressor import CustomerLTVRegressor
from sklearn.model_selection import train_test_split
from src.visualization.dimensionality import DimensionalityReducer


ALIAS_MAP = {
    "recency": "Recency_Days",
    "recency_days": "Recency_Days",
    "last_order_days": "Recency_Days",
    "days_since_last_purchase": "Recency_Days",
    "frequency": "Frequency_Orders",
    "frequency_orders": "Frequency_Orders",
    "orders": "Frequency_Orders",
    "total_orders": "Frequency_Orders",
    "purchases": "Frequency_Orders",
    "monetary": "Monetary_Spend",
    "monetary_spend": "Monetary_Spend",
    "spend": "Monetary_Spend",
    "total_spend": "Monetary_Spend",
    "revenue": "Monetary_Spend",
    "amount": "Monetary_Spend",
    "customer_id": "Customer_ID",
    "id": "Customer_ID",
    "user_id": "Customer_ID",
}

CORE_REQUIRED = {"Recency_Days", "Frequency_Orders", "Monetary_Spend"}
REQUIRED_COLUMNS = CORE_REQUIRED

FEATURE_DEFAULTS = {
    "Category_Diversity": 3,
    "Engagement_Score": 50,
    "Support_Tickets": 1,
    "Discount_Ratio": 0.15,
    "Return_Rate": 0.05,
    "Preferred_Channel": "Web",
    "Gender": "Unknown",
    "Age": 35,
}


def validate_customer_dataframe(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    df_clean = df.copy()
    warnings = []

    # Map column aliases
    rename_dict = {}
    for col in df_clean.columns:
        col_lower = str(col).strip().lower().replace(" ", "_")
        if col_lower in ALIAS_MAP:
            rename_dict[col] = ALIAS_MAP[col_lower]
    if rename_dict:
        df_clean.rename(columns=rename_dict, inplace=True)
        warnings.append(f"Auto-mapped column names: {rename_dict}")

    missing_core = sorted(CORE_REQUIRED - set(df_clean.columns))
    if missing_core:
        raise ValueError(
            f"Missing required core RFM columns: {', '.join(missing_core)}. "
            "Please include columns for Recency, Frequency (orders), and Monetary (spend)."
        )

    if "Customer_ID" not in df_clean.columns:
        df_clean.insert(0, "Customer_ID", [f"UPL-{10001 + i}" for i in range(len(df_clean))])
        warnings.append("Customer_ID was missing, so stable IDs were generated.")

    for col, default in FEATURE_DEFAULTS.items():
        if col not in df_clean.columns:
            df_clean[col] = default
            warnings.append(f"Optional column '{col}' was missing, populated with default '{default}'.")

    numeric_cols = [
        "Recency_Days",
        "Frequency_Orders",
        "Monetary_Spend",
        "Category_Diversity",
        "Engagement_Score",
        "Support_Tickets",
        "Discount_Ratio",
        "Return_Rate",
        "Age",
    ]
    for col in numeric_cols:
        df_clean[col] = pd.to_numeric(df_clean[col], errors="coerce")

    # Impute any NaNs in numeric columns with column median or default
    for col in numeric_cols:
        if df_clean[col].isna().any():
            median_val = df_clean[col].median()
            default_val = median_val if pd.notna(median_val) else FEATURE_DEFAULTS.get(col, 1)
            df_clean[col] = df_clean[col].fillna(default_val)
            warnings.append(f"Imputed missing numeric values in '{col}' with median {default_val}.")

    df_clean["Recency_Days"] = df_clean["Recency_Days"].clip(1, 365).round().astype(int)
    df_clean["Frequency_Orders"] = df_clean["Frequency_Orders"].clip(1, 200).round().astype(int)
    df_clean["Monetary_Spend"] = df_clean["Monetary_Spend"].clip(lower=1.0)
    df_clean["Category_Diversity"] = df_clean["Category_Diversity"].clip(1, 10).round().astype(int)
    df_clean["Engagement_Score"] = df_clean["Engagement_Score"].clip(1, 100)
    df_clean["Support_Tickets"] = df_clean["Support_Tickets"].clip(0, 20).round().astype(int)
    df_clean["Discount_Ratio"] = df_clean["Discount_Ratio"].clip(0, 1)
    df_clean["Return_Rate"] = df_clean["Return_Rate"].clip(0, 0.5)
    df_clean["Age"] = df_clean["Age"].clip(18, 90).round().astype(int)

    df_clean["Preferred_Channel"] = df_clean["Preferred_Channel"].fillna("Web").astype(str)
    df_clean["Gender"] = df_clean["Gender"].fillna("Unknown").astype(str)

    if "Avg_Order_Value" not in df_clean.columns:
        df_clean["Avg_Order_Value"] = (df_clean["Monetary_Spend"] / df_clean["Frequency_Orders"].replace(0, 1)).round(2)
        warnings.append("Calculated 'Avg_Order_Value' from Monetary_Spend and Frequency_Orders.")

    if "Churn_Risk_Index" not in df_clean.columns:
        rec_factor = (df_clean["Recency_Days"] / 120.0).clip(upper=1.0)
        freq_factor = (1.0 - (df_clean["Frequency_Orders"] / 20.0)).clip(lower=0.0)
        eng_factor = (1.0 - (df_clean["Engagement_Score"] / 100.0)).clip(lower=0.0)
        support_factor = (df_clean["Support_Tickets"] / 5.0).clip(upper=1.0)
        df_clean["Churn_Risk_Index"] = (0.4 * rec_factor + 0.3 * freq_factor + 0.2 * eng_factor + 0.1 * support_factor).round(3)
        warnings.append("Calculated 'Churn_Risk_Index' using heuristic RFM-engagement model.")

    return df_clean, warnings


def train_customer_segmentation_pipeline(df_raw: pd.DataFrame, data_source: str = "synthetic") -> dict:
    df_raw, validation_warnings = validate_customer_dataframe(df_raw)

    preprocessor = CustomerDataPreprocessor(scaler_save_path=SCALER_PATH)
    X_matrix, df_processed = preprocessor.fit_transform(df_raw)

    os.makedirs(os.path.dirname(DATA_PROCESSED_PATH), exist_ok=True)
    df_processed.to_csv(DATA_PROCESSED_PATH, index=False)

    kmeans = KMeansSegmentationModel()
    k_search = kmeans.find_optimal_k(X_matrix, range(2, 10))
    optimal_k = k_search["best_k"]
    kmeans.n_clusters = optimal_k
    kmeans.fit(X_matrix)
    joblib.dump(kmeans.model, KMEANS_MODEL_PATH)

    dbscan = DBSCANSegmentationModel()
    db_search = dbscan.tune_hyperparameters(X_matrix)
    dbscan.eps = db_search["best_params"]["eps"]
    dbscan.min_samples = db_search["best_params"]["min_samples"]
    dbscan.fit(X_matrix)

    hac = HierarchicalSegmentationModel(n_clusters=optimal_k, linkage="ward")
    hac.fit(X_matrix)

    gmm = GMMSegmentationModel(n_components=optimal_k)
    gmm.fit(X_matrix)

    reducer = DimensionalityReducer(n_components_pca=3, pca_save_path=PCA_PATH)
    X_pca, pca_meta = reducer.fit_transform_pca(X_matrix)

    df_processed["KMeans_Cluster"] = kmeans.labels_
    df_processed["DBSCAN_Cluster"] = dbscan.labels_
    df_processed["HAC_Cluster"] = hac.labels_
    df_processed["GMM_Cluster"] = gmm.labels_
    df_processed["PCA_1"] = X_pca[:, 0]
    df_processed["PCA_2"] = X_pca[:, 1]
    df_processed["PCA_3"] = X_pca[:, 2]
    df_processed.to_csv(DATA_SEGMENTS_PATH, index=False)

    # Train Supervised Segment Classifier for Real-time Single Customer Scoring
    X_clf_train, X_clf_test, y_clf_train, y_clf_test = train_test_split(
        X_matrix, kmeans.labels_, test_size=0.20, random_state=42, stratify=kmeans.labels_
    )
    classifier = SegmentClassifierModel(n_estimators=100, max_depth=12, random_state=42)
    classifier.fit(X_clf_train, y_clf_train)
    classifier_eval = classifier.evaluate(X_clf_test, y_clf_test)
    classifier.save(CLASSIFIER_PATH)

    # Train Supervised Customer LTV Regressor
    y_ltv = df_processed["Customer_LTV"].values if "Customer_LTV" in df_processed.columns else df_processed["Monetary_Spend"].values * 1.5
    ltv_regressor = CustomerLTVRegressor(n_estimators=100, max_depth=6, random_state=42)
    ltv_eval = ltv_regressor.fit(X_matrix, y_ltv, feature_names=preprocessor.feature_names)
    ltv_regressor.save(LTV_REGRESSOR_PATH)

    feature_names = [col for col in df_processed.columns if col not in [
        "Customer_ID", "KMeans_Cluster", "DBSCAN_Cluster", "HAC_Cluster", "GMM_Cluster",
        "PCA_1", "PCA_2", "PCA_3", "Preferred_Channel", "Gender", "Region"
    ]]
    feature_importances = classifier.get_feature_importances(feature_names)

    models_dict = {
        "K-Means Clustering": kmeans,
        "Gaussian Mixture Model (GMM)": gmm,
        "Hierarchical Agglomerative": hac,
        "DBSCAN (Density-Based)": dbscan,
    }
    df_benchmark = ModelBenchmarkEvaluator.evaluate_all(X_matrix, models_dict)

    personas_data = CustomerPersonaGenerator.analyze_clusters(df_processed, cluster_col="KMeans_Cluster")
    metadata = {
        "project_positioning": "Production-style customer intelligence platform for segmentation, anomaly detection, real-time prediction, and marketing decisioning.",
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "data_source": data_source,
        "validation_warnings": validation_warnings,
        "total_samples": len(df_processed),
        "optimal_k": int(optimal_k),
        "production_model": "K-Means Clustering",
        "realtime_classifier": "RandomForestClassifier",
        "anomaly_model": "DBSCAN (Density-Based)",
        "classifier_metrics": classifier_eval,
        "classifier_feature_importances": feature_importances,
        "model_artifacts": {
            "preprocessing_pipeline": SCALER_PATH,
            "kmeans_model": KMEANS_MODEL_PATH,
            "classifier_model": CLASSIFIER_PATH,
            "pca_model": PCA_PATH,
        },
        "k_search_grid": k_search["grid_search"],
        "dbscan_search_grid": db_search["grid_search"],
        "kmeans_centroids": kmeans.cluster_centers_.tolist(),
        "pca_metadata": pca_meta,
        "benchmark_comparison": df_benchmark.to_dict(orient="records"),
        "persona_summary": personas_data,
        "segment_explanations": SegmentIntelligenceEngine.build_segment_explanations(df_processed, "KMeans_Cluster"),
        "campaign_recommendations": SegmentIntelligenceEngine.build_campaign_recommendations(df_processed, personas_data),
        "anomaly_summary": SegmentIntelligenceEngine.summarize_anomalies(df_processed, top_n=25),
        "stability_report": SegmentIntelligenceEngine.estimate_kmeans_stability(
            X_matrix, n_clusters=optimal_k, random_state=42, runs=5
        ),
    }

    os.makedirs(os.path.dirname(METADATA_PATH), exist_ok=True)
    with open(METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=2)

    return metadata
