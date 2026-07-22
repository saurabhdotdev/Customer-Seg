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
    "days_since_last_order": "Recency_Days",
    "last_purchase_days": "Recency_Days",
    "recency_in_days": "Recency_Days",
    "days_ago": "Recency_Days",
    "days_inactive": "Recency_Days",
    "last_seen_days": "Recency_Days",
    "frequency": "Frequency_Orders",
    "frequency_orders": "Frequency_Orders",
    "orders": "Frequency_Orders",
    "total_orders": "Frequency_Orders",
    "order_count": "Frequency_Orders",
    "num_orders": "Frequency_Orders",
    "number_of_orders": "Frequency_Orders",
    "purchases": "Frequency_Orders",
    "total_purchases": "Frequency_Orders",
    "purchase_count": "Frequency_Orders",
    "transactions": "Frequency_Orders",
    "total_transactions": "Frequency_Orders",
    "num_transactions": "Frequency_Orders",
    "monetary": "Monetary_Spend",
    "monetary_spend": "Monetary_Spend",
    "spend": "Monetary_Spend",
    "total_spend": "Monetary_Spend",
    "revenue": "Monetary_Spend",
    "total_revenue": "Monetary_Spend",
    "amount": "Monetary_Spend",
    "total_amount": "Monetary_Spend",
    "sales": "Monetary_Spend",
    "total_sales": "Monetary_Spend",
    "gross_revenue": "Monetary_Spend",
    "spent": "Monetary_Spend",
    "customer_id": "Customer_ID",
    "id": "Customer_ID",
    "user_id": "Customer_ID",
    "client_id": "Customer_ID",
    "account_id": "Customer_ID",
    "cust_id": "Customer_ID",
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

    # 1. Exact & normalized alias mapping
    rename_dict = {}
    for col in df_clean.columns:
        col_clean = str(col).strip().lower().replace(" ", "_").replace("-", "_")
        col_clean = "".join([c for c in col_clean if c.isalnum() or c == "_"])
        if col_clean in ALIAS_MAP:
            rename_dict[col] = ALIAS_MAP[col_clean]

    # 2. Fuzzy substring fallback if core RFM columns unmapped
    for col in df_clean.columns:
        if col in rename_dict or col in CORE_REQUIRED:
            continue
        c_lower = str(col).strip().lower()
        mapped_targets = set(rename_dict.values()).union(set(df_clean.columns))
        
        if "Monetary_Spend" not in mapped_targets and ("monetar" in c_lower or "spend" in c_lower or "revenu" in c_lower or "amount" in c_lower or "sales" in c_lower or "val" in c_lower or "money" in c_lower or "paid" in c_lower or "price" in c_lower or "cost" in c_lower or "dollar" in c_lower):
            rename_dict[col] = "Monetary_Spend"
        elif "Frequency_Orders" not in mapped_targets and ("frequenc" in c_lower or "order" in c_lower or "purchase" in c_lower or "transact" in c_lower or "bought" in c_lower or "qty" in c_lower or "quantity" in c_lower or "count" in c_lower or "times" in c_lower):
            rename_dict[col] = "Frequency_Orders"
        elif "Recency_Days" not in mapped_targets and ("recenc" in c_lower or "last" in c_lower or "days" in c_lower or "recent" in c_lower or "ago" in c_lower or "since" in c_lower or "date" in c_lower or "time" in c_lower):
            rename_dict[col] = "Recency_Days"
        elif "Customer_ID" not in mapped_targets and ("cust" in c_lower or "user" in c_lower or "client" in c_lower or "account" in c_lower or "email" in c_lower or "name" in c_lower):
            rename_dict[col] = "Customer_ID"

    if rename_dict:
        df_clean.rename(columns=rename_dict, inplace=True)
        warnings.append(f"Auto-mapped column names: {rename_dict}")

    # 3. Position & magnitude fallback if numeric columns exist
    missing_core = CORE_REQUIRED - set(df_clean.columns)
    if missing_core:
        num_cols = df_clean.select_dtypes(include=["number"]).columns.tolist()
        if len(num_cols) >= 3:
            col_means = {c: df_clean[c].mean() for c in num_cols}
            sorted_by_mean = sorted(num_cols, key=lambda c: col_means[c], reverse=True)
            
            if "Monetary_Spend" in missing_core:
                spend_col = sorted_by_mean[0]
                df_clean.rename(columns={spend_col: "Monetary_Spend"}, inplace=True)
                warnings.append(f"Auto-inferred '{spend_col}' as Monetary_Spend based on value magnitude.")
                sorted_by_mean.remove(spend_col)
                missing_core.remove("Monetary_Spend")

            if "Frequency_Orders" in missing_core and sorted_by_mean:
                freq_col = sorted_by_mean[-1]
                df_clean.rename(columns={freq_col: "Frequency_Orders"}, inplace=True)
                warnings.append(f"Auto-inferred '{freq_col}' as Frequency_Orders based on order count scale.")
                sorted_by_mean.remove(freq_col)
                missing_core.remove("Frequency_Orders")

            if "Recency_Days" in missing_core and sorted_by_mean:
                rec_col = sorted_by_mean[0]
                df_clean.rename(columns={rec_col: "Recency_Days"}, inplace=True)
                warnings.append(f"Auto-inferred '{rec_col}' as Recency_Days based on days scale.")
                missing_core.remove("Recency_Days")

    missing_core_final = sorted(CORE_REQUIRED - set(df_clean.columns))
    if missing_core_final:
        raise ValueError(
            f"Missing required core RFM columns: {', '.join(missing_core_final)}. "
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


def train_customer_segmentation_pipeline(
    df_raw: pd.DataFrame,
    data_source: str = "synthetic",
    scaler_path: str = None,
    kmeans_path: str = None,
    classifier_path: str = None,
    ltv_path: str = None,
    pca_path: str = None,
    processed_path: str = None,
    segments_path: str = None,
    metadata_path: str = None,
) -> dict:
    scaler_target = scaler_path or SCALER_PATH
    kmeans_target = kmeans_path or KMEANS_MODEL_PATH
    classifier_target = classifier_path or CLASSIFIER_PATH
    ltv_target = ltv_path or LTV_REGRESSOR_PATH
    pca_target = pca_path or PCA_PATH
    processed_target = processed_path or DATA_PROCESSED_PATH
    segments_target = segments_path or DATA_SEGMENTS_PATH
    metadata_target = metadata_path or METADATA_PATH

    df_raw, validation_warnings = validate_customer_dataframe(df_raw)

    preprocessor = CustomerDataPreprocessor(scaler_save_path=scaler_target)
    X_matrix, df_processed = preprocessor.fit_transform(df_raw)

    os.makedirs(os.path.dirname(processed_target), exist_ok=True)
    df_processed.to_csv(processed_target, index=False)

    kmeans = KMeansSegmentationModel()
    k_search = kmeans.find_optimal_k(X_matrix, range(2, 10))
    optimal_k = k_search["best_k"]
    kmeans.n_clusters = optimal_k
    kmeans.fit(X_matrix)
    os.makedirs(os.path.dirname(kmeans_target), exist_ok=True)
    joblib.dump(kmeans.model, kmeans_target)

    dbscan = DBSCANSegmentationModel()
    db_search = dbscan.tune_hyperparameters(X_matrix)
    dbscan.eps = db_search["best_params"]["eps"]
    dbscan.min_samples = db_search["best_params"]["min_samples"]
    dbscan.fit(X_matrix)

    hac = HierarchicalSegmentationModel(n_clusters=optimal_k, linkage="ward")
    hac.fit(X_matrix)

    gmm = GMMSegmentationModel(n_components=optimal_k)
    gmm.fit(X_matrix)

    reducer = DimensionalityReducer(n_components_pca=3, pca_save_path=pca_target)
    X_pca, pca_meta = reducer.fit_transform_pca(X_matrix)

    df_processed["KMeans_Cluster"] = kmeans.labels_
    df_processed["DBSCAN_Cluster"] = dbscan.labels_
    df_processed["HAC_Cluster"] = hac.labels_
    df_processed["GMM_Cluster"] = gmm.labels_
    df_processed["PCA_1"] = X_pca[:, 0]
    df_processed["PCA_2"] = X_pca[:, 1]
    df_processed["PCA_3"] = X_pca[:, 2]
    os.makedirs(os.path.dirname(segments_target), exist_ok=True)
    df_processed.to_csv(segments_target, index=False)

    # Train Supervised Segment Classifier for Real-time Single Customer Scoring
    X_clf_train, X_clf_test, y_clf_train, y_clf_test = train_test_split(
        X_matrix, kmeans.labels_, test_size=0.20, random_state=42, stratify=kmeans.labels_
    )
    classifier = SegmentClassifierModel(n_estimators=100, max_depth=12, random_state=42)
    classifier.fit(X_clf_train, y_clf_train)
    classifier_eval = classifier.evaluate(X_clf_test, y_clf_test)
    classifier.save(classifier_target)

    # Train Supervised Customer LTV Regressor
    y_ltv = df_processed["Customer_LTV"].values if "Customer_LTV" in df_processed.columns else df_processed["Monetary_Spend"].values * 1.5
    ltv_regressor = CustomerLTVRegressor(n_estimators=100, max_depth=6, random_state=42)
    ltv_eval = ltv_regressor.fit(X_matrix, y_ltv, feature_names=preprocessor.feature_names)
    ltv_regressor.save(ltv_target)

    feature_names = [col for col in df_processed.columns if col not in [
        "Customer_ID", "KMeans_Cluster", "DBSCAN_Cluster", "HAC_Cluster", "GMM_Cluster",
        "PCA_1", "PCA_2", "PCA_3", "Preferred_Channel", "Gender", "Region"
    ]]
    feature_importances = classifier.get_feature_importances(feature_names)

    from src.models.anomaly_detector import CustomerAnomalyDetector
    anomaly_detector = CustomerAnomalyDetector(contamination=0.03)
    df_processed = anomaly_detector.fit_predict(df_processed)
    anomaly_target = os.path.join(os.path.dirname(kmeans_target), "anomaly_detector.joblib")
    anomaly_detector.save(anomaly_target)

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
        "anomaly_model": "IsolationForest Anomaly Detector",
        "anomaly_summary": {
            "count": int(df_processed["Is_Anomaly"].sum()),
            "percentage": round(float(df_processed["Is_Anomaly"].mean() * 100), 2),
            "types_breakdown": df_processed[df_processed["Is_Anomaly"]]["Anomaly_Type"].value_counts().to_dict()
        },
        "classifier_metrics": classifier_eval,
        "classifier_feature_importances": feature_importances,
        "model_artifacts": {
            "preprocessing_pipeline": scaler_target,
            "kmeans_model": kmeans_target,
            "classifier_model": classifier_target,
            "anomaly_model": anomaly_target,
            "pca_model": pca_target,
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

    os.makedirs(os.path.dirname(metadata_target), exist_ok=True)
    with open(metadata_target, "w") as f:
        json.dump(metadata, f, indent=2)

    return metadata
