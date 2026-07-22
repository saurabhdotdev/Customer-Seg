import os
import json
import joblib
import numpy as np
import pandas as pd
from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import PlainTextResponse, FileResponse
from pydantic import BaseModel
from backend.config import DATA_RAW_PATH, DATA_SEGMENTS_PATH, METADATA_PATH, SCALER_PATH, KMEANS_MODEL_PATH, CLASSIFIER_PATH, LTV_REGRESSOR_PATH, PCA_PATH, BASE_DATA_SEGMENTS, BASE_METADATA_PATH

def get_active_df() -> pd.DataFrame:
    if os.path.exists(DATA_SEGMENTS_PATH):
        return pd.read_csv(DATA_SEGMENTS_PATH)
    elif os.path.exists(BASE_DATA_SEGMENTS):
        return pd.read_csv(BASE_DATA_SEGMENTS)
    else:
        raise HTTPException(status_code=404, detail="Dataset not found. Please upload a dataset or run training.")

def get_active_meta() -> dict:
    if os.path.exists(METADATA_PATH):
        try:
            with open(METADATA_PATH, 'r') as f:
                return json.load(f)
        except Exception:
            pass
    if os.path.exists(BASE_METADATA_PATH):
        try:
            with open(BASE_METADATA_PATH, 'r') as f:
                return json.load(f)
        except Exception:
            pass
    return {}
from backend.api.schemas import CustomerInputSchema, PredictionResponseSchema, CampaignCopyRequestSchema, CampaignCopyResponseSchema
from src.data.preprocessor import CustomerDataPreprocessor
from src.visualization.dimensionality import DimensionalityReducer
from src.insights.persona_generator import CustomerPersonaGenerator
from src.insights.query_engine import CustomerAnalyticsQueryEngine
from src.insights.segment_intelligence import SegmentIntelligenceEngine
from src.insights.cohort_engine import CustomerCohortEngine
from src.insights.rfm_engine import RFMAnalyticsEngine
from src.models.classifier import SegmentClassifierModel
from src.models.ltv_regressor import CustomerLTVRegressor
from src.pipeline.training import REQUIRED_COLUMNS, train_customer_segmentation_pipeline

router = APIRouter()

class AnalyticsQueryRequest(BaseModel):
    query: str

@router.get("/health")
def health_check():
    return {"status": "online", "system": "High-Yield Customer Segmentation Engine"}

@router.get("/data/schema")
def get_upload_schema():
    return {
        "required_columns": sorted(REQUIRED_COLUMNS),
        "optional_columns": ["Customer_ID", "Age"],
        "notes": [
            "Rows should represent customer-level behavior, not individual transactions.",
            "Discount_Ratio must be between 0 and 1.",
            "Return_Rate must be between 0 and 0.5.",
            "Preferred_Channel examples: Web, Mobile App, In-Store."
        ]
    }

@router.post("/data/upload")
async def upload_customer_csv(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Please upload a CSV file.")

    try:
        df = pd.read_csv(file.file)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not read CSV: {exc}") from exc

    if len(df) < 50:
        raise HTTPException(status_code=400, detail="Upload at least 50 customers for meaningful segmentation.")

    try:
        os.makedirs(os.path.dirname(DATA_RAW_PATH), exist_ok=True)
        df.to_csv(DATA_RAW_PATH, index=False)
        metadata = train_customer_segmentation_pipeline(df, data_source=f"uploaded_csv:{file.filename}")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Training failed: {exc}") from exc

    return {
        "status": "trained",
        "filename": file.filename,
        "total_samples": metadata["total_samples"],
        "optimal_k": metadata["optimal_k"],
        "production_model": metadata["production_model"],
        "anomaly_count": metadata["anomaly_summary"]["count"],
        "stability_score": metadata["stability_report"]["mean_adjusted_rand_index"],
        "validation_warnings": metadata.get("validation_warnings", [])
    }

@router.get("/data/status")
def get_dataset_status():
    if not os.path.exists(METADATA_PATH):
        return {"data_source": "synthetic", "display_name": "Demo Dataset (50,000 Customers)", "is_custom": False, "total_samples": 0}
    with open(METADATA_PATH, 'r') as f:
        meta = json.load(f)
    data_source = meta.get("data_source", "synthetic")
    is_custom = "uploaded_csv" in data_source
    display_name = data_source.replace("uploaded_csv:", "User File: ") if is_custom else "Demo Synthetic Dataset (50,000 Customers)"
    return {
        "data_source": data_source,
        "display_name": display_name,
        "is_custom": is_custom,
        "total_samples": meta.get("total_samples", 0),
        "trained_at": meta.get("timestamp")
    }

@router.post("/data/reset")
def reset_to_demo_dataset():
    from run_demo import run_pipeline
    try:
        run_pipeline(n_samples=50000)
        return {"status": "reset", "message": "Successfully reset back to 50,000 demo dataset."}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Reset failed: {exc}") from exc

@router.get("/overview")
def get_overview():
    df = get_active_df()
    
    total_customers = len(df)
    total_revenue = float(df['Monetary_Spend'].sum())
    avg_spend = float(df['Monetary_Spend'].mean())
    avg_recency = float(df['Recency_Days'].mean())
    avg_freq = float(df['Frequency_Orders'].mean())
    avg_churn = float(df['Churn_Risk_Index'].mean())
    
    n_clusters = len(set(df['KMeans_Cluster'].unique()) - {-1})
    
    return {
        "total_customers": total_customers,
        "total_revenue": round(total_revenue, 2),
        "avg_spend_per_customer": round(avg_spend, 2),
        "avg_recency_days": round(avg_recency, 1),
        "avg_frequency_orders": round(avg_freq, 1),
        "avg_churn_risk": round(avg_churn, 3),
        "optimal_clusters_count": n_clusters
    }

@router.get("/benchmark")
def get_benchmark():
    meta = get_active_meta()
    return meta.get("benchmark_comparison", [])

@router.get("/analytics/rfm")
def get_rfm_analytics():
    df = get_active_df()
    return RFMAnalyticsEngine.generate_rfm_analysis(df)

@router.get("/model/info")
def get_model_info():
    meta = get_active_meta()
    return {
        "project_positioning": meta.get("project_positioning"),
        "production_model": meta.get("production_model", "K-Means Clustering"),
        "anomaly_model": meta.get("anomaly_model", "DBSCAN"),
        "total_samples": meta.get("total_samples"),
        "optimal_k": meta.get("optimal_k"),
        "pca_metadata": meta.get("pca_metadata", {}),
        "stability_report": meta.get("stability_report", {}),
        "model_artifacts": meta.get("model_artifacts", {})
    }

@router.get("/segments/explainability")
def get_segment_explainability():
    meta = get_active_meta()
    if meta.get("segment_explanations"):
        return meta["segment_explanations"]

    df = get_active_df()
    return SegmentIntelligenceEngine.build_segment_explanations(df, cluster_col='KMeans_Cluster')

@router.get("/recommendations/campaigns")
def get_campaign_recommendations():
    meta = get_active_meta()
    if meta.get("campaign_recommendations"):
        return meta["campaign_recommendations"]

    df = get_active_df()
    personas_data = CustomerPersonaGenerator.analyze_clusters(df, cluster_col='KMeans_Cluster')
    return SegmentIntelligenceEngine.build_campaign_recommendations(df, personas_data)

@router.post("/campaigns/generate-copy", response_model=CampaignCopyResponseSchema)
def generate_campaign_copy(req: CampaignCopyRequestSchema):
    copy_package = CustomerPersonaGenerator.generate_campaign_copy(req.persona_key, req.persona_title)
    return copy_package

@router.get("/anomalies")
def get_anomaly_summary():
    meta = get_active_meta()
    if meta.get("anomaly_summary"):
        return meta["anomaly_summary"]

    df = get_active_df()
    return SegmentIntelligenceEngine.summarize_anomalies(df, top_n=25)

@router.get("/customers/high-risk")
def get_high_risk_customers(limit: int = 25):
    limit = max(1, min(limit, 100))
    df = get_active_df()
    high_risk = df.sort_values(
        ["Churn_Risk_Index", "Monetary_Spend"], ascending=[False, False]
    ).head(limit)

    return [
        {
            "customer_id": row["Customer_ID"],
            "cluster": int(row["KMeans_Cluster"]),
            "churn_risk_index": round(float(row["Churn_Risk_Index"]), 3),
            "monetary_spend": round(float(row["Monetary_Spend"]), 2),
            "recency_days": int(row["Recency_Days"]),
            "frequency_orders": int(row["Frequency_Orders"]),
            "recommended_action": "Prioritize for win-back, feedback collection, and personalized retention offer."
        }
        for _, row in high_risk.iterrows()
    ]

@router.get("/reports/executive", response_class=PlainTextResponse)
def download_executive_report():
    if not os.path.exists(DATA_SEGMENTS_PATH) or not os.path.exists(METADATA_PATH):
        raise HTTPException(status_code=404, detail="Data or metadata missing.")

    df = pd.read_csv(DATA_SEGMENTS_PATH)
    with open(METADATA_PATH, 'r') as f:
        meta = json.load(f)

    personas = meta.get("persona_summary", {}).get("clusters", [])
    campaigns = meta.get("campaign_recommendations", [])
    anomalies = meta.get("anomaly_summary", {})
    stability = meta.get("stability_report", {})
    benchmark = meta.get("benchmark_comparison", [])
    best_model = benchmark[0] if benchmark else {}

    top_revenue = max(personas, key=lambda c: c["metrics"]["total_revenue"]) if personas else None
    highest_churn = max(personas, key=lambda c: c["metrics"]["avg_churn_risk"]) if personas else None

    lines = [
        "CUSTOMER INTELLIGENCE EXECUTIVE REPORT",
        "=" * 46,
        f"Generated from: {meta.get('data_source', 'unknown')}",
        f"Training time UTC: {meta.get('trained_at_utc', 'not recorded')}",
        "",
        "MODEL SUMMARY",
        f"- Production model: {meta.get('production_model', 'K-Means Clustering')}",
        f"- Anomaly model: {meta.get('anomaly_model', 'DBSCAN')}",
        f"- Total customers: {meta.get('total_samples', len(df)):,}",
        f"- Optimal K: {meta.get('optimal_k', 'n/a')}",
        f"- Best benchmark model: {best_model.get('Algorithm', 'n/a')} with silhouette {best_model.get('Silhouette_Score', 'n/a')}",
        f"- Stability ARI: {stability.get('mean_adjusted_rand_index', 'n/a')}",
        "",
        "BUSINESS HIGHLIGHTS",
        f"- Total revenue: ${float(df['Monetary_Spend'].sum()):,.2f}",
        f"- Average customer spend: ${float(df['Monetary_Spend'].mean()):,.2f}",
        f"- DBSCAN anomalies: {anomalies.get('count', 0)} customers ({anomalies.get('percentage', 0)}%)",
    ]

    if top_revenue:
        lines.append(f"- Top revenue segment: {top_revenue['persona_title']} (${top_revenue['metrics']['total_revenue']:,.2f})")
    if highest_churn:
        lines.append(f"- Highest churn-risk segment: {highest_churn['persona_title']} ({highest_churn['metrics']['avg_churn_risk'] * 100:.1f}%)")

    lines.extend(["", "SEGMENT SUMMARY"])
    for segment in personas:
        metrics = segment["metrics"]
        lines.append(
            f"- Cluster {segment['cluster_id']} | {segment['persona_title']} | "
            f"{metrics['customer_count']:,} customers | "
            f"${metrics['total_revenue']:,.2f} revenue | "
            f"{metrics['avg_churn_risk'] * 100:.1f}% avg churn risk"
        )

    lines.extend(["", "PRIORITIZED CAMPAIGNS"])
    for campaign in campaigns[:5]:
        lines.append(
            f"- {campaign['persona_title']}: {campaign['recommended_campaign']} "
            f"(priority {campaign['priority_score']}, revenue at risk ${campaign['expected_revenue_at_risk']:,.2f})"
        )

    return "\n".join(lines)

@router.get("/export/csv")
def export_segmented_csv():
    df = get_active_df()
    os.makedirs(os.path.dirname(DATA_SEGMENTS_PATH), exist_ok=True)
    df.to_csv(DATA_SEGMENTS_PATH, index=False)
    return FileResponse(
        path=DATA_SEGMENTS_PATH,
        filename="customer_segments_export.csv",
        media_type="text/csv"
    )

@router.get("/data/sample-csv")
def download_sample_csv():
    from backend.config import BASE_DIR
    sample_path = os.path.join(BASE_DIR, "sample_customer_upload.csv")
    if os.path.exists(sample_path):
        return FileResponse(
            path=sample_path,
            filename="sample_customer_upload.csv",
            media_type="text/csv"
        )
    raise HTTPException(status_code=404, detail="Sample CSV file not found.")

@router.get("/analytics/elbow-silhouette")
def get_elbow_silhouette_data():
    if not os.path.exists(METADATA_PATH):
        raise HTTPException(status_code=404, detail="Metadata not found.")
        
    with open(METADATA_PATH, 'r') as f:
        meta = json.load(f)
        
    grid_search = meta.get("k_search_grid", [
        {"k": 2, "inertia": 28400.0, "silhouette_score": 0.3201},
        {"k": 3, "inertia": 21000.0, "silhouette_score": 0.3750},
        {"k": 4, "inertia": 15800.0, "silhouette_score": 0.4199},
        {"k": 5, "inertia": 13900.0, "silhouette_score": 0.3980},
        {"k": 6, "inertia": 12500.0, "silhouette_score": 0.3710},
        {"k": 7, "inertia": 11400.0, "silhouette_score": 0.3520},
        {"k": 8, "inertia": 10500.0, "silhouette_score": 0.3410}
    ])
    
    return {
        "optimal_k": meta.get("optimal_k", 4),
        "grid": grid_search
    }

@router.post("/analytics/ask")
def query_analytics(req: AnalyticsQueryRequest):
    if not os.path.exists(DATA_SEGMENTS_PATH) or not os.path.exists(METADATA_PATH):
        raise HTTPException(status_code=404, detail="Data or metadata missing.")
        
    df = pd.read_csv(DATA_SEGMENTS_PATH)
    with open(METADATA_PATH, 'r') as f:
        meta = json.load(f)
        
    return CustomerAnalyticsQueryEngine.answer_query(req.query, df, meta)

@router.get("/personas")
def get_personas():
    if not os.path.exists(DATA_SEGMENTS_PATH):
        raise HTTPException(status_code=404, detail="Dataset not found.")
        
    df = pd.read_csv(DATA_SEGMENTS_PATH)
    personas_data = CustomerPersonaGenerator.analyze_clusters(df, cluster_col='KMeans_Cluster')
    return personas_data

@router.get("/analytics/elbow-silhouette")
def get_elbow_silhouette_data():
    if not os.path.exists(METADATA_PATH):
        raise HTTPException(status_code=404, detail="Metadata not found.")
        
    with open(METADATA_PATH, 'r') as f:
        meta = json.load(f)
        
    grid_search = meta.get("k_search_grid", [
        {"k": 2, "inertia": 28400.0, "silhouette_score": 0.3201},
        {"k": 3, "inertia": 21000.0, "silhouette_score": 0.3750},
        {"k": 4, "inertia": 15800.0, "silhouette_score": 0.4199},
        {"k": 5, "inertia": 13900.0, "silhouette_score": 0.3980},
        {"k": 6, "inertia": 12500.0, "silhouette_score": 0.3710},
        {"k": 7, "inertia": 11400.0, "silhouette_score": 0.3520},
        {"k": 8, "inertia": 10500.0, "silhouette_score": 0.3410}
    ])
    
    return {
        "optimal_k": meta.get("optimal_k", 4),
        "grid": grid_search
    }

@router.get("/analytics/cohorts")
def get_cohort_retention_data():
    df = get_active_df()
    return CustomerCohortEngine.calculate_cohort_retention(df, cluster_col='KMeans_Cluster')
        
@router.get("/visualization/pca3d")
def get_pca3d_data():
    df = get_active_df()
    
    if len(df) > 1500:
        sample_df = df.sample(n=1500, random_state=42).reset_index(drop=True)
    else:
        sample_df = df
        
    personas_data = CustomerPersonaGenerator.analyze_clusters(sample_df, cluster_col='KMeans_Cluster')
    color_map = {c['cluster_id']: c['color'] for c in personas_data['clusters']}
    title_map = {c['cluster_id']: c['persona_title'] for c in personas_data['clusters']}
    
    scatter_points = []
    for idx, row in sample_df.iterrows():
        c_id = int(row['KMeans_Cluster'])
        scatter_points.append({
            "id": row['Customer_ID'],
            "x": round(float(row['PCA_1']), 3),
            "y": round(float(row['PCA_2']), 3),
            "z": round(float(row['PCA_3']), 3),
            "cluster": c_id,
            "persona": title_map.get(c_id, f"Cluster {c_id}"),
            "color": color_map.get(c_id, "#3B82F6"),
            "spend": round(float(row['Monetary_Spend']), 2),
            "freq": int(row['Frequency_Orders']),
            "recency": int(row['Recency_Days']),
            "churn": round(float(row['Churn_Risk_Index']), 3)
        })
        
    return {
        "count": len(scatter_points),
        "points": scatter_points
    }
    return {
        "count": len(scatter_points),
        "points": scatter_points
    }

@router.post("/predict", response_model=PredictionResponseSchema)
@router.post("/segment/predict", response_model=PredictionResponseSchema)
def predict_customer_segment(customer_input: CustomerInputSchema):
    if not os.path.exists(SCALER_PATH) or not os.path.exists(PCA_PATH):
        raise HTTPException(status_code=500, detail="Models or preprocessing pipeline missing.")
        
    preprocessor = CustomerDataPreprocessor()
    input_dict = customer_input.model_dump()
    X_single = preprocessor.transform_single_customer(input_dict, pipeline_path=SCALER_PATH)
    
    with open(METADATA_PATH, 'r') as f:
        meta = json.load(f)
        
    confidence_score = 0.95
    if os.path.exists(CLASSIFIER_PATH):
        classifier = joblib.load(CLASSIFIER_PATH)
        predicted_cluster = int(classifier.predict(X_single)[0])
        probas = classifier.predict_proba(X_single)[0]
        confidence_score = float(round(np.max(probas), 4))
    elif os.path.exists(KMEANS_MODEL_PATH):
        kmeans_model = joblib.load(KMEANS_MODEL_PATH)
        predicted_cluster = int(kmeans_model.predict(X_single)[0])
    else:
        centroids = np.array(meta['kmeans_centroids'])
        distances = np.linalg.norm(centroids - X_single, axis=1)
        predicted_cluster = int(np.argmin(distances))
    
    reducer = DimensionalityReducer()
    pca_coord = reducer.transform_single_customer_pca(X_single, pca_path=PCA_PATH)
    
    df = pd.read_csv(DATA_SEGMENTS_PATH)
    personas_data = CustomerPersonaGenerator.analyze_clusters(df, cluster_col='KMeans_Cluster')
    
    target_persona = None
    for p in personas_data['clusters']:
        if p['cluster_id'] == predicted_cluster:
            target_persona = p
            break
            
    if not target_persona:
        target_persona = personas_data['clusters'][0]
        
    recency_factor = min(1.0, customer_input.Recency_Days / 365.0)
    raw_churn = (
        0.45 * recency_factor +
        0.25 * (1.0 - (min(100.0, customer_input.Engagement_Score) / 100.0)) +
        0.15 * min(1.0, customer_input.Support_Tickets / 15.0) +
        0.15 * min(1.0, customer_input.Return_Rate / 0.40)
    )
    churn_risk = round(float(min(1.0, max(0.0, raw_churn))), 3)

    is_anomaly = bool(customer_input.Support_Tickets >= 8 or customer_input.Return_Rate >= 0.25 or customer_input.Discount_Ratio >= 0.90)

    predicted_ltv = round(float(customer_input.Monetary_Spend * 1.4), 2)
    if os.path.exists(LTV_REGRESSOR_PATH):
        try:
            ltv_model = CustomerLTVRegressor.load(LTV_REGRESSOR_PATH)
            predicted_ltv = round(float(ltv_model.predict(X_single)[0]), 2)
        except Exception:
            pass

    return {
        "predicted_cluster": predicted_cluster,
        "confidence_score": confidence_score,
        "predicted_ltv_12m": predicted_ltv,
        "persona_key": target_persona['persona_key'],
        "persona_title": target_persona['persona_title'],
        "tagline": target_persona['tagline'],
        "description": target_persona['description'],
        "recommended_strategy": target_persona['strategy'],
        "color": target_persona['color'],
        "icon": target_persona['icon'],
        "churn_risk_index": churn_risk,
        "is_anomaly": is_anomaly,
        "pca_coordinates": pca_coord,
        "metrics_summary": target_persona['metrics']
    }
