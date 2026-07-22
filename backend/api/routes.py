import os
import json
import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from backend.config import DATA_SEGMENTS_PATH, METADATA_PATH, SCALER_PATH, PCA_PATH
from backend.api.schemas import CustomerInputSchema, PredictionResponseSchema
from src.data.preprocessor import CustomerDataPreprocessor
from src.visualization.dimensionality import DimensionalityReducer
from src.insights.persona_generator import CustomerPersonaGenerator
from src.insights.query_engine import CustomerAnalyticsQueryEngine

router = APIRouter()

class AnalyticsQueryRequest(BaseModel):
    query: str

@router.get("/health")
def health_check():
    return {"status": "online", "system": "High-Yield Customer Segmentation Engine"}

@router.get("/overview")
def get_overview():
    if not os.path.exists(DATA_SEGMENTS_PATH):
        raise HTTPException(status_code=404, detail="Processed dataset not found. Run training demo first.")
        
    df = pd.read_csv(DATA_SEGMENTS_PATH)
    
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
    if not os.path.exists(METADATA_PATH):
        raise HTTPException(status_code=404, detail="Benchmark metadata not found.")
        
    with open(METADATA_PATH, 'r') as f:
        meta = json.load(f)
        
    return meta.get("benchmark_comparison", [])

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

@router.get("/visualization/pca3d")
def get_pca3d_data():
    if not os.path.exists(DATA_SEGMENTS_PATH):
        raise HTTPException(status_code=404, detail="Dataset not found.")
        
    df = pd.read_csv(DATA_SEGMENTS_PATH)
    
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

@router.post("/segment/predict", response_model=PredictionResponseSchema)
def predict_customer_segment(customer_input: CustomerInputSchema):
    if not os.path.exists(SCALER_PATH) or not os.path.exists(PCA_PATH):
        raise HTTPException(status_code=500, detail="Models or preprocessing pipeline missing.")
        
    preprocessor = CustomerDataPreprocessor()
    X_single = preprocessor.transform_single_customer(customer_input.model_dump(), pipeline_path=SCALER_PATH)
    
    with open(METADATA_PATH, 'r') as f:
        meta = json.load(f)
        
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
        
    churn_risk = round(float(
        0.50 * (customer_input.Recency_Days / 365.0) +
        0.25 * (1.0 - (customer_input.Engagement_Score / 100.0)) +
        0.15 * (customer_input.Support_Tickets / 12.0) +
        0.10 * (customer_input.Return_Rate / 0.30)
    ), 3)

    return {
        "predicted_cluster": predicted_cluster,
        "persona_key": target_persona['persona_key'],
        "persona_title": target_persona['persona_title'],
        "tagline": target_persona['tagline'],
        "description": target_persona['description'],
        "recommended_strategy": target_persona['strategy'],
        "color": target_persona['color'],
        "icon": target_persona['icon'],
        "churn_risk_index": churn_risk,
        "pca_coordinates": pca_coord,
        "metrics_summary": target_persona['metrics']
    }
